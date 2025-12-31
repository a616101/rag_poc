"""
Chunk Expander 節點：對檢索結果進行 Adaptive Chunk Expansion。

功能：
1. 根據文件大小智慧決定需要補取哪些相鄰 chunks
2. 從 Qdrant 取得缺少的 chunks
3. 將擴展後的 chunks 格式化為 context 字串

策略（calculate_needed_chunk_indices）：
- 小文件（≤3 chunks）：取全部
- 中型文件（≤8 chunks）：每個命中 ±1
- 大型文件（>8 chunks）：智慧窗口
    - 開頭命中：往後擴展
    - 結尾命中：往前擴展
    - 中間命中：±1

前置節點：reranker
後續節點：result_evaluator
"""

from collections import defaultdict
from typing import Callable, Dict, List, cast

from langgraph.config import get_stream_writer
from langfuse import get_client
from loguru import logger

from chatbot_rag.llm import State
from chatbot_rag.services.qdrant_service import qdrant_service
from ...types import StateDict
from ...constants import AskStreamStages
from ...events import emit_node_event
from ...utils import calculate_needed_chunk_indices


def _get_langfuse_client():
    """安全取得 Langfuse client，避免初始化失敗影響主流程。"""
    try:
        return get_client()
    except Exception:
        return None


def _adaptive_expand_chunks(
    matched_docs: List[dict],
) -> List[dict]:
    """
    對檢索結果進行 Adaptive Chunk Expansion。

    根據每個文件的大小（total_chunks）和命中位置，
    智慧決定需要擴展哪些 chunks。

    Args:
        matched_docs: 檢索到的文件列表

    Returns:
        List[dict]: 擴展後的文件列表，按 filename 和 chunk_index 排序
    """
    if not matched_docs:
        return []

    # 按檔案分組
    by_file: Dict[str, List[dict]] = defaultdict(list)
    for doc in matched_docs:
        filename = doc.get("filename", "unknown")
        by_file[filename].append(doc)

    expanded_results: List[dict] = []

    for filename, file_docs in by_file.items():
        # 取得文件的 total_chunks（從第一個 doc 的 metadata）
        first_doc = file_docs[0]
        metadata = first_doc.get("metadata") or {}
        total_chunks = metadata.get("total_chunks", 1)
        hit_indices = [doc.get("chunk_index", 0) for doc in file_docs]

        # 使用共用函數計算需要的 indices
        needed_indices = calculate_needed_chunk_indices(hit_indices, total_chunks)

        # 已經命中的 indices
        existing_indices = set(hit_indices)

        # 需要額外取得的 indices
        missing_indices = needed_indices - existing_indices

        logger.debug(
            "[CHUNK_EXPANDER] Adaptive expansion for {}: "
            "total={}, hits={}, needed={}, missing={}",
            filename,
            total_chunks,
            sorted(existing_indices),
            sorted(needed_indices),
            sorted(missing_indices),
        )

        # 取得缺少的 chunks
        if missing_indices:
            additional_chunks = qdrant_service.fetch_chunks_by_indices(
                filename=filename,
                indices=missing_indices,
            )

            # 將額外取得的 chunks 轉換為與 matched_docs 相同的格式
            for chunk in additional_chunks:
                payload = chunk.get("payload", {})
                expanded_results.append({
                    "content": payload.get("contextualized_text") or payload.get("text", ""),
                    "source": payload.get("source", "unknown"),
                    "filename": filename,
                    "chunk_index": chunk.get("chunk_index", 0),
                    "score": 0.0,  # 額外取得的 chunks 沒有相似度分數
                    "metadata": payload,
                    "_expanded": True,  # 標記為擴展取得
                })

        # 加入原本命中的 docs
        expanded_results.extend(file_docs)

    # 按 filename 和 chunk_index 排序，確保同文件的 chunks 連續
    expanded_results.sort(key=lambda d: (d.get("filename", ""), d.get("chunk_index", 0)))

    if len(expanded_results) > len(matched_docs):
        logger.info(
            "[CHUNK_EXPANDER] Adaptive expansion: {} -> {} chunks",
            len(matched_docs),
            len(expanded_results),
        )

    return expanded_results


def _format_expanded_docs_to_context(docs: List[dict]) -> str:
    """
    將擴展後的 docs 格式化為 context 字串。

    同一文件的 chunks 會合併顯示，避免重複的 header。

    Args:
        docs: 擴展後的文件列表（已按 filename, chunk_index 排序）

    Returns:
        str: 格式化的 context 字串
    """
    if not docs:
        return ""

    # 按檔案分組
    by_file: Dict[str, List[dict]] = defaultdict(list)
    for doc in docs:
        filename = doc.get("filename", "unknown")
        by_file[filename].append(doc)

    context_parts = []
    doc_idx = 1

    for filename, file_docs in by_file.items():
        # 取得代表性的 metadata（使用第一個有分數的 doc）
        scored_docs = [d for d in file_docs if d.get("score", 0) > 0 or d.get("rerank_score", 0) > 0]
        representative = scored_docs[0] if scored_docs else file_docs[0]

        metadata = representative.get("metadata") or {}
        entry_type = metadata.get("entry_type", "")
        module = metadata.get("module", "")
        # 優先使用 rerank_score，否則使用 embedding score
        max_score = max(
            (d.get("rerank_score", 0) or d.get("score", 0) for d in file_docs),
            default=0
        )
        total_chunks = metadata.get("total_chunks", len(file_docs))
        chunk_indices = sorted(d.get("chunk_index", 0) for d in file_docs)

        # Header
        header_parts = [f"[Document {doc_idx}]"]
        header_parts.append(f"(來源: {filename}")
        if entry_type:
            header_parts.append(f" 類型: {entry_type}")
        if module:
            header_parts.append(f" 模組: {module}")
        if max_score > 0:
            header_parts.append(f" 相關度: {max_score:.2f}")
        header_parts.append(f" chunks: {chunk_indices}/{total_chunks})")

        # 合併 chunks 內容（按 chunk_index 排序）
        sorted_docs = sorted(file_docs, key=lambda d: d.get("chunk_index", 0))
        contents = [d.get("content", "") for d in sorted_docs]
        body = "\n".join(contents)

        context_parts.append(f"{' '.join(header_parts)}\n{body}\n")
        doc_idx += 1

    return "\n".join(context_parts)


def build_chunk_expander_node() -> Callable[[State], State]:
    """
    建立 Chunk Expander 節點。

    此節點負責：
    1. 對 reranker（或 tool_executor）的結果進行 Adaptive Chunk Expansion
    2. 格式化為 context 字串
    3. 更新 state["context"] 和 state["retrieval"]["expanded_docs"]

    Returns:
        Callable: Chunk Expander 節點函數
    """

    async def chunk_expander_node(state: State) -> State:
        writer = get_stream_writer()
        langfuse = _get_langfuse_client()

        state_dict = cast(StateDict, state)
        new_state = cast(State, dict(state))

        # 取得檢索狀態
        retrieval_state = dict(state_dict.get("retrieval") or {})

        # 優先使用 reranked_docs（來自 reranker 節點），fallback 到 raw_docs
        docs = retrieval_state.get("reranked_docs") or retrieval_state.get("raw_docs") or []
        input_source = "reranked_docs" if retrieval_state.get("reranked_docs") else "raw_docs"

        # 收集 input 資訊
        input_files = list({doc.get("filename", "unknown") for doc in docs})
        input_count = len(docs)

        # 準備 Langfuse span input
        span_input = {
            "source": input_source,
            "input_chunks_count": input_count,
            "input_files_count": len(input_files),
            "input_files": input_files[:10],
            "input_chunks": [
                {
                    "filename": doc.get("filename", "unknown"),
                    "chunk_index": doc.get("chunk_index", 0),
                    "score": round(doc.get("rerank_score", 0) or doc.get("score", 0), 3),
                    "total_chunks": (doc.get("metadata") or {}).get("total_chunks", 1),
                }
                for doc in docs[:20]  # 最多顯示 20 個 chunks
            ],
        }

        emit_node_event(
            writer,
            node="chunk_expander",
            phase="retrieval",
            payload={
                "channel": "status",
                "stage": AskStreamStages.CHUNK_EXPANDER_START,
                "input_source": input_source,
                "input_chunks_count": input_count,
                "input_files_count": len(input_files),
                "input_files": input_files[:10],
            },
        )

        if not docs:
            logger.debug("[CHUNK_EXPANDER] No documents to expand")
            # 記錄空輸入到 Langfuse
            if langfuse:
                with langfuse.start_as_current_span(
                    name="chunk_expander",
                    input=span_input,
                    metadata={"skipped": True, "reason": "no_documents"},
                ) as span:
                    span.update(output={"skipped": True, "reason": "no_documents"})

            emit_node_event(
                writer,
                node="chunk_expander",
                phase="retrieval",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.CHUNK_EXPANDER_DONE,
                    "skipped": True,
                    "reason": "no_documents",
                    "input_chunks_count": 0,
                    "output_chunks_count": 0,
                },
            )
            return new_state

        # 使用 Langfuse span 包裝主要處理邏輯
        if langfuse:
            with langfuse.start_as_current_span(
                name="chunk_expander",
                input=span_input,
                metadata={
                    "input_chunks_count": input_count,
                    "input_files_count": len(input_files),
                },
            ) as span:
                # Adaptive Chunk Expansion
                expanded_docs = _adaptive_expand_chunks(docs)
                output_count = len(expanded_docs)
                expanded_count = output_count - input_count

                # 收集 output 資訊
                output_files = list({doc.get("filename", "unknown") for doc in expanded_docs})
                file_chunk_counts = {}
                for doc in expanded_docs:
                    fname = doc.get("filename", "unknown")
                    file_chunk_counts[fname] = file_chunk_counts.get(fname, 0) + 1

                # 格式化為 context 字串
                context_text = _format_expanded_docs_to_context(expanded_docs)
                context_length = len(context_text)

                # 更新 Langfuse span output
                span.update(
                    output={
                        "output_chunks_count": output_count,
                        "output_files_count": len(output_files),
                        "expanded_chunks_count": expanded_count,
                        "context_length": context_length,
                        "file_details": [
                            {"filename": fname, "chunks": count}
                            for fname, count in sorted(file_chunk_counts.items())
                        ],
                        "output_chunks": [
                            {
                                "filename": doc.get("filename", "unknown"),
                                "chunk_index": doc.get("chunk_index", 0),
                                "is_expanded": doc.get("_expanded", False),
                            }
                            for doc in expanded_docs[:30]  # 最多顯示 30 個
                        ],
                        "context_preview": context_text[:500] + "..." if len(context_text) > 500 else context_text,
                    },
                    metadata={
                        "output_chunks_count": output_count,
                        "expanded_chunks_count": expanded_count,
                        "context_length": context_length,
                    },
                )
        else:
            # Langfuse 不可用時直接執行
            expanded_docs = _adaptive_expand_chunks(docs)
            output_count = len(expanded_docs)
            expanded_count = output_count - input_count

            output_files = list({doc.get("filename", "unknown") for doc in expanded_docs})
            file_chunk_counts = {}
            for doc in expanded_docs:
                fname = doc.get("filename", "unknown")
                file_chunk_counts[fname] = file_chunk_counts.get(fname, 0) + 1

            context_text = _format_expanded_docs_to_context(expanded_docs)
            context_length = len(context_text)

        # 設定 context（僅當尚未設定時）
        existing_context = state_dict.get("context") or ""
        if not existing_context and context_text.strip():
            new_state["context"] = context_text

        # 更新 retrieval state
        retrieval_state["expanded_docs"] = expanded_docs
        retrieval_state["raw_chunks"] = [context_text] if context_text else []  # 向後相容
        retrieval_state["expansion_stats"] = {
            "input_count": input_count,
            "output_count": output_count,
            "expanded_count": expanded_count,
            "context_length": context_length,
        }
        new_state["retrieval"] = retrieval_state

        logger.info(
            "[CHUNK_EXPANDER] Expanded {} -> {} chunks (+{}), context_length={}",
            input_count,
            output_count,
            expanded_count,
            context_length,
        )

        emit_node_event(
            writer,
            node="chunk_expander",
            phase="retrieval",
            payload={
                "channel": "status",
                "stage": AskStreamStages.CHUNK_EXPANDER_DONE,
                "input_source": input_source,
                "input_chunks_count": input_count,
                "input_files_count": len(input_files),
                "output_chunks_count": output_count,
                "output_files_count": len(output_files),
                "expanded_chunks_count": expanded_count,
                "context_length": context_length,
                "file_details": [
                    {"filename": fname, "chunks": count}
                    for fname, count in sorted(file_chunk_counts.items())
                ][:10],
            },
        )

        return new_state

    return chunk_expander_node
