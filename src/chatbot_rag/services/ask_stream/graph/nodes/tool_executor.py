"""
Tool executor 節點：執行工具呼叫並回傳原始文件。

功能：
- Multi-Query Retrieval：平行執行多個查詢並合併結果
- 文件去重（相同 filename + chunk_index）
- 漸進式檢索閾值（0.65 → 0.50 → 0.35）
- Adaptive Chunk Expansion（根據文件大小智慧擴展 chunks）

後續節點：
- reranker: Cross-encoder 重新排序
- result_evaluator: 結果評估與重試邏輯
"""

import asyncio
import hashlib
from collections import defaultdict
from typing import Callable, Dict, List, Set, cast

from langgraph.config import get_stream_writer
from loguru import logger

from chatbot_rag.llm import State
from chatbot_rag.llm.graph_nodes import retrieval_top_k_var
from chatbot_rag.services.retriever_service import retriever_service
from chatbot_rag.services.qdrant_service import qdrant_service
from ...types import StateDict, tool_executor_inputs_from_state
from ...constants import AskStreamStages, UNIFIED_AGENT_TOOLS_BY_NAME
from ...events import emit_node_event, truncate_tool_output_for_sse
from ...utils import extract_text_from_payload, calculate_needed_chunk_indices


# 取得工具名稱
retrieve_documents_tool_name = list(UNIFIED_AGENT_TOOLS_BY_NAME.keys())[0]
get_form_download_links_tool_name = list(UNIFIED_AGENT_TOOLS_BY_NAME.keys())[1]

# 漸進式檢索閾值
RETRIEVAL_THRESHOLDS = [0.65, 0.50, 0.35]


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
            "[TOOL_EXECUTOR] Adaptive expansion for {}: "
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
            "[TOOL_EXECUTOR] Adaptive expansion: {} -> {} chunks",
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
        scored_docs = [d for d in file_docs if d.get("score", 0) > 0]
        representative = scored_docs[0] if scored_docs else file_docs[0]

        metadata = representative.get("metadata") or {}
        entry_type = metadata.get("entry_type", "")
        module = metadata.get("module", "")
        max_score = max((d.get("score", 0) for d in file_docs), default=0)
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
            header_parts.append(f" 相似度: {max_score:.2f}")
        header_parts.append(f" chunks: {chunk_indices}/{total_chunks})")

        # 合併 chunks 內容（按 chunk_index 排序）
        sorted_docs = sorted(file_docs, key=lambda d: d.get("chunk_index", 0))
        contents = [d.get("content", "") for d in sorted_docs]
        body = "\n".join(contents)

        context_parts.append(f"{' '.join(header_parts)}\n{body}\n")
        doc_idx += 1

    return "\n".join(context_parts)


def _retrieve_with_progressive_threshold(
    query: str,
    top_k: int = 3,
) -> List[dict]:
    """
    使用漸進式閾值進行檢索。

    Args:
        query: 查詢文字
        top_k: 最大返回數量

    Returns:
        List[dict]: 檢索到的文件列表
    """
    for threshold in RETRIEVAL_THRESHOLDS:
        docs = retriever_service.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=threshold,
            expand_context=False,  # 不使用舊的擴展機制
        )
        if docs:
            logger.debug(
                "[TOOL_EXECUTOR] Retrieved {} docs at threshold={}",
                len(docs),
                threshold,
            )
            return docs

    return []


def _deduplicate_chunks(chunks: List[str]) -> List[str]:
    """
    對檢索結果進行去重。

    使用內容 hash 來識別重複內容。

    Args:
        chunks: 原始 chunks 列表

    Returns:
        List[str]: 去重後的 chunks
    """
    seen_hashes: Set[str] = set()
    unique_chunks: List[str] = []

    for chunk in chunks:
        # 使用 MD5 hash 識別重複內容
        chunk_hash = hashlib.md5(chunk.strip().encode("utf-8")).hexdigest()[:16]
        if chunk_hash not in seen_hashes:
            seen_hashes.add(chunk_hash)
            unique_chunks.append(chunk)

    if len(chunks) != len(unique_chunks):
        logger.debug(
            "[TOOL_EXECUTOR] Deduplicated chunks: {} -> {}",
            len(chunks),
            len(unique_chunks),
        )

    return unique_chunks


def _rrf_merge(
    result_lists: List[List[str]],
    k: int = 60,
) -> List[str]:
    """
    使用 Reciprocal Rank Fusion (RRF) 合併多個檢索結果。

    RRF score = sum(1 / (k + rank)) for each result list

    Args:
        result_lists: 多個結果列表
        k: RRF 常數（預設 60）

    Returns:
        List[str]: 合併後的結果
    """
    if not result_lists:
        return []

    if len(result_lists) == 1:
        return result_lists[0]

    # 計算每個 chunk 的 RRF 分數
    chunk_scores: Dict[str, float] = {}
    chunk_content: Dict[str, str] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            chunk_hash = hashlib.md5(chunk.strip().encode("utf-8")).hexdigest()[:16]
            score = 1.0 / (k + rank)
            chunk_scores[chunk_hash] = chunk_scores.get(chunk_hash, 0) + score
            chunk_content[chunk_hash] = chunk

    # 按分數排序
    sorted_hashes = sorted(
        chunk_scores.keys(),
        key=lambda h: chunk_scores[h],
        reverse=True,
    )

    return [chunk_content[h] for h in sorted_hashes]


def build_tool_executor_node(
    *,
    agent_backend: str = "chat",
    enable_multi_query: bool = True,
) -> Callable[[State], State]:
    """
    Tool executor 節點：依計畫呼叫工具並回傳原始文件。

    職責：
    1. Multi-Query Retrieval：平行執行多個查詢
    2. 去重（相同 filename + chunk_index）
    3. 將原始文件存入 retrieval["raw_docs"] 供後續節點使用

    Reranking 和 Adaptive Expansion 由後續節點（reranker、result_evaluator）負責。

    Args:
        agent_backend: Agent 後端類型
        enable_multi_query: 是否啟用多查詢檢索

    Returns:
        Callable: Tool executor 節點函數
    """

    async def _execute_retrieve_with_tool(query: str, top_k: int) -> tuple[str, List[dict]]:
        """
        使用 tool.ainvoke() 執行檢索，保持 Langfuse trace 可見性。

        同時返回：
        1. 原始 tool 輸出（字串）- 用於 SSE event
        2. 結構化文件列表 - 用於 rerank 和 adaptive expansion

        Args:
            query: 查詢文字
            top_k: 最大返回數量

        Returns:
            tuple[str, List[dict]]: (tool 輸出字串, 結構化文件列表)
        """
        tool = UNIFIED_AGENT_TOOLS_BY_NAME.get(retrieve_documents_tool_name)
        if tool is None:
            logger.error("[TOOL_EXECUTOR] retrieve_documents tool not found")
            return "", []

        try:
            # 使用 tool.ainvoke() 保持 Langfuse trace 可見性
            tool_output = await tool.ainvoke({"query": query})
            tool_output_str = str(tool_output)

            # 同時取得結構化資料用於 rerank
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(
                None,
                lambda: _retrieve_with_progressive_threshold(query, top_k),
            )

            return tool_output_str, docs

        except Exception as exc:
            logger.error("[TOOL_EXECUTOR] Retrieve error: %s", exc)
            return "", []

    async def tool_executor(state: State) -> State:
        writer = get_stream_writer()
        emit_node_event(
            writer,
            node="tool_executor",
            phase="retrieval",
            payload={
                "channel": "status",
                "stage": AskStreamStages.TOOL_EXECUTOR_START,
                "agent_backend": agent_backend,
            },
        )
        state_dict = cast(StateDict, state)
        exec_inputs = tool_executor_inputs_from_state(state_dict)
        tool_calls = exec_inputs["tool_calls"]
        user_question = exec_inputs["user_question"]
        raw_chunks: List[str] = []
        used_tools = exec_inputs["used_tools"]
        new_state = cast(State, dict(state))

        # 取得 top_k 參數
        top_k = retrieval_top_k_var.get()

        # 檢查是否有分解的查詢（Multi-Query Retrieval）
        retrieval_state = dict(new_state.get("retrieval") or {})
        decomposed_queries: List[str] = retrieval_state.get("decomposed_queries", [])

        for planned_call in tool_calls:
            tool_name = planned_call.get("name")
            if not tool_name:
                continue
            tool = UNIFIED_AGENT_TOOLS_BY_NAME.get(tool_name)
            if tool is None:
                logger.warning("[TOOL_EXECUTOR] Unknown tool requested: {}", tool_name)
                continue

            call_args = dict(planned_call.get("args") or {})
            if tool_name == retrieve_documents_tool_name:
                preferred_query = extract_text_from_payload(
                    call_args.get("content")
                ) or extract_text_from_payload(call_args.get("query"))
                fallback_query = exec_inputs["retrieval_query"] or user_question
                primary_query = preferred_query or fallback_query

                # Multi-Query Retrieval: 如果有分解查詢，平行執行
                if enable_multi_query and decomposed_queries and len(decomposed_queries) > 1:
                    logger.info(
                        "[TOOL_EXECUTOR] Multi-query retrieval with {} queries",
                        len(decomposed_queries),
                    )

                    emit_node_event(
                        writer,
                        node="tool_executor",
                        phase="retrieval",
                        payload={
                            "channel": "status",
                            "stage": AskStreamStages.TOOL_EXECUTOR_CALL,
                            "tool_name": tool_name,
                            "tool_args": {"queries": decomposed_queries},
                            "multi_query": True,
                            "agent_backend": agent_backend,
                        },
                    )

                    # 使用 asyncio.gather 平行執行多個查詢
                    # 每個查詢都會呼叫 tool.ainvoke() 以保持 Langfuse trace 可見性
                    all_docs: List[dict] = []
                    all_tool_outputs: List[str] = []
                    tasks = [
                        _execute_retrieve_with_tool(q, top_k)
                        for q in decomposed_queries
                    ]
                    try:
                        results = await asyncio.wait_for(
                            asyncio.gather(*tasks, return_exceptions=True),
                            timeout=15.0,
                        )
                        for result in results:
                            if isinstance(result, Exception):
                                logger.warning(
                                    "[TOOL_EXECUTOR] Multi-query sub-task error: %s",
                                    result,
                                )
                            elif result:
                                tool_output_str, docs = result
                                if tool_output_str:
                                    all_tool_outputs.append(tool_output_str)
                                if docs:
                                    all_docs.extend(docs)
                    except asyncio.TimeoutError:
                        logger.warning("[TOOL_EXECUTOR] Multi-query timeout")

                    if all_docs:
                        # 先去重（相同 filename + chunk_index）
                        seen_keys: Set[str] = set()
                        unique_docs: List[dict] = []
                        for doc in all_docs:
                            key = f"{doc.get('filename', '')}:{doc.get('chunk_index', 0)}"
                            if key not in seen_keys:
                                seen_keys.add(key)
                                unique_docs.append(doc)

                        logger.info(
                            "[TOOL_EXECUTOR] Multi-query: {} total docs -> {} unique",
                            len(all_docs),
                            len(unique_docs),
                        )

                        # 將原始文件存入 retrieval state，供 reranker 節點使用
                        retrieval_state["raw_docs"] = unique_docs
                        retrieval_state["query"] = primary_query

                        observation_text = (
                            f"[Multi-Query] Retrieved {len(unique_docs)} unique docs from {len(decomposed_queries)} queries"
                        )
                    else:
                        observation_text = "[Multi-Query] No results"
                        retrieval_state["raw_docs"] = []

                    if tool_name not in used_tools:
                        used_tools.append(tool_name)

                    emit_node_event(
                        writer,
                        node="tool_executor",
                        phase="retrieval",
                        payload={
                            "channel": "status",
                            "stage": AskStreamStages.TOOL_EXECUTOR_RESULT,
                            "tool_name": tool_name,
                            "tool_output": truncate_tool_output_for_sse(observation_text),
                            "multi_query": True,
                            "query_count": len(decomposed_queries),
                            "documents_count": len(retrieval_state.get("raw_docs", [])),
                            "agent_backend": agent_backend,
                        },
                    )
                    continue  # 跳過正常的單一查詢執行

                # 單一查詢執行（使用 tool.ainvoke 保持 Langfuse trace 可見性）
                emit_node_event(
                    writer,
                    node="tool_executor",
                    phase="retrieval",
                    payload={
                        "channel": "status",
                        "stage": AskStreamStages.TOOL_EXECUTOR_CALL,
                        "tool_name": tool_name,
                        "tool_args": {"query": primary_query},
                        "agent_backend": agent_backend,
                    },
                )

                # 使用 tool.ainvoke 保持 Langfuse trace 可見性
                tool_output_str, docs = await _execute_retrieve_with_tool(primary_query, top_k)

                if docs:
                    logger.info(
                        "[TOOL_EXECUTOR] Single query: retrieved {} docs",
                        len(docs),
                    )

                    # 將原始文件存入 retrieval state，供 reranker 節點使用
                    retrieval_state["raw_docs"] = docs
                    retrieval_state["query"] = primary_query

                    observation_text = f"[Single Query] Retrieved {len(docs)} docs"
                else:
                    observation_text = "[Single Query] No results"
                    retrieval_state["raw_docs"] = []

                if tool_name not in used_tools:
                    used_tools.append(tool_name)

                emit_node_event(
                    writer,
                    node="tool_executor",
                    phase="retrieval",
                    payload={
                        "channel": "status",
                        "stage": AskStreamStages.TOOL_EXECUTOR_RESULT,
                        "tool_name": tool_name,
                        "tool_output": truncate_tool_output_for_sse(observation_text),
                        "documents_count": len(retrieval_state.get("raw_docs", [])),
                        "agent_backend": agent_backend,
                    },
                )
                continue

            elif tool_name == get_form_download_links_tool_name:
                preferred_query = extract_text_from_payload(
                    call_args.get("content")
                ) or extract_text_from_payload(call_args.get("query"))
                fallback_query = exec_inputs["retrieval_query"] or user_question
                call_args["query"] = preferred_query or fallback_query
                call_args.pop("content", None)

            # 其他工具：標準執行（如 get_form_download_links）
            emit_node_event(
                writer,
                node="tool_executor",
                phase="retrieval",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.TOOL_EXECUTOR_CALL,
                    "tool_name": tool_name,
                    "tool_args": call_args,
                    "agent_backend": agent_backend,
                },
            )

            try:
                observation = await tool.ainvoke(call_args)
            except Exception as exc:  # noqa: BLE001 pylint: disable=broad-exception-caught
                logger.error("[TOOL_EXECUTOR] Tool %s error: %s", tool_name, exc)
                observation = f"[工具執行錯誤] {tool_name}: {str(exc)}"

            observation_text = str(observation)
            if observation_text.strip():
                raw_chunks.append(observation_text)
            if tool_name not in used_tools:
                used_tools.append(tool_name)

            emit_node_event(
                writer,
                node="tool_executor",
                phase="retrieval",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.TOOL_EXECUTOR_RESULT,
                    "tool_name": tool_name,
                    "tool_output": truncate_tool_output_for_sse(observation_text),
                    "agent_backend": agent_backend,
                },
            )

        # 去重處理
        if len(raw_chunks) > 1:
            raw_chunks = _deduplicate_chunks(raw_chunks)

        retrieval_state["raw_chunks"] = raw_chunks
        new_state["retrieval"] = retrieval_state
        new_state["used_tools"] = used_tools

        emit_node_event(
            writer,
            node="tool_executor",
            phase="retrieval",
            payload={
                "channel": "status",
                "stage": AskStreamStages.TOOL_EXECUTOR_DONE,
                "used_tools": used_tools,
                "documents_count": len(raw_chunks),
                "agent_backend": agent_backend,
            },
        )
        return new_state

    return tool_executor
