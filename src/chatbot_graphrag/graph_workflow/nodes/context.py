"""
上下文建構節點

Chunk 擴展、上下文打包和證據表生成。
對檢索的內容實現 OWASP LLM01（提示注入）緩解措施。

主要節點：
- chunk_expander_node: 擴展 chunk 以獲得更好的理解
- context_packer_node: 將 chunk 打包成 LLM 上下文
- evidence_table_node: 建構用於落地性檢查的證據表
"""

import logging
import re
from typing import Any

from chatbot_graphrag.graph_workflow.types import EvidenceItem, GraphRAGState
from chatbot_graphrag.graph_workflow.budget import estimate_tokens

logger = logging.getLogger(__name__)

# ==================== OWASP LLM01: 上下文跳脫 ====================

# 要從檢索內容中跳脫/剝離的模式（通過上下文的潛在提示注入）
CONTEXT_STRIP_PATTERNS = [
    # 可能覆蓋提示的系統/指令標記
    r"<\|system\|>",
    r"<\|/system\|>",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<\|endoftext\|>",
    r"<\|assistant\|>",
    r"<\|user\|>",
    r"<system>",
    r"</system>",
    r"\[INST\]",
    r"\[/INST\]",
    r"\[SYS\]",
    r"\[/SYS\]",
    # 指令覆蓋嘗試
    r"###\s*(System|Human|Assistant)\s*:",
    r"忽略.*?(指令|規則|說明)",
    r"ignore\s+(previous|all|above)\s+instructions?",
    r"disregard\s+(previous|all|above)\s+instructions?",
]

COMPILED_CONTEXT_STRIP = [re.compile(p, re.IGNORECASE) for p in CONTEXT_STRIP_PATTERNS]

# 每個 chunk 的最大內容長度，以防止上下文洪氾
MAX_CHUNK_CONTENT_LENGTH = 3000


def escape_for_prompt(text: str, max_length: int = MAX_CHUNK_CONTENT_LENGTH) -> str:
    """
    跳脫檢索的內容以安全插入提示（OWASP LLM01）。

    此函數在將檢索文件中的潛在不受信任內容包含到 LLM 提示之前進行清理。
    這可以防止間接提示注入攻擊，其中惡意指令被嵌入到文件中。

    Args:
        text: 從檢索文件中獲取的原始文字
        max_length: 允許的最大長度（防止上下文洪氾）

    Returns:
        安全用於提示插入的清理過的文字
    """
    if not text:
        return text

    # 1. 剝離危險模式
    for pattern in COMPILED_CONTEXT_STRIP:
        text = pattern.sub("[FILTERED]", text)

    # 2. 限制長度以防止上下文洪氾
    if len(text) > max_length:
        text = text[:max_length] + "\n...(內容已截斷)"
        logger.debug(f"Context chunk truncated to {max_length} chars")

    # 3. 正規化過多的空白
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    # 4. 添加內容邊界標記（幫助 LLM 區分文件內容）
    # 這些標記清楚表明這是檢索的內容，而不是指令
    text = text.strip()

    return text


def escape_metadata_for_prompt(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    跳脫 metadata 值以安全顯示（防止通過 metadata 注入）。

    Args:
        metadata: 來自文件的原始 metadata 字典

    Returns:
        清理過的 metadata 字典
    """
    if not metadata:
        return metadata

    safe_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            # 對字串值應用輕度跳脫
            safe_value = value[:500]  # Limit length
            for pattern in COMPILED_CONTEXT_STRIP:
                safe_value = pattern.sub("", safe_value)
            safe_metadata[key] = safe_value
        elif isinstance(value, (int, float, bool)):
            safe_metadata[key] = value
        elif isinstance(value, list):
            # 只包含簡單的列表項目
            safe_metadata[key] = [
                str(v)[:100] for v in value[:10] if isinstance(v, (str, int, float))
            ]
        # 跳過複雜的巢狀結構

    return safe_metadata


async def chunk_expander_node(state: GraphRAGState) -> dict[str, Any]:
    """
    Chunk 擴展節點。

    擴展 chunk 以包含周圍的上下文，以便更好地理解。

    Returns:
        更新後的狀態，包含 expanded_chunks
    """
    import time
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("chunk_expander", "START")

    start_time = time.time()
    reranked_chunks = state.get("reranked_chunks", [])
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))
    budget = state.get("budget")

    if not reranked_chunks:
        return {
            "expanded_chunks": [],
            "retrieval_path": retrieval_path + ["chunk_expander:no_input"],
            "timing": {**timing, "chunk_expander_ms": (time.time() - start_time) * 1000},
        }

    logger.info(f"Expanding {len(reranked_chunks)} chunks...")

    # 在生產環境中，這會：
    # 1. 為每個 chunk 載入父 chunk
    # 2. 添加兄弟 chunk 作為上下文
    # 3. 合併重疊的擴展

    # 目前，通過 metadata 豐富化傳遞
    expanded_chunks = []
    total_tokens = 0

    for chunk in reranked_chunks:
        # 估計 token 數
        chunk_tokens = estimate_tokens(chunk.content if hasattr(chunk, "content") else "")
        total_tokens += chunk_tokens

        # 檢查 token 預算
        if budget and hasattr(budget, "max_tokens"):
            if total_tokens > budget.max_tokens:
                logger.info(f"Token budget reached: {total_tokens} > {budget.max_tokens}")
                break

        expanded_chunks.append(chunk)

    # 更新預算中的 token 計數
    if budget and hasattr(budget, "add_tokens"):
        budget.add_tokens(total_tokens)

    logger.info(f"Expanded to {len(expanded_chunks)} chunks ({total_tokens} tokens)")

    emit_status("chunk_expander", "DONE")
    return {
        "expanded_chunks": expanded_chunks,
        "context_tokens": total_tokens,
        "retrieval_path": retrieval_path + [f"chunk_expander:{len(expanded_chunks)}"],
        "timing": {**timing, "chunk_expander_ms": (time.time() - start_time) * 1000},
        "budget": budget,
    }


async def context_packer_node(state: GraphRAGState) -> dict[str, Any]:
    """
    上下文打包節點。

    將 chunk 打包成 LLM 的上下文文字。

    Returns:
        更新後的狀態，包含 context_text
    """
    import time
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("context_packer", "START")

    start_time = time.time()
    expanded_chunks = state.get("expanded_chunks", [])
    community_reports = state.get("community_reports", [])
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))

    logger.debug(f"Packing {len(expanded_chunks)} chunks into context...")

    context_parts = []

    # 如果可用則添加社群報告摘要
    # 注意：社群報告是內部生成的，但仍然應用輕度跳脫
    if community_reports:
        context_parts.append("=== 相關主題概述 ===\n")
        for i, report in enumerate(community_reports[:3], 1):
            title = escape_for_prompt(report.get("title", ""), max_length=200)
            summary = escape_for_prompt(report.get("summary", ""), max_length=500)
            if title or summary:
                context_parts.append(f"{i}. {title}\n{summary}\n")
        context_parts.append("\n")

    # 添加文件 chunk
    # 重要：檢索的內容是不受信任的 - 應用完整跳脫（OWASP LLM01）
    context_parts.append("=== 相關文件內容 ===\n")
    for i, chunk in enumerate(expanded_chunks, 1):
        raw_content = chunk.content if hasattr(chunk, "content") else str(chunk)
        # 跳脫內容以防止間接提示注入
        content = escape_for_prompt(raw_content)
        doc_id = chunk.doc_id if hasattr(chunk, "doc_id") else "unknown"

        # 抽取 metadata 以便更好的歸因（例如，醫師姓名、文件標題）
        metadata = chunk.metadata if hasattr(chunk, "metadata") else {}
        title = metadata.get("title", "")
        section_title = metadata.get("section_title", "")
        department = metadata.get("department", "")
        doc_type = metadata.get("doc_type", "")

        # 跳脫 metadata 值
        if title:
            title = escape_for_prompt(title, max_length=200)
        if section_title:
            section_title = escape_for_prompt(section_title, max_length=100)
        if department:
            department = escape_for_prompt(department, max_length=100)

        # 建構帶有標題和章節的來源資訊以便更好的歸因
        source_parts = []
        if title:
            source_parts.append(title)
        if section_title and section_title != title:
            source_parts.append(f"章節: {section_title}")
        if department:
            source_parts.append(f"科別: {department}")

        if source_parts:
            source_info = f"來源: {' | '.join(source_parts)} ({doc_id})"
        else:
            source_info = f"來源: {doc_id}"

        # 對於醫師文件，在內容前加上醫師姓名以便清楚歸因
        # 這確保 LLM 知道每個 chunk 屬於哪位醫師
        if doc_type == "physician" and title:
            content = f"【{title}醫師】\n{content}"

        # 使用清晰的邊界標記幫助 LLM 區分文件內容
        context_parts.append(f"[文件 {i}] {source_info}\n---\n{content}\n---\n\n")

    context_text = "".join(context_parts)
    context_tokens = estimate_tokens(context_text)

    logger.info(f"Context packed: {len(context_text)} chars, ~{context_tokens} tokens")

    emit_status("context_packer", "DONE")
    return {
        "context_text": context_text,
        "context_tokens": context_tokens,
        "retrieval_path": retrieval_path + [f"context_packer:{context_tokens}tok"],
        "timing": {**timing, "context_packer_ms": (time.time() - start_time) * 1000},
    }


async def evidence_table_node(state: GraphRAGState) -> dict[str, Any]:
    """
    證據表節點。

    建構用於落地性檢查的結構化證據表。

    Returns:
        更新後的狀態，包含 evidence_table
    """
    import time
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("evidence_table", "START")

    start_time = time.time()
    expanded_chunks = state.get("expanded_chunks", [])
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))

    logger.debug(f"Building evidence table from {len(expanded_chunks)} chunks...")

    evidence_table = []

    for i, chunk in enumerate(expanded_chunks, 1):
        chunk_id = chunk.chunk_id if hasattr(chunk, "chunk_id") else f"chunk_{i}"
        raw_content = chunk.content if hasattr(chunk, "content") else str(chunk)
        doc_id = chunk.doc_id if hasattr(chunk, "doc_id") else "unknown"
        score = chunk.score if hasattr(chunk, "score") else 0.0
        raw_metadata = chunk.metadata if hasattr(chunk, "metadata") else {}

        # 跳脫內容和 metadata（OWASP LLM01 - 證據可能被顯示/使用）
        content = escape_for_prompt(raw_content, max_length=1000)
        metadata = escape_metadata_for_prompt(raw_metadata)

        # 對於醫師文件，在內容前加上醫師姓名以便清楚歸因
        doc_type = raw_metadata.get("doc_type", "")
        title = raw_metadata.get("title", "")
        if doc_type == "physician" and title:
            title = escape_for_prompt(title, max_length=200)
            content = f"【{title}醫師】\n{content}"

        evidence = EvidenceItem(
            chunk_id=chunk_id,
            content=content,
            relevance_score=score,
            source_doc=doc_id,
            citation_index=i,
            metadata=metadata,
        )
        evidence_table.append(evidence)

    logger.info(f"Built evidence table with {len(evidence_table)} items")

    emit_status("evidence_table", "DONE")
    return {
        "evidence_table": evidence_table,
        "retrieval_path": retrieval_path + [f"evidence_table:{len(evidence_table)}"],
        "timing": {**timing, "evidence_table_ms": (time.time() - start_time) * 1000},
    }
