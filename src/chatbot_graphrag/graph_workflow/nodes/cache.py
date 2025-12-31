"""
快取節點

查詢回應的語意快取查詢和儲存。
第 6 階段：為快取鍵添加版本欄位以進行正確的失效處理。

主要功能：
- cache_lookup_node: 語意快取查詢
- cache_response_node: 提供快取回應
- cache_store_node: 儲存回應到快取
"""

import hashlib
import logging
from datetime import datetime
from typing import Any, Optional

from chatbot_graphrag.graph_workflow.types import GraphRAGState
from chatbot_graphrag.core.config import settings

logger = logging.getLogger(__name__)

# 語意快取相似度閾值
CACHE_SIMILARITY_THRESHOLD = 0.90


def compute_cache_key(
    question: str,
    acl_groups: list[str],
    tenant_id: Optional[str] = None,
    index_version: Optional[str] = None,
    prompt_version: Optional[str] = None,
) -> str:
    """
    計算問題的快取鍵。

    第 6 階段：包含版本欄位，當索引或提示版本改變時
    自動失效快取。

    Args:
        question: 正規化的問題文字
        acl_groups: 使用者的 ACL 群組
        tenant_id: 租戶識別碼，用於多租戶隔離
        index_version: 索引版本（重新索引時失效）
        prompt_version: 提示版本（提示改變時失效）

    Returns:
        32 字元的快取鍵雜湊
    """
    # 如果未提供則使用設定預設值
    tenant_id = tenant_id or "default"
    index_version = index_version or settings.index_version
    prompt_version = prompt_version or settings.langfuse_prompt_label or "default"

    # 建構快取鍵元件
    acl_str = ",".join(sorted(acl_groups))
    components = [
        question.lower().strip(),
        acl_str,
        tenant_id,
        f"idx:{index_version}",
        f"pmt:{prompt_version}",
    ]
    combined = "|".join(components)
    return hashlib.sha256(combined.encode()).hexdigest()[:32]


def compute_cache_key_with_state(state: GraphRAGState) -> str:
    """
    從 GraphRAG 狀態計算快取鍵。

    第 6 階段：從狀態中抽取所有相關欄位作為快取鍵。

    Args:
        state: 當前工作流程狀態

    Returns:
        32 字元的快取鍵雜湊
    """
    question = state.get("normalized_question") or state.get("question", "")
    acl_groups = state.get("acl_groups", ["public"])
    tenant_id = state.get("tenant_id")
    index_version = state.get("index_version")
    prompt_version = state.get("prompt_version")

    return compute_cache_key(
        question=question,
        acl_groups=acl_groups,
        tenant_id=tenant_id,
        index_version=index_version,
        prompt_version=prompt_version,
    )


async def cache_lookup_node(state: GraphRAGState) -> dict[str, Any]:
    """
    語意查詢快取的快取查詢節點。

    第 6 階段：使用版本感知的快取鍵進行自動失效。

    嘗試使用語意相似度為類似查詢找到快取的回應。
    遵守 semantic_cache_enabled 設定和 TTL 配置。

    Returns:
        如果找到則更新狀態與快取的回應
    """
    import time
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("cache_lookup", "START")

    start_time = time.time()
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))

    # 檢查快取是否啟用
    if not settings.semantic_cache_enabled:
        emit_status("cache_lookup", "DONE")
        return {
            "retrieval_path": retrieval_path + ["cache:disabled"],
            "timing": {**timing, "cache_lookup_ms": (time.time() - start_time) * 1000},
        }

    # 取得用於語意搜尋的問題
    question = state.get("normalized_question") or state.get("question", "")
    tenant_id = state.get("tenant_id")
    index_version = state.get("index_version") or settings.index_version
    prompt_version = state.get("prompt_version") or settings.langfuse_prompt_label or "default"

    if not question:
        logger.debug("No question for cache lookup")
        emit_status("cache_lookup", "DONE")
        return {
            "retrieval_path": retrieval_path + ["cache:no_question"],
            "timing": {**timing, "cache_lookup_ms": (time.time() - start_time) * 1000},
        }

    # 第 6 階段：使用版本感知的快取鍵進行精確匹配回退
    cache_key = compute_cache_key_with_state(state)
    logger.debug(f"Cache lookup: key={cache_key[:8]}...")

    try:
        # Import services lazily to avoid circular imports
        from chatbot_graphrag.services.vector import qdrant_service, embedding_service

        # Ensure services are initialized
        await embedding_service.initialize()
        await qdrant_service.initialize()

        # 1. Compute embedding for the question
        question_embedding = await embedding_service.embed_text(question)

        # 2. 在快取中搜尋相似嵌入向量，並進行版本過濾
        cache_result = await qdrant_service.cache_lookup(
            query_embedding=question_embedding,
            threshold=CACHE_SIMILARITY_THRESHOLD,
            tenant_id=tenant_id,
        )

        if cache_result:
            payload = cache_result.get("payload", {})
            cached_response = payload.get("response")
            cached_index_version = payload.get("index_version")
            cached_prompt_version = payload.get("prompt_version")
            cached_created_at = payload.get("created_at")

            # 3. 驗證版本相容性（第 6 階段：版本感知失效）
            version_match = (
                cached_index_version == index_version and
                cached_prompt_version == prompt_version
            )

            # 4. 如果配置了 TTL 則驗證（0 = 無過期）
            ttl_valid = True
            if settings.semantic_cache_ttl_seconds > 0 and cached_created_at:
                try:
                    created_time = datetime.fromisoformat(cached_created_at.replace("Z", "+00:00"))
                    age_seconds = (datetime.utcnow() - created_time.replace(tzinfo=None)).total_seconds()
                    if age_seconds > settings.semantic_cache_ttl_seconds:
                        ttl_valid = False
                        logger.debug(f"Cache entry expired (age={age_seconds:.0f}s, ttl={settings.semantic_cache_ttl_seconds}s)")
                except Exception as e:
                    logger.warning(f"Failed to parse cache created_at: {e}")

            if cached_response and version_match and ttl_valid:
                similarity = cache_result.get("score", 0.0)
                logger.info(
                    f"Cache hit (similarity={similarity:.3f}): {question[:50]}..."
                )
                emit_status("cache_lookup", "DONE")
                return {
                    "final_answer": cached_response,
                    "cache_hit": True,
                    "cache_similarity": similarity,
                    "retrieval_path": ["cache:hit"],
                    "timing": {**timing, "cache_lookup_ms": (time.time() - start_time) * 1000},
                }
            elif cached_response:
                if not version_match:
                    logger.debug(
                        f"Cache version mismatch: idx={cached_index_version} vs {index_version}, "
                        f"pmt={cached_prompt_version} vs {prompt_version}"
                    )
                if not ttl_valid:
                    logger.debug("Cache entry expired by TTL")

        logger.debug("Cache miss")
        emit_status("cache_lookup", "DONE")
        return {
            "retrieval_path": retrieval_path + ["cache:miss"],
            "timing": {**timing, "cache_lookup_ms": (time.time() - start_time) * 1000},
        }

    except Exception as e:
        logger.warning(f"Cache lookup error: {e}")
        return {
            "retrieval_path": retrieval_path + ["cache:error"],
            "timing": {**timing, "cache_lookup_ms": (time.time() - start_time) * 1000},
        }


async def cache_response_node(state: GraphRAGState) -> dict[str, Any]:
    """
    用於提供快取回應的快取回應節點。

    簡單地將快取的回應傳遞到輸出。

    Returns:
        帶有快取回應的狀態
    """
    import time
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("cache_response", "START")

    start_time = time.time()
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))

    # 記錄快取的答案以便除錯
    cached_answer = state.get("final_answer", "")
    cache_hit = state.get("cache_hit", False)
    logger.info(f"cache_response_node: cache_hit={cache_hit}, answer_len={len(cached_answer) if cached_answer else 0}")

    emit_status("cache_response", "DONE")

    # 回應已經從 cache_lookup 存入 final_answer
    return {
        "retrieval_path": retrieval_path + ["cache:served"],
        "timing": {**timing, "cache_response_ms": (time.time() - start_time) * 1000},
    }


async def cache_store_node(state: GraphRAGState) -> dict[str, Any]:
    """
    用於將回應儲存到快取的快取儲存節點。

    第 6 階段：使用版本感知的快取鍵進行正確的版本控制。

    將生成的回應與語意嵌入向量一起儲存，供未來查詢使用。

    Returns:
        更新後的狀態
    """
    import time
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("cache_store", "START")

    start_time = time.time()
    final_answer = state.get("final_answer", "")
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))

    # 如果禁用、空白、被阻擋、HITL 回應或已快取的回應則不快取
    skip_conditions = [
        not settings.semantic_cache_enabled,  # Cache disabled
        not final_answer,
        state.get("guard_blocked"),
        state.get("acl_denied"),
        state.get("hitl_timed_out"),
        state.get("hitl_rejected"),
        state.get("cache_hit"),  # Don't re-cache hit responses
    ]

    if any(skip_conditions):
        emit_status("cache_store", "DONE")
        return {
            "retrieval_path": retrieval_path + ["cache:skip"],
            "timing": {**timing, "cache_store_ms": (time.time() - start_time) * 1000},
        }

    # 取得問題和版本資訊
    # 優先使用 resolved_question（重寫後的完整問題），這樣快取可以被未來相同問題命中
    # 例如：「那李醫師呢？」重寫為「李醫師什麼時候看診？」後存入快取
    # 下次有人問「李醫師什麼時候看診？」就能命中快取
    question = (
        state.get("resolved_question")
        or state.get("normalized_question")
        or state.get("question", "")
    )
    tenant_id = state.get("tenant_id")
    acl_groups = state.get("acl_groups", ["public"])
    index_version = state.get("index_version") or settings.index_version
    prompt_version = state.get("prompt_version") or settings.langfuse_prompt_label or "default"

    # 記錄使用的問題
    original_q = state.get("question", "")
    resolved_q = state.get("resolved_question", "")
    if resolved_q and resolved_q != original_q:
        logger.info(f"Cache store using resolved question: '{resolved_q[:50]}...' (original: '{original_q[:30]}...')")

    if not question:
        emit_status("cache_store", "DONE")
        return {
            "retrieval_path": retrieval_path + ["cache:no_question"],
            "timing": {**timing, "cache_store_ms": (time.time() - start_time) * 1000},
        }

    # 第 6 階段：使用版本感知的快取鍵
    cache_key = compute_cache_key_with_state(state)
    logger.debug(f"Cache store: key={cache_key[:8]}...")

    try:
        # Import services lazily to avoid circular imports
        from chatbot_graphrag.services.vector import qdrant_service, embedding_service

        # Ensure services are initialized
        await embedding_service.initialize()
        await qdrant_service.initialize()

        # 1. Compute embedding for the question
        question_embedding = await embedding_service.embed_text(question)

        # 2. 收集來源文件 ID 用於快取失效
        evidence_table = state.get("evidence_table", [])
        source_doc_ids = list(set(
            e.source_doc for e in evidence_table
            if hasattr(e, "source_doc") and e.source_doc
        ))

        # 3. 建構帶有版本資訊的快取負載
        cache_payload = {
            "query_hash": cache_key,
            "question": question,
            "response": final_answer,
            "tenant_id": tenant_id,
            "acl_groups": acl_groups,
            "index_version": index_version,
            "prompt_version": prompt_version,
            "source_doc_ids": source_doc_ids,
            "created_at": datetime.utcnow().isoformat(),
            "retrieval_path": retrieval_path,
        }

        # 4. 儲存到語意快取
        cache_id = await qdrant_service.cache_store(
            query_embedding=question_embedding,
            payload=cache_payload,
        )

        logger.info(
            f"Response cached (id={cache_id[:8]}...): {len(final_answer)} chars, "
            f"idx_ver={index_version}, pmt_ver={prompt_version}"
        )
        emit_status("cache_store", "DONE")
        return {
            "cache_id": cache_id,
            "retrieval_path": retrieval_path + ["cache:stored"],
            "timing": {**timing, "cache_store_ms": (time.time() - start_time) * 1000},
        }

    except Exception as e:
        logger.warning(f"Cache store error: {e}")
        return {
            "retrieval_path": retrieval_path + ["cache:store_error"],
            "timing": {**timing, "cache_store_ms": (time.time() - start_time) * 1000},
        }
