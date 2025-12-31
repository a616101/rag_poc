"""
檢索節點

混合搜尋、社群報告和追蹤查詢生成。

主要節點：
- hybrid_seed_node: 初始混合搜尋
- community_reports_node: 社群報告檢索
- followups_node: 追蹤查詢生成
- rrf_merge_node: RRF 融合
- hop_hybrid_node: 跳躍混合搜尋
"""

import logging
from typing import Any

from chatbot_graphrag.graph_workflow.types import (
    GraphRAGState,
    QueryMode,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


async def hybrid_seed_node(state: GraphRAGState, config: dict | None = None) -> dict[str, Any]:
    """
    混合種子搜尋節點。

    執行初始混合搜尋，結合稠密、稀疏和全文檢索搜尋。
    對實體特定查詢使用查詢分解（例如，醫師姓名 + 屬性）。

    Args:
        state: 當前圖譜狀態
        config: 包含 Langfuse 追蹤回調的可選 LangGraph 配置

    Returns:
        更新後的狀態，包含 seed_results
    """
    import asyncio
    import time
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("hybrid_seed", "START")

    start_time = time.time()
    question = state.get("normalized_question") or state.get("question", "")
    messages = state.get("messages", [])  # Get conversation history

    # Extract callbacks from config for Langfuse tracing
    callbacks = config.get("callbacks", []) if config else []
    acl_groups = state.get("acl_groups", ["public"])
    tenant_id = state.get("tenant_id", "default")
    filter_context = state.get("filter_context")
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))
    budget = state.get("budget")

    logger.info(f"Hybrid seed search: {question[:50]}...")

    try:
        # Import services lazily
        from chatbot_graphrag.services.search.hybrid_search import (
            HybridSearchConfig,
            hybrid_search_service,
        )
        from chatbot_graphrag.services.search.query_decomposer import decompose_query_with_llm
        from chatbot_graphrag.services.vector.embedding_service import embedding_service

        # 如有需要則初始化服務
        await hybrid_search_service.initialize()
        await embedding_service.initialize()

        # 使用 LLM 分解查詢以進行智能子查詢生成
        # 傳遞聊天歷史以解析追蹤問題的上下文
        # 傳遞用於 Langfuse 追蹤的回調
        decomposed = await decompose_query_with_llm(
            question,
            chat_history=messages,
            callbacks=callbacks,
        )
        logger.info(f"Query decomposition: type={decomposed.query_type}, reasoning={decomposed.reasoning}")
        if decomposed.resolved_query:
            logger.info(f"Resolved query (with context): {decomposed.resolved_query}")

        # 收集所有要搜尋的查詢
        # 如果可用則使用 resolved_query（帶上下文）作為主要查詢，否則使用原始問題
        primary_query = decomposed.resolved_query or decomposed.primary_query or question
        queries_to_search = [primary_query]

        # 如果與解析的不同則添加原始問題
        if decomposed.resolved_query and question != decomposed.resolved_query:
            queries_to_search.append(question)

        if decomposed.sub_queries:
            queries_to_search.extend(decomposed.sub_queries)
            logger.info(f"Sub-queries: {decomposed.sub_queries}")

        logger.info(f"Total queries to search: {len(queries_to_search)}")

        # 並行為所有查詢生成嵌入向量
        embedding_tasks = [embedding_service.embed_text(q) for q in queries_to_search]
        embeddings = await asyncio.gather(*embedding_tasks)

        # 如果可用則使用 filter_context 建構搜尋配置
        if filter_context:
            config_acl = filter_context.acl_groups
            config_tenant = filter_context.tenant_id
        else:
            config_acl = acl_groups
            config_tenant = tenant_id

        # 第 1 階段：只在明確設定（非預設）時按 tenant_id 過濾
        # 這維持了與添加 tenant_id 之前攝取的資料的向後相容性
        effective_tenant = config_tenant if config_tenant != "default" else None

        base_config = HybridSearchConfig(
            dense_limit=40,
            sparse_limit=40,
            fts_limit=40,
            final_limit=20,
            acl_groups=config_acl,
            tenant_id=effective_tenant,  # Phase 1: multi-tenant filtering (None = no filter)
        )

        # 對於 entity_lookup 查詢，如果偵測到醫師姓名則也按標題過濾
        if decomposed.query_type == "entity_lookup" and decomposed.physician_name:
            base_config.title_contains = decomposed.physician_name
            logger.info(f"Applied title filter: title_contains={decomposed.physician_name}")

        # Execute searches for all queries in parallel
        search_tasks = []
        for query, embedding in zip(queries_to_search, embeddings):
            search_tasks.append(
                hybrid_search_service.search(
                    query=query,
                    query_embedding=embedding,
                    config=base_config,
                )
            )

        all_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # 使用 RRF 風格的去重合併結果
        seen_chunks = set()
        merged_results = []
        chunk_scores: dict[str, float] = {}

        for i, results in enumerate(all_results):
            if isinstance(results, Exception):
                logger.error(f"Search {i} failed: {results}")
                continue

            # 權重：主要查詢獲得完整權重，子查詢獲得遞減權重
            weight = 1.0 if i == 0 else 0.7

            for result in results:
                chunk_id = result.chunk_id
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    merged_results.append(result)
                    chunk_scores[chunk_id] = result.score * weight
                else:
                    # 對被多個查詢找到的 chunk 提升分數
                    chunk_scores[chunk_id] = max(
                        chunk_scores.get(chunk_id, 0),
                        result.score * weight
                    )

        # 按組合分數排序並限制數量
        for result in merged_results:
            result.score = chunk_scores.get(result.chunk_id, result.score)
        merged_results.sort(key=lambda r: r.score, reverse=True)
        merged_results = merged_results[:base_config.final_limit]

        # Update budget
        if budget and hasattr(budget, "increment_queries"):
            budget.increment_queries(len(queries_to_search))

        seed_results = RetrievalResult(
            chunks=merged_results,
            query_used=question,
            mode=QueryMode.LOCAL,
            retrieval_path=["hybrid_seed"],
        )

        logger.info(f"Hybrid seed found {len(merged_results)} results (from {len(queries_to_search)} queries)")

        emit_status("hybrid_seed", "DONE")

        # 儲存解析的問題用於最終答案生成
        # 如果分解器解析了追蹤問題，則使用它；否則回退到原始問題
        resolved_question = decomposed.resolved_query or decomposed.primary_query or question

        return {
            "seed_results": seed_results,
            "resolved_question": resolved_question,
            "query_decomposition": {
                "physician_name": decomposed.physician_name,
                "property_type": decomposed.property_type,
                "sub_queries": decomposed.sub_queries,
                "resolved_query": decomposed.resolved_query,
                "reasoning": decomposed.reasoning,
            },
            "retrieval_path": retrieval_path + [f"hybrid_seed:{len(merged_results)}"],
            "timing": {**timing, "hybrid_seed_ms": (time.time() - start_time) * 1000},
            "budget": budget,
        }

    except Exception as e:
        logger.error(f"Hybrid seed error: {e}")
        return {
            "seed_results": RetrievalResult(
                chunks=[],
                query_used=question,
                mode=QueryMode.LOCAL,
            ),
            "error": str(e),
            "retrieval_path": retrieval_path + ["hybrid_seed:error"],
            "timing": {**timing, "hybrid_seed_ms": (time.time() - start_time) * 1000},
        }


async def community_reports_node(state: GraphRAGState) -> dict[str, Any]:
    """
    社群報告節點。

    為 global/drift 模式檢索相關的社群報告。

    Returns:
        更新後的狀態，包含 community_reports
    """
    import time
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("community_reports", "START")

    start_time = time.time()
    question = state.get("normalized_question") or state.get("question", "")
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))
    budget = state.get("budget")

    logger.info(f"Community reports search: {question[:50]}...")

    try:
        # Import services lazily
        from chatbot_graphrag.services.retrieval.global_mode import (
            GlobalModeConfig,
            global_mode_retriever,
        )
        from chatbot_graphrag.services.vector.embedding_service import embedding_service

        await global_mode_retriever.initialize()
        await embedding_service.initialize()

        # Generate embedding
        query_embedding = await embedding_service.embed_text(question)

        # 配置全局模式
        config = GlobalModeConfig(
            max_communities=5,
            min_community_score=0.3,
            enable_drill_down=False,  # We'll handle drill-down separately
        )

        # 取得社群報告
        result = await global_mode_retriever.retrieve(
            query=question,
            query_embedding=query_embedding,
            config=config,
        )

        # Update budget
        if budget and hasattr(budget, "increment_queries"):
            budget.increment_queries()

        community_reports = [
            {
                "community_id": cr.community_id,
                "level": cr.level,
                "title": cr.title,
                "summary": cr.summary,
                "key_entities": cr.key_entities,
                "key_themes": cr.key_themes,
                "score": cr.score,
            }
            for cr in result.community_reports
        ]

        logger.info(f"Found {len(community_reports)} community reports")

        emit_status("community_reports", "DONE")
        return {
            "community_reports": community_reports,
            "retrieval_path": retrieval_path + [f"community_reports:{len(community_reports)}"],
            "timing": {**timing, "community_reports_ms": (time.time() - start_time) * 1000},
            "budget": budget,
        }

    except Exception as e:
        logger.error(f"Community reports error: {e}")
        return {
            "community_reports": [],
            "error": str(e),
            "retrieval_path": retrieval_path + ["community_reports:error"],
            "timing": {**timing, "community_reports_ms": (time.time() - start_time) * 1000},
        }


async def followups_node(state: GraphRAGState) -> dict[str, Any]:
    """
    追蹤查詢生成節點。

    根據社群報告生成追蹤查詢。

    Returns:
        更新後的狀態，包含 followup_queries
    """
    import time
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("followups", "START")

    start_time = time.time()
    question = state.get("normalized_question") or state.get("question", "")
    community_reports = state.get("community_reports", [])
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))

    logger.debug("Generating follow-up queries...")

    # 從社群報告中抽取主題和實體
    all_themes = set()
    all_entities = set()

    for report in community_reports:
        all_themes.update(report.get("key_themes", []))
        all_entities.update(report.get("key_entities", []))

    # 根據主題生成追蹤查詢
    followup_queries = []
    for theme in list(all_themes)[:3]:
        followup_queries.append(f"{question} 關於 {theme}")

    logger.info(f"Generated {len(followup_queries)} follow-up queries")

    emit_status("followups", "DONE")
    return {
        "followup_queries": followup_queries,
        "retrieval_path": retrieval_path + [f"followups:{len(followup_queries)}"],
        "timing": {**timing, "followups_ms": (time.time() - start_time) * 1000},
    }


async def rrf_merge_node(state: GraphRAGState) -> dict[str, Any]:
    """
    RRF 融合節點。

    使用倒數排名融合（Reciprocal Rank Fusion）合併來自多個檢索來源的結果。

    Returns:
        更新後的狀態，包含 merged_results
    """
    import time
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("rrf_merge", "START")

    start_time = time.time()
    seed_results = state.get("seed_results")
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))

    logger.debug("Merging retrieval results...")

    # 從種子結果取得所有 chunk
    all_chunks = []
    if seed_results and hasattr(seed_results, "chunks"):
        all_chunks = list(seed_results.chunks)

    # 在完整實現中，我們還會合併：
    # - 社群深入結果
    # - 追蹤查詢結果
    # - 圖譜遍歷結果

    merged_results = RetrievalResult(
        chunks=all_chunks,
        query_used=seed_results.query_used if seed_results else "",
        mode=QueryMode.LOCAL,
        retrieval_path=["rrf_merge"],
    )

    logger.info(f"Merged {len(all_chunks)} chunks")

    emit_status("rrf_merge", "DONE")
    return {
        "merged_results": merged_results,
        "retrieval_path": retrieval_path + [f"rrf_merge:{len(all_chunks)}"],
        "timing": {**timing, "rrf_merge_ms": (time.time() - start_time) * 1000},
    }


async def hop_hybrid_node(state: GraphRAGState) -> dict[str, Any]:
    """
    跳躍混合搜尋節點。

    根據圖譜遍歷結果執行額外的混合搜尋。
    這實現了 GraphRAG 的「跳躍」部分 - 使用實體衍生的
    追蹤查詢來找到額外的相關文件。

    該節點執行從以下來源生成的追蹤查詢：
    - 圖譜子圖實體（subgraph_to_queries_node）
    - 社群報告主題
    - 實體關係

    Returns:
        更新後的狀態，將額外的 chunk 合併到 merged_results 中
    """
    import asyncio
    import time
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("hop_hybrid", "START")

    start_time = time.time()
    followup_queries = state.get("followup_queries", [])

    # Handle merged_results which may be a RetrievalResult object or a list
    raw_merged_results = state.get("merged_results")
    if raw_merged_results is None:
        merged_results = []
    elif hasattr(raw_merged_results, "chunks"):
        # It's a RetrievalResult object - extract chunks
        merged_results = list(raw_merged_results.chunks)
    elif isinstance(raw_merged_results, list):
        merged_results = list(raw_merged_results)
    else:
        merged_results = []

    acl_groups = state.get("acl_groups", ["public"])
    tenant_id = state.get("tenant_id", "default")
    filter_context = state.get("filter_context")
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))
    budget = state.get("budget")

    # 在額外搜尋之前檢查預算
    if budget and hasattr(budget, "is_exhausted") and budget.is_exhausted():
        logger.info("Budget exhausted - skipping hop hybrid")
        emit_status("hop_hybrid", "DONE")
        return {
            "retrieval_path": retrieval_path + ["hop_hybrid:budget_exhausted"],
            "timing": {**timing, "hop_hybrid_ms": (time.time() - start_time) * 1000},
        }

    # 如果沒有追蹤查詢則跳過
    if not followup_queries:
        logger.debug("No follow-up queries - skipping hop hybrid")
        emit_status("hop_hybrid", "DONE")
        return {
            "retrieval_path": retrieval_path + ["hop_hybrid:no_queries"],
            "timing": {**timing, "hop_hybrid_ms": (time.time() - start_time) * 1000},
        }

    # 限制查詢以防止過多的 API 呼叫
    max_hop_queries = 3
    queries_to_search = followup_queries[:max_hop_queries]

    logger.info(f"Hop hybrid search: executing {len(queries_to_search)} follow-up queries")

    try:
        # Import services lazily
        from chatbot_graphrag.services.search.hybrid_search import (
            HybridSearchConfig,
            hybrid_search_service,
        )
        from chatbot_graphrag.services.vector.embedding_service import embedding_service

        # Initialize services
        await hybrid_search_service.initialize()
        await embedding_service.initialize()

        # 並行為所有追蹤查詢生成嵌入向量
        embedding_tasks = [embedding_service.embed_text(q) for q in queries_to_search]
        embeddings = await asyncio.gather(*embedding_tasks)

        # 如果可用則使用 filter_context 建構搜尋配置
        if filter_context:
            config_acl = filter_context.acl_groups
            config_tenant = filter_context.tenant_id
        else:
            config_acl = acl_groups
            config_tenant = tenant_id

        # 第 1 階段：只在明確設定時按 tenant_id 過濾
        effective_tenant = config_tenant if config_tenant != "default" else None

        hop_config = HybridSearchConfig(
            dense_limit=20,   # Smaller limits for follow-up queries
            sparse_limit=20,
            fts_limit=20,
            final_limit=10,   # Return fewer results per hop query
            acl_groups=config_acl,
            tenant_id=effective_tenant,
        )

        # Execute searches for all queries in parallel
        search_tasks = []
        for query, embedding in zip(queries_to_search, embeddings):
            search_tasks.append(
                hybrid_search_service.search(
                    query=query,
                    query_embedding=embedding,
                    config=hop_config,
                )
            )

        all_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # 追蹤現有的 chunk ID 以避免重複
        existing_chunk_ids = {r.chunk_id for r in merged_results}
        new_chunks = []
        chunk_scores: dict[str, float] = {}

        for i, results in enumerate(all_results):
            if isinstance(results, Exception):
                logger.warning(f"Hop search {i} failed: {results}")
                continue

            # 跳躍查詢獲得遞減權重（0.6, 0.5, 0.4...）
            weight = 0.6 - (i * 0.1)
            weight = max(weight, 0.3)  # Minimum weight

            for result in results:
                chunk_id = result.chunk_id
                if chunk_id not in existing_chunk_ids:
                    existing_chunk_ids.add(chunk_id)
                    # 根據跳躍權重調整分數
                    result.score *= weight
                    new_chunks.append(result)
                    chunk_scores[chunk_id] = result.score
                elif chunk_id in chunk_scores:
                    # 對被多個跳躍查詢找到的 chunk 提升分數
                    chunk_scores[chunk_id] = max(
                        chunk_scores[chunk_id],
                        result.score * weight
                    )

        # 按分數排序新的 chunk
        new_chunks.sort(key=lambda r: r.score, reverse=True)

        # 限制新 chunk 數量以防止上下文溢出
        max_new_chunks = 5
        new_chunks = new_chunks[:max_new_chunks]

        # 將新 chunk 與現有結果合併
        if new_chunks:
            merged_results.extend(new_chunks)
            logger.info(f"Hop hybrid added {len(new_chunks)} new chunks")

        # Update budget
        if budget and hasattr(budget, "increment_queries"):
            budget.increment_queries(len(queries_to_search))

        # 將結果包裝回 RetrievalResult 供下游節點使用（例如 rerank_node）
        updated_result = RetrievalResult(
            chunks=merged_results,
            query_used=raw_merged_results.query_used if hasattr(raw_merged_results, "query_used") else "",
            mode=raw_merged_results.mode if hasattr(raw_merged_results, "mode") else QueryMode.LOCAL,
            retrieval_path=["hop_hybrid"],
        )

        emit_status("hop_hybrid", "DONE")
        return {
            "merged_results": updated_result,
            "retrieval_path": retrieval_path + [f"hop_hybrid:{len(new_chunks)}_new"],
            "timing": {**timing, "hop_hybrid_ms": (time.time() - start_time) * 1000},
            "budget": budget,
        }

    except Exception as e:
        logger.error(f"Hop hybrid search error: {e}")
        emit_status("hop_hybrid", "DONE")
        return {
            "error": str(e),
            "retrieval_path": retrieval_path + ["hop_hybrid:error"],
            "timing": {**timing, "hop_hybrid_ms": (time.time() - start_time) * 1000},
        }
