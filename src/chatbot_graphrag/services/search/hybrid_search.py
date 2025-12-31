"""
混合搜尋服務

結合密集向量搜尋、稀疏向量搜尋和全文搜尋，
使用 Reciprocal Rank Fusion (RRF) 進行融合。

第 5 階段：支援基於查詢分類的查詢感知權重調整。

主要功能：
- 三路混合搜尋（dense + sparse + FTS）
- RRF 分數融合
- 查詢感知的動態權重調整
- 漸進式搜尋（逐步降低閾值）
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from chatbot_graphrag.core.constants import (
    DEFAULT_RRF_K,
    DEFAULT_RRF_WEIGHTS,
    SCORE_THRESHOLD_HIGH,
    SCORE_THRESHOLD_LOW,
    SCORE_THRESHOLD_MEDIUM,
)

logger = logging.getLogger(__name__)


# 延遲載入的查詢分類器
_query_classifier: Optional[Any] = None


def _get_query_classifier():
    """取得延遲載入的查詢分類器實例。"""
    global _query_classifier
    if _query_classifier is None:
        try:
            from chatbot_graphrag.services.search.query_classifier import get_query_classifier
            _query_classifier = get_query_classifier()
        except ImportError:
            logger.warning("Query classifier not available")
            _query_classifier = False  # Mark as unavailable
    return _query_classifier if _query_classifier else None


@dataclass
class SearchResult:
    """所有搜尋方法的統一搜尋結果。"""

    chunk_id: str  # chunk 唯一識別碼
    doc_id: str  # 所屬文件識別碼
    content: str  # chunk 內容
    score: float  # 最終分數（正規化後的 RRF 分數）
    source: str  # 來源："dense", "sparse", "fts", "hybrid"
    metadata: dict[str, Any] = field(default_factory=dict)  # 額外元資料

    # 個別搜尋方法的分數
    dense_score: float = 0.0  # 密集向量搜尋分數
    sparse_score: float = 0.0  # 稀疏向量搜尋分數
    fts_score: float = 0.0  # 全文搜尋分數
    rrf_score: float = 0.0  # RRF 融合分數（未正規化）


@dataclass
class HybridSearchConfig:
    """混合搜尋配置。"""

    # 結果數量限制
    dense_limit: int = 40  # 密集向量搜尋結果數
    sparse_limit: int = 40  # 稀疏向量搜尋結果數
    fts_limit: int = 40  # 全文搜尋結果數
    final_limit: int = 20  # 最終返回結果數

    # RRF 參數
    rrf_k: int = DEFAULT_RRF_K  # RRF k 常數（預設 60）
    dense_weight: float = DEFAULT_RRF_WEIGHTS[0]  # 密集搜尋權重
    sparse_weight: float = DEFAULT_RRF_WEIGHTS[1]  # 稀疏搜尋權重
    fts_weight: float = DEFAULT_RRF_WEIGHTS[2]  # 全文搜尋權重

    # 分數閾值
    min_score: float = SCORE_THRESHOLD_LOW  # 最低分數閾值

    # 過濾器
    doc_types: Optional[list[str]] = None  # 文件類型過濾
    acl_groups: Optional[list[str]] = None  # ACL 群組過濾
    tenant_id: Optional[str] = None  # 第 1 階段：多租戶過濾
    department: Optional[str] = None  # 部門過濾
    title_contains: Optional[str] = None  # 標題包含字串過濾

    # 第 5 階段：查詢感知權重調整
    use_query_aware_weights: bool = True  # 根據查詢類型自動調整權重
    query_type: Optional[str] = None  # 偵測到的查詢類型（用於除錯）


class HybridSearchService:
    """
    結合多種搜尋模式的混合搜尋服務。

    使用 Reciprocal Rank Fusion (RRF) 合併以下來源的結果：
    - 密集向量搜尋（Qdrant）
    - 稀疏向量搜尋（Qdrant SPLADE）
    - 全文搜尋（OpenSearch）
    """

    def __init__(self):
        """初始化混合搜尋服務。"""
        self._initialized = False

    async def initialize(self) -> None:
        """初始化依賴服務。"""
        if self._initialized:
            return

        # 延遲匯入服務
        from chatbot_graphrag.services.vector import qdrant_service
        from chatbot_graphrag.services.search import opensearch_service

        self._qdrant = qdrant_service
        self._opensearch = opensearch_service

        await self._qdrant.initialize()
        await self._opensearch.initialize()

        self._initialized = True
        logger.info("混合搜尋服務已初始化")

    async def search(
        self,
        query: str,
        query_embedding: list[float],
        sparse_embedding: Optional[dict[int, float]] = None,
        config: Optional[HybridSearchConfig] = None,
    ) -> list[SearchResult]:
        """
        執行結合所有模式的混合搜尋。

        Args:
            query: 原始查詢文字
            query_embedding: 密集嵌入向量
            sparse_embedding: 稀疏嵌入（token_id -> 權重）
            config: 搜尋配置

        Returns:
            按 RRF 分數排序的 SearchResult 列表
        """
        if not self._initialized:
            await self.initialize()

        config = config or HybridSearchConfig()

        # 第 5 階段：應用查詢感知權重調整
        if config.use_query_aware_weights:
            classifier = _get_query_classifier()
            if classifier:
                classification = classifier.classify(query)
                config.dense_weight = classification.recommended_weights[0]
                config.sparse_weight = classification.recommended_weights[1]
                config.fts_weight = classification.recommended_weights[2]
                config.query_type = classification.query_type.value
                logger.debug(
                    f"Query-aware weights applied: type={classification.query_type.value}, "
                    f"weights=({config.dense_weight}, {config.sparse_weight}, {config.fts_weight})"
                )

        # 建構過濾器
        filters = self._build_filters(config)

        # 並行執行所有搜尋
        dense_task = self._dense_search(query_embedding, config.dense_limit, filters)
        sparse_task = self._sparse_search(sparse_embedding, config.sparse_limit, filters)
        fts_task = self._fts_search(query, config.fts_limit, filters)

        results = await asyncio.gather(dense_task, sparse_task, fts_task, return_exceptions=True)

        # 處理錯誤
        dense_results = results[0] if not isinstance(results[0], Exception) else []
        sparse_results = results[1] if not isinstance(results[1], Exception) else []
        fts_results = results[2] if not isinstance(results[2], Exception) else []

        if isinstance(results[0], Exception):
            logger.error(f"Dense search failed: {results[0]}")
        else:
            logger.info(f"Dense search returned {len(dense_results)} results")
        if isinstance(results[1], Exception):
            logger.error(f"Sparse search failed: {results[1]}")
        else:
            logger.info(f"Sparse search returned {len(sparse_results)} results")
        if isinstance(results[2], Exception):
            logger.error(f"FTS search failed: {results[2]}")
        else:
            logger.info(f"FTS search returned {len(fts_results)} results")

        # 應用 RRF 融合
        fused_results = self._rrf_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            fts_results=fts_results,
            config=config,
        )
        logger.info(f"RRF fusion produced {len(fused_results)} results")
        if fused_results:
            logger.info(f"Top normalized score: {fused_results[0].score:.4f}, min_score threshold: {config.min_score}")

        # 按最低正規化分數過濾（非原始 RRF 分數）
        filtered_results = [
            r for r in fused_results if r.score >= config.min_score
        ]
        logger.info(f"After min_score filter: {len(filtered_results)} results")

        # 限制結果數量
        return filtered_results[: config.final_limit]

    async def _dense_search(
        self,
        embedding: list[float],
        limit: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """執行密集向量搜尋。"""
        try:
            # Convert our filter format to Qdrant format
            qdrant_filters = None
            title_contains = None

            if filters:
                qdrant_filters = {}
                if "doc_types" in filters:
                    qdrant_filters["doc_type"] = filters["doc_types"]
                if "acl_groups" in filters:
                    qdrant_filters["acl_groups"] = filters["acl_groups"]
                if "tenant_id" in filters:  # Phase 1: multi-tenant
                    qdrant_filters["tenant_id"] = filters["tenant_id"]
                if "department" in filters:
                    qdrant_filters["department"] = filters["department"]
                # Handle title_contains separately for post-filtering
                title_contains = filters.get("title_contains")

            # Fetch more results if we need to post-filter by title
            fetch_limit = limit * 3 if title_contains else limit

            results = await self._qdrant.search_dense(
                query_embedding=embedding,
                limit=fetch_limit,
                score_threshold=0.0,  # 在 Qdrant 層級不進行過濾 - 讓 RRF 處理評分
                filter_conditions=qdrant_filters if qdrant_filters else None,
            )

            # Convert ScoredPoint objects to SearchResult
            search_results = []
            for r in results:
                # 按標題包含進行後過濾（不區分大小寫）
                if title_contains:
                    title = r.payload.get("title", "")
                    if title_contains not in title:
                        continue

                search_results.append(
                    SearchResult(
                        chunk_id=r.payload.get("chunk_id", str(r.id)),
                        doc_id=r.payload.get("doc_id", ""),
                        content=r.payload.get("content", ""),
                        score=r.score,
                        source="dense",
                        metadata=r.payload,
                        dense_score=r.score,
                    )
                )

                # Stop once we have enough results
                if len(search_results) >= limit:
                    break

            return search_results
        except Exception as e:
            logger.error(f"Dense search error: {e}")
            return []

    async def _sparse_search(
        self,
        sparse_embedding: Optional[dict[int, float]],
        limit: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """執行稀疏向量搜尋。"""
        if not sparse_embedding:
            return []

        try:
            # 轉換過濾格式到 Qdrant 格式
            qdrant_filters = None
            title_contains = None

            if filters:
                qdrant_filters = {}
                if "doc_types" in filters:
                    qdrant_filters["doc_type"] = filters["doc_types"]
                if "acl_groups" in filters:
                    qdrant_filters["acl_groups"] = filters["acl_groups"]
                if "tenant_id" in filters:  # 第 1 階段：多租戶
                    qdrant_filters["tenant_id"] = filters["tenant_id"]
                if "department" in filters:
                    qdrant_filters["department"] = filters["department"]
                # 單獨處理 title_contains 用於後過濾
                title_contains = filters.get("title_contains")

            # 如果需要按標題後過濾則取得更多結果
            fetch_limit = limit * 3 if title_contains else limit

            results = await self._qdrant.search_sparse(
                sparse_embedding=sparse_embedding,
                limit=fetch_limit,
                filter_conditions=qdrant_filters if qdrant_filters else None,
            )

            # 轉換 ScoredPoint 物件為 SearchResult
            search_results = []
            for r in results:
                # 按標題包含進行後過濾
                if title_contains:
                    title = r.payload.get("title", "")
                    if title_contains not in title:
                        continue

                search_results.append(
                    SearchResult(
                        chunk_id=r.payload.get("chunk_id", str(r.id)),
                        doc_id=r.payload.get("doc_id", ""),
                        content=r.payload.get("content", ""),
                        score=r.score,
                        source="sparse",
                        metadata=r.payload,
                        sparse_score=r.score,
                    )
                )

                # Stop once we have enough results
                if len(search_results) >= limit:
                    break

            return search_results
        except Exception as e:
            logger.error(f"Sparse search error: {e}")
            return []

    async def _fts_search(
        self,
        query: str,
        limit: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """執行全文搜尋。"""
        try:
            # 建構 OpenSearch 過濾器字典（不是列表 - OpenSearchService._build_filters 處理字典）
            os_filters = {}
            title_contains = None

            if filters:
                if "doc_types" in filters:
                    os_filters["doc_type"] = filters["doc_types"]
                if "acl_groups" in filters:
                    os_filters["acl_groups"] = filters["acl_groups"]
                if "tenant_id" in filters:  # 第 1 階段：多租戶
                    os_filters["tenant_id"] = filters["tenant_id"]
                if "department" in filters:
                    os_filters["department"] = filters["department"]
                # 單獨處理 title_contains 用於後過濾
                title_contains = filters.get("title_contains")

            # 如果需要按標題後過濾則取得更多結果
            fetch_limit = limit * 3 if title_contains else limit

            results = await self._opensearch.search(
                query=query,
                limit=fetch_limit,
                filters=os_filters if os_filters else None,
            )

            search_results = []
            for r in results:
                # 按標題包含進行後過濾
                if title_contains:
                    title = r.get("title", "")
                    if title_contains not in title:
                        continue

                search_results.append(
                    SearchResult(
                        chunk_id=r.get("chunk_id", r.get("_id", "")),
                        doc_id=r.get("doc_id", ""),
                        content=r.get("content", ""),
                        score=r.get("score", 0.0),
                        source="fts",
                        metadata={k: v for k, v in r.items() if k not in ("chunk_id", "doc_id", "content", "score")},
                        fts_score=r.get("score", 0.0),
                    )
                )

                # Stop once we have enough results
                if len(search_results) >= limit:
                    break

            return search_results
        except Exception as e:
            logger.error(f"FTS search error: {e}")
            return []

    def _rrf_fusion(
        self,
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
        fts_results: list[SearchResult],
        config: HybridSearchConfig,
    ) -> list[SearchResult]:
        """
        應用 Reciprocal Rank Fusion 合併結果。

        RRF 分數 = sum(weight / (k + rank)) 對於每個結果集
        """
        # 建立 chunk_id -> 結果的映射
        result_map: dict[str, SearchResult] = {}

        # 處理密集搜尋結果
        for rank, result in enumerate(dense_results, start=1):
            if result.chunk_id not in result_map:
                result_map[result.chunk_id] = result
                result_map[result.chunk_id].source = "hybrid"
            result_map[result.chunk_id].dense_score = result.dense_score
            rrf_contribution = config.dense_weight / (config.rrf_k + rank)
            result_map[result.chunk_id].rrf_score += rrf_contribution

        # 處理稀疏搜尋結果
        for rank, result in enumerate(sparse_results, start=1):
            if result.chunk_id not in result_map:
                result_map[result.chunk_id] = result
                result_map[result.chunk_id].source = "hybrid"
            result_map[result.chunk_id].sparse_score = result.sparse_score
            rrf_contribution = config.sparse_weight / (config.rrf_k + rank)
            result_map[result.chunk_id].rrf_score += rrf_contribution

        # 處理全文搜尋結果
        for rank, result in enumerate(fts_results, start=1):
            if result.chunk_id not in result_map:
                result_map[result.chunk_id] = result
                result_map[result.chunk_id].source = "hybrid"
            result_map[result.chunk_id].fts_score = result.fts_score
            rrf_contribution = config.fts_weight / (config.rrf_k + rank)
            result_map[result.chunk_id].rrf_score += rrf_contribution

        # 按 RRF 分數排序
        sorted_results = sorted(
            result_map.values(),
            key=lambda x: x.rrf_score,
            reverse=True,
        )

        # 將分數正規化為最終分數
        if sorted_results:
            max_rrf = sorted_results[0].rrf_score
            for result in sorted_results:
                result.score = result.rrf_score / max_rrf if max_rrf > 0 else 0

        return sorted_results

    def _build_filters(
        self,
        config: HybridSearchConfig,
    ) -> dict[str, Any]:
        """為搜尋查詢建構過濾器。"""
        filters = {}

        if config.doc_types:
            filters["doc_types"] = config.doc_types
        if config.acl_groups:
            filters["acl_groups"] = config.acl_groups
        if config.tenant_id:  # 第 1 階段：多租戶過濾
            filters["tenant_id"] = config.tenant_id
        if config.department:
            filters["department"] = config.department
        if config.title_contains:
            filters["title_contains"] = config.title_contains

        return filters if filters else None

    async def search_progressive(
        self,
        query: str,
        query_embedding: list[float],
        sparse_embedding: Optional[dict[int, float]] = None,
        config: Optional[HybridSearchConfig] = None,
        thresholds: tuple[float, float, float] = (
            SCORE_THRESHOLD_HIGH,
            SCORE_THRESHOLD_MEDIUM,
            SCORE_THRESHOLD_LOW,
        ),
    ) -> tuple[list[SearchResult], str]:
        """
        漸進式搜尋，逐步降低閾值。

        先嘗試高閾值，然後中閾值，最後低閾值。

        Args:
            query: 查詢文字
            query_embedding: 密集嵌入
            sparse_embedding: 稀疏嵌入
            config: 搜尋配置
            thresholds: (高, 中, 低) 分數閾值

        Returns:
            (結果列表, 使用的閾值等級) 元組
        """
        config = config or HybridSearchConfig()
        threshold_names = ("high", "medium", "low")

        for threshold, name in zip(thresholds, threshold_names):
            config.min_score = threshold
            results = await self.search(
                query=query,
                query_embedding=query_embedding,
                sparse_embedding=sparse_embedding,
                config=config,
            )

            if len(results) >= 3:  # 最少有用結果數
                logger.info(f"Progressive search succeeded at {name} threshold ({threshold})")
                return results, name

        # 返回在最低閾值下取得的結果
        logger.info("Progressive search exhausted all thresholds")
        return results, "low"


# 單例實例
hybrid_search_service = HybridSearchService()
