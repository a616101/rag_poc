"""
GraphRAG 全域模式 (Global Mode)

社區報告 → 追問查詢 → 本地深入。

流程：
1. 搜索社區報告找到相關社區
2. 從社區上下文生成追問查詢
3. 使用本地模式深入執行追問
4. 合併和排序最終結果
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from chatbot_graphrag.core.constants import MAX_COMMUNITY_LEVEL
from chatbot_graphrag.services.search.hybrid_search import (
    HybridSearchConfig,
    SearchResult,
    hybrid_search_service,
)
from chatbot_graphrag.services.retrieval.local_mode import (
    LocalModeConfig,
    LocalModeResult,
    local_mode_retriever,
)

logger = logging.getLogger(__name__)


@dataclass
class GlobalModeConfig:
    """全域模式檢索配置。"""

    # 社區檢索
    max_communities: int = 5               # 最大社區數
    min_community_score: float = 0.3       # 最低社區分數
    community_levels: list[int] = field(default_factory=lambda: [1, 2])  # 社區層級

    # 追問生成
    max_followups: int = 3                 # 最大追問數

    # 本地深入
    enable_drill_down: bool = True         # 是否啟用本地深入
    drill_down_config: Optional[LocalModeConfig] = None  # 本地模式配置

    # 過濾
    doc_types: Optional[list[str]] = None  # 限制文件類型
    acl_groups: Optional[list[str]] = None # ACL 群組過濾


@dataclass
class CommunityReport:
    """社區報告資料。"""

    community_id: str              # 社區 ID
    level: int                     # 社區層級
    title: str                     # 報告標題
    summary: str                   # 報告摘要
    key_entities: list[str]        # 關鍵實體
    key_themes: list[str]          # 關鍵主題
    score: float                   # 相關性分數
    metadata: dict[str, Any] = field(default_factory=dict)  # 元資料


@dataclass
class FollowUpQuery:
    """生成的追問查詢。"""

    query: str                          # 查詢內容
    target_community_ids: list[str]     # 目標社區 ID
    reasoning: str                      # 推理說明


@dataclass
class GlobalModeResult:
    """全域模式檢索結果。"""

    community_reports: list[CommunityReport]       # 社區報告列表
    followup_queries: list[FollowUpQuery]          # 追問查詢列表
    drill_down_results: list[LocalModeResult]      # 深入結果列表
    final_chunks: list[SearchResult]               # 最終 chunks
    retrieval_path: list[str] = field(default_factory=list)  # 檢索路徑


class GlobalModeRetriever:
    """
    GraphRAG 全域模式檢索器。

    流程：
    1. 搜索社區報告找到相關社區
    2. 從社區上下文生成追問查詢
    3. 使用本地模式深入執行追問
    4. 合併和排序最終結果
    """

    def __init__(self):
        """初始化全域模式檢索器。"""
        self._initialized = False

    async def initialize(self) -> None:
        """初始化依賴服務。"""
        if self._initialized:
            return

        self._hybrid_search = hybrid_search_service
        await self._hybrid_search.initialize()

        self._local_retriever = local_mode_retriever
        await self._local_retriever.initialize()

        # 延遲導入服務
        from chatbot_graphrag.services.vector import qdrant_service
        from chatbot_graphrag.services.storage import minio_service

        self._qdrant = qdrant_service
        self._minio = minio_service

        self._initialized = True
        logger.info("全域模式檢索器初始化完成")

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        sparse_embedding: Optional[dict[int, float]] = None,
        config: Optional[GlobalModeConfig] = None,
    ) -> GlobalModeResult:
        """
        使用全域模式策略檢索上下文。

        Args:
            query: 使用者查詢
            query_embedding: 密集嵌入向量
            sparse_embedding: 稀疏嵌入向量
            config: 檢索配置

        Returns:
            包含所有上下文的 GlobalModeResult
        """
        if not self._initialized:
            await self.initialize()

        config = config or GlobalModeConfig()
        retrieval_path = ["global_mode_start"]

        # 步驟 1: 搜索社區報告
        community_reports = await self._search_communities(
            query=query,
            query_embedding=query_embedding,
            config=config,
        )
        retrieval_path.append(f"community_search:{len(community_reports)}")

        if not community_reports:
            # 如果沒有找到社區，回退到本地模式
            local_result = await self._local_retriever.retrieve(
                query=query,
                query_embedding=query_embedding,
                sparse_embedding=sparse_embedding,
                config=config.drill_down_config,
            )
            retrieval_path.append("fallback_to_local")

            return GlobalModeResult(
                community_reports=[],
                followup_queries=[],
                drill_down_results=[local_result],
                final_chunks=local_result.final_chunks,
                retrieval_path=retrieval_path + local_result.retrieval_path,
            )

        # 步驟 2: 從社區上下文生成追問查詢
        followup_queries = await self._generate_followups(
            original_query=query,
            community_reports=community_reports,
            config=config,
        )
        retrieval_path.append(f"followups_generated:{len(followup_queries)}")

        # 步驟 3: 執行深入查詢
        drill_down_results: list[LocalModeResult] = []

        if config.enable_drill_down and followup_queries:
            # 並行執行追問
            tasks = [
                self._execute_followup(fq, query_embedding, config)
                for fq in followup_queries
            ]
            drill_down_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 過濾掉例外
            drill_down_results = [
                r for r in drill_down_results if not isinstance(r, Exception)
            ]
            retrieval_path.append(f"drill_downs:{len(drill_down_results)}")

        # 步驟 4: 合併和排序最終 chunks
        final_chunks = self._merge_results(
            community_reports=community_reports,
            drill_down_results=drill_down_results,
        )
        retrieval_path.append(f"final_chunks:{len(final_chunks)}")

        return GlobalModeResult(
            community_reports=community_reports,
            followup_queries=followup_queries,
            drill_down_results=drill_down_results,
            final_chunks=final_chunks,
            retrieval_path=retrieval_path,
        )

    async def _search_communities(
        self,
        query: str,
        query_embedding: list[float],
        config: GlobalModeConfig,
    ) -> list[CommunityReport]:
        """使用嵌入向量搜索社區報告。"""
        community_reports: list[CommunityReport] = []

        try:
            # 在 Qdrant 中搜索社區嵌入
            results = await self._qdrant.search_communities(
                query_embedding=query_embedding,
                limit=config.max_communities,
                min_score=config.min_community_score,
                levels=config.community_levels,
            )

            for result in results:
                # 從 MinIO 載入完整報告
                community_id = result["community_id"]
                level = result.get("level", 1)

                report_data = await self._minio.load_community_report(
                    community_id=community_id,
                    level=level,
                )

                if report_data:
                    community_reports.append(
                        CommunityReport(
                            community_id=community_id,
                            level=level,
                            title=report_data.get("title", ""),
                            summary=report_data.get("summary", ""),
                            key_entities=report_data.get("key_entities", []),
                            key_themes=report_data.get("key_themes", []),
                            score=result["score"],
                            metadata=report_data.get("metadata", {}),
                        )
                    )

        except Exception as e:
            logger.error(f"Community search error: {e}")

        return community_reports

    async def _generate_followups(
        self,
        original_query: str,
        community_reports: list[CommunityReport],
        config: GlobalModeConfig,
    ) -> list[FollowUpQuery]:
        """從社區上下文生成追問查詢。"""
        followup_queries: list[FollowUpQuery] = []

        # 從社區抽取關鍵主題和實體
        all_themes = set()
        all_entities = set()
        community_ids = []

        for report in community_reports:
            all_themes.update(report.key_themes)
            all_entities.update(report.key_entities)
            community_ids.append(report.community_id)

        # 基於主題和實體生成追問
        # 在生產環境中，這會使用 LLM 生成針對性查詢
        for i, theme in enumerate(list(all_themes)[: config.max_followups]):
            followup_queries.append(
                FollowUpQuery(
                    query=f"{original_query} 關於 {theme}",
                    target_community_ids=community_ids,
                    reasoning=f"探索社群報告中的主題: {theme}",
                )
            )

        return followup_queries

    async def _execute_followup(
        self,
        followup: FollowUpQuery,
        query_embedding: list[float],
        config: GlobalModeConfig,
    ) -> LocalModeResult:
        """使用本地模式執行追問查詢。"""
        # 在生產環境中，您會為追問查詢生成新的嵌入
        # 目前重用原始嵌入

        local_config = config.drill_down_config or LocalModeConfig(
            seed_limit=10,
            max_hops=1,
            doc_types=config.doc_types,
            acl_groups=config.acl_groups,
        )

        return await self._local_retriever.retrieve(
            query=followup.query,
            query_embedding=query_embedding,
            config=local_config,
        )

    def _merge_results(
        self,
        community_reports: list[CommunityReport],
        drill_down_results: list[LocalModeResult],
    ) -> list[SearchResult]:
        """合併和排序所有來源的結果。"""
        all_chunks: dict[str, SearchResult] = {}

        # 添加深入結果的 chunks
        for result in drill_down_results:
            for chunk in result.final_chunks:
                if chunk.chunk_id not in all_chunks:
                    all_chunks[chunk.chunk_id] = chunk
                else:
                    # 如果在多個深入結果中找到，提升分數
                    all_chunks[chunk.chunk_id].score = max(
                        all_chunks[chunk.chunk_id].score,
                        chunk.score,
                    )

        # 按分數排序
        sorted_chunks = sorted(
            all_chunks.values(),
            key=lambda x: x.score,
            reverse=True,
        )

        return sorted_chunks[:12]  # 限制最終結果數量


# 單例實例
global_mode_retriever = GlobalModeRetriever()
