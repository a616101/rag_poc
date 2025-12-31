"""
GraphRAG 漂移模式 (DRIFT Mode)

通過聚焦遍歷進行動態推理和推斷。
社區擴展 → 多次追問 → 深度探索。

流程：
1. 從基於社區的全域搜索開始
2. 迭代擴展到相關社區
3. 追蹤新資訊的新穎度
4. 當預算耗盡或新穎度下降時終止
5. 將所有發現合併到最終上下文
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from chatbot_graphrag.core.constants import DEFAULT_LOOP_BUDGET, MAX_COMMUNITY_LEVEL
from chatbot_graphrag.services.search.hybrid_search import (
    SearchResult,
    hybrid_search_service,
)
from chatbot_graphrag.services.retrieval.local_mode import (
    LocalModeConfig,
    LocalModeResult,
    local_mode_retriever,
)
from chatbot_graphrag.services.retrieval.global_mode import (
    CommunityReport,
    GlobalModeConfig,
    global_mode_retriever,
)

logger = logging.getLogger(__name__)


@dataclass
class DriftModeConfig:
    """漂移模式檢索配置。"""

    # 迴圈預算
    max_iterations: int = DEFAULT_LOOP_BUDGET["max_loops"]            # 最大迭代次數
    max_queries: int = DEFAULT_LOOP_BUDGET["max_new_queries"]         # 最大查詢次數
    max_wall_time_seconds: float = DEFAULT_LOOP_BUDGET["max_wall_time_seconds"]  # 最大時間

    # 社區擴展
    initial_communities: int = 3       # 初始社區數
    expansion_factor: float = 1.5      # 每次迭代的社區擴展係數
    max_communities: int = 8           # 最大社區數

    # 探索深度
    explore_depth: int = 2             # 探索深度
    min_novelty_score: float = 0.3     # 繼續擴展的最低新穎度

    # 結果限制
    max_final_chunks: int = 15         # 最終 chunks 上限

    # 過濾
    doc_types: Optional[list[str]] = None  # 限制文件類型
    acl_groups: Optional[list[str]] = None # ACL 群組過濾


@dataclass
class DriftIteration:
    """單次 DRIFT 迭代結果。"""

    iteration: int                      # 迭代編號
    query: str                          # 查詢內容
    communities_explored: list[str]     # 探索的社區
    chunks_found: list[SearchResult]    # 找到的 chunks
    novelty_score: float                # 新穎度分數
    duration_ms: float                  # 持續時間（毫秒）


@dataclass
class DriftModeResult:
    """漂移模式檢索結果。"""

    iterations: list[DriftIteration]          # 迭代列表
    community_reports: list[CommunityReport]  # 社區報告列表
    final_chunks: list[SearchResult]          # 最終 chunks
    total_queries: int                        # 總查詢次數
    total_duration_ms: float                  # 總持續時間（毫秒）
    terminated_reason: str  # 終止原因: "max_iterations", "max_queries", "timeout", "low_novelty", "complete"
    retrieval_path: list[str] = field(default_factory=list)  # 檢索路徑


class DriftModeRetriever:
    """
    GraphRAG 漂移模式檢索器。

    DRIFT = Dynamic Reasoning and Inference through Focused Traversal
            (通過聚焦遍歷進行動態推理和推斷)

    流程：
    1. 從基於社區的全域搜索開始
    2. 迭代擴展到相關社區
    3. 追蹤新資訊的新穎度
    4. 當預算耗盡或新穎度下降時終止
    5. 將所有發現合併到最終上下文
    """

    def __init__(self):
        """初始化漂移模式檢索器。"""
        self._initialized = False

    async def initialize(self) -> None:
        """初始化依賴服務。"""
        if self._initialized:
            return

        self._hybrid_search = hybrid_search_service
        await self._hybrid_search.initialize()

        self._local_retriever = local_mode_retriever
        await self._local_retriever.initialize()

        self._global_retriever = global_mode_retriever
        await self._global_retriever.initialize()

        # 延遲導入圖譜服務
        from chatbot_graphrag.services.graph import nebula_client

        self._nebula = nebula_client

        self._initialized = True
        logger.info("漂移模式檢索器初始化完成")

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        sparse_embedding: Optional[dict[int, float]] = None,
        config: Optional[DriftModeConfig] = None,
    ) -> DriftModeResult:
        """
        使用漂移模式策略檢索上下文。

        Args:
            query: 使用者查詢
            query_embedding: 密集嵌入向量
            sparse_embedding: 稀疏嵌入向量
            config: 檢索配置

        Returns:
            包含所有上下文的 DriftModeResult
        """
        if not self._initialized:
            await self.initialize()

        config = config or DriftModeConfig()
        retrieval_path = ["drift_mode_start"]

        import time

        start_time = time.time()

        iterations: list[DriftIteration] = []
        all_communities: list[CommunityReport] = []
        all_chunks: dict[str, SearchResult] = {}
        explored_community_ids: set[str] = set()
        total_queries = 0

        # 初始全域搜索
        global_config = GlobalModeConfig(
            max_communities=config.initial_communities,
            enable_drill_down=True,
            doc_types=config.doc_types,
            acl_groups=config.acl_groups,
        )

        global_result = await self._global_retriever.retrieve(
            query=query,
            query_embedding=query_embedding,
            sparse_embedding=sparse_embedding,
            config=global_config,
        )

        # 記錄初始迭代
        initial_duration = (time.time() - start_time) * 1000
        iterations.append(
            DriftIteration(
                iteration=0,
                query=query,
                communities_explored=[c.community_id for c in global_result.community_reports],
                chunks_found=global_result.final_chunks,
                novelty_score=1.0,  # Initial results are all novel
                duration_ms=initial_duration,
            )
        )

        # Update tracking
        all_communities.extend(global_result.community_reports)
        for chunk in global_result.final_chunks:
            all_chunks[chunk.chunk_id] = chunk
        explored_community_ids.update(c.community_id for c in global_result.community_reports)
        total_queries += 1

        retrieval_path.append(f"initial_global:{len(global_result.final_chunks)}")

        # DRIFT 迭代
        terminated_reason = "complete"
        current_query = query

        for iteration in range(1, config.max_iterations + 1):
            # 檢查終止條件
            elapsed = (time.time() - start_time)
            if elapsed > config.max_wall_time_seconds:
                terminated_reason = "timeout"
                break

            if total_queries >= config.max_queries:
                terminated_reason = "max_queries"
                break

            # 擴展到相鄰社區
            neighboring_communities = await self._get_neighboring_communities(
                explored_community_ids,
                config,
            )

            new_communities = [
                c for c in neighboring_communities
                if c not in explored_community_ids
            ]

            if not new_communities:
                terminated_reason = "complete"
                break

            # 探索新社區
            iteration_start = time.time()
            expansion_query = await self._generate_expansion_query(
                original_query=query,
                current_communities=all_communities,
            )

            # 在新社區區域執行本地搜索
            local_config = LocalModeConfig(
                seed_limit=10,
                max_hops=config.explore_depth,
                doc_types=config.doc_types,
                acl_groups=config.acl_groups,
            )

            local_result = await self._local_retriever.retrieve(
                query=expansion_query,
                query_embedding=query_embedding,
                sparse_embedding=sparse_embedding,
                config=local_config,
            )

            total_queries += 1

            # 計算新穎度
            new_chunk_ids = set(c.chunk_id for c in local_result.final_chunks)
            existing_chunk_ids = set(all_chunks.keys())
            novel_ids = new_chunk_ids - existing_chunk_ids
            novelty_score = len(novel_ids) / len(new_chunk_ids) if new_chunk_ids else 0

            iteration_duration = (time.time() - iteration_start) * 1000

            iterations.append(
                DriftIteration(
                    iteration=iteration,
                    query=expansion_query,
                    communities_explored=list(new_communities)[:5],
                    chunks_found=local_result.final_chunks,
                    novelty_score=novelty_score,
                    duration_ms=iteration_duration,
                )
            )

            # Update tracking
            for chunk in local_result.final_chunks:
                if chunk.chunk_id not in all_chunks:
                    all_chunks[chunk.chunk_id] = chunk
            explored_community_ids.update(new_communities)

            retrieval_path.append(f"drift_iter_{iteration}:{len(novel_ids)}_novel")

            # 檢查新穎度閾值
            if novelty_score < config.min_novelty_score:
                terminated_reason = "low_novelty"
                break

        # 合併最終結果
        total_duration = (time.time() - start_time) * 1000

        final_chunks = self._rank_final_chunks(
            all_chunks=all_chunks,
            iterations=iterations,
            max_chunks=config.max_final_chunks,
        )

        retrieval_path.append(f"drift_complete:{terminated_reason}")

        return DriftModeResult(
            iterations=iterations,
            community_reports=all_communities,
            final_chunks=final_chunks,
            total_queries=total_queries,
            total_duration_ms=total_duration,
            terminated_reason=terminated_reason,
            retrieval_path=retrieval_path,
        )

    async def _get_neighboring_communities(
        self,
        current_community_ids: set[str],
        config: DriftModeConfig,
    ) -> list[str]:
        """取得當前集合的相鄰社區。"""
        neighboring = set()

        try:
            for community_id in current_community_ids:
                # 從圖譜取得連接的社區
                connected = await self._nebula.get_connected_communities(
                    community_id=community_id,
                    limit=int(config.expansion_factor * 2),
                )
                neighboring.update(connected)
        except Exception as e:
            logger.warning(f"Error getting neighboring communities: {e}")

        return list(neighboring - current_community_ids)[: config.max_communities]

    async def _generate_expansion_query(
        self,
        original_query: str,
        current_communities: list[CommunityReport],
    ) -> str:
        """基於當前上下文生成擴展查詢。"""
        # 在生產環境中，使用 LLM 生成針對性擴展查詢
        # 目前將原始查詢與關鍵主題結合

        themes = set()
        for community in current_communities[:3]:
            themes.update(community.key_themes[:2])

        if themes:
            theme_str = "、".join(list(themes)[:3])
            return f"{original_query} (探索: {theme_str})"

        return original_query

    def _rank_final_chunks(
        self,
        all_chunks: dict[str, SearchResult],
        iterations: list[DriftIteration],
        max_chunks: int,
    ) -> list[SearchResult]:
        """排序和選擇最終 chunks。"""
        # 提升在多次迭代中找到的 chunks 的分數
        chunk_counts: dict[str, int] = {}
        for iteration in iterations:
            for chunk in iteration.chunks_found:
                chunk_counts[chunk.chunk_id] = chunk_counts.get(chunk.chunk_id, 0) + 1

        # 根據迭代次數和新近度調整分數
        for chunk_id, chunk in all_chunks.items():
            count = chunk_counts.get(chunk_id, 1)
            # 出現在多次迭代中的額外加分
            chunk.score *= (1 + 0.1 * (count - 1))

        # 按調整後的分數排序
        sorted_chunks = sorted(
            all_chunks.values(),
            key=lambda x: x.score,
            reverse=True,
        )

        return sorted_chunks[:max_chunks]


# 單例實例
drift_mode_retriever = DriftModeRetriever()
