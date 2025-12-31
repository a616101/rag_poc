"""
GraphRAG 本地模式 (Local Mode)

實體/事件種子 → 圖譜遍歷 → 子圖抽取 → 上下文建構。

流程：
1. 混合搜索取得初始種子 chunks
2. 從 chunks 抽取實體種子
3. 從實體種子遍歷圖譜
4. 收集子圖上下文
5. 擴展和排序最終 chunks
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from chatbot_graphrag.core.constants import MAX_GRAPH_HOPS
from chatbot_graphrag.services.search.hybrid_search import (
    HybridSearchConfig,
    HybridSearchService,
    SearchResult,
    hybrid_search_service,
)

logger = logging.getLogger(__name__)


@dataclass
class LocalModeConfig:
    """本地模式檢索配置。"""

    # 種子檢索
    seed_limit: int = 20           # 種子數量上限
    min_seed_score: float = 0.35   # 最低種子分數

    # 圖譜遍歷
    max_hops: int = MAX_GRAPH_HOPS           # 最大跳數
    max_entities_per_hop: int = 10           # 每跳最多實體數
    entity_types: Optional[list[str]] = None # 限制實體類型
    relation_types: Optional[list[str]] = None  # 限制關係類型

    # 擴展
    expand_chunks: bool = True        # 是否擴展 chunks
    max_expanded_chunks: int = 12     # 擴展後最大 chunks 數

    # 過濾
    doc_types: Optional[list[str]] = None  # 限制文件類型
    acl_groups: Optional[list[str]] = None # ACL 群組過濾


@dataclass
class EntitySeed:
    """從初始種子結果中抽取的實體。"""

    entity_id: str                # 實體 ID
    entity_type: str              # 實體類型
    name: str                     # 實體名稱
    score: float                  # 來源 chunk 分數
    source_chunk_ids: list[str] = field(default_factory=list)  # 來源 chunk ID 列表
    properties: dict[str, Any] = field(default_factory=dict)   # 實體屬性


@dataclass
class GraphContext:
    """從圖譜遍歷中抽取的上下文。"""

    entities: list[dict[str, Any]]          # 實體列表
    relations: list[dict[str, Any]]         # 關係列表
    subgraph_chunks: list[SearchResult]     # 子圖相關 chunks
    community_ids: list[str]                # 相關社區 ID


@dataclass
class LocalModeResult:
    """本地模式檢索結果。"""

    seed_results: list[SearchResult]            # 種子搜索結果
    entity_seeds: list[EntitySeed]              # 實體種子列表
    graph_context: Optional[GraphContext]       # 圖譜上下文
    final_chunks: list[SearchResult]            # 最終 chunks
    total_hops: int = 0                         # 總跳數
    retrieval_path: list[str] = field(default_factory=list)  # 檢索路徑


class LocalModeRetriever:
    """
    GraphRAG 本地模式檢索器。

    流程：
    1. 混合搜索取得初始種子 chunks
    2. 從 chunks 抽取實體種子
    3. 從實體種子遍歷圖譜
    4. 收集子圖上下文
    5. 擴展和排序最終 chunks
    """

    def __init__(self):
        """初始化本地模式檢索器。"""
        self._initialized = False
        self._hybrid_search: Optional[HybridSearchService] = None

    async def initialize(self) -> None:
        """初始化依賴服務。"""
        if self._initialized:
            return

        self._hybrid_search = hybrid_search_service
        await self._hybrid_search.initialize()

        # 延遲導入圖譜服務
        from chatbot_graphrag.services.graph import nebula_client

        self._nebula = nebula_client

        self._initialized = True
        logger.info("本地模式檢索器初始化完成")

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        sparse_embedding: Optional[dict[int, float]] = None,
        config: Optional[LocalModeConfig] = None,
    ) -> LocalModeResult:
        """
        使用本地模式策略檢索上下文。

        Args:
            query: 使用者查詢
            query_embedding: 密集嵌入向量
            sparse_embedding: 稀疏嵌入向量
            config: 檢索配置

        Returns:
            包含所有上下文的 LocalModeResult
        """
        if not self._initialized:
            await self.initialize()

        config = config or LocalModeConfig()
        retrieval_path = ["local_mode_start"]

        # 步驟 1: 混合種子搜索
        seed_config = HybridSearchConfig(
            final_limit=config.seed_limit,
            min_score=config.min_seed_score,
            doc_types=config.doc_types,
            acl_groups=config.acl_groups,
        )

        seed_results = await self._hybrid_search.search(
            query=query,
            query_embedding=query_embedding,
            sparse_embedding=sparse_embedding,
            config=seed_config,
        )
        retrieval_path.append(f"seed_search:{len(seed_results)}")

        if not seed_results:
            return LocalModeResult(
                seed_results=[],
                entity_seeds=[],
                graph_context=None,
                final_chunks=[],
                retrieval_path=retrieval_path,
            )

        # 步驟 2: 從 chunks 抽取實體種子
        entity_seeds = await self._extract_entity_seeds(seed_results, config)
        retrieval_path.append(f"entity_seeds:{len(entity_seeds)}")

        # 步驟 3: 圖譜遍歷
        graph_context = None
        total_hops = 0

        if entity_seeds and config.max_hops > 0:
            graph_context = await self._traverse_graph(entity_seeds, config)
            total_hops = config.max_hops
            retrieval_path.append(f"graph_traverse:{len(graph_context.entities) if graph_context else 0}")

        # 步驟 4: 擴展和合併 chunks
        final_chunks = await self._expand_and_merge(
            seed_results,
            graph_context,
            config,
        )
        retrieval_path.append(f"final_chunks:{len(final_chunks)}")

        return LocalModeResult(
            seed_results=seed_results,
            entity_seeds=entity_seeds,
            graph_context=graph_context,
            final_chunks=final_chunks,
            total_hops=total_hops,
            retrieval_path=retrieval_path,
        )

    async def _extract_entity_seeds(
        self,
        seed_results: list[SearchResult],
        config: LocalModeConfig,
    ) -> list[EntitySeed]:
        """從 chunk 元資料中抽取實體種子。"""
        entity_seeds: list[EntitySeed] = []
        seen_entities: set[str] = set()

        for result in seed_results:
                # 從 chunk 元資料取得實體 ID
            entity_ids = result.metadata.get("entity_ids", [])

            for entity_id in entity_ids:
                if entity_id in seen_entities:
                    continue

                seen_entities.add(entity_id)

                # Look up entity details from graph
                try:
                    entity_data = await self._nebula.get_entity(entity_id)
                    if entity_data:
                        entity_type = entity_data.get("entity_type", "unknown")

                        # 如果有指定，依實體類型過濾
                        if config.entity_types and entity_type not in config.entity_types:
                            continue

                        entity_seeds.append(
                            EntitySeed(
                                entity_id=entity_id,
                                entity_type=entity_type,
                                name=entity_data.get("name", ""),
                                score=result.score,
                                source_chunk_ids=[result.chunk_id],
                                properties=entity_data,
                            )
                        )
                except Exception as e:
                    logger.warning(f"Error looking up entity {entity_id}: {e}")

        # Sort by score and limit
        entity_seeds.sort(key=lambda x: x.score, reverse=True)
        return entity_seeds[: config.max_entities_per_hop]

    async def _traverse_graph(
        self,
        entity_seeds: list[EntitySeed],
        config: LocalModeConfig,
    ) -> GraphContext:
        """從實體種子遍歷圖譜。"""
        # entity_id 已包含完整的 VID 格式 (e_{type}_{hash})
        # 由 entity_extractor._generate_entity_id() 設定
        seed_vids = [e.entity_id for e in entity_seeds]

        try:
            traversal_result = await self._nebula.traverse(
                seed_vids=seed_vids,
                max_hops=config.max_hops,
                relation_types=config.relation_types,
                max_vertices=config.max_entities_per_hop * config.max_hops,
            )

            # GraphTraversalResult 是 Pydantic 模型 - 存取子圖
            entities = traversal_result.subgraph.entities
            relations = traversal_result.subgraph.relations

            # 轉換為字典供下游處理
            entity_dicts = [e.model_dump() for e in entities]
            relation_dicts = [r.model_dump() for r in relations]

            # 從實體中抽取社區 ID
            community_ids = set()
            for entity in entities:
                if hasattr(entity, 'doc_ids') and entity.doc_ids:
                    community_ids.update(entity.doc_ids)

            # 取得與遍歷實體相關的 chunks
            subgraph_chunks = await self._get_entity_chunks(
                entity_dicts,
                config,
            )

            return GraphContext(
                entities=entity_dicts,
                relations=relation_dicts,
                subgraph_chunks=subgraph_chunks,
                community_ids=list(community_ids),
            )
        except Exception as e:
            logger.error(f"Graph traversal error: {e}")
            return GraphContext(
                entities=[],
                relations=[],
                subgraph_chunks=[],
                community_ids=[],
            )

    async def _get_entity_chunks(
        self,
        entities: list[dict[str, Any]],
        config: LocalModeConfig,
    ) -> list[SearchResult]:
        """
        取得與圖譜遍歷實體相關的 chunks。

        這是 GraphRAG 的關鍵部分 - 在圖譜遍歷識別出相關實體後，
        我們需要檢索提及這些實體的原始 chunks。

        Args:
            entities: 來自圖譜遍歷的實體字典列表
            config: 本地模式配置

        Returns:
            實體相關 chunks 的 SearchResult 物件列表
        """
        # 收集所有實體的 chunk ID
        chunk_ids = set()
        for entity in entities:
            # 攝取期間存儲在 chunk 元資料中的 entity_ids
            entity_chunk_ids = entity.get("source_chunk_ids", [])
            chunk_ids.update(entity_chunk_ids)

        if not chunk_ids:
            logger.debug("No chunk IDs found in entities")
            return []

        logger.debug(f"Fetching {len(chunk_ids)} chunks for {len(entities)} entities")

        try:
            # 從 Qdrant 取得 chunk 詳細資料
            from chatbot_graphrag.services.vector import qdrant_service

            await qdrant_service.initialize()

            chunk_data_list = await qdrant_service.get_chunks_by_ids(list(chunk_ids))

            # 轉換為 SearchResult 物件
            results = []
            for chunk_data in chunk_data_list:
                result = SearchResult(
                    chunk_id=chunk_data.get("chunk_id", ""),
                    doc_id=chunk_data.get("doc_id", ""),
                    content=chunk_data.get("content", ""),
                    score=0.8,  # Base score for graph-connected chunks
                    source="graph",  # Mark as coming from graph traversal
                    metadata=chunk_data.get("metadata", {}),
                )
                results.append(result)

            logger.info(f"Retrieved {len(results)} chunks from graph entities")
            return results

        except Exception as e:
            logger.error(f"Failed to fetch entity chunks: {e}")
            return []

    async def _expand_and_merge(
        self,
        seed_results: list[SearchResult],
        graph_context: Optional[GraphContext],
        config: LocalModeConfig,
    ) -> list[SearchResult]:
        """擴展和合併所有 chunks。"""
        all_chunks: dict[str, SearchResult] = {}

        # 添加種子結果
        for result in seed_results:
            all_chunks[result.chunk_id] = result

        # 添加圖譜上下文 chunks
        if graph_context:
            for result in graph_context.subgraph_chunks:
                if result.chunk_id not in all_chunks:
                    # 提升圖譜連接 chunks 的分數
                    result.score *= 0.9  # Slightly lower than direct hits
                    all_chunks[result.chunk_id] = result

        # 按分數排序並限制數量
        sorted_chunks = sorted(
            all_chunks.values(),
            key=lambda x: x.score,
            reverse=True,
        )

        return sorted_chunks[: config.max_expanded_chunks]


# 單例實例
local_mode_retriever = LocalModeRetriever()
