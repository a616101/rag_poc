"""
社區偵測服務

使用 Leiden 演算法在知識圖譜中偵測社區。

主要功能：
- 使用 Leiden 演算法進行階層式社區偵測
- 支援多層級社區結構
- 後備連通分量偵測
- 將社區儲存到 NebulaGraph
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from chatbot_graphrag.core.constants import MAX_COMMUNITY_LEVEL
from chatbot_graphrag.models.pydantic.graph import Community, Entity, Relation

logger = logging.getLogger(__name__)


@dataclass
class CommunityDetectionResult:
    """社區偵測結果。"""

    communities: list[Community] = field(default_factory=list)  # 偵測到的社區列表
    total_communities: int = 0  # 社區總數
    levels: int = 0  # 層級數
    modularity: float = 0.0  # 模組化分數
    execution_time_ms: float = 0.0  # 執行時間（毫秒）
    errors: list[str] = field(default_factory=list)  # 錯誤列表


class CommunityDetector:
    """
    使用 Leiden 演算法在知識圖譜中偵測社區。

    使用階層式社區偵測來識別多個粒度層級的
    相關實體群集。
    """

    def __init__(self):
        """初始化社區偵測器。"""
        self._initialized = False

    async def initialize(self) -> None:
        """初始化社區偵測器。"""
        if self._initialized:
            return

        try:
            import igraph as ig
            import leidenalg
            self._ig = ig
            self._leidenalg = leidenalg
            logger.info("Community detector initialized with leidenalg")
        except ImportError:
            logger.warning(
                "leidenalg not installed. Community detection will use fallback. "
                "Install with: pip install leidenalg python-igraph"
            )
            self._ig = None
            self._leidenalg = None

        self._initialized = True

    async def detect_communities(
        self,
        entities: list[Entity],
        relations: list[Relation],
        max_levels: int = MAX_COMMUNITY_LEVEL,
        resolution: float = 1.0,
    ) -> CommunityDetectionResult:
        """
        從實體和關係中偵測社區。

        Args:
            entities: 實體列表
            relations: 實體間的關係列表
            max_levels: 要偵測的最大階層層級數
            resolution: Leiden 的解析度參數（越高 = 越多社區）

        Returns:
            包含偵測到社區的 CommunityDetectionResult
        """
        if not self._initialized:
            await self.initialize()

        import time
        start_time = time.time()
        result = CommunityDetectionResult()

        if len(entities) < 2:
            result.execution_time_ms = (time.time() - start_time) * 1000
            return result

        try:
            if self._ig and self._leidenalg:
                # 使用 Leiden 演算法
                communities = await self._detect_with_leiden(
                    entities, relations, max_levels, resolution
                )
            else:
                # 使用後備連通分量
                communities = await self._detect_fallback(entities, relations)

            result.communities = communities
            result.total_communities = len(communities)
            result.levels = max(c.level for c in communities) + 1 if communities else 0

            # 計算整體模組化分數
            if communities:
                modularities = [
                    c.modularity_score
                    for c in communities
                    if c.modularity_score is not None
                ]
                result.modularity = sum(modularities) / len(modularities) if modularities else 0.0

        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            result.errors.append(str(e))

        result.execution_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Detected {result.total_communities} communities across "
            f"{result.levels} levels in {result.execution_time_ms:.1f}ms"
        )

        return result

    async def _detect_with_leiden(
        self,
        entities: list[Entity],
        relations: list[Relation],
        max_levels: int,
        resolution: float,
    ) -> list[Community]:
        """使用 Leiden 演算法偵測社區。"""
        # 建構 igraph 圖
        entity_id_to_idx = {e.id: idx for idx, e in enumerate(entities)}

        # 建立邊列表
        edges = []
        weights = []
        for relation in relations:
            src_idx = entity_id_to_idx.get(relation.source_id)
            tgt_idx = entity_id_to_idx.get(relation.target_id)
            if src_idx is not None and tgt_idx is not None and src_idx != tgt_idx:
                edges.append((src_idx, tgt_idx))
                weights.append(relation.weight)

        if not edges:
            # 沒有邊 - 用所有實體建立單一社區
            return [self._create_community(
                entity_ids=[e.id for e in entities],
                level=0,
                modularity=0.0,
            )]

        # 建立 igraph 圖
        g = self._ig.Graph(n=len(entities), edges=edges, directed=False)
        g.es['weight'] = weights if weights else [1.0] * len(edges)

        # 設定頂點屬性
        for idx, entity in enumerate(entities):
            g.vs[idx]['id'] = entity.id
            g.vs[idx]['name'] = entity.name
            g.vs[idx]['type'] = entity.entity_type.value

        communities = []

        # 在不同解析度下偵測社區以建立階層
        resolutions = [resolution * (2 ** level) for level in range(max_levels)]

        for level, res in enumerate(resolutions):
            # 執行 Leiden 演算法
            partition = self._leidenalg.find_partition(
                g,
                self._leidenalg.RBConfigurationVertexPartition,
                weights='weight',
                resolution_parameter=res,
            )

            # 將分區轉換為社區
            level_communities = self._partition_to_communities(
                partition, entities, entity_id_to_idx, level
            )

            # 只有在有有意義的社區時才添加
            if level_communities and (
                level == 0 or
                len(level_communities) != len(communities)
            ):
                communities.extend(level_communities)

            # 如果已收斂到單一社區則停止
            if len(partition) <= 1:
                break

        return communities

    def _partition_to_communities(
        self,
        partition,
        entities: list[Entity],
        entity_id_to_idx: dict[str, int],
        level: int,
    ) -> list[Community]:
        """將 Leiden 分區轉換為 Community 物件。"""
        communities = []
        idx_to_entity_id = {idx: eid for eid, idx in entity_id_to_idx.items()}

        for cluster_idx, cluster in enumerate(partition):
            if len(cluster) < 2:
                # 跳過單一成員的社區
                continue

            entity_ids = [idx_to_entity_id[idx] for idx in cluster]

            community = self._create_community(
                entity_ids=entity_ids,
                level=level,
                modularity=partition.modularity if hasattr(partition, 'modularity') else None,
            )
            communities.append(community)

        return communities

    async def _detect_fallback(
        self,
        entities: list[Entity],
        relations: list[Relation],
    ) -> list[Community]:
        """使用連通分量的後備社區偵測。"""
        # 使用 union-find 建構鄰接關係
        parent = {e.id: e.id for e in entities}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # 處理關係
        for relation in relations:
            if relation.source_id in parent and relation.target_id in parent:
                union(relation.source_id, relation.target_id)

        # 按分量分組實體
        components: dict[str, list[str]] = {}
        for entity in entities:
            root = find(entity.id)
            if root not in components:
                components[root] = []
            components[root].append(entity.id)

        # 建立社區
        communities = []
        for entity_ids in components.values():
            if len(entity_ids) >= 2:
                community = self._create_community(
                    entity_ids=entity_ids,
                    level=0,
                    modularity=0.0,
                )
                communities.append(community)

        return communities

    def _create_community(
        self,
        entity_ids: list[str],
        level: int,
        modularity: Optional[float] = None,
    ) -> Community:
        """建立 Community 物件。"""
        community_id = uuid.uuid4().hex[:12]

        return Community(
            id=community_id,
            level=level,
            entity_ids=entity_ids,
            entity_count=len(entity_ids),
            edge_count=0,  # 載入到圖譜時會計算
            modularity_score=modularity,
        )

    async def detect_and_store(
        self,
        entities: list[Entity],
        relations: list[Relation],
        max_levels: int = MAX_COMMUNITY_LEVEL,
        resolution: float = 1.0,
    ) -> CommunityDetectionResult:
        """
        偵測社區並儲存到 NebulaGraph。

        Args:
            entities: 實體列表
            relations: 關係列表
            max_levels: 最大階層層級數
            resolution: Leiden 解析度參數

        Returns:
            CommunityDetectionResult
        """
        # 偵測社區
        result = await self.detect_communities(
            entities, relations, max_levels, resolution
        )

        if not result.communities:
            return result

        # 將社區儲存到 NebulaGraph
        from chatbot_graphrag.services.graph.nebula_client import nebula_client

        try:
            await nebula_client.initialize()

            for community in result.communities:
                try:
                    await nebula_client.upsert_community(community)
                except Exception as e:
                    result.errors.append(f"Failed to store community {community.id}: {e}")
                    logger.warning(f"Failed to store community: {e}")

            logger.info(f"Stored {len(result.communities)} communities in NebulaGraph")

        except Exception as e:
            result.errors.append(f"NebulaGraph connection failed: {e}")
            logger.error(f"Failed to connect to NebulaGraph: {e}")

        return result

    async def rebuild_communities(
        self,
        max_levels: int = MAX_COMMUNITY_LEVEL,
        resolution: float = 1.0,
    ) -> CommunityDetectionResult:
        """
        從當前圖譜資料重建所有社區。

        從 NebulaGraph 載入所有實體和關係，
        偵測社區，並儲存它們。
        """
        from chatbot_graphrag.services.graph.nebula_client import nebula_client

        await nebula_client.initialize()

        # 載入所有實體
        entities = []
        for entity_type in ["person", "department", "procedure", "location",
                           "building", "floor", "room", "form", "medication",
                           "equipment", "service", "condition", "contact"]:
            try:
                from chatbot_graphrag.core.constants import EntityType
                results = await nebula_client.find_entities_by_type(
                    EntityType(entity_type), limit=10000
                )
                for row in results:
                    props = row.get("props", {})
                    entities.append(Entity(
                        id=row.get("vid", ""),
                        name=props.get("name", ""),
                        entity_type=EntityType(props.get("entity_type", "service")),
                        description=props.get("description"),
                        mention_count=props.get("mention_count", 1),
                    ))
            except Exception as e:
                logger.warning(f"Failed to load {entity_type} entities: {e}")

        if len(entities) < 2:
            logger.info("Not enough entities for community detection")
            return CommunityDetectionResult()

        # 載入關係（簡化版 - 只取得邊數）
        # 在生產環境中，您會想要載入實際的邊
        relations = []

        logger.info(f"Loaded {len(entities)} entities for community detection")

        return await self.detect_and_store(
            entities, relations, max_levels, resolution
        )


# 單例實例
community_detector = CommunityDetector()
