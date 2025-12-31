"""
圖譜資料庫服務（NebulaGraph、實體/關係抽取、社區偵測）。

提供知識圖譜的核心功能：
- NebulaGraphClient: NebulaGraph 連接和操作
- EntityExtractor: 從文本抽取實體
- RelationExtractor: 從文本抽取實體間的關係
- CommunityDetector: 使用 Leiden 演算法偵測社區
- CommunitySummarizer: 生成社區摘要報告
- GraphBatchLoader: 批次載入實體和關係到圖譜
"""

from chatbot_graphrag.services.graph.nebula_client import (
    NebulaGraphClient,
    nebula_client,
)
from chatbot_graphrag.services.graph.entity_extractor import (
    EntityExtractor,
    entity_extractor,
)
from chatbot_graphrag.services.graph.relation_extractor import (
    RelationExtractor,
    relation_extractor,
)
from chatbot_graphrag.services.graph.batch_loader import (
    GraphBatchLoader,
    BatchLoadResult,
    graph_batch_loader,
)
from chatbot_graphrag.services.graph.community_detector import (
    CommunityDetector,
    CommunityDetectionResult,
    community_detector,
)
from chatbot_graphrag.services.graph.community_summarizer import (
    CommunitySummarizer,
    CommunitySummaryResult,
    community_summarizer,
)

__all__ = [
    # NebulaGraph 客戶端
    "NebulaGraphClient",
    "nebula_client",
    # 實體抽取
    "EntityExtractor",
    "entity_extractor",
    # 關係抽取
    "RelationExtractor",
    "relation_extractor",
    # 批次載入器
    "GraphBatchLoader",
    "BatchLoadResult",
    "graph_batch_loader",
    # 社區偵測
    "CommunityDetector",
    "CommunityDetectionResult",
    "community_detector",
    # 社區摘要
    "CommunitySummarizer",
    "CommunitySummaryResult",
    "community_summarizer",
]
