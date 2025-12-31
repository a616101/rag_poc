"""
檢索服務

提供三種 GraphRAG 檢索模式：
- LocalMode: 實體種子 → 圖譜遍歷 → 子圖抽取 → 上下文建構
- GlobalMode: 社區報告 → 追問查詢 → 本地深入
- DriftMode: 動態推理和推斷的聚焦遍歷 (DRIFT)
"""

from chatbot_graphrag.services.retrieval.local_mode import (
    LocalModeConfig,
    LocalModeResult,
    LocalModeRetriever,
    EntitySeed,
    GraphContext,
    local_mode_retriever,
)
from chatbot_graphrag.services.retrieval.global_mode import (
    GlobalModeConfig,
    GlobalModeResult,
    GlobalModeRetriever,
    CommunityReport,
    FollowUpQuery,
    global_mode_retriever,
)
from chatbot_graphrag.services.retrieval.drift_mode import (
    DriftModeConfig,
    DriftModeResult,
    DriftModeRetriever,
    DriftIteration,
    drift_mode_retriever,
)

__all__ = [
    # 本地模式 (Local mode)
    "LocalModeConfig",
    "LocalModeResult",
    "LocalModeRetriever",
    "EntitySeed",
    "GraphContext",
    "local_mode_retriever",
    # 全域模式 (Global mode)
    "GlobalModeConfig",
    "GlobalModeResult",
    "GlobalModeRetriever",
    "CommunityReport",
    "FollowUpQuery",
    "global_mode_retriever",
    # 漂移模式 (Drift mode)
    "DriftModeConfig",
    "DriftModeResult",
    "DriftModeRetriever",
    "DriftIteration",
    "drift_mode_retriever",
]
