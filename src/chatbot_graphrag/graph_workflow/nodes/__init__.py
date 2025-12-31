"""
GraphRAG 的 LangGraph 工作流程節點。

此模組匯出 GraphRAG 工作流程中使用的所有節點函數。

節點類型：
- 輸入處理節點：guard, acl, normalize, cache
- 意圖和路由節點：intent_router, direct_answer
- 檢索節點：hybrid_seed, community_reports, followups, rrf_merge, hop_hybrid
- 重排序節點：rerank
- 圖譜節點：graph_seed_extract, graph_traverse, subgraph_to_queries
- 上下文建構節點：chunk_expander, context_packer, evidence_table
- 品質節點：groundedness, targeted_retry, interrupt_hitl
- 輸出節點：final_answer, telemetry
"""

# 輸入處理節點
from chatbot_graphrag.graph_workflow.nodes.guard import guard_node
from chatbot_graphrag.graph_workflow.nodes.acl import acl_node
from chatbot_graphrag.graph_workflow.nodes.normalize import normalize_node
from chatbot_graphrag.graph_workflow.nodes.cache import (
    cache_lookup_node,
    cache_response_node,
    cache_store_node,
)

# 意圖和路由節點
from chatbot_graphrag.graph_workflow.nodes.intent import (
    intent_router_node,
    direct_answer_node,
)

# 檢索節點
from chatbot_graphrag.graph_workflow.nodes.retrieval import (
    hybrid_seed_node,
    community_reports_node,
    followups_node,
    rrf_merge_node,
    hop_hybrid_node,
)

# 重排序節點
from chatbot_graphrag.graph_workflow.nodes.rerank import rerank_node

# 圖譜節點
from chatbot_graphrag.graph_workflow.nodes.graph import (
    graph_seed_extract_node,
    graph_traverse_node,
    subgraph_to_queries_node,
)

# 上下文建構節點
from chatbot_graphrag.graph_workflow.nodes.context import (
    chunk_expander_node,
    context_packer_node,
    evidence_table_node,
)

# 品質節點
from chatbot_graphrag.graph_workflow.nodes.quality import (
    groundedness_node,
    targeted_retry_node,
    interrupt_hitl_node,
)

# 輸出節點
from chatbot_graphrag.graph_workflow.nodes.output import (
    final_answer_node,
    telemetry_node,
)

__all__ = [
    # 輸入處理
    "guard_node",  # 守衛節點 - 安全檢查
    "acl_node",  # 存取控制節點
    "normalize_node",  # 正規化節點
    "cache_lookup_node",  # 快取查詢節點
    "cache_response_node",  # 快取回應節點
    "cache_store_node",  # 快取儲存節點
    # 意圖和路由
    "intent_router_node",  # 意圖路由節點
    "direct_answer_node",  # 直接回答節點
    # 檢索
    "hybrid_seed_node",  # 混合種子檢索節點
    "community_reports_node",  # 社群報告節點
    "followups_node",  # 追蹤查詢節點
    "rrf_merge_node",  # RRF 融合節點
    "hop_hybrid_node",  # 跳躍混合檢索節點
    # 重排序
    "rerank_node",  # 重排序節點
    # 圖譜
    "graph_seed_extract_node",  # 圖譜種子抽取節點
    "graph_traverse_node",  # 圖譜遍歷節點
    "subgraph_to_queries_node",  # 子圖轉查詢節點
    # 上下文建構
    "chunk_expander_node",  # Chunk 擴展節點
    "context_packer_node",  # 上下文打包節點
    "evidence_table_node",  # 證據表建構節點
    # 品質
    "groundedness_node",  # 落地性檢查節點
    "targeted_retry_node",  # 針對性重試節點
    "interrupt_hitl_node",  # HITL 中斷節點
    # 輸出
    "final_answer_node",  # 最終答案節點
    "telemetry_node",  # 遙測節點
]
