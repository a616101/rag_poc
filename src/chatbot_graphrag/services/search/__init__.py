"""
搜尋服務模組（混合搜尋、全文搜尋、查詢分類）。

提供多種搜尋策略：
- OpenSearchService: 基於 OpenSearch 的全文搜尋（BM25）
- QueryClassifier: 查詢類型分類器，用於動態調整 RRF 權重
- HybridSearchService: 結合向量搜尋和全文搜尋的混合搜尋
- QueryDecomposer: 查詢分解服務，生成子查詢
"""

from chatbot_graphrag.services.search.opensearch_service import (
    OpenSearchService,
    opensearch_service,
)
from chatbot_graphrag.services.search.query_classifier import (
    QueryType,
    QueryClassification,
    QueryClassifier,
    classify_query,
    get_weights_for_query,
    get_query_classifier,
)

__all__ = [
    "OpenSearchService",
    "opensearch_service",
    "QueryType",
    "QueryClassification",
    "QueryClassifier",
    "classify_query",
    "get_weights_for_query",
    "get_query_classifier",
]
