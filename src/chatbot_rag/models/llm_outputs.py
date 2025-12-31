"""
LLM 輸出的 Pydantic 模型定義。

用於結構化輸出驗證和 JSON schema 生成。

設計原則：
- 領域無關：不硬編碼特定任務類型，由 prompt 定義
- 路由驅動：LLM 輸出直接驅動流程路由
"""

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


# ============================================================================
# 領域無關的路由提示類型
# ============================================================================

RoutingHint = Literal["continue", "direct_response", "followup"]
"""
路由提示類型：
- continue: 需要檢索知識庫後再回應
- direct_response: 不需要檢索，直接回應
- followup: 這是追問，需要參考上一輪回答
"""

# 查詢類型
QueryType = Literal["list", "detail", "hybrid"]
"""
查詢類型：
- list: 列表型查詢（如「有哪些醫師」「所有科別」）- 使用 metadata filter
- detail: 細節型查詢（如「吳榮州醫師的門診時間」）- 使用向量檢索
- hybrid: 混合型查詢（同時需要列表和細節）
"""

# 檢索策略
RetrievalStrategy = Literal["vector", "metadata_filter", "hybrid"]
"""
檢索策略：
- vector: 向量相似度搜尋（傳統 RAG）
- metadata_filter: 使用 metadata filter 精確查詢（聚合型查詢）
- hybrid: 先 filter 確定範圍，再 vector 取得相關內容
"""


# ============================================================================
# Intent Analyzer 輸出
# ============================================================================

class IntentAnalyzerOutput(BaseModel):
    """
    Intent Analyzer 節點的結構化輸出。

    領域無關設計：
    - intent: 由 prompt 定義的意圖類型，非程式碼硬編碼
    - routing_hint: 決定流程路由的提示
    - needs_retrieval/needs_followup: 布林欄位驅動路由決策
    - query_type/retrieval_strategy: 查詢類型和檢索策略（用於聚合型查詢優化）
    """

    intent: str = Field(
        default="general_inquiry",
        description="意圖類型（由 prompt 定義，如 simple_faq, symptom_inquiry 等）",
    )
    needs_retrieval: bool = Field(
        default=True,
        description="是否需要檢索知識庫",
    )
    needs_followup: bool = Field(
        default=False,
        description="是否是追問（需要參考上一輪回答）",
    )
    routing_hint: RoutingHint = Field(
        default="continue",
        description="路由提示：continue（檢索）、direct_response（直接回應）、followup（追問處理）",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="意圖分析的信心度",
    )
    reasoning: str = Field(
        default="",
        description="判斷理由（用於 debug 和 telemetry）",
    )

    # 查詢類型和檢索策略（用於聚合型查詢優化）
    query_type: QueryType = Field(
        default="detail",
        description="查詢類型：list（列表型）、detail（細節型）、hybrid（混合型）",
    )
    retrieval_strategy: RetrievalStrategy = Field(
        default="vector",
        description="檢索策略：vector（向量搜尋）、metadata_filter（metadata 過濾）、hybrid（混合）",
    )
    extracted_entities: dict[str, str] = Field(
        default_factory=dict,
        description="從問題中識別的實體，如 {'department': '心臟血管科', 'doctor': '吳榮州'}",
    )

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典格式。"""
        return {
            "intent": self.intent,
            "needs_retrieval": self.needs_retrieval,
            "needs_followup": self.needs_followup,
            "routing_hint": self.routing_hint,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "query_type": self.query_type,
            "retrieval_strategy": self.retrieval_strategy,
            "extracted_entities": self.extracted_entities,
        }

    @classmethod
    def get_json_schema(cls) -> dict[str, Any]:
        """取得 OpenAI Structured Output 相容的 JSON schema。"""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "intent_analysis",
                "strict": True,
                "schema": cls.model_json_schema(),
            },
        }


class ResultEvaluatorOutput(BaseModel):
    """
    Result Evaluator 節點的結構化輸出。

    用於評估檢索結果是否足以回答問題。
    """

    sufficient: bool = Field(
        default=True,
        description="檢索結果是否足以回答問題",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="評估的信心度",
    )
    missing_aspects: List[str] = Field(
        default_factory=list,
        description="缺少的資訊面向",
    )
    suggested_query: Optional[str] = Field(
        default=None,
        description="建議的補充查詢（若 sufficient=false）",
    )
    reasoning: str = Field(
        default="",
        description="判斷理由",
    )

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典格式。"""
        return {
            "sufficient": self.sufficient,
            "confidence": self.confidence,
            "missing_aspects": self.missing_aspects,
            "suggested_query": self.suggested_query,
            "reasoning": self.reasoning,
        }

    @classmethod
    def get_json_schema(cls) -> dict[str, Any]:
        """取得 OpenAI Structured Output 相容的 JSON schema。"""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "result_evaluation",
                "strict": True,
                "schema": cls.model_json_schema(),
            },
        }
