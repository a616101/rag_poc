"""
GraphRAG API 回應模型

定義所有 API 回應載荷的 Pydantic 模型，包含：
- BaseResponse/ErrorResponse：基礎回應模型
- VectorizeResponse：攝取回應
- AskResponse：問答回應
- SSE 事件模型：串流回應事件
- HITL/Feedback/Cache 回應
- Health 回應
"""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field

from chatbot_graphrag.core.constants import (
    JobStatus,
    QueryMode,
    GroundednessStatus,
    SSEStage,
)


# ==================== 通用回應模型 ====================


class BaseResponse(BaseModel):
    """
    帶有通用欄位的基礎回應模型。
    """

    success: bool = Field(default=True, description="請求是否成功")
    message: Optional[str] = Field(default=None, description="可選訊息")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """
    錯誤回應模型。
    """

    success: bool = Field(default=False)
    error: str = Field(..., description="錯誤訊息")
    error_code: Optional[str] = Field(default=None, description="錯誤代碼")
    details: Optional[dict[str, Any]] = Field(default=None, description="錯誤詳情")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ==================== 向量化端點 ====================


class VectorizeResponse(BaseModel):
    """
    向量化端點的回應模型。
    """

    success: bool = True
    job_id: str = Field(..., description="攝取工作 ID")
    status: JobStatus = Field(default=JobStatus.PENDING, description="工作狀態")
    message: str = Field(default="攝取工作已排入佇列")
    estimated_documents: Optional[int] = Field(
        default=None,
        description="預估要處理的文件數量"
    )


class VectorizeStatusResponse(BaseModel):
    """
    向量化狀態端點的回應模型。
    """

    job_id: str = Field(..., description="攝取工作 ID")
    status: JobStatus = Field(..., description="目前工作狀態")
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="進度百分比"
    )
    total_documents: int = Field(default=0, description="待處理的總文件數")
    processed_documents: int = Field(default=0, description="目前已處理的文件數")
    failed_documents: int = Field(default=0, description="失敗的文件數")
    chunks_created: int = Field(default=0, description="建立的總 chunk 數")
    entities_extracted: int = Field(default=0, description="抽取的總實體數")
    relations_extracted: int = Field(default=0, description="抽取的總關係數")
    communities_detected: int = Field(default=0, description="偵測的社群數")
    started_at: Optional[datetime] = Field(default=None, description="工作開始時間")
    completed_at: Optional[datetime] = Field(default=None, description="工作完成時間")
    error_message: Optional[str] = Field(default=None, description="失敗時的錯誤訊息")
    error_details: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="文件層級的錯誤列表"
    )


# ==================== 問答/串流端點 ====================


class Citation(BaseModel):
    """
    來源引用模型。

    表示回答中引用的來源文件資訊。
    """

    chunk_id: str = Field(..., description="Chunk ID")
    doc_id: str = Field(..., description="文件 ID")
    doc_title: str = Field(..., description="文件標題")
    doc_type: str = Field(..., description="文件類型")
    content_preview: str = Field(..., description="內容預覽（前 200 字元）")
    relevance_score: float = Field(..., description="相關性分數")
    page_number: Optional[int] = Field(default=None, description="頁碼（如適用）")
    section: Optional[str] = Field(default=None, description="章節標題（如適用）")


class EvidenceItem(BaseModel):
    """
    落地性追蹤的證據項目。

    用於驗證回答中的主張是否有來源支持。
    """

    claim: str = Field(..., description="回答中的主張")
    evidence: str = Field(..., description="來源中的支持證據")
    chunk_id: str = Field(..., description="來源 chunk ID")
    confidence: float = Field(..., ge=0.0, le=1.0, description="信心分數")
    grounded: bool = Field(..., description="主張是否已落地")


class RetrievalMetrics(BaseModel):
    """
    可觀測性的檢索指標。

    追蹤檢索過程的各階段統計。
    """

    seed_count: int = Field(default=0, description="初始種子結果數")
    reranked_count: int = Field(default=0, description="重排序後的結果數")
    graph_expanded_count: int = Field(default=0, description="圖譜擴展後的結果數")
    final_count: int = Field(default=0, description="最終上下文 chunk 數")
    avg_relevance_score: float = Field(default=0.0, description="平均相關性分數")
    dense_weight: float = Field(default=0.4, description="使用的密集向量權重")
    sparse_weight: float = Field(default=0.3, description="使用的稀疏向量權重")
    fts_weight: float = Field(default=0.3, description="使用的全文檢索權重")


class LoopMetrics(BaseModel):
    """
    循環執行指標。

    追蹤 Agentic RAG 循環的執行統計。
    """

    total_loops: int = Field(default=0, description="執行的總循環數")
    total_queries: int = Field(default=0, description="生成的總查詢數")
    context_tokens: int = Field(default=0, description="使用的總上下文 token 數")
    wall_time_seconds: float = Field(default=0.0, description="總牆鐘時間")


class AskResponse(BaseModel):
    """
    問答端點的非串流回應模型。

    包含完整的問答結果和所有相關指標。
    """

    success: bool = True
    answer: str = Field(..., description="生成的回答")
    query_mode: QueryMode = Field(..., description="使用的查詢模式")
    language: str = Field(..., description="回應語言")
    citations: list[Citation] = Field(default_factory=list, description="來源引用")
    evidence_table: Optional[list[EvidenceItem]] = Field(
        default=None,
        description="落地性證據表"
    )
    groundedness_status: GroundednessStatus = Field(
        default=GroundednessStatus.PASS,
        description="落地性檢查結果"
    )
    hitl_required: bool = Field(default=False, description="是否需要 HITL 審核")
    retrieval_metrics: Optional[RetrievalMetrics] = Field(
        default=None,
        description="檢索指標"
    )
    loop_metrics: Optional[LoopMetrics] = Field(
        default=None,
        description="循環執行指標"
    )
    trace_id: str = Field(..., description="可觀測性追蹤 ID")
    conversation_id: Optional[str] = Field(default=None, description="對話 ID")


# ==================== SSE 事件模型 ====================


class SSEEvent(BaseModel):
    """
    基礎 SSE 事件模型。

    所有 SSE 事件類型的基礎類別。
    """

    stage: SSEStage = Field(..., description="管道階段")
    data: dict[str, Any] = Field(default_factory=dict, description="事件資料")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SSEGuardEvent(SSEEvent):
    """守衛階段的 SSE 事件。"""

    stage: SSEStage = SSEStage.GUARD_START
    data: dict[str, Any] = Field(
        default_factory=lambda: {"status": "checking"}
    )


class SSEIntentEvent(SSEEvent):
    """意圖分析的 SSE 事件。"""

    stage: SSEStage = SSEStage.INTENT_END
    data: dict[str, Any] = Field(
        default_factory=lambda: {
            "intent": "retrieval",
            "query_mode": "local",
            "confidence": 0.0,
        }
    )


class SSERetrievalEvent(SSEEvent):
    """檢索進度的 SSE 事件。"""

    stage: SSEStage = SSEStage.RETRIEVAL_START
    data: dict[str, Any] = Field(
        default_factory=lambda: {
            "phase": "hybrid_seed",
            "results_count": 0,
        }
    )


class SSEGroundednessEvent(SSEEvent):
    """落地性檢查的 SSE 事件。"""

    stage: SSEStage = SSEStage.GROUNDEDNESS_END
    data: dict[str, Any] = Field(
        default_factory=lambda: {
            "status": "pass",
            "grounded_claims": 0,
            "total_claims": 0,
        }
    )


class SSEResponseChunk(BaseModel):
    """
    串流回應的 SSE chunk。

    用於串流回應文字。
    """

    stage: SSEStage = SSEStage.RESPONSE_GENERATING
    content: str = Field(..., description="回應 chunk 內容")
    is_final: bool = Field(default=False, description="是否為最後一個 chunk")


class SSEMetaSummary(BaseModel):
    """
    SSE 元摘要事件（最終事件）。

    包含整個請求的摘要資訊和指標。
    """

    stage: SSEStage = SSEStage.META_SUMMARY
    trace_id: str = Field(..., description="追蹤 ID")
    query_mode: QueryMode = Field(..., description="使用的查詢模式")
    groundedness_status: GroundednessStatus = Field(..., description="落地性結果")
    hitl_required: bool = Field(default=False, description="是否需要 HITL")
    citations: list[Citation] = Field(default_factory=list, description="引用")
    retrieval_metrics: Optional[RetrievalMetrics] = None
    loop_metrics: Optional[LoopMetrics] = None
    total_duration_ms: float = Field(..., description="總執行時間（毫秒）")


# ==================== HITL 回應 ====================


class HITLResolveResponse(BaseModel):
    """HITL 解決的回應模型。"""

    success: bool = True
    trace_id: str = Field(..., description="追蹤 ID")
    resolution: str = Field(..., description="套用的解決方案")
    message: str = Field(default="HITL 解決已記錄")


class HITLPendingResponse(BaseModel):
    """列出待處理 HITL 項目的回應模型。"""

    items: list[dict[str, Any]] = Field(default_factory=list)
    total_count: int = Field(default=0)
    page: int = Field(default=1)
    page_size: int = Field(default=20)


# ==================== 回饋回應 ====================


class FeedbackResponse(BaseModel):
    """回饋端點的回應模型。"""

    success: bool = True
    message: str = Field(default="回饋已記錄")
    trace_id: str = Field(..., description="關聯的追蹤 ID")


# ==================== 快取回應 ====================


class CacheInvalidateResponse(BaseModel):
    """快取失效的回應模型。"""

    success: bool = True
    invalidated_count: int = Field(..., description="失效的快取項目數")
    message: str = Field(default="快取失效已完成")


class CacheStatsResponse(BaseModel):
    """快取統計的回應模型。"""

    total_entries: int = Field(default=0, description="總快取項目數")
    hit_count: int = Field(default=0, description="快取命中次數")
    miss_count: int = Field(default=0, description="快取未命中次數")
    hit_rate: float = Field(default=0.0, description="快取命中率")
    memory_usage_mb: float = Field(default=0.0, description="記憶體使用量（MB）")


# ==================== 圖譜回應 ====================


class GraphQueryResponse(BaseModel):
    """圖譜查詢端點的回應模型。"""

    success: bool = True
    data: list[dict[str, Any]] = Field(default_factory=list, description="查詢結果")
    row_count: int = Field(default=0, description="返回的列數")
    execution_time_ms: float = Field(default=0.0, description="查詢執行時間")


class CommunityReportResponse(BaseModel):
    """社群報告生成的回應模型。"""

    success: bool = True
    job_id: str = Field(..., description="背景工作 ID")
    communities_queued: int = Field(..., description="已排入佇列的社群數")
    message: str = Field(default="社群報告生成已開始")


# ==================== 健康檢查回應 ====================


class HealthResponse(BaseModel):
    """健康檢查端點的回應模型。"""

    status: str = Field(default="healthy", description="總體狀態")
    version: str = Field(..., description="API 版本")
    services: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="服務健康狀態"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ServiceHealth(BaseModel):
    """單一服務的健康狀態。"""

    name: str = Field(..., description="服務名稱")
    status: str = Field(..., description="服務狀態：healthy、degraded、unhealthy")
    latency_ms: Optional[float] = Field(default=None, description="延遲（毫秒）")
    details: Optional[dict[str, Any]] = Field(default=None, description="額外詳情")
