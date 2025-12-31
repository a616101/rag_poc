"""
GraphRAG API 請求模型

定義所有 API 請求載荷的 Pydantic 模型，包含：
- VectorizeRequest：文件攝取請求
- AskRequest：問答請求
- HITLResolveRequest：HITL 解決請求
- FeedbackRequest：使用者回饋請求
- CacheInvalidateRequest：快取失效請求
- GraphQueryRequest：圖譜查詢請求
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator

from chatbot_graphrag.core.constants import PipelineType, ProcessMode, QueryMode


# ==================== 向量化端點 ====================


class VectorizeRequest(BaseModel):
    """
    文件攝取請求模型（vectorize 端點）。

    用於觸發文件向量化和圖譜建構。
    """

    pipeline: PipelineType = Field(
        default=PipelineType.CURATED,
        description="攝取管道類型：'curated'（YAML+MD）或 'raw'（PDF/DOCX/HTML）"
    )
    source_path: Optional[str] = Field(
        default=None,
        description="文件目錄或檔案的本地路徑"
    )
    source_urls: Optional[list[str]] = Field(
        default=None,
        description="要爬取和攝取的 URL 列表"
    )
    mode: ProcessMode = Field(
        default=ProcessMode.UPDATE,
        description="處理模式：'update'（新增/更新）或 'override'（刪除並重建）"
    )
    collection_name: Optional[str] = Field(
        default=None,
        description="目標 Qdrant 集合名稱（預設使用配置值）"
    )
    enable_graph: bool = Field(
        default=True,
        description="是否抽取實體/關係並建構圖譜"
    )
    enable_community_detection: bool = Field(
        default=True,
        description="是否在圖譜建構後執行社群偵測"
    )
    doc_type_filter: Optional[list[str]] = Field(
        default=None,
        description="只處理特定文件類型的過濾器"
    )

    @field_validator("source_path", "source_urls", mode="after")
    @classmethod
    def validate_source(cls, v, info):
        """確保至少提供一個來源。"""
        if info.field_name == "source_urls" and v is None:
            source_path = info.data.get("source_path")
            if source_path is None:
                raise ValueError("必須提供 source_path 或 source_urls")
        return v


class VectorizeStatusRequest(BaseModel):
    """
    檢查攝取工作狀態的請求模型。
    """

    job_id: str = Field(
        ...,
        description="要檢查狀態的攝取工作 ID"
    )


# ==================== 問答/串流端點 ====================


class AskRequest(BaseModel):
    """
    串流問答請求模型（ask/stream 端點）。

    包含問題和所有相關配置參數。
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="使用者問題"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="對話 ID，用於多輪對話上下文"
    )
    query_mode: Optional[QueryMode] = Field(
        default=None,
        description="強制指定查詢模式（若為 None 則自動偵測）"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="使用者 ID，用於 ACL 過濾"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="租戶 ID，用於多租戶隔離"
    )
    language: Optional[str] = Field(
        default=None,
        description="偏好的回應語言（若為 None 則自動偵測）"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=100,
        le=8000,
        description="回應的最大 token 數"
    )
    include_sources: bool = Field(
        default=True,
        description="是否包含來源引用"
    )
    include_reasoning: bool = Field(
        default=False,
        description="是否包含推理步驟（用於 stream_chat）"
    )
    stream_events: bool = Field(
        default=True,
        description="是否串流中間 SSE 事件"
    )

    # 循環預算覆蓋
    max_loops: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="覆蓋最大檢索循環次數"
    )
    max_wall_time_seconds: Optional[float] = Field(
        default=None,
        ge=5.0,
        le=60.0,
        description="覆蓋最大牆鐘時間（秒）"
    )


class AskStreamChatRequest(AskRequest):
    """
    帶推理的串流問答請求模型（ask/stream_chat 端點）。

    擴展 AskRequest，增加推理相關配置。
    """

    include_reasoning: bool = Field(
        default=True,
        description="是否包含推理步驟"
    )
    reasoning_effort: Optional[str] = Field(
        default="medium",
        description="推理努力程度：'low'、'medium'、'high'"
    )

    @field_validator("reasoning_effort")
    @classmethod
    def validate_reasoning_effort(cls, v):
        """驗證推理努力程度。"""
        valid_levels = {"low", "medium", "high"}
        if v and v not in valid_levels:
            raise ValueError(f"reasoning_effort 必須是以下之一：{valid_levels}")
        return v


# ==================== HITL 端點 ====================


class HITLResolveRequest(BaseModel):
    """
    HITL 解決請求模型。

    用於人工審核和解決需要人工介入的查詢。
    """

    trace_id: str = Field(
        ...,
        description="需要 HITL 的查詢的追蹤 ID"
    )
    resolution: str = Field(
        ...,
        description="解決動作：'approve'、'reject'、'edit'"
    )
    edited_response: Optional[str] = Field(
        default=None,
        description="編輯後的回應內容（resolution='edit' 時必填）"
    )
    reviewer_id: str = Field(
        ...,
        description="人工審核者的 ID"
    )
    notes: Optional[str] = Field(
        default=None,
        description="審核者備註"
    )

    @field_validator("edited_response", mode="after")
    @classmethod
    def validate_edited_response(cls, v, info):
        """確保 resolution 為 'edit' 時提供 edited_response。"""
        resolution = info.data.get("resolution")
        if resolution == "edit" and not v:
            raise ValueError("resolution 為 'edit' 時必須提供 edited_response")
        return v


# ==================== 回饋端點 ====================


class FeedbackRequest(BaseModel):
    """
    使用者回饋請求模型。

    用於收集使用者對回答品質的回饋。
    """

    trace_id: str = Field(
        ...,
        description="查詢的追蹤 ID"
    )
    rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="使用者評分（1-5）"
    )
    comment: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="可選的使用者評論"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="使用者 ID（用於歸屬）"
    )


# ==================== 快取端點 ====================


class CacheInvalidateRequest(BaseModel):
    """
    快取失效請求模型。

    用於使語意快取中的特定項目失效。
    """

    query_pattern: Optional[str] = Field(
        default=None,
        description="用於匹配要失效查詢的模式"
    )
    doc_ids: Optional[list[str]] = Field(
        default=None,
        description="要失效快取的文件 ID 列表"
    )
    invalidate_all: bool = Field(
        default=False,
        description="是否使整個快取失效"
    )

    @field_validator("invalidate_all", mode="after")
    @classmethod
    def validate_invalidation(cls, v, info):
        """確保指定至少一個失效目標。"""
        if not v:
            query_pattern = info.data.get("query_pattern")
            doc_ids = info.data.get("doc_ids")
            if not query_pattern and not doc_ids:
                raise ValueError("必須指定 query_pattern、doc_ids，或設定 invalidate_all=True")
        return v


# ==================== 管理端點 ====================


class GraphQueryRequest(BaseModel):
    """
    直接圖譜查詢請求模型。

    用於執行原生 nGQL 查詢。
    """

    query: str = Field(
        ...,
        description="nGQL 查詢字串"
    )
    space: Optional[str] = Field(
        default=None,
        description="NebulaGraph 空間（預設使用配置值）"
    )
    timeout: Optional[int] = Field(
        default=30000,
        ge=1000,
        le=300000,
        description="查詢逾時（毫秒）"
    )


class CommunityReportRequest(BaseModel):
    """
    社群報告生成請求模型。

    用於觸發社群報告的生成或重新生成。
    """

    level: int = Field(
        default=0,
        ge=0,
        le=5,
        description="社群層級"
    )
    regenerate: bool = Field(
        default=False,
        description="是否重新生成現有報告"
    )
    community_ids: Optional[list[str]] = Field(
        default=None,
        description="要處理的特定社群 ID（若為 None 則處理全部）"
    )
