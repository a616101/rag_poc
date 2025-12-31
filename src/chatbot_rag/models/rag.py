"""
RAG 相關數據模型。

本模組定義了所有 RAG 功能相關的請求和響應數據模型，
使用 Pydantic 進行資料驗證和序列化。
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ProcessMode(str, Enum):
    """
    文檔處理模式枚舉。

    定義向量化時的處理方式：
    - OVERRIDE: 覆蓋模式，刪除現有向量後重新建立
    - UPDATE: 更新模式，保留現有向量並新增或更新
    """

    OVERRIDE = "override"  # 覆蓋所有現有向量
    UPDATE = "update"  # 增量更新向量


class DocumentSource(str, Enum):
    """
    文檔來源枚舉。

    定義文檔的來源類型：
    - DEFAULT: 預設來源，從指定目錄讀取文檔
    - UPLOADED: 上傳來源，處理用戶上傳的文件
    """

    DEFAULT = "default"  # 從預設目錄讀取
    UPLOADED = "uploaded"  # 處理上傳文件


class VectorizeRequest(BaseModel):
    """
    文檔向量化請求模型。

    用於請求將文檔轉換為向量並儲存到 Qdrant 向量數據庫。

    Attributes:
        source: 文檔來源（預設或上傳）
        mode: 處理模式（覆蓋或更新）
        directory: 自訂目錄路徑（僅在來源為 'default' 時使用）
        use_llm_context: 是否使用 LLM 生成 chunk 脈絡摘要
    """

    source: DocumentSource = Field(
        default=DocumentSource.DEFAULT,
        description="文檔來源：'default' 使用預設文檔，'uploaded' 處理上傳文件",
    )
    mode: ProcessMode = Field(
        default=ProcessMode.UPDATE,
        description="處理模式：'override' 覆蓋所有向量，'update' 增量更新",
    )
    directory: Optional[str] = Field(
        default=None,
        description="自訂目錄路徑（僅當 source='default' 時使用）",
    )
    use_llm_context: Optional[bool] = Field(
        default=None,
        description="是否使用 LLM 生成 chunk 脈絡摘要。None 表示使用系統預設值（contextual_chunking_use_llm）",
    )


class VectorizeResponse(BaseModel):
    """
    文檔向量化響應模型。

    返回向量化操作的結果和統計資訊。

    Attributes:
        status: 操作狀態（success/error）
        message: 額外訊息（可選）
        mode: 使用的處理模式
        source: 使用的文檔來源
        documents_processed: 處理的文檔數量
        chunks_created: 建立的文本塊數量
        vectors_stored: 儲存到 Qdrant 的向量數量
        collection_name: Qdrant 集合名稱
    """

    status: str = Field(..., description="操作狀態")
    message: Optional[str] = Field(None, description="額外訊息")
    mode: str = Field(..., description="使用的處理模式")
    source: str = Field(..., description="使用的文檔來源")
    documents_processed: int = Field(..., description="處理的文檔數量")
    chunks_created: int = Field(..., description="建立的文本塊數量")
    vectors_stored: int = Field(..., description="儲存到 Qdrant 的向量數量")
    collection_name: str = Field(..., description="Qdrant 集合名稱")


class CollectionInfoResponse(BaseModel):
    """
    集合資訊響應模型。

    返回 Qdrant 集合的詳細資訊和狀態。

    Attributes:
        name: 集合名稱
        points_count: 集合中的點數量
        vectors_count: 向量數量
        status: 集合狀態
        optimizer_status: 優化器狀態
    """

    name: str = Field(..., description="集合名稱")
    points_count: int = Field(..., description="集合中的點數量")
    vectors_count: Optional[int] = Field(None, description="向量數量")
    status: str = Field(..., description="集合狀態")
    optimizer_status: str = Field(..., description="優化器狀態")


class HealthCheckResponse(BaseModel):
    """
    健康檢查響應模型。

    返回系統各組件的健康狀態和連接資訊。

    Attributes:
        qdrant_connected: Qdrant 數據庫連接狀態
        embedding_service_connected: 嵌入服務連接狀態
        collection_exists: 集合是否存在
        embedding_dimension: 嵌入服務的向量維度
        expected_dimension: 預期的向量維度
        dimension_match: 向量維度是否匹配
    """

    qdrant_connected: bool = Field(..., description="Qdrant 數據庫連接狀態")
    embedding_service_connected: bool = Field(
        ..., description="嵌入服務（OpenAI）連接狀態"
    )
    collection_exists: bool = Field(..., description="集合是否存在")
    embedding_dimension: Optional[int] = Field(
        None, description="從嵌入服務獲取的向量維度"
    )
    expected_dimension: int = Field(..., description="預期的向量維度（系統配置）")
    dimension_match: Optional[bool] = Field(None, description="向量維度是否匹配")


class ConversationMessage(BaseModel):
    """
    對話訊息模型。

    用於表示單條對話訊息，支援多輪對話上下文。

    Attributes:
        role: 訊息角色（'user' 或 'assistant'）
        content: 訊息內容
    """

    role: str = Field(..., description="訊息角色：'user'（用戶）或 'assistant'（助手）")
    content: str = Field(..., description="訊息內容")


class LLMConfig(BaseModel):
    """
    LLM 設定模型。

    用於在「單次請求」層級調整 LLM 的行為，例如：
    - 指定使用的模型名稱
    - 調整 reasoning 強度
    - 調整 reasoning summary 詳細程度

    任何欄位為 None 時，會回退到系統預設值。
    """

    model: Optional[str] = Field(
        default=None,
        description="要使用的 LLM 模型名稱，未設定時使用系統預設 chat_model。",
    )
    reasoning_effort: Optional[str] = Field(
        default="medium",
        description="Reasoning 強度：'low'、'medium'、'high' 之一；未設定時使用預設值。",
    )
    reasoning_summary: Optional[str] = Field(
        default="auto",
        description="Reasoning summary 模式：'concise'、'detailed'、'auto' 之一；未設定時使用預設值。",
    )


class QuestionRequest(BaseModel):
    """
    問題請求模型。

    用於向 RAG 系統提出問題，支援多輪對話。

    Attributes:
        question: 用戶問題（必填，至少 1 字元）
        conversation_history: 對話歷史記錄（可選）
        top_k: 檢索文檔數量（預設 3，範圍 1-10）
    """

    question: str = Field(..., description="用戶問題", min_length=1)
    conversation_history: Optional[list[ConversationMessage]] = Field(
        default=None,
        description="先前的對話訊息（用於上下文理解）",
    )
    top_k: int = Field(
        default=3,
        description="要檢索的文檔數量",
        ge=1,
        le=10,
    )
    llm_config: Optional[LLMConfig] = Field(
        default=None,
        description="本次請求欲覆寫的 LLM 設定（模型 / reasoning 等），未提供時使用系統預設值。",
    )
    enable_conversation_summary: bool = Field(
        default=True,
        description="是否啟用對話摘要記憶。關閉時系統不會產生或傳遞 conversation_summary。",
    )
    conversation_summary: Optional[str] = Field(
        default=None,
        description="先前對話內容的摘要，若提供則可延續上下文記憶。",
        max_length=4000,
    )


class QuestionResponse(BaseModel):
    """
    問題響應模型。

    返回問答結果及相關元數據。

    Attributes:
        answer: 生成的答案
        question: 原始問題
        documents_used: 是否使用了文檔檢索
        steps: 推理步驟數量（反映 Agent 的決策過程）
    """

    answer: str = Field(..., description="生成的答案")
    question: str = Field(..., description="原始問題")
    documents_used: bool = Field(
        ..., description="是否檢索並使用了文檔"
    )
    steps: int = Field(..., description="Agent 執行的推理步驟數量")


class RetrievalTestRequest(BaseModel):
    """
    檢索測試請求模型。

    用於測試文檔檢索功能，不生成答案。

    Attributes:
        query: 測試查詢文本
        top_k: 要檢索的文檔數量
    """

    query: str = Field(..., description="測試查詢", min_length=1)
    top_k: int = Field(default=3, description="要檢索的文檔數量", ge=1, le=10)


class RetrievalTestResponse(BaseModel):
    """
    檢索測試響應模型。

    返回檢索測試的結果。

    Attributes:
        status: 測試狀態
        query: 測試查詢
        documents_found: 找到的文檔數量
        documents: 檢索到的文檔列表
    """

    status: str = Field(..., description="測試狀態")
    query: str = Field(..., description="測試查詢")
    documents_found: int = Field(..., description="找到的文檔數量")
    documents: list[dict] = Field(..., description="檢索到的文檔列表（包含內容和分數）")


class InputValidationResult(BaseModel):
    """
    輸入驗證結果模型。

    記錄輸入驗證的結果，用於安全防護。

    Attributes:
        is_valid: 輸入是否有效
        reason: 拒絕原因（當無效時）
        risk_type: 檢測到的風險類型
    """

    is_valid: bool = Field(..., description="輸入是否有效")
    reason: Optional[str] = Field(None, description="拒絕原因（當無效時提供友善說明）")
    risk_type: Optional[str] = Field(
        None,
        description=(
            "檢測到的風險類型："
            "'irrelevant'（無關）、'injection'（注入攻擊）、"
            "'malicious'（惡意輸入）、'greeting'（問候語）、"
            "'service_inquiry'（服務詢問）"
        ),
    )