"""
GraphRAG 攝取模型

定義文件攝取、分塊和管道工件的 Pydantic 模型，包含：
- Document：處理後的文件模型
- Chunk：文件分塊模型
- IngestJob：攝取工作追蹤
- PipelineResult：管道執行結果
- CuratedDocSchema：YAML frontmatter 結構定義
"""

from datetime import datetime
from typing import Any, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from chatbot_graphrag.core.constants import (
    DocType,
    PipelineType,
    ProcessMode,
    JobStatus,
    DocStatus,
    ChunkType,
)


# ==================== 文件模型 ====================


class DocumentMetadata(BaseModel):
    """
    從文件 YAML frontmatter 抽取或推斷的 metadata。

    這些 metadata 用於分類、搜尋和存取控制。
    """

    doc_type: DocType = Field(..., description="文件類型分類")
    title: str = Field(..., description="文件標題")
    description: Optional[str] = Field(default=None, description="文件描述")
    language: str = Field(default="zh-TW", description="文件語言")
    source_url: Optional[str] = Field(default=None, description="原始來源 URL")
    author: Optional[str] = Field(default=None, description="文件作者")
    department: Optional[str] = Field(default=None, description="部門/類別")
    tags: list[str] = Field(default_factory=list, description="文件標籤")
    version: str = Field(default="1.0.0", description="文件版本")
    effective_date: Optional[datetime] = Field(default=None, description="生效日期")
    expiry_date: Optional[datetime] = Field(default=None, description="到期日期")
    acl_groups: list[str] = Field(
        default_factory=lambda: ["public"],
        description="存取控制群組"
    )
    tenant_id: str = Field(
        default="default",
        description="租戶識別碼（用於多租戶隔離）"
    )
    custom_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="額外自訂 metadata"
    )


class DocumentInput(BaseModel):
    """
    攝取的輸入文件。

    可透過檔案路徑、原始內容或 URL 提供文件。
    """

    file_path: Optional[str] = Field(default=None, description="本地檔案路徑")
    content: Optional[str] = Field(default=None, description="原始內容（替代 file_path）")
    url: Optional[str] = Field(default=None, description="來源 URL")
    metadata: Optional[DocumentMetadata] = Field(
        default=None,
        description="預定義 metadata（覆蓋抽取結果）"
    )
    pipeline: PipelineType = Field(
        default=PipelineType.CURATED,
        description="管道類型"
    )


class Document(BaseModel):
    """
    處理後的文件模型。

    包含完整的文件資訊，包括 metadata、內容雜湊、儲存路徑等。
    """

    id: str = Field(..., description="唯一文件 ID")
    metadata: DocumentMetadata = Field(..., description="文件 metadata")
    content_hash: str = Field(..., description="內容的 SHA-256 雜湊")
    raw_content: str = Field(..., description="原始內容")
    canonical_path: Optional[str] = Field(
        default=None,
        description="MinIO 中規範 JSON 的路徑"
    )
    asset_paths: list[str] = Field(
        default_factory=list,
        description="MinIO 中關聯資產的路徑（圖片等）"
    )
    pipeline: PipelineType = Field(..., description="用於攝取的管道")
    status: DocStatus = Field(default=DocStatus.ACTIVE, description="文件狀態")
    version: int = Field(default=1, description="版本號")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentVersion(BaseModel):
    """
    文件版本追蹤。

    用於追蹤文件的歷史版本，支援回滾和審計。
    """

    id: str = Field(..., description="版本 ID")
    doc_id: str = Field(..., description="父文件 ID")
    version: int = Field(..., description="版本號")
    content_hash: str = Field(..., description="此版本的內容雜湊")
    canonical_path: str = Field(..., description="MinIO 中規範 JSON 的路徑")
    change_summary: Optional[str] = Field(default=None, description="變更摘要")
    status: DocStatus = Field(default=DocStatus.ACTIVE)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(default=None, description="建立此版本的使用者")


# ==================== Chunk 模型 ====================


class ChunkMetadata(BaseModel):
    """
    Chunk 層級的 metadata。

    包含 chunk 的分類、位置和關聯資訊。
    """

    chunk_type: ChunkType = Field(..., description="Chunk 類型分類")
    section_title: Optional[str] = Field(default=None, description="章節標題")
    page_number: Optional[int] = Field(default=None, description="來源頁碼")
    position_in_doc: int = Field(default=0, description="文件中的位置索引")
    parent_chunk_id: Optional[str] = Field(
        default=None,
        description="父 chunk ID（用於階層式 chunk）"
    )
    sibling_chunk_ids: list[str] = Field(
        default_factory=list,
        description="兄弟 chunk ID 列表"
    )
    language: str = Field(default="zh-TW", description="Chunk 語言")
    entity_ids: list[str] = Field(
        default_factory=list,
        description="抽取的實體 ID 列表"
    )
    relation_ids: list[str] = Field(
        default_factory=list,
        description="抽取的關係 ID 列表"
    )
    custom_fields: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """
    處理後的 chunk 模型。

    Chunk 是文件分塊後的最小檢索單位，包含文字內容和嵌入向量。
    """

    id: str = Field(..., description="唯一 chunk ID")
    doc_id: str = Field(..., description="父文件 ID")
    doc_version: int = Field(default=1, description="文件版本")
    content: str = Field(..., description="Chunk 文字內容")
    contextual_content: Optional[str] = Field(
        default=None,
        description="上下文內容（前置文件上下文）"
    )
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")
    token_count: int = Field(default=0, description="Token 數量")
    char_count: int = Field(default=0, description="字元數量")
    dense_embedding: Optional[list[float]] = Field(
        default=None,
        description="密集嵌入向量"
    )
    sparse_embedding: Optional[dict[int, float]] = Field(
        default=None,
        description="稀疏嵌入（token_id -> 權重）"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def effective_content(self) -> str:
        """取得用於檢索的有效內容（優先使用上下文內容）。"""
        return self.contextual_content or self.content


class ChunkCreate(BaseModel):
    """
    建立新 chunk 的模型。

    用於 chunker 輸出的中間資料。
    """

    doc_id: str = Field(..., description="父文件 ID")
    content: str = Field(..., description="Chunk 文字內容")
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")
    doc_version: int = Field(default=1)


class ChunkArtifact(BaseModel):
    """
    Chunk 工件，用於儲存/序列化。

    包含 chunk 資料和各儲存系統中的參考 ID。
    """

    chunk: Chunk = Field(..., description="Chunk 資料")
    qdrant_point_id: Optional[str] = Field(
        default=None,
        description="Qdrant point ID"
    )
    opensearch_doc_id: Optional[str] = Field(
        default=None,
        description="OpenSearch 文件 ID"
    )
    nebula_chunk_vid: Optional[str] = Field(
        default=None,
        description="NebulaGraph chunk 頂點 ID"
    )


# ==================== 攝取工作模型 ====================


class IngestJobConfig(BaseModel):
    """
    攝取工作配置。

    定義攝取工作的所有參數，包括管道類型、處理模式、目標集合等。
    """

    pipeline: PipelineType = Field(..., description="管道類型")
    mode: ProcessMode = Field(default=ProcessMode.UPDATE, description="處理模式")
    source_path: Optional[str] = Field(default=None, description="來源路徑")
    source_urls: Optional[list[str]] = Field(default=None, description="來源 URL 列表")
    collection_name: str = Field(..., description="目標 Qdrant 集合")
    enable_graph: bool = Field(default=True, description="啟用圖譜抽取")
    enable_community_detection: bool = Field(default=True, description="啟用社群偵測")
    doc_type_filter: Optional[list[str]] = Field(default=None, description="文件類型過濾器")
    chunk_size: int = Field(default=500, description="目標 chunk 大小")
    chunk_overlap: int = Field(default=50, description="Chunk 重疊大小")
    contextual_chunking: bool = Field(default=True, description="啟用上下文分塊")


class IngestJob(BaseModel):
    """
    攝取工作模型。

    追蹤攝取工作的狀態、進度和統計資訊。
    """

    id: str = Field(..., description="唯一工作 ID")
    config: IngestJobConfig = Field(..., description="工作配置")
    status: JobStatus = Field(default=JobStatus.PENDING, description="工作狀態")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="進度百分比")

    # 計數器
    total_documents: int = Field(default=0, description="待處理的總文件數")
    processed_documents: int = Field(default=0, description="已處理的文件數")
    failed_documents: int = Field(default=0, description="失敗的文件數")
    skipped_documents: int = Field(default=0, description="跳過的文件數（內容未變更）")
    chunks_created: int = Field(default=0, description="建立的 chunk 數")
    entities_extracted: int = Field(default=0, description="抽取的實體數")
    relations_extracted: int = Field(default=0, description="抽取的關係數")
    communities_detected: int = Field(default=0, description="偵測的社群數")

    # 時間追蹤
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(default=None, description="工作開始時間")
    completed_at: Optional[datetime] = Field(default=None, description="工作完成時間")

    # 錯誤追蹤
    error_message: Optional[str] = Field(default=None, description="總體錯誤訊息")
    document_errors: list[dict[str, Any]] = Field(
        default_factory=list,
        description="每個文件的錯誤詳情"
    )

    # 建立者
    created_by: Optional[str] = Field(default=None, description="建立工作的使用者")

    @computed_field
    @property
    def duration_seconds(self) -> Optional[float]:
        """計算工作執行時間（秒）。"""
        if self.started_at:
            end_time = self.completed_at or datetime.utcnow()
            return (end_time - self.started_at).total_seconds()
        return None

    @computed_field
    @property
    def success_rate(self) -> float:
        """計算文件處理成功率。"""
        if self.processed_documents + self.failed_documents == 0:
            return 0.0
        return self.processed_documents / (self.processed_documents + self.failed_documents)


class IngestJobUpdate(BaseModel):
    """
    更新攝取工作狀態的模型。

    用於部分更新工作的狀態和統計資訊。
    """

    status: Optional[JobStatus] = None
    progress: Optional[float] = None
    processed_documents: Optional[int] = None
    failed_documents: Optional[int] = None
    skipped_documents: Optional[int] = None
    chunks_created: Optional[int] = None
    entities_extracted: Optional[int] = None
    relations_extracted: Optional[int] = None
    communities_detected: Optional[int] = None
    error_message: Optional[str] = None
    document_errors: Optional[list[dict[str, Any]]] = None


# ==================== 管道工件模型 ====================


class PipelineStageResult(BaseModel):
    """
    管道階段執行結果。

    記錄單一管道階段的執行統計和錯誤資訊。
    """

    stage_name: str = Field(..., description="階段名稱")
    success: bool = Field(..., description="階段是否成功")
    input_count: int = Field(default=0, description="輸入項目數")
    output_count: int = Field(default=0, description="輸出項目數")
    duration_ms: float = Field(default=0.0, description="階段執行時間（毫秒）")
    error_message: Optional[str] = Field(default=None, description="失敗時的錯誤訊息")
    metadata: dict[str, Any] = Field(default_factory=dict, description="階段 metadata")


class PipelineResult(BaseModel):
    """
    完整管道執行結果。

    包含整個管道的執行統計、輸出和各階段結果。
    """

    job_id: str = Field(..., description="工作 ID")
    pipeline: PipelineType = Field(..., description="管道類型")
    success: bool = Field(..., description="總體是否成功")
    stages: list[PipelineStageResult] = Field(default_factory=list)

    # 輸出
    documents: list[Document] = Field(default_factory=list, description="處理後的文件")
    chunks: list[Chunk] = Field(default_factory=list, description="建立的 chunk")
    entity_count: int = Field(default=0)
    relation_count: int = Field(default=0)
    community_count: int = Field(default=0)

    # 時間追蹤
    total_duration_ms: float = Field(default=0.0, description="管道總執行時間")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)


# ==================== 索引版本模型 ====================


class IndexVersion(BaseModel):
    """
    索引版本，用於追蹤 schema/資料版本。

    支援藍綠部署和版本回滾。
    """

    id: str = Field(..., description="版本 ID")
    version: str = Field(..., description="語意版本字串")
    index_type: str = Field(..., description="索引類型（qdrant, opensearch, nebula）")
    collection_name: str = Field(..., description="集合/索引名稱")
    schema_hash: str = Field(..., description="用於相容性檢查的 schema 雜湊")
    doc_count: int = Field(default=0, description="此版本的文件數")
    chunk_count: int = Field(default=0, description="此版本的 chunk 數")
    is_active: bool = Field(default=True, description="是否為當前活躍版本")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    activated_at: Optional[datetime] = Field(default=None)
    deactivated_at: Optional[datetime] = Field(default=None)
    notes: Optional[str] = Field(default=None, description="版本備註")


# ==================== 追蹤連結模型 ====================


class TraceLink(BaseModel):
    """
    用於血統追蹤的 chunk、實體和文件之間的連結。

    支援從實體追溯到來源 chunk 和文件。
    """

    id: str = Field(..., description="連結 ID")
    chunk_id: str = Field(..., description="Chunk ID")
    doc_id: str = Field(..., description="文件 ID")
    doc_version: int = Field(default=1, description="文件版本")
    entity_ids: list[str] = Field(default_factory=list, description="關聯的實體 ID")
    relation_ids: list[str] = Field(default_factory=list, description="關聯的關係 ID")
    community_ids: list[str] = Field(default_factory=list, description="關聯的社群 ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ==================== YAML Schema 模型 ====================


class CuratedDocSchema(BaseModel):
    """
    精選文件的 YAML frontmatter schema。

    支援多種文件格式：
    1. 標準格式：doc_type, title
    2. 醫師格式：type, name（含 zh-Hant/en 子欄位）
    3. 衛教單格式：type, title（dict）, education（複雜物件）
    4. 指南/流程格式：type, title（dict）

    使用 extra='allow' 接受未明確定義的欄位而不會驗證失敗。
    """

    # 允許未明確定義的額外欄位
    model_config = ConfigDict(extra='allow')

    # 核心欄位 - 同時支援 type 和 doc_type
    type: Optional[str] = Field(default=None, description="文件類型（doc_type 的別名）")
    doc_type: Optional[str] = Field(default=None, description="文件類型")

    # 標題欄位 - 可選，從 name 欄位推導；支援 str 或 dict（含 zh-Hant/en）
    title: Optional[str | dict[str, str]] = Field(default=None, description="文件標題（字串或 dict）")
    id: Optional[str] = Field(default=None, description="文件 ID")

    # 通用欄位
    description: Optional[str | dict[str, str]] = Field(default=None, description="描述（str 或 dict）")
    summary: Optional[str | dict[str, str]] = Field(default=None, description="摘要（str 或 dict）")
    language: str = Field(default="zh-TW")
    lang: Optional[str] = Field(default=None, description="語言代碼（別名）")
    department: Optional[str] = Field(default=None)
    tags: list[str] = Field(default_factory=list)
    version: str = Field(default="1.0.0")
    effective_date: Optional[str] = Field(default=None, description="ISO 日期字串")
    acl_groups: list[str] = Field(default_factory=lambda: ["public"])

    # 類型特定欄位（根據 doc_type 驗證）
    # 程序類
    steps: Optional[list[dict[str, Any]]] = Field(default=None)
    requirements: Optional[list[str]] = Field(default=None)
    fees: Optional[dict[str, Any]] = Field(default=None)
    forms: Optional[list[str]] = Field(default=None)

    # 位置指南類
    building: Optional[str] = Field(default=None)
    floor: Optional[str] = Field(default=None)
    room: Optional[str] = Field(default=None)
    coordinates: Optional[dict[str, float]] = Field(default=None)

    # 交通指南類
    transport_modes: Optional[list[dict[str, Any]]] = Field(default=None)
    parking: Optional[dict[str, Any]] = Field(default=None)

    # 醫師類 - 支援複雜結構
    name: Optional[str | dict[str, str]] = Field(default=None, description="姓名字串或含 zh-Hant/en 的 dict")
    specialties: Optional[list[str]] = Field(default=None)
    schedule: Optional[list[dict[str, Any]]] = Field(default=None)
    contact: Optional[dict[str, str]] = Field(default=None)

    # 醫師額外欄位
    org: Optional[dict[str, Any]] = Field(default=None, description="組織資訊")
    departments: Optional[list[dict[str, Any]]] = Field(default=None, description="部門列表")
    role: Optional[dict[str, Any]] = Field(default=None, description="職位/職稱資訊")
    expertise: Optional[list[str]] = Field(default=None, description="專長領域")
    # education 可以是 list[str]（醫師學歷）或 dict（衛教單元資訊）
    education: Optional[list[str] | dict[str, Any]] = Field(default=None, description="學歷或衛教單元 metadata")
    experience: Optional[list[str]] = Field(default=None, description="工作經歷")
    certifications: Optional[list[str]] = Field(default=None, description="證照")
    memberships: Optional[list[str]] = Field(default=None, description="專業會員資格")
    languages: Optional[list[str]] = Field(default=None, description="語言能力")
    retrieval: Optional[dict[str, Any]] = Field(default=None, description="檢索 metadata")
    source: Optional[dict[str, Any]] = Field(default=None, description="來源資訊")
    updated_at: Optional[str | datetime] = Field(default=None, description="最後更新日期")
    last_reviewed: Optional[str | datetime] = Field(default=None, description="最後審核日期")

    # 醫療團隊類
    team_name: Optional[str] = Field(default=None)
    members: Optional[list[dict[str, Any]]] = Field(default=None)
    services: Optional[list[str]] = Field(default=None)

    # 衛教單欄位
    audience: Optional[list[str]] = Field(default=None, description="目標對象（病患、家屬等）")

    @field_validator('tags', 'acl_groups', mode='before')
    @classmethod
    def coerce_list_to_strings(cls, v):
        """將所有列表項目轉換為字串（YAML 可能將 2025 解析為 int）。"""
        if v is None:
            return []
        if isinstance(v, list):
            return [str(item) for item in v if item is not None and not isinstance(item, list)]
        return v

    @field_validator('certifications', 'memberships', 'specialties', 'expertise', mode='before')
    @classmethod
    def filter_nested_empty_lists(cls, v):
        """過濾掉像 [[]] 這樣的巢狀空列表（來自 YAML - [] 標記）。"""
        if v is None:
            return None
        if isinstance(v, list):
            # 過濾空列表和 None 值，轉換為字串
            cleaned = [str(item) for item in v if item is not None and item != [] and not isinstance(item, list)]
            return cleaned if cleaned else None
        return v

    @field_validator('education', mode='before')
    @classmethod
    def normalize_education_field(cls, v):
        """處理可以是 list[str] 或 dict 或 [None]（來自 YAML）的 education 欄位。"""
        if v is None:
            return None
        # 處理 dict（衛教單 metadata）- 直接傳遞
        if isinstance(v, dict):
            return v
        # 處理 list（醫師學歷）
        if isinstance(v, list):
            # 過濾 None 和空值
            cleaned = [str(item) for item in v if item is not None and item != '' and not isinstance(item, list)]
            return cleaned if cleaned else None
        return v

    @model_validator(mode="after")
    def normalize_fields(self) -> "CuratedDocSchema":
        """
        正規化欄位：
        1. type → doc_type（別名處理）
        2. title dict → title str（多語言處理）
        3. name dict → title str（fallback）
        4. expertise → specialties（推導）
        """
        # 處理 doc_type / type 別名
        if self.doc_type is None and self.type is not None:
            self.doc_type = self.type
        elif self.doc_type is None and self.type is None:
            raise ValueError("必須提供 'doc_type' 或 'type' 欄位")

        # 處理 title dict → str
        if isinstance(self.title, dict):
            # 優先使用 zh-Hant，其次 zh，最後取第一個值
            self.title = (
                self.title.get("zh-Hant")
                or self.title.get("zh")
                or next(iter(self.title.values()), None)
            )

        # 若 title 仍為 None，嘗試從 name 推導
        if self.title is None:
            if isinstance(self.name, dict):
                self.title = (
                    self.name.get("zh-Hant")
                    or self.name.get("zh")
                    or next(iter(self.name.values()), None)
                )
            elif isinstance(self.name, str):
                self.title = self.name

        if self.title is None:
            raise ValueError("必須提供 'title' 或 'name' 欄位")

        # 從 expertise 推導 specialties（如果沒有 specialties 但有 expertise）
        if self.specialties is None and self.expertise is not None:
            self.specialties = self.expertise

        return self
