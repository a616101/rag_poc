"""
GraphRAG 圖譜資料模型

定義知識圖譜相關的 Pydantic 模型，包含：
- 實體（Entity）：圖譜中的節點，如人員、部門、程序、位置等
- 關係（Relation）：實體之間的連結，如 belongs_to, works_in 等
- 事件（Event）：時間相關的發生事項
- 社群（Community）：透過 Leiden 演算法偵測的實體群組
- 子圖（Subgraph）：查詢結果的圖譜片段
- 抽取結果（ExtractionResult）：LLM 從文字中抽取的實體/關係
"""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field, computed_field

from chatbot_graphrag.core.constants import EntityType, RelationType


# ==================== 實體模型 ====================


class EntityBase(BaseModel):
    """
    實體基礎模型。

    定義圖譜實體的共用欄位，所有實體模型都繼承此類別。
    """

    name: str = Field(..., min_length=1, max_length=500, description="實體名稱")
    entity_type: EntityType = Field(..., description="實體類型分類")
    description: Optional[str] = Field(default=None, description="實體描述")
    properties: dict[str, Any] = Field(default_factory=dict, description="額外屬性")


class Entity(EntityBase):
    """
    完整實體模型，包含 ID 和 metadata。

    用於表示已儲存在圖譜中的實體，包含來源追蹤和時間戳記。
    """

    id: str = Field(..., description="唯一實體 ID（NebulaGraph 中的 vid）")
    embedding: Optional[list[float]] = Field(
        default=None,
        description="實體名稱/描述的嵌入向量"
    )
    source_chunk_ids: list[str] = Field(
        default_factory=list,
        description="抽取此實體的來源 chunk ID 列表"
    )
    doc_ids: list[str] = Field(
        default_factory=list,
        description="來源文件 ID 列表"
    )
    mention_count: int = Field(default=1, description="跨 chunk 的提及次數")
    aliases: list[str] = Field(
        default_factory=list,
        description="實體的替代名稱/別名"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def nebula_vid(self) -> str:
        """
        取得 NebulaGraph 頂點 ID。

        注意：Entity.id 已包含完整的 VID 格式（e_{type}_{hash}），
        由 entity_extractor._generate_entity_id() 設定，因此直接返回。
        """
        return self.id


class EntityCreate(EntityBase):
    """
    建立新實體的模型。

    用於從 chunk 中抽取新實體時的輸入資料。
    """

    source_chunk_id: str = Field(..., description="抽取來源的 chunk ID")
    doc_id: str = Field(..., description="來源文件 ID")


class EntityMerge(BaseModel):
    """
    合併重複實體的模型。

    用於實體解析（Entity Resolution），將重複的實體合併為一個。
    """

    primary_id: str = Field(..., description="要保留的主要實體 ID")
    merge_ids: list[str] = Field(..., description="要合併到主要實體的實體 ID 列表")
    merged_properties: Optional[dict[str, Any]] = Field(
        default=None,
        description="合併後要套用的屬性"
    )


# ==================== 關係模型 ====================


class RelationBase(BaseModel):
    """
    關係基礎模型。

    定義圖譜關係（邊）的共用欄位，連接兩個實體。
    """

    source_id: str = Field(..., description="來源實體 ID")
    target_id: str = Field(..., description="目標實體 ID")
    relation_type: RelationType = Field(..., description="關係類型")
    description: Optional[str] = Field(default=None, description="關係描述")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="關係權重/強度")
    properties: dict[str, Any] = Field(default_factory=dict, description="額外屬性")


class Relation(RelationBase):
    """
    完整關係模型，包含 metadata。

    用於表示已儲存在圖譜中的關係。
    """

    id: str = Field(..., description="唯一關係 ID")
    source_chunk_ids: list[str] = Field(
        default_factory=list,
        description="抽取此關係的來源 chunk ID 列表"
    )
    doc_ids: list[str] = Field(
        default_factory=list,
        description="來源文件 ID 列表"
    )
    mention_count: int = Field(default=1, description="提及次數")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def nebula_edge_type(self) -> str:
        """取得 NebulaGraph 邊類型。"""
        return self.relation_type.value


class RelationCreate(RelationBase):
    """
    建立新關係的模型。

    用於從 chunk 中抽取新關係時的輸入資料。
    """

    source_chunk_id: str = Field(..., description="抽取來源的 chunk ID")
    doc_id: str = Field(..., description="來源文件 ID")


# ==================== 事件模型 ====================


class EventBase(BaseModel):
    """
    事件基礎模型（時間相關的發生事項）。

    事件是圖譜中具有時間維度的節點，用於記錄程序更新、
    時間表變更等時間相關的資訊。
    """

    name: str = Field(..., min_length=1, max_length=500, description="事件名稱")
    event_type: str = Field(..., description="事件類型（如 'procedure_update', 'schedule_change'）")
    description: Optional[str] = Field(default=None, description="事件描述")
    participants: list[str] = Field(
        default_factory=list,
        description="參與此事件的實體 ID 列表"
    )
    location_id: Optional[str] = Field(default=None, description="位置實體 ID")
    start_time: Optional[datetime] = Field(default=None, description="事件開始時間")
    end_time: Optional[datetime] = Field(default=None, description="事件結束時間")
    properties: dict[str, Any] = Field(default_factory=dict, description="額外屬性")


class Event(EventBase):
    """
    完整事件模型，包含 metadata。

    用於表示已儲存在圖譜中的事件。
    """

    id: str = Field(..., description="唯一事件 ID")
    source_chunk_ids: list[str] = Field(
        default_factory=list,
        description="抽取此事件的來源 chunk ID 列表"
    )
    doc_ids: list[str] = Field(
        default_factory=list,
        description="來源文件 ID 列表"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EventCreate(EventBase):
    """
    建立新事件的模型。

    用於從 chunk 中抽取新事件時的輸入資料。
    """

    source_chunk_id: str = Field(..., description="抽取來源的 chunk ID")
    doc_id: str = Field(..., description="來源文件 ID")


# ==================== 社群模型 ====================


class CommunityBase(BaseModel):
    """
    社群基礎模型（Leiden/Louvain 聚類）。

    社群是透過圖譜社群偵測演算法（如 Leiden）識別的實體群組，
    用於 GraphRAG 的 Global 和 DRIFT 查詢模式。
    """

    level: int = Field(..., ge=0, description="社群層級（0 = 葉節點）")
    title: Optional[str] = Field(default=None, description="社群標題（自動生成）")
    summary: Optional[str] = Field(default=None, description="社群摘要（LLM 生成）")
    entity_ids: list[str] = Field(
        default_factory=list,
        description="此社群中的實體 ID 列表"
    )
    sub_community_ids: list[str] = Field(
        default_factory=list,
        description="子社群 ID 列表（用於階層式社群）"
    )
    parent_community_id: Optional[str] = Field(
        default=None,
        description="父社群 ID"
    )


class Community(CommunityBase):
    """
    完整社群模型，包含 metadata。

    用於表示已儲存的社群，包含統計資訊和嵌入向量。
    """

    id: str = Field(..., description="唯一社群 ID")
    embedding: Optional[list[float]] = Field(
        default=None,
        description="社群摘要的嵌入向量"
    )
    entity_count: int = Field(default=0, description="社群中的實體數量")
    edge_count: int = Field(default=0, description="社群內部的邊數量")
    modularity_score: Optional[float] = Field(default=None, description="社群模組度分數")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def nebula_vid(self) -> str:
        """生成社群的 NebulaGraph 頂點 ID。"""
        return f"c_{self.level}_{self.id}"


class CommunityReport(BaseModel):
    """
    社群報告模型（LLM 生成的摘要）。

    社群報告是 GraphRAG Global 模式的核心，包含社群的
    關鍵實體、關係和主題摘要，用於回答全局性問題。
    """

    community_id: str = Field(..., description="社群 ID")
    level: int = Field(..., ge=0, description="社群層級")
    title: str = Field(..., description="報告標題")
    summary: str = Field(..., description="LLM 生成的摘要")
    key_entities: list[str] = Field(
        default_factory=list,
        description="社群中的關鍵實體名稱"
    )
    key_relations: list[str] = Field(
        default_factory=list,
        description="關鍵關係描述"
    )
    themes: list[str] = Field(
        default_factory=list,
        description="識別的主題/話題"
    )
    importance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="重要性分數（用於排序）"
    )
    token_count: int = Field(default=0, description="報告的 token 數量")
    embedding: Optional[list[float]] = Field(
        default=None,
        description="報告的嵌入向量（用於語意搜尋）"
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: str = Field(default="", description="用於生成的 LLM 模型")


class CommunityCreate(CommunityBase):
    """建立新社群的模型。"""

    pass


# ==================== 子圖模型 ====================


class Subgraph(BaseModel):
    """
    子圖模型，用於查詢結果。

    表示從圖譜中提取的子集，包含查詢相關的實體、關係、
    事件和社群。
    """

    entities: list[Entity] = Field(default_factory=list, description="子圖中的實體")
    relations: list[Relation] = Field(default_factory=list, description="子圖中的關係")
    events: list[Event] = Field(default_factory=list, description="子圖中的事件")
    communities: list[Community] = Field(
        default_factory=list,
        description="相關社群"
    )
    seed_entity_ids: list[str] = Field(
        default_factory=list,
        description="啟動遍歷的種子實體 ID"
    )
    hop_count: int = Field(default=0, description="遍歷的跳數")


class GraphTraversalResult(BaseModel):
    """
    圖譜遍歷操作結果。

    包含遍歷產生的子圖和執行統計資訊。
    """

    subgraph: Subgraph = Field(..., description="提取的子圖")
    traversal_path: list[list[str]] = Field(
        default_factory=list,
        description="遍歷路徑（每跳的實體 ID 列表）"
    )
    total_entities_visited: int = Field(default=0, description="訪問的總實體數")
    total_edges_traversed: int = Field(default=0, description="遍歷的總邊數")
    execution_time_ms: float = Field(default=0.0, description="執行時間（毫秒）")


# ==================== 實體抽取模型 ====================


class ExtractedEntity(BaseModel):
    """
    LLM 從文字中抽取的實體。

    這是抽取過程的中間結果，尚未寫入圖譜。
    """

    name: str = Field(..., description="實體名稱")
    entity_type: str = Field(..., description="實體類型字串")
    description: str = Field(default="", description="實體描述")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="抽取信心度")


class ExtractedRelation(BaseModel):
    """
    LLM 從文字中抽取的關係。

    這是抽取過程的中間結果，尚未寫入圖譜。
    """

    source: str = Field(..., description="來源實體名稱")
    target: str = Field(..., description="目標實體名稱")
    relation_type: str = Field(..., description="關係類型字串")
    description: str = Field(default="", description="關係描述")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="抽取信心度")


class ExtractionResult(BaseModel):
    """
    從 chunk 中抽取實體/關係的結果。

    包含單一 chunk 的完整抽取結果和執行統計。
    """

    chunk_id: str = Field(..., description="來源 chunk ID")
    doc_id: str = Field(..., description="來源文件 ID")
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    model_used: str = Field(default="", description="使用的 LLM 模型")
    extraction_time_ms: float = Field(default=0.0)
