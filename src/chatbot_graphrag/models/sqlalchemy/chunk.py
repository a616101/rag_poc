"""
Chunk SQLAlchemy 模型

定義 Chunk 和 TraceLink 模型，用於 chunk metadata 和血統追蹤。

包含：
- Chunk: Chunk metadata 儲存（嵌入向量在 Qdrant）
- TraceLink: 血統追蹤連結
- IngestJob: 攝取工作追蹤
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import (
    String,
    Text,
    Integer,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from chatbot_graphrag.models.sqlalchemy.base import Base, TimestampMixin

if TYPE_CHECKING:
    from chatbot_graphrag.models.sqlalchemy.doc import Doc


class Chunk(Base, TimestampMixin):
    """
    Chunk metadata 模型。

    儲存 chunk metadata 和向量嵌入的參考。
    實際嵌入儲存在 Qdrant，內容索引在 OpenSearch。
    """

    __tablename__ = "chunks"

    # Primary key
    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    # Foreign key to parent doc
    doc_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("docs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    doc_version: Mapped[int] = mapped_column(Integer, default=1)

    # Chunk classification
    chunk_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Chunk type (paragraph, table, steps, faq, etc.)",
    )

    # Position tracking
    position_in_doc: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Position index within document",
    )
    section_title: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="Section/heading this chunk belongs to",
    )
    page_number: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Source page number (for PDFs)",
    )

    # Content (stored for reference, actual content in OpenSearch)
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Raw chunk content",
    )
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="SHA-256 hash of content",
    )

    # Contextual content (with document context prepended)
    contextual_content: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Content with document context prepended",
    )

    # Size metrics
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    char_count: Mapped[int] = mapped_column(Integer, default=0)

    # Hierarchy
    parent_chunk_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("chunks.id", ondelete="SET NULL"),
        nullable=True,
        comment="Parent chunk ID for hierarchical chunks",
    )
    sibling_chunk_ids: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)),
        default=list,
        comment="Sibling chunk IDs for context expansion",
    )

    # Vector store references
    qdrant_point_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="Qdrant point ID",
    )
    opensearch_doc_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        comment="OpenSearch document ID",
    )

    # Graph references
    entity_ids: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)),
        default=list,
        comment="Extracted entity IDs",
    )
    relation_ids: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)),
        default=list,
        comment="Extracted relation IDs",
    )
    community_ids: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)),
        default=list,
        comment="Associated community IDs",
    )

    # NebulaGraph reference
    nebula_chunk_vid: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="NebulaGraph vertex ID for chunk",
    )

    # Status
    status: Mapped[str] = mapped_column(
        String(20),
        default="active",
        index=True,
    )
    is_indexed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="Whether chunk is indexed in vector stores",
    )
    indexed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Language
    language: Mapped[str] = mapped_column(String(10), default="zh-TW")

    # Extended metadata (Python attr 'extra_metadata' maps to DB column 'metadata')
    # Note: 'metadata' is reserved in SQLAlchemy, so we use a different Python attr name
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",  # DB column name
        JSONB,
        default=dict,
        comment="Extended chunk metadata",
    )

    # Relationship
    doc: Mapped["Doc"] = relationship("Doc", back_populates="chunks")
    parent_chunk: Mapped[Optional["Chunk"]] = relationship(
        "Chunk",
        remote_side="Chunk.id",
        foreign_keys=[parent_chunk_id],
    )
    trace_links: Mapped[list["TraceLink"]] = relationship(
        "TraceLink",
        back_populates="chunk",
        cascade="all, delete-orphan",
    )

    # Indexes
    __table_args__ = (
        Index("ix_chunks_doc_id_position", "doc_id", "position_in_doc"),
        Index("ix_chunks_doc_id_version", "doc_id", "doc_version"),
        Index("ix_chunks_chunk_type_status", "chunk_type", "status"),
        Index("ix_chunks_entity_ids", "entity_ids", postgresql_using="gin"),
        Index("ix_chunks_community_ids", "community_ids", postgresql_using="gin"),
        Index("ix_chunks_metadata", "metadata", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<Chunk(id={self.id}, doc_id={self.doc_id}, type={self.chunk_type})>"


class TraceLink(Base, TimestampMixin):
    """
    血統追蹤的追蹤連結模型。

    連結 chunk 與實體、關係和社群，
    用於來源追蹤和證據引用。
    """

    __tablename__ = "trace_links"

    # Primary key
    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    # Foreign key to chunk
    chunk_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Document reference
    doc_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("docs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    doc_version: Mapped[int] = mapped_column(Integer, default=1)

    # Link type
    link_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Link type: entity, relation, community, event",
    )

    # Target references
    entity_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="Linked entity ID",
    )
    relation_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="Linked relation ID",
    )
    community_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="Linked community ID",
    )
    event_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="Linked event ID",
    )

    # Extraction metadata
    extraction_method: Mapped[str] = mapped_column(
        String(50),
        default="llm",
        comment="Extraction method: llm, rule, manual",
    )
    extraction_model: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Model used for extraction",
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        default=1.0,
        comment="Extraction confidence score",
    )

    # Position in chunk
    start_offset: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Start character offset in chunk",
    )
    end_offset: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="End character offset in chunk",
    )
    matched_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Matched text span",
    )

    # Verification
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="Whether link has been verified",
    )
    verified_by: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )
    verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationship
    chunk: Mapped["Chunk"] = relationship("Chunk", back_populates="trace_links")

    # Indexes
    __table_args__ = (
        Index("ix_trace_links_chunk_link_type", "chunk_id", "link_type"),
        Index("ix_trace_links_entity_id", "entity_id"),
        Index("ix_trace_links_relation_id", "relation_id"),
        Index("ix_trace_links_community_id", "community_id"),
    )

    def __repr__(self) -> str:
        return f"<TraceLink(id={self.id}, chunk_id={self.chunk_id}, type={self.link_type})>"


class IngestJob(Base, TimestampMixin):
    """
    攝取工作追蹤模型。

    追蹤文件攝取工作的狀態、進度和錯誤。
    """

    __tablename__ = "ingest_jobs"

    # Primary key
    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    # Job configuration
    pipeline_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
    )
    process_mode: Mapped[str] = mapped_column(
        String(20),
        default="update",
    )
    source_path: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    source_urls: Mapped[list[str]] = mapped_column(
        ARRAY(String(2000)),
        default=list,
    )
    collection_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Configuration JSON
    config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        comment="Full job configuration",
    )

    # Status
    status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        index=True,
    )
    progress: Mapped[float] = mapped_column(Float, default=0.0)

    # Counters
    total_documents: Mapped[int] = mapped_column(Integer, default=0)
    processed_documents: Mapped[int] = mapped_column(Integer, default=0)
    failed_documents: Mapped[int] = mapped_column(Integer, default=0)
    skipped_documents: Mapped[int] = mapped_column(Integer, default=0)
    chunks_created: Mapped[int] = mapped_column(Integer, default=0)
    entities_extracted: Mapped[int] = mapped_column(Integer, default=0)
    relations_extracted: Mapped[int] = mapped_column(Integer, default=0)
    communities_detected: Mapped[int] = mapped_column(Integer, default=0)

    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Error tracking
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    document_errors: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        default=list,
    )

    # Audit
    created_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Indexes
    __table_args__ = (
        Index("ix_ingest_jobs_status_created", "status", "created_at"),
        Index("ix_ingest_jobs_pipeline_status", "pipeline_type", "status"),
    )

    def __repr__(self) -> str:
        return f"<IngestJob(id={self.id}, status={self.status}, progress={self.progress}%)>"
