"""
文件 SQLAlchemy 模型

定義 Doc 和 DocVersion 模型，用於在 PostgreSQL 中儲存文件 metadata。

包含：
- Doc: 文件 metadata（實際內容在 MinIO）
- DocVersion: 文件版本追蹤
- Asset: 文件相關資產（圖片、PDF 等）
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import (
    String,
    Text,
    Integer,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from chatbot_graphrag.models.sqlalchemy.base import Base, TimestampMixin

if TYPE_CHECKING:
    from chatbot_graphrag.models.sqlalchemy.chunk import Chunk


class Doc(Base, TimestampMixin):
    """
    文件 metadata 模型。

    儲存文件 metadata 和物件儲存的參考。
    實際內容以規範 JSON 格式儲存在 MinIO。
    """

    __tablename__ = "docs"

    # Primary key
    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    # Document classification
    doc_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Document type (procedure, guide.location, physician, etc.)",
    )
    pipeline_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="curated",
        comment="Pipeline type (curated, raw)",
    )

    # Basic metadata
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    language: Mapped[str] = mapped_column(String(10), default="zh-TW")
    source_url: Mapped[Optional[str]] = mapped_column(String(2000), nullable=True)
    source_path: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)

    # Categorization
    department: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(String(100)),
        default=list,
        nullable=False,
    )

    # Version tracking
    current_version: Mapped[int] = mapped_column(Integer, default=1)
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="SHA-256 hash of current content",
    )

    # Storage references
    canonical_path: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="MinIO path to canonical JSON",
    )
    asset_paths: Mapped[list[str]] = mapped_column(
        ARRAY(String(500)),
        default=list,
        comment="MinIO paths to assets (images, PDFs, etc.)",
    )

    # Status
    status: Mapped[str] = mapped_column(
        String(20),
        default="active",
        index=True,
        comment="Document status (active, deprecated, deleted)",
    )
    is_published: Mapped[bool] = mapped_column(Boolean, default=True)

    # Validity period
    effective_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    expiry_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Extended metadata (flexible JSON storage)
    # Note: Python attr 'extra_metadata' maps to DB column 'metadata'
    # ('metadata' is reserved in SQLAlchemy)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",  # DB column name
        JSONB,
        default=dict,
        comment="Extended metadata as JSON",
    )

    # ACL groups for access control
    acl_groups: Mapped[list[str]] = mapped_column(
        ARRAY(String(100)),
        default=lambda: ["public"],
        comment="Access control groups",
    )

    # Relationships
    versions: Mapped[list["DocVersion"]] = relationship(
        "DocVersion",
        back_populates="doc",
        cascade="all, delete-orphan",
        order_by="DocVersion.version.desc()",
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="doc",
        cascade="all, delete-orphan",
    )

    # Indexes
    __table_args__ = (
        Index("ix_docs_doc_type_status", "doc_type", "status"),
        Index("ix_docs_department_status", "department", "status"),
        Index("ix_docs_tags", "tags", postgresql_using="gin"),
        Index("ix_docs_acl_groups", "acl_groups", postgresql_using="gin"),
        Index("ix_docs_metadata", "metadata", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<Doc(id={self.id}, title={self.title[:50]}..., type={self.doc_type})>"


class DocVersion(Base, TimestampMixin):
    """
    文件版本模型。

    追蹤文件版本，用於審計追蹤和回滾功能。
    """

    __tablename__ = "doc_versions"

    # Primary key
    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    # Foreign key to parent doc
    doc_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("docs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Version info
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="SHA-256 hash of this version's content",
    )

    # Storage reference
    canonical_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="MinIO path to this version's canonical JSON",
    )

    # Change tracking
    change_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    changed_sections: Mapped[list[str]] = mapped_column(
        ARRAY(String(100)),
        default=list,
        comment="Sections that changed in this version",
    )

    # Status
    status: Mapped[str] = mapped_column(
        String(20),
        default="active",
        index=True,
        comment="Version status (active, deprecated, deleted)",
    )
    is_current: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="Whether this is the current active version",
    )

    # Audit
    created_by: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="User who created this version",
    )
    approved_by: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="User who approved this version",
    )
    approved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationship
    doc: Mapped["Doc"] = relationship("Doc", back_populates="versions")

    # Constraints
    __table_args__ = (
        UniqueConstraint("doc_id", "version", name="uq_doc_versions_doc_version"),
        Index("ix_doc_versions_doc_id_version", "doc_id", "version"),
        Index("ix_doc_versions_content_hash", "content_hash"),
    )

    def __repr__(self) -> str:
        return f"<DocVersion(id={self.id}, doc_id={self.doc_id}, version={self.version})>"


class Asset(Base, TimestampMixin):
    """
    文件相關檔案的資產模型（圖片、PDF 等）。

    追蹤儲存在 MinIO 中的資產檔案。
    """

    __tablename__ = "assets"

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

    # Asset info
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # Storage
    minio_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="MinIO object path",
    )
    minio_bucket: Mapped[str] = mapped_column(
        String(100),
        default="assets",
        nullable=False,
    )

    # Metadata
    alt_text: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    caption: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Note: Python attr 'extra_metadata' maps to DB column 'metadata'
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",  # DB column name
        JSONB,
        default=dict,
    )

    # Indexes
    __table_args__ = (
        Index("ix_assets_doc_id_version", "doc_id", "doc_version"),
        Index("ix_assets_content_hash", "content_hash"),
    )

    def __repr__(self) -> str:
        return f"<Asset(id={self.id}, filename={self.filename})>"
