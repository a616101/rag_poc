"""
存取控制 SQLAlchemy 模型

定義多租戶存取控制相關的模型，包含：
- Tenant: 租戶模型
- TenantScope: 租戶範圍（文件/集合存取）
- ACLGroup: ACL 群組
- ACLEntry: ACL 條目（使用者-群組-租戶映射）
- SemanticCache: 語意快取
- CommunityReport: 社群報告儲存
- IndexVersion: 索引版本追蹤
"""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    String,
    Text,
    Integer,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from chatbot_graphrag.models.sqlalchemy.base import Base, TimestampMixin


class Tenant(Base, TimestampMixin):
    """
    多租戶隔離的租戶模型。

    每個租戶擁有獨立的資料空間和配置。
    """

    __tablename__ = "tenants"

    # Primary key
    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    # Tenant info
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Configuration
    settings: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        comment="Tenant-specific settings",
    )
    features: Mapped[list[str]] = mapped_column(
        ARRAY(String(100)),
        default=list,
        comment="Enabled features for this tenant",
    )

    # Limits
    max_documents: Mapped[int] = mapped_column(Integer, default=10000)
    max_storage_mb: Mapped[int] = mapped_column(Integer, default=10240)
    max_queries_per_day: Mapped[int] = mapped_column(Integer, default=10000)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    suspended_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    suspension_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Audit
    created_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Relationships
    scopes: Mapped[list["TenantScope"]] = relationship(
        "TenantScope",
        back_populates="tenant",
        cascade="all, delete-orphan",
    )
    acl_entries: Mapped[list["ACLEntry"]] = relationship(
        "ACLEntry",
        back_populates="tenant",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Tenant(id={self.id}, name={self.name})>"


class TenantScope(Base, TimestampMixin):
    """
    文件/集合存取的租戶範圍。

    定義租戶可以存取的文件集合。
    """

    __tablename__ = "tenant_scopes"

    # Primary key
    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    # Foreign key to tenant
    tenant_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Scope definition
    scope_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Scope type: collection, doc_type, department, tag",
    )
    scope_value: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Scope value (collection name, doc type, etc.)",
    )

    # Access level
    access_level: Mapped[str] = mapped_column(
        String(20),
        default="read",
        comment="Access level: read, write, admin",
    )

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationship
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="scopes")

    # Constraints
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "scope_type", "scope_value",
            name="uq_tenant_scopes_tenant_scope",
        ),
        Index("ix_tenant_scopes_scope_type_value", "scope_type", "scope_value"),
    )

    def __repr__(self) -> str:
        return f"<TenantScope(tenant_id={self.tenant_id}, type={self.scope_type}, value={self.scope_value})>"


class ACLGroup(Base, TimestampMixin):
    """
    ACL 群組模型。

    定義存取控制群組（如 'public'、'staff'、'admin'）。
    支援階層式群組結構。
    """

    __tablename__ = "acl_groups"

    # Primary key
    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    # Group info
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Hierarchy
    parent_group_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("acl_groups.id", ondelete="SET NULL"),
        nullable=True,
    )
    level: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="Hierarchy level (0 = root)",
    )

    # Permissions
    permissions: Mapped[list[str]] = mapped_column(
        ARRAY(String(100)),
        default=list,
        comment="Permissions granted to this group",
    )

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    is_system: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="System groups cannot be deleted",
    )

    # Relationships
    parent_group: Mapped[Optional["ACLGroup"]] = relationship(
        "ACLGroup",
        remote_side="ACLGroup.id",
        foreign_keys=[parent_group_id],
    )
    entries: Mapped[list["ACLEntry"]] = relationship(
        "ACLEntry",
        back_populates="group",
        cascade="all, delete-orphan",
    )

    # Indexes
    __table_args__ = (
        Index("ix_acl_groups_parent", "parent_group_id"),
    )

    def __repr__(self) -> str:
        return f"<ACLGroup(id={self.id}, name={self.name})>"


class ACLEntry(Base, TimestampMixin):
    """
    ACL 條目模型。

    將使用者映射到群組和租戶，並授予特定權限。
    """

    __tablename__ = "acl_entries"

    # Primary key
    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    # Subject
    user_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="User ID (external identity)",
    )

    # Group membership
    group_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("acl_groups.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Tenant (optional, for multi-tenant)
    tenant_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Role within group
    role: Mapped[str] = mapped_column(
        String(50),
        default="member",
        comment="Role: member, admin",
    )

    # Additional permissions (override group permissions)
    extra_permissions: Mapped[list[str]] = mapped_column(
        ARRAY(String(100)),
        default=list,
    )
    denied_permissions: Mapped[list[str]] = mapped_column(
        ARRAY(String(100)),
        default=list,
        comment="Explicitly denied permissions",
    )

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Audit
    granted_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default="now()",
    )

    # Relationships
    group: Mapped["ACLGroup"] = relationship("ACLGroup", back_populates="entries")
    tenant: Mapped[Optional["Tenant"]] = relationship("Tenant", back_populates="acl_entries")

    # Constraints
    __table_args__ = (
        UniqueConstraint(
            "user_id", "group_id", "tenant_id",
            name="uq_acl_entries_user_group_tenant",
        ),
        Index("ix_acl_entries_user_tenant", "user_id", "tenant_id"),
    )

    def __repr__(self) -> str:
        return f"<ACLEntry(user_id={self.user_id}, group_id={self.group_id})>"


class SemanticCache(Base, TimestampMixin):
    """
    語意快取條目模型。

    儲存帶有嵌入向量的查詢-回應配對，用於相似度匹配。
    可減少重複查詢的 LLM 呼叫。
    """

    __tablename__ = "semantic_cache"

    # Primary key
    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    # Query
    query: Mapped[str] = mapped_column(Text, nullable=False)
    query_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="SHA-256 hash of normalized query",
    )
    normalized_query: Mapped[str] = mapped_column(Text, nullable=False)

    # Response
    response: Mapped[str] = mapped_column(Text, nullable=False)
    response_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        comment="Response metadata (citations, trace_id, etc.)",
    )

    # Context
    conversation_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )
    tenant_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    # Qdrant reference (for vector similarity)
    qdrant_point_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        comment="Qdrant point ID for similarity lookup",
    )

    # Usage stats
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    last_hit_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Expiry
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    # Status
    is_valid: Mapped[bool] = mapped_column(Boolean, default=True)
    invalidated_reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Indexes
    __table_args__ = (
        Index("ix_semantic_cache_query_hash_tenant", "query_hash", "tenant_id"),
        Index("ix_semantic_cache_valid_expires", "is_valid", "expires_at"),
    )

    def __repr__(self) -> str:
        return f"<SemanticCache(id={self.id}, hits={self.hit_count})>"


class CommunityReport(Base, TimestampMixin):
    """
    社群報告儲存模型。

    儲存 LLM 生成的社群報告，用於 GraphRAG Global/DRIFT 模式。
    每個報告包含社群摘要、關鍵實體和主題。
    """

    __tablename__ = "community_reports"

    # Primary key
    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    # Community reference
    community_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="NebulaGraph community vertex ID",
    )
    level: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
        comment="Community hierarchy level",
    )

    # Report content
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    key_entities: Mapped[list[str]] = mapped_column(
        ARRAY(String(255)),
        default=list,
    )
    key_relations: Mapped[list[str]] = mapped_column(
        ARRAY(String(500)),
        default=list,
    )
    themes: Mapped[list[str]] = mapped_column(
        ARRAY(String(255)),
        default=list,
    )

    # Scoring
    importance_score: Mapped[float] = mapped_column(Float, default=0.0)
    entity_count: Mapped[int] = mapped_column(Integer, default=0)
    edge_count: Mapped[int] = mapped_column(Integer, default=0)

    # Token tracking
    token_count: Mapped[int] = mapped_column(Integer, default=0)

    # Vector store reference
    qdrant_point_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        comment="Qdrant point ID for report embedding",
    )

    # Generation metadata
    model_used: Mapped[str] = mapped_column(String(100), nullable=False)
    generation_time_ms: Mapped[float] = mapped_column(Float, default=0.0)

    # Version tracking
    version: Mapped[int] = mapped_column(Integer, default=1)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    # Indexes
    __table_args__ = (
        Index("ix_community_reports_community_level", "community_id", "level"),
        Index("ix_community_reports_level_current", "level", "is_current"),
        UniqueConstraint(
            "community_id", "version",
            name="uq_community_reports_community_version",
        ),
    )

    def __repr__(self) -> str:
        return f"<CommunityReport(id={self.id}, community={self.community_id}, level={self.level})>"


class IndexVersion(Base, TimestampMixin):
    """
    索引版本追蹤模型。

    追蹤向量儲存和搜尋索引的 schema 和資料版本。
    支援藍綠部署和版本回滾。
    """

    __tablename__ = "index_versions"

    # Primary key
    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    # Index identification
    index_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Index type: qdrant, opensearch, nebula",
    )
    collection_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Collection/index name",
    )

    # Version info
    version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Semantic version string",
    )
    schema_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Hash of schema for compatibility checking",
    )

    # Counts
    doc_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    entity_count: Mapped[int] = mapped_column(Integer, default=0)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    activated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    deactivated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Notes
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Indexes
    __table_args__ = (
        Index("ix_index_versions_type_collection", "index_type", "collection_name"),
        UniqueConstraint(
            "index_type", "collection_name", "version",
            name="uq_index_versions_type_collection_version",
        ),
    )

    def __repr__(self) -> str:
        return f"<IndexVersion(type={self.index_type}, collection={self.collection_name}, version={self.version})>"
