"""Initial schema - 13 tables for GraphRAG

Revision ID: 001
Revises:
Create Date: 2025-12-31

Tables:
- docs, doc_versions, assets (Document management)
- chunks, trace_links, ingest_jobs (Chunk and lineage)
- tenants, tenant_scopes, acl_groups, acl_entries (Access control)
- semantic_cache, community_reports, index_versions (Cache and indexing)
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """建立所有 GraphRAG 資料表"""

    # ==================== docs ====================
    op.create_table(
        "docs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("doc_type", sa.String(50), nullable=False, index=True),
        sa.Column("pipeline_type", sa.String(20), nullable=False, server_default="curated"),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("language", sa.String(10), server_default="zh-TW"),
        sa.Column("source_url", sa.String(2000), nullable=True),
        sa.Column("source_path", sa.String(1000), nullable=True),
        sa.Column("department", sa.String(100), nullable=True, index=True),
        sa.Column("tags", postgresql.ARRAY(sa.String(100)), nullable=False, server_default="{}"),
        sa.Column("current_version", sa.Integer, server_default="1"),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("canonical_path", sa.String(500), nullable=True),
        sa.Column("asset_paths", postgresql.ARRAY(sa.String(500)), server_default="{}"),
        sa.Column("status", sa.String(20), server_default="active", index=True),
        sa.Column("is_published", sa.Boolean, server_default="true"),
        sa.Column("effective_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expiry_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("acl_groups", postgresql.ARRAY(sa.String(100)), server_default="{public}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_docs_doc_type_status", "docs", ["doc_type", "status"])
    op.create_index("ix_docs_department_status", "docs", ["department", "status"])
    op.create_index("ix_docs_tags", "docs", ["tags"], postgresql_using="gin")
    op.create_index("ix_docs_acl_groups", "docs", ["acl_groups"], postgresql_using="gin")
    op.create_index("ix_docs_metadata", "docs", ["metadata"], postgresql_using="gin")

    # ==================== doc_versions ====================
    op.create_table(
        "doc_versions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("doc_id", sa.String(64), sa.ForeignKey("docs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("canonical_path", sa.String(500), nullable=False),
        sa.Column("change_summary", sa.Text, nullable=True),
        sa.Column("changed_sections", postgresql.ARRAY(sa.String(100)), server_default="{}"),
        sa.Column("status", sa.String(20), server_default="active", index=True),
        sa.Column("is_current", sa.Boolean, server_default="false"),
        sa.Column("created_by", sa.String(100), nullable=True),
        sa.Column("approved_by", sa.String(100), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_unique_constraint("uq_doc_versions_doc_version", "doc_versions", ["doc_id", "version"])
    op.create_index("ix_doc_versions_doc_id_version", "doc_versions", ["doc_id", "version"])
    op.create_index("ix_doc_versions_content_hash", "doc_versions", ["content_hash"])

    # ==================== assets ====================
    op.create_table(
        "assets",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("doc_id", sa.String(64), sa.ForeignKey("docs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("doc_version", sa.Integer, server_default="1"),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("content_type", sa.String(100), nullable=False),
        sa.Column("size_bytes", sa.Integer, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("minio_path", sa.String(500), nullable=False),
        sa.Column("minio_bucket", sa.String(100), server_default="assets", nullable=False),
        sa.Column("alt_text", sa.String(500), nullable=True),
        sa.Column("caption", sa.Text, nullable=True),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_assets_doc_id_version", "assets", ["doc_id", "doc_version"])
    op.create_index("ix_assets_content_hash", "assets", ["content_hash"])

    # ==================== chunks ====================
    op.create_table(
        "chunks",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("doc_id", sa.String(64), sa.ForeignKey("docs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("doc_version", sa.Integer, server_default="1"),
        sa.Column("chunk_type", sa.String(50), nullable=False, index=True),
        sa.Column("position_in_doc", sa.Integer, nullable=False),
        sa.Column("section_title", sa.String(500), nullable=True),
        sa.Column("page_number", sa.Integer, nullable=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False, index=True),
        sa.Column("contextual_content", sa.Text, nullable=True),
        sa.Column("token_count", sa.Integer, server_default="0"),
        sa.Column("char_count", sa.Integer, server_default="0"),
        sa.Column("parent_chunk_id", sa.String(64), sa.ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True),
        sa.Column("sibling_chunk_ids", postgresql.ARRAY(sa.String(64)), server_default="{}"),
        sa.Column("qdrant_point_id", sa.String(64), nullable=True, index=True),
        sa.Column("opensearch_doc_id", sa.String(64), nullable=True),
        sa.Column("entity_ids", postgresql.ARRAY(sa.String(64)), server_default="{}"),
        sa.Column("relation_ids", postgresql.ARRAY(sa.String(64)), server_default="{}"),
        sa.Column("community_ids", postgresql.ARRAY(sa.String(64)), server_default="{}"),
        sa.Column("nebula_chunk_vid", sa.String(100), nullable=True),
        sa.Column("status", sa.String(20), server_default="active", index=True),
        sa.Column("is_indexed", sa.Boolean, server_default="false"),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("language", sa.String(10), server_default="zh-TW"),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_chunks_doc_id_position", "chunks", ["doc_id", "position_in_doc"])
    op.create_index("ix_chunks_doc_id_version", "chunks", ["doc_id", "doc_version"])
    op.create_index("ix_chunks_chunk_type_status", "chunks", ["chunk_type", "status"])
    op.create_index("ix_chunks_entity_ids", "chunks", ["entity_ids"], postgresql_using="gin")
    op.create_index("ix_chunks_community_ids", "chunks", ["community_ids"], postgresql_using="gin")
    op.create_index("ix_chunks_metadata", "chunks", ["metadata"], postgresql_using="gin")

    # ==================== trace_links ====================
    op.create_table(
        "trace_links",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("chunk_id", sa.String(64), sa.ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("doc_id", sa.String(64), sa.ForeignKey("docs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("doc_version", sa.Integer, server_default="1"),
        sa.Column("link_type", sa.String(50), nullable=False, index=True),
        sa.Column("entity_id", sa.String(64), nullable=True, index=True),
        sa.Column("relation_id", sa.String(64), nullable=True, index=True),
        sa.Column("community_id", sa.String(64), nullable=True, index=True),
        sa.Column("event_id", sa.String(64), nullable=True, index=True),
        sa.Column("extraction_method", sa.String(50), server_default="llm"),
        sa.Column("extraction_model", sa.String(100), nullable=True),
        sa.Column("confidence", sa.Float, server_default="1.0"),
        sa.Column("start_offset", sa.Integer, nullable=True),
        sa.Column("end_offset", sa.Integer, nullable=True),
        sa.Column("matched_text", sa.Text, nullable=True),
        sa.Column("is_verified", sa.Boolean, server_default="false"),
        sa.Column("verified_by", sa.String(100), nullable=True),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_trace_links_chunk_link_type", "trace_links", ["chunk_id", "link_type"])

    # ==================== ingest_jobs ====================
    op.create_table(
        "ingest_jobs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("pipeline_type", sa.String(20), nullable=False, index=True),
        sa.Column("process_mode", sa.String(20), server_default="update"),
        sa.Column("source_path", sa.String(1000), nullable=True),
        sa.Column("source_urls", postgresql.ARRAY(sa.String(2000)), server_default="{}"),
        sa.Column("collection_name", sa.String(100), nullable=False),
        sa.Column("config", postgresql.JSONB, server_default="{}"),
        sa.Column("status", sa.String(20), server_default="pending", index=True),
        sa.Column("progress", sa.Float, server_default="0.0"),
        sa.Column("total_documents", sa.Integer, server_default="0"),
        sa.Column("processed_documents", sa.Integer, server_default="0"),
        sa.Column("failed_documents", sa.Integer, server_default="0"),
        sa.Column("skipped_documents", sa.Integer, server_default="0"),
        sa.Column("chunks_created", sa.Integer, server_default="0"),
        sa.Column("entities_extracted", sa.Integer, server_default="0"),
        sa.Column("relations_extracted", sa.Integer, server_default="0"),
        sa.Column("communities_detected", sa.Integer, server_default="0"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("document_errors", postgresql.JSONB, server_default="[]"),
        sa.Column("created_by", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_ingest_jobs_status_created", "ingest_jobs", ["status", "created_at"])
    op.create_index("ix_ingest_jobs_pipeline_status", "ingest_jobs", ["pipeline_type", "status"])

    # ==================== tenants ====================
    op.create_table(
        "tenants",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("settings", postgresql.JSONB, server_default="{}"),
        sa.Column("features", postgresql.ARRAY(sa.String(100)), server_default="{}"),
        sa.Column("max_documents", sa.Integer, server_default="10000"),
        sa.Column("max_storage_mb", sa.Integer, server_default="10240"),
        sa.Column("max_queries_per_day", sa.Integer, server_default="10000"),
        sa.Column("is_active", sa.Boolean, server_default="true", index=True),
        sa.Column("suspended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("suspension_reason", sa.Text, nullable=True),
        sa.Column("created_by", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # ==================== tenant_scopes ====================
    op.create_table(
        "tenant_scopes",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(64), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("scope_type", sa.String(50), nullable=False, index=True),
        sa.Column("scope_value", sa.String(255), nullable=False),
        sa.Column("access_level", sa.String(20), server_default="read"),
        sa.Column("is_active", sa.Boolean, server_default="true"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_unique_constraint("uq_tenant_scopes_tenant_scope", "tenant_scopes", ["tenant_id", "scope_type", "scope_value"])
    op.create_index("ix_tenant_scopes_scope_type_value", "tenant_scopes", ["scope_type", "scope_value"])

    # ==================== acl_groups ====================
    op.create_table(
        "acl_groups",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("parent_group_id", sa.String(64), sa.ForeignKey("acl_groups.id", ondelete="SET NULL"), nullable=True),
        sa.Column("level", sa.Integer, server_default="0"),
        sa.Column("permissions", postgresql.ARRAY(sa.String(100)), server_default="{}"),
        sa.Column("is_active", sa.Boolean, server_default="true", index=True),
        sa.Column("is_system", sa.Boolean, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_acl_groups_parent", "acl_groups", ["parent_group_id"])

    # ==================== acl_entries ====================
    op.create_table(
        "acl_entries",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(100), nullable=False, index=True),
        sa.Column("group_id", sa.String(64), sa.ForeignKey("acl_groups.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("tenant_id", sa.String(64), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=True, index=True),
        sa.Column("role", sa.String(50), server_default="member"),
        sa.Column("extra_permissions", postgresql.ARRAY(sa.String(100)), server_default="{}"),
        sa.Column("denied_permissions", postgresql.ARRAY(sa.String(100)), server_default="{}"),
        sa.Column("is_active", sa.Boolean, server_default="true"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("granted_by", sa.String(100), nullable=True),
        sa.Column("granted_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_unique_constraint("uq_acl_entries_user_group_tenant", "acl_entries", ["user_id", "group_id", "tenant_id"])
    op.create_index("ix_acl_entries_user_tenant", "acl_entries", ["user_id", "tenant_id"])

    # ==================== semantic_cache ====================
    op.create_table(
        "semantic_cache",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column("query_hash", sa.String(64), nullable=False, index=True),
        sa.Column("normalized_query", sa.Text, nullable=False),
        sa.Column("response", sa.Text, nullable=False),
        sa.Column("response_metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("conversation_id", sa.String(64), nullable=True, index=True),
        sa.Column("tenant_id", sa.String(64), nullable=True, index=True),
        sa.Column("qdrant_point_id", sa.String(64), nullable=True),
        sa.Column("hit_count", sa.Integer, server_default="0"),
        sa.Column("last_hit_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True, index=True),
        sa.Column("is_valid", sa.Boolean, server_default="true"),
        sa.Column("invalidated_reason", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_semantic_cache_query_hash_tenant", "semantic_cache", ["query_hash", "tenant_id"])
    op.create_index("ix_semantic_cache_valid_expires", "semantic_cache", ["is_valid", "expires_at"])

    # ==================== community_reports ====================
    op.create_table(
        "community_reports",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("community_id", sa.String(64), nullable=False, index=True),
        sa.Column("level", sa.Integer, nullable=False, index=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("summary", sa.Text, nullable=False),
        sa.Column("key_entities", postgresql.ARRAY(sa.String(255)), server_default="{}"),
        sa.Column("key_relations", postgresql.ARRAY(sa.String(500)), server_default="{}"),
        sa.Column("themes", postgresql.ARRAY(sa.String(255)), server_default="{}"),
        sa.Column("importance_score", sa.Float, server_default="0.0"),
        sa.Column("entity_count", sa.Integer, server_default="0"),
        sa.Column("edge_count", sa.Integer, server_default="0"),
        sa.Column("token_count", sa.Integer, server_default="0"),
        sa.Column("qdrant_point_id", sa.String(64), nullable=True),
        sa.Column("model_used", sa.String(100), nullable=False),
        sa.Column("generation_time_ms", sa.Float, server_default="0.0"),
        sa.Column("version", sa.Integer, server_default="1"),
        sa.Column("is_current", sa.Boolean, server_default="true", index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_community_reports_community_level", "community_reports", ["community_id", "level"])
    op.create_index("ix_community_reports_level_current", "community_reports", ["level", "is_current"])
    op.create_unique_constraint("uq_community_reports_community_version", "community_reports", ["community_id", "version"])

    # ==================== index_versions ====================
    op.create_table(
        "index_versions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("index_type", sa.String(50), nullable=False, index=True),
        sa.Column("collection_name", sa.String(100), nullable=False),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("schema_hash", sa.String(64), nullable=False),
        sa.Column("doc_count", sa.Integer, server_default="0"),
        sa.Column("chunk_count", sa.Integer, server_default="0"),
        sa.Column("entity_count", sa.Integer, server_default="0"),
        sa.Column("is_active", sa.Boolean, server_default="true", index=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deactivated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("created_by", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_index_versions_type_collection", "index_versions", ["index_type", "collection_name"])
    op.create_unique_constraint("uq_index_versions_type_collection_version", "index_versions", ["index_type", "collection_name", "version"])


def downgrade() -> None:
    """刪除所有 GraphRAG 資料表"""
    # 按相反順序刪除（先刪除有外鍵依賴的表）
    op.drop_table("index_versions")
    op.drop_table("community_reports")
    op.drop_table("semantic_cache")
    op.drop_table("acl_entries")
    op.drop_table("acl_groups")
    op.drop_table("tenant_scopes")
    op.drop_table("tenants")
    op.drop_table("ingest_jobs")
    op.drop_table("trace_links")
    op.drop_table("chunks")
    op.drop_table("assets")
    op.drop_table("doc_versions")
    op.drop_table("docs")
