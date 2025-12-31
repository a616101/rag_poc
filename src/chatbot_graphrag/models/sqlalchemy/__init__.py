"""
GraphRAG SQLAlchemy 模型

用於 PostgreSQL 資料庫儲存的所有 SQLAlchemy 模型。

模型分類：
- base: 基礎配置和 Mixin
- doc: 文件和版本追蹤
- chunk: Chunk 和血統追蹤
- acl: 存取控制和租戶管理
"""

from chatbot_graphrag.models.sqlalchemy.base import (
    Base,
    TimestampMixin,
    SoftDeleteMixin,
    model_to_dict,
)

from chatbot_graphrag.models.sqlalchemy.doc import (
    Doc,
    DocVersion,
    Asset,
)

from chatbot_graphrag.models.sqlalchemy.chunk import (
    Chunk,
    TraceLink,
    IngestJob,
)

from chatbot_graphrag.models.sqlalchemy.acl import (
    Tenant,
    TenantScope,
    ACLGroup,
    ACLEntry,
    SemanticCache,
    CommunityReport,
    IndexVersion,
)

__all__ = [
    # ==================== 基礎 ====================
    "Base",
    "TimestampMixin",
    "SoftDeleteMixin",
    "model_to_dict",
    # ==================== 文件 ====================
    "Doc",
    "DocVersion",
    "Asset",
    # ==================== Chunk ====================
    "Chunk",
    "TraceLink",
    "IngestJob",
    # ==================== 存取控制 ====================
    "Tenant",
    "TenantScope",
    "ACLGroup",
    "ACLEntry",
    "SemanticCache",
    "CommunityReport",
    "IndexVersion",
]
