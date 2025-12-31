"""
GraphRAG Repository 層

提供資料存取層，封裝 SQLAlchemy ORM 操作。

Repositories:
    - DocumentRepository: 文件 CRUD 操作
    - JobRepository: 攝取工作持久化
"""

from chatbot_graphrag.db.repositories.document_repo import DocumentRepository
from chatbot_graphrag.db.repositories.job_repo import JobRepository

__all__ = ["DocumentRepository", "JobRepository"]
