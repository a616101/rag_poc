"""
GraphRAG 資料庫模組

提供 PostgreSQL 資料庫連線管理和 Repository 層。

使用方式：
    from chatbot_graphrag.db import init_db, close_db, get_async_session

    # 在應用程式啟動時初始化
    await init_db()

    # 使用 session
    async with get_async_session() as session:
        repo = DocumentRepository(session)
        doc = await repo.get_by_id("doc_123")

    # 在應用程式關閉時釋放連線
    await close_db()
"""

from chatbot_graphrag.db.connection import (
    get_async_engine,
    get_async_session,
    init_db,
    close_db,
    is_db_initialized,
)
from chatbot_graphrag.db.repositories import (
    DocumentRepository,
    JobRepository,
)

__all__ = [
    "get_async_engine",
    "get_async_session",
    "init_db",
    "close_db",
    "is_db_initialized",
    "DocumentRepository",
    "JobRepository",
]
