"""
Async SQLAlchemy 連線管理

提供 PostgreSQL 資料庫的非同步連線管理，支援：
- 連線池管理
- Context manager 封裝的 session 管理

注意：資料庫 schema 遷移由 Alembic 管理，請使用：
  alembic -c src/chatbot_graphrag/alembic.ini upgrade head
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from chatbot_graphrag.core.config import settings

logger = logging.getLogger(__name__)

# 全域引擎（單例）
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def _get_async_url() -> str:
    """
    將 postgresql:// 轉換為 postgresql+asyncpg://

    SQLAlchemy async 需要使用 asyncpg 驅動，URL 格式需要調整。

    Returns:
        轉換後的 async URL
    """
    url = settings.postgres_url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


async def init_db() -> None:
    """
    初始化資料庫連線池。

    此函數應在應用程式啟動時呼叫一次。
    會建立連線池和 session factory。

    注意：此函數不執行 schema 遷移，遷移由 Alembic 管理。

    Raises:
        Exception: 資料庫連線失敗時
    """
    global _engine, _session_factory

    if _engine is not None:
        logger.debug("PostgreSQL 資料庫連線池已存在")
        return

    async_url = _get_async_url()
    # 隱藏密碼顯示
    display_url = async_url.split("@")[-1] if "@" in async_url else "localhost"
    logger.info(f"正在連線到 PostgreSQL: {display_url}")

    try:
        _engine = create_async_engine(
            async_url,
            pool_size=settings.postgres_pool_size,
            max_overflow=settings.postgres_max_overflow,
            pool_recycle=settings.postgres_pool_recycle,
            pool_pre_ping=True,
            echo=settings.debug,
        )

        _session_factory = async_sessionmaker(
            bind=_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

        # 驗證連線
        async with _engine.begin() as conn:
            await conn.execute(text("SELECT 1"))

        logger.info("PostgreSQL 連線池已初始化")

    except Exception as e:
        logger.error(f"PostgreSQL 連線失敗: {e}")
        _engine = None
        _session_factory = None
        raise


async def close_db() -> None:
    """
    關閉資料庫連線。

    此函數應在應用程式關閉時呼叫，釋放所有連線資源。
    """
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("PostgreSQL 連線池已關閉")
    else:
        logger.debug("PostgreSQL 連線池已關閉或未初始化")


def get_async_engine() -> AsyncEngine:
    """
    取得資料庫引擎。

    Returns:
        AsyncEngine 實例

    Raises:
        RuntimeError: 資料庫未初始化時
    """
    if _engine is None:
        raise RuntimeError("資料庫未初始化，請先呼叫 init_db()")
    return _engine


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    取得資料庫 Session（Context Manager）。

    使用 context manager 封裝，自動處理 commit 和 rollback：
    - 正常結束時自動 commit
    - 發生例外時自動 rollback

    Yields:
        AsyncSession 實例

    Raises:
        RuntimeError: 資料庫未初始化時

    Example:
        async with get_async_session() as session:
            repo = DocumentRepository(session)
            doc = await repo.get_by_id("doc_123")
    """
    if _session_factory is None:
        raise RuntimeError("資料庫未初始化，請先呼叫 init_db()")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def is_db_initialized() -> bool:
    """
    檢查資料庫是否已初始化。

    Returns:
        True 如果資料庫已初始化，否則 False
    """
    return _engine is not None and _session_factory is not None
