"""
Alembic 環境配置 - Async SQLAlchemy 支援

使用 asyncpg 驅動執行非同步遷移。
支援從環境變數讀取資料庫配置。
"""
import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from chatbot_graphrag.core.config import settings
from chatbot_graphrag.models.sqlalchemy import Base

# Alembic Config 物件
config = context.config


def get_url() -> str:
    """
    取得資料庫 URL，轉換為 asyncpg 格式。

    從 settings 讀取 PostgreSQL URL，並轉換為 asyncpg 驅動格式。
    """
    url = settings.postgres_url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


# 動態設定資料庫 URL
config.set_main_option("sqlalchemy.url", get_url())

# 設定 logging（從 alembic.ini 讀取）
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 目標 metadata（用於 autogenerate）
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    離線模式：生成 SQL 腳本而不連接資料庫。

    適用於需要預覽或審查 SQL 變更的場景。
    使用 --sql 參數觸發此模式。
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """
    執行遷移的同步回調函數。

    由 async context 內部呼叫。
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    非同步執行遷移。

    建立 async engine 並執行遷移操作。
    使用 NullPool 避免連線池管理問題。
    """
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """
    線上模式：連接資料庫執行遷移。

    這是預設模式，直接連接資料庫執行遷移。
    """
    asyncio.run(run_async_migrations())


# 根據模式執行遷移
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
