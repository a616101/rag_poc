"""
SQLAlchemy 基礎配置

定義所有 SQLAlchemy 模型的宣告式基礎類別和通用 Mixin。

包含：
- Base: 宣告式基礎類別
- TimestampMixin: created_at/updated_at 時間戳記
- SoftDeleteMixin: 軟刪除支援
- model_to_dict: 模型轉字典工具函數
"""

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """所有模型的 SQLAlchemy 宣告式基礎類別。"""

    type_annotation_map = {
        datetime: DateTime(timezone=True),
    }


class TimestampMixin:
    """
    created_at 和 updated_at 時間戳記的 Mixin。

    自動追蹤記錄的建立和更新時間。
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class SoftDeleteMixin:
    """
    軟刪除支援的 Mixin。

    記錄不會真正從資料庫刪除，而是標記為已刪除。
    """

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )
    is_deleted: Mapped[bool] = mapped_column(
        default=False,
        nullable=False,
    )


def model_to_dict(model: Any) -> dict[str, Any]:
    """將 SQLAlchemy 模型轉換為字典。"""
    return {
        column.name: getattr(model, column.name)
        for column in model.__table__.columns
    }
