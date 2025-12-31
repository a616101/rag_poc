"""
語意快取管理 API 路由。

提供以下功能：
- 查看快取統計資訊
- 清除所有快取
- 依條件清除快取（檔案名稱）

所有端點都在 /api/v1/cache 路徑下。
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from chatbot_rag.core.config import settings
from chatbot_rag.services.semantic_cache_service import semantic_cache_service


router = APIRouter(prefix="/api/v1/cache", tags=["Cache"])


# ==============================================================================
# Pydantic Models
# ==============================================================================


class CacheStatsResponse(BaseModel):
    """快取統計資訊"""

    enabled: bool
    collection_name: str
    total_entries: int
    similarity_threshold: float
    ttl_seconds: int


class ClearCacheResponse(BaseModel):
    """清除快取回應"""

    success: bool
    message: str
    deleted_count: int


class InvalidateCacheRequest(BaseModel):
    """依條件清除快取請求"""

    filenames: List[str]


class InvalidateCacheResponse(BaseModel):
    """依條件清除快取回應"""

    success: bool
    message: str
    invalidated_count: int
    filenames: List[str]


# ==============================================================================
# 快取統計
# ==============================================================================


@router.get("/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """
    取得快取統計資訊。

    Returns:
        快取統計資訊，包含總數、設定等
    """
    if not settings.semantic_cache_enabled:
        return CacheStatsResponse(
            enabled=False,
            collection_name=settings.semantic_cache_collection_name,
            total_entries=0,
            similarity_threshold=settings.semantic_cache_similarity_threshold,
            ttl_seconds=settings.semantic_cache_ttl_seconds,
        )

    try:
        stats = semantic_cache_service.get_stats()
        return CacheStatsResponse(
            enabled=True,
            collection_name=stats.get(
                "collection_name", settings.semantic_cache_collection_name
            ),
            total_entries=stats.get("total_entries", 0),
            similarity_threshold=stats.get(
                "similarity_threshold", settings.semantic_cache_similarity_threshold
            ),
            ttl_seconds=stats.get("ttl_seconds", settings.semantic_cache_ttl_seconds),
        )
    except Exception as e:
        logger.error(f"[CacheAPI] 取得統計資訊失敗: {e}")
        raise HTTPException(status_code=500, detail=f"取得快取統計失敗: {e}")


# ==============================================================================
# 快取清除
# ==============================================================================


@router.post("/clear", response_model=ClearCacheResponse)
async def clear_cache():
    """
    清除所有快取。

    警告：此操作會刪除所有快取資料，無法復原。

    Returns:
        清除結果
    """
    if not settings.semantic_cache_enabled:
        return ClearCacheResponse(
            success=True,
            message="語意快取已停用，無需清除",
            deleted_count=0,
        )

    try:
        logger.info("[CacheAPI] 執行清除所有快取")
        deleted_count = semantic_cache_service.clear_all()
        return ClearCacheResponse(
            success=True,
            message=f"已清除 {deleted_count} 筆快取",
            deleted_count=deleted_count,
        )
    except Exception as e:
        logger.error(f"[CacheAPI] 清除快取失敗: {e}")
        raise HTTPException(status_code=500, detail=f"清除快取失敗: {e}")


@router.post("/invalidate", response_model=InvalidateCacheResponse)
async def invalidate_cache(request: InvalidateCacheRequest):
    """
    依檔案名稱清除快取。

    當文件更新時，可使用此端點清除相關快取，確保不會返回過時資料。

    Args:
        request: 包含要清除的檔案名稱列表

    Returns:
        清除結果
    """
    if not settings.semantic_cache_enabled:
        return InvalidateCacheResponse(
            success=True,
            message="語意快取已停用，無需清除",
            invalidated_count=0,
            filenames=request.filenames,
        )

    if not request.filenames:
        raise HTTPException(status_code=400, detail="至少需要提供一個檔案名稱")

    try:
        logger.info(f"[CacheAPI] 依檔案清除快取: {request.filenames}")
        invalidated_count = semantic_cache_service.invalidate_by_filenames(
            request.filenames
        )
        return InvalidateCacheResponse(
            success=True,
            message=f"已清除 {invalidated_count} 筆與指定檔案相關的快取",
            invalidated_count=invalidated_count,
            filenames=request.filenames,
        )
    except Exception as e:
        logger.error(f"[CacheAPI] 依檔案清除快取失敗: {e}")
        raise HTTPException(status_code=500, detail=f"清除快取失敗: {e}")


# ==============================================================================
# 快取狀態
# ==============================================================================


@router.get("/status")
async def get_cache_status():
    """
    檢查語意快取服務狀態。

    Returns:
        服務狀態
    """
    try:
        if not settings.semantic_cache_enabled:
            return {
                "status": "disabled",
                "message": "語意快取已停用",
            }

        # 嘗試取得統計資訊以確認服務正常
        stats = semantic_cache_service.get_stats()
        return {
            "status": "healthy",
            "collection_exists": stats.get("collection_exists", False),
            "total_entries": stats.get("total_entries", 0),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }
