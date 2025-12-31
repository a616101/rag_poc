"""
GraphRAG 快取管理 API

Phase 6: 提供快取版本管理與失效功能。

端點總覽：
- POST /api/v1/admin/cache/invalidate - 依版本失效快取
- GET  /api/v1/admin/cache/stats      - 取得快取統計資訊
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from chatbot_graphrag.core.config import settings

logger = logging.getLogger(__name__)

# ============================================================
# API 路由器配置
# ============================================================

router = APIRouter(prefix="/api/v1/admin", tags=["cache-admin"])
"""
GraphRAG 快取管理 API 路由器

前綴：/api/v1/admin
標籤：cache-admin

快取版本管理相關端點都在此路由器下管理。
"""


# ============================================================
# 請求/響應模型定義
# ============================================================


class CacheInvalidateRequest(BaseModel):
    """
    快取失效請求模型。

    Phase 6: 支援按版本欄位選擇性失效快取。

    Attributes:
        index_version: 索引版本
            - 若指定，將失效不匹配此版本的快取項目
            - 若為 None，不按索引版本過濾
        prompt_version: 提示詞版本
            - 若指定，將失效不匹配此版本的快取項目
            - 若為 None，不按提示詞版本過濾
        tenant_id: 租戶識別碼
            - 若指定，僅處理該租戶的快取
            - 若為 None，處理所有租戶的快取
        invalidate_all: 是否失效所有快取
            - True: 忽略版本欄位，清空所有快取
            - False: 按版本欄位選擇性失效（預設）

    Example:
        ```json
        {
            "index_version": "v2.0",
            "prompt_version": "production",
            "tenant_id": "hospital_a"
        }
        ```
    """

    index_version: Optional[str] = Field(
        None,
        description="索引版本。失效不匹配此版本的快取項目",
        examples=["v1.0", "v2.0", "2024-01-15"],
    )
    prompt_version: Optional[str] = Field(
        None,
        description="提示詞版本（對應 Langfuse prompt label）",
        examples=["production", "staging", "v1.2"],
    )
    tenant_id: Optional[str] = Field(
        None,
        description="租戶識別碼。若指定，僅處理該租戶的快取",
        examples=["default", "hospital_a", "clinic_b"],
    )
    invalidate_all: bool = Field(
        False,
        description="是否失效所有快取。True=清空全部，False=按版本選擇性失效",
    )


class CacheInvalidateResponse(BaseModel):
    """
    快取失效響應模型。

    Attributes:
        success: 操作是否成功
        invalidated_count: 失效的快取項目數量
        message: 操作訊息
        index_version: 目前使用的索引版本
        prompt_version: 目前使用的提示詞版本
    """

    success: bool = Field(
        ...,
        description="操作是否成功",
    )
    invalidated_count: int = Field(
        ...,
        description="失效的快取項目數量",
    )
    message: str = Field(
        ...,
        description="操作訊息",
    )
    index_version: str = Field(
        ...,
        description="目前系統使用的索引版本",
    )
    prompt_version: str = Field(
        ...,
        description="目前系統使用的提示詞版本",
    )


class CacheStatsResponse(BaseModel):
    """
    快取統計資訊響應模型。

    Attributes:
        enabled: 快取是否啟用
        total_entries: 快取總項目數（估計值）
        index_version: 目前索引版本
        prompt_version: 目前提示詞版本
        collection_name: 快取集合名稱
    """

    enabled: bool = Field(
        ...,
        description="快取功能是否啟用",
    )
    total_entries: int = Field(
        ...,
        description="快取總項目數（估計值）",
    )
    index_version: str = Field(
        ...,
        description="目前系統使用的索引版本",
    )
    prompt_version: str = Field(
        ...,
        description="目前系統使用的提示詞版本",
    )
    collection_name: str = Field(
        ...,
        description="快取集合名稱",
    )


# ============================================================
# API 端點定義
# ============================================================


@router.post("/cache/invalidate", response_model=CacheInvalidateResponse)
async def invalidate_cache(
    request: CacheInvalidateRequest,
) -> CacheInvalidateResponse:
    """
    依版本失效快取端點。

    Phase 6: 支援按索引版本、提示詞版本選擇性失效快取。
    當系統升級（重新索引或更新提示詞）時，使用此端點清除舊版快取。

    使用場景：
    1. 重新索引文檔後，失效舊索引版本的快取
    2. 更新 Langfuse 提示詞後，失效舊提示詞版本的快取
    3. 清空特定租戶的所有快取

    Args:
        request (CacheInvalidateRequest): 失效請求
            - index_version: 保留此版本，失效其他版本
            - prompt_version: 保留此版本，失效其他版本
            - invalidate_all: True 則清空所有快取

    Returns:
        CacheInvalidateResponse: 失效結果
            - invalidated_count: 失效的項目數量
            - success: 操作是否成功

    Raises:
        HTTPException:
            - 500: 快取操作發生錯誤

    Example:
        **失效舊索引版本的快取**:
        ```bash
        curl -X POST "http://localhost:18000/api/v1/admin/cache/invalidate" \\
             -H "Content-Type: application/json" \\
             -d '{
               "index_version": "v2.0"
             }'
        ```

        **失效所有快取**:
        ```bash
        curl -X POST "http://localhost:18000/api/v1/admin/cache/invalidate" \\
             -H "Content-Type: application/json" \\
             -d '{
               "invalidate_all": true
             }'
        ```

        **響應範例**:
        ```json
        {
          "success": true,
          "invalidated_count": 150,
          "message": "已失效 150 個舊版本快取項目",
          "index_version": "v2.0",
          "prompt_version": "production"
        }
        ```

    Note:
        - 此操作是不可逆的，失效的快取項目會被刪除
        - 失效後，後續查詢會重新計算並快取結果
        - 建議在低流量時段執行大規模失效操作
    """
    logger.info(
        f"快取失效請求: index_version={request.index_version}, "
        f"prompt_version={request.prompt_version}, "
        f"tenant_id={request.tenant_id}, "
        f"invalidate_all={request.invalidate_all}"
    )

    try:
        from chatbot_graphrag.services.vector import qdrant_service

        current_index_version = settings.index_version
        current_prompt_version = settings.langfuse_prompt_label or "default"

        if request.invalidate_all:
            # 清空所有快取
            invalidated = await qdrant_service.cache_invalidate_by_version(
                tenant_id=request.tenant_id,
            )
            message = f"已清空所有快取（{invalidated} 個項目）"

        else:
            # 按版本選擇性失效
            # 使用請求中的版本，或使用目前系統版本
            target_index = request.index_version or current_index_version
            target_prompt = request.prompt_version or current_prompt_version

            invalidated = await qdrant_service.cache_invalidate_by_version(
                index_version=target_index,
                prompt_version=target_prompt,
                tenant_id=request.tenant_id,
            )
            message = f"已失效 {invalidated} 個舊版本快取項目"

        logger.info(f"快取失效完成: {message}")

        return CacheInvalidateResponse(
            success=True,
            invalidated_count=invalidated,
            message=message,
            index_version=current_index_version,
            prompt_version=current_prompt_version,
        )

    except Exception as e:
        logger.error(f"快取失效錯誤: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats() -> CacheStatsResponse:
    """
    取得快取統計資訊端點。

    返回目前快取系統的基本統計資訊和版本資訊。

    Returns:
        CacheStatsResponse: 快取統計
            - enabled: 快取是否啟用
            - total_entries: 快取總項目數
            - index_version: 目前索引版本
            - prompt_version: 目前提示詞版本

    Raises:
        HTTPException:
            - 500: 查詢過程發生錯誤

    Example:
        **查詢快取統計**:
        ```bash
        curl "http://localhost:18000/api/v1/admin/cache/stats"
        ```

        **響應範例**:
        ```json
        {
          "enabled": true,
          "total_entries": 1500,
          "index_version": "v2.0",
          "prompt_version": "production",
          "collection_name": "semantic_cache"
        }
        ```
    """
    try:
        from chatbot_graphrag.services.vector import qdrant_service

        # 取得快取集合統計
        try:
            count = await qdrant_service.get_cache_count()
        except Exception:
            count = 0

        return CacheStatsResponse(
            enabled=settings.semantic_cache_enabled,
            total_entries=count,
            index_version=settings.index_version,
            prompt_version=settings.langfuse_prompt_label or "default",
            collection_name=settings.qdrant_collection_cache,
        )

    except Exception as e:
        logger.error(f"取得快取統計錯誤: {e}")
        raise HTTPException(status_code=500, detail=str(e))
