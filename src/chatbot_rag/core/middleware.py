"""
請求處理和效能優化中介軟體模組

此模組提供用於請求處理、日誌記錄和效能監控的中介軟體，
包括請求追蹤、處理時間計算和慢請求偵測等功能。
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from chatbot_rag.core.logging import get_logger

# 取得日誌記錄器實例
logger = get_logger()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    請求日誌記錄中介軟體

    此中介軟體負責記錄所有進入的 HTTP 請求和回應，
    包括請求方法、URL、處理時間、狀態碼等資訊。
    為每個請求生成唯一的請求 ID，方便追蹤和除錯。
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        處理請求並記錄詳細資訊

        此方法會為每個請求生成唯一 ID，記錄請求開始和結束的資訊，
        計算處理時間，並在發生錯誤時記錄錯誤詳情。

        Args:
            request: 進入的 HTTP 請求物件
            call_next: 下一個中介軟體或路由處理器

        Returns:
            Response: 應用程式返回的 HTTP 回應

        Raises:
            Exception: 重新拋出處理過程中發生的任何異常
        """
        # 生成唯一的請求 ID，用於追蹤整個請求生命週期
        request_id = str(uuid.uuid4())

        # 將請求 ID 添加到請求狀態中，供後續處理器使用
        request.state.request_id = request_id

        # 啟動計時器，記錄請求開始時間
        start_time = time.perf_counter()

        # 記錄請求開始的日誌
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client": request.client.host if request.client else None,
            },
        )

        # 處理請求
        try:
            # 呼叫下一個處理器處理請求
            response = await call_next(request)

            # 計算請求處理時間
            process_time = time.perf_counter() - start_time

            # 在回應標頭中添加自訂資訊
            response.headers["X-Request-ID"] = request_id  # 請求追蹤 ID
            response.headers["X-Process-Time"] = f"{process_time:.4f}"  # 處理時間（秒）

            # 記錄請求完成的日誌
            logger.info(
                f"Request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "status_code": response.status_code,
                    "process_time": f"{process_time:.4f}s",
                },
            )

            return response

        except Exception as e:
            # 如果發生異常，計算到異常發生時的處理時間
            process_time = time.perf_counter() - start_time

            # 記錄請求失敗的錯誤日誌
            logger.error(
                f"Request failed: {str(e)}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "process_time": f"{process_time:.4f}s",
                    "error": str(e),
                },
            )
            # 重新拋出異常，讓 FastAPI 的異常處理器處理
            raise


class PerformanceMiddleware(BaseHTTPMiddleware):
    """
    效能監控中介軟體

    此中介軟體專門用於監控請求處理效能，
    當偵測到處理時間過長的慢請求時，會記錄警告日誌。
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        監控請求處理效能

        測量請求處理時間，並在偵測到慢請求（處理時間超過 1 秒）時記錄警告。

        Args:
            request: 進入的 HTTP 請求物件
            call_next: 下一個中介軟體或路由處理器

        Returns:
            Response: 應用程式返回的 HTTP 回應
        """
        # 記錄開始時間
        start_time = time.perf_counter()

        # 處理請求
        response = await call_next(request)

        # 計算處理時間
        process_time = time.perf_counter() - start_time

        # 偵測慢請求（處理時間超過 1 秒）
        if process_time > 1.0:
            logger.warning(
                f"Slow request detected",
                extra={
                    "method": request.method,
                    "url": str(request.url),
                    "process_time": f"{process_time:.4f}s",
                },
            )

        return response
