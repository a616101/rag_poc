"""
基礎 API 路由模組。

本模組定義了 Chatbot RAG API 的基礎路由端點，包括：
- 根路徑和健康檢查端點：用於基本的 API 狀態查詢
- 問候端點：演示路徑參數的使用和日誌上下文功能
- 演示端點：包括日誌級別、慢速請求和錯誤處理的測試端點

這些端點主要用於：
1. API 基礎功能測試
2. 日誌系統演示
3. 性能監控中間件測試
4. 錯誤處理機制驗證

所有端點都使用 FastAPI 的 APIRouter 進行組織，並整合了統一的日誌記錄功能。
"""

import asyncio

from fastapi import APIRouter, HTTPException

from chatbot_rag.core.logging import LogContext, get_logger

# 建立路由器實例，用於組織和管理所有的 API 端點
router = APIRouter()

# 取得全局日誌記錄器，用於記錄 API 操作和錯誤
logger = get_logger()


@router.get("/")
async def root():
    """
    API 根端點。

    返回 API 的基本信息和運行狀態。此端點通常用於快速檢查 API 是否正常啟動。

    Returns:
        dict: 包含歡迎訊息和狀態的字典
            - message (str): 歡迎訊息
            - status (str): 運行狀態

    Example:
        ```bash
        curl http://localhost:8000/
        ```

        Response:
        ```json
        {
            "message": "Welcome to Chatbot RAG API",
            "status": "running"
        }
        ```
    """
    # 記錄根端點被訪問的日誌
    logger.info("API root endpoint accessed")

    # 返回基本的歡迎訊息和狀態
    return {
        "message": "Welcome to Chatbot RAG API",
        "status": "running",
    }


@router.get("/health")
async def health_check():
    """
    健康檢查端點。

    用於監控系統是否正常運行，通常被負載均衡器或監控工具調用。
    在生產環境中，此端點應該快速響應，避免執行耗時的檢查。

    Returns:
        dict: 健康狀態
            - status (str): "healthy" 表示系統正常

    Example:
        ```bash
        curl http://localhost:8000/health
        ```

        Response:
        ```json
        {
            "status": "healthy"
        }
        ```

    Note:
        此端點使用 DEBUG 級別記錄，避免過多的日誌輸出。
    """
    # 使用 debug 級別記錄健康檢查，避免過多日誌
    logger.debug("Health check performed")

    # 返回簡單的健康狀態
    return {"status": "healthy"}


@router.get("/hello/{name}")
async def say_hello(name: str):
    """
    個人化問候端點。

    根據提供的名字返回個人化的問候訊息，同時演示日誌上下文的使用。
    此端點展示了如何使用 LogContext 為特定用戶添加上下文信息。

    Args:
        name (str): 要問候的用戶名稱，從 URL 路徑中獲取

    Returns:
        dict: 包含問候訊息的字典
            - message (str): 個人化問候訊息

    Example:
        ```bash
        curl http://localhost:8000/hello/Alice
        ```

        Response:
        ```json
        {
            "message": "Hello Alice"
        }
        ```

    Note:
        使用 LogContext 可以在日誌中自動附加用戶相關的上下文信息，
        便於追蹤特定用戶的操作。
    """
    # 使用日誌上下文記錄用戶資訊
    # LogContext 會自動在日誌中添加 user 欄位
    with LogContext(user=name) as ctx_logger:
        ctx_logger.info(f"Greeting user: {name}")

    # 返回個人化問候訊息
    return {"message": f"Hello {name}"}


@router.get("/demo/log-levels")
async def demo_log_levels():
    """
    日誌級別演示端點。

    演示不同級別的日誌記錄，用於測試日誌系統配置。
    依序記錄 DEBUG、INFO、WARNING、ERROR 級別的訊息。

    Returns:
        dict: 包含訊息和記錄的日誌級別
            - message (str): 說明訊息
            - levels (list[str]): 已記錄的日誌級別列表

    Example:
        ```bash
        curl http://localhost:8000/demo/log-levels
        ```

        Response:
        ```json
        {
            "message": "Logged messages at different levels",
            "levels": ["DEBUG", "INFO", "WARNING", "ERROR"]
        }
        ```

    Note:
        實際看到的日誌訊息取決於系統配置的日誌級別。
        例如，若日誌級別設為 INFO，則 DEBUG 訊息不會顯示。
    """
    # 記錄 DEBUG 級別訊息（通常用於開發除錯）
    logger.debug("This is a DEBUG message")

    # 記錄 INFO 級別訊息（一般資訊性訊息）
    logger.info("This is an INFO message")

    # 記錄 WARNING 級別訊息（警告但不影響運行）
    logger.warning("This is a WARNING message")

    # 記錄 ERROR 級別訊息（錯誤但不中斷程式）
    logger.error("This is an ERROR message")

    # 返回已記錄的日誌級別列表
    return {
        "message": "Logged messages at different levels",
        "levels": ["DEBUG", "INFO", "WARNING", "ERROR"],
    }


@router.get("/demo/slow-request")
async def demo_slow_request():
    """
    慢速請求模擬端點。

    模擬一個耗時的請求，用於測試性能監控中間件。
    該端點會延遲 1.5 秒後才返回響應，觸發慢速請求警告。

    Returns:
        dict: 包含說明訊息的字典
            - message (str): 說明此請求耗時超過 1 秒

    Example:
        ```bash
        curl http://localhost:8000/demo/slow-request
        ```

        Response (延遲 1.5 秒後):
        ```json
        {
            "message": "This request took more than 1 second"
        }
        ```

    Note:
        當請求耗時超過閾值時，性能監控中間件會記錄警告。
        此端點可用於驗證性能監控、超時設定等功能。
    """
    # 記錄開始模擬慢速請求
    logger.info("Starting slow request simulation")

    # 使用 asyncio.sleep 模擬耗時操作（非阻塞式延遲 1.5 秒）
    # 這會觸發性能監控中間件的慢速請求警告（通常閾值為 1 秒）
    await asyncio.sleep(1.5)

    # 記錄慢速請求完成
    logger.info("Slow request completed")

    # 返回說明訊息
    return {"message": "This request took more than 1 second"}


@router.get("/demo/error")
async def demo_error():
    """
    錯誤處理演示端點。

    故意拋出 HTTP 異常，用於測試錯誤處理和日誌記錄機制。
    此端點用於驗證全局異常處理器是否正確捕獲和記錄錯誤。

    Raises:
        HTTPException: 狀態碼 500 的演示錯誤

    Example:
        ```bash
        curl http://localhost:8000/demo/error
        ```

        Response (HTTP 500):
        ```json
        {
            "detail": "This is a demo error"
        }
        ```

    Note:
        錯誤會被全局異常處理器捕獲並記錄到日誌系統中。
        在實際應用中，應該根據錯誤類型使用適當的 HTTP 狀態碼。
    """
    # 記錄即將拋出錯誤的警告
    logger.warning("About to raise an error for demonstration")

    # 拋出 HTTP 500 錯誤
    # 這將被 FastAPI 的全局異常處理器捕獲
    # 並觸發錯誤日誌記錄和適當的錯誤響應
    raise HTTPException(status_code=500, detail="This is a demo error")
