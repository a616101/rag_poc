"""
FastAPI 主應用程序入口點。

本模組是整個 Chatbot RAG 系統的核心入口，負責建立和配置 FastAPI 應用程序。

主要功能：
- 日誌系統初始化：配置 Loguru 日誌記錄器，支援控制台和檔案輸出
- 中間件配置：
  * RequestLoggingMiddleware：記錄每個 HTTP 請求的詳細資訊
  * PerformanceMiddleware：監控請求處理時間，識別性能瓶頸
  * CORSMiddleware：處理跨域資源共享（CORS）
- API 路由註冊：
  * /api/v1/*：基礎 API 路由（健康檢查、測試端點）
  * /api/v1/rag/*：RAG API 路由（文檔向量化、問答查詢）
- 生命週期管理：處理應用程序啟動和關閉事件

使用方式：
    # 使用 uvicorn 啟動服務器
    uvicorn chatbot_rag.main:app --host 0.0.0.0 --port 8000 --reload

    # 或直接執行模組
    python -m chatbot_rag.main

Author: Bruce
Date: 2025
"""

from contextlib import asynccontextmanager  # 用於管理異步上下文（應用程序生命週期）

# FastAPI 核心模組
from fastapi import FastAPI  # FastAPI 主應用程序類別
from fastapi.middleware.cors import CORSMiddleware  # CORS 中間件
from fastapi.responses import ORJSONResponse  # 高性能 JSON 講應類別（使用 ORJSON）

# 應用程序內部模組
from chatbot_rag.api.admin_routes import router as admin_router  # 管理 API 路由（Langfuse 管理）
from chatbot_rag.api.cache_routes import router as cache_router  # 語意快取管理路由
from chatbot_rag.api.file_routes import router as files_router  # 檔案下載路由（/files/*）
from chatbot_rag.api.rag_routes import router as rag_router  # RAG API 路由（問答、向量化）
from chatbot_rag.api.report_routes import router as report_router  # 報表 API 路由
from chatbot_rag.api.routes import router  # 基礎 API 路由（測試、演示）
from chatbot_rag.api.scraper_routes import router as scraper_router  # 網頁爬取 API 路由
from chatbot_rag.core.concurrency import llm_concurrency  # LLM 並發控制
from chatbot_rag.core.telemetry_sampler import initialize_telemetry_sampler  # 遙測採樣初始化
from chatbot_rag.core.config import settings  # 全局配置設定
from chatbot_rag.core.logging import LogConfig, get_logger  # 日誌配置和記錄器
from chatbot_rag.core.middleware import PerformanceMiddleware, RequestLoggingMiddleware  # 自訂中間件

# ============================================================================
# 日誌系統初始化
# ============================================================================
# 從全局配置（settings）讀取日誌設定，建立 LogConfig 物件並執行初始化。
# 此設定會影響整個應用程序的日誌行為，包括輸出格式、目標位置、檔案管理等。
log_config = LogConfig(
    level=settings.log_level,  # 日誌級別（DEBUG/INFO/WARNING/ERROR/CRITICAL）
    log_to_console=settings.log_to_console,  # 是否輸出到控制台（終端機）
    log_to_file=settings.log_to_file,  # 是否輸出到檔案
    log_file_path=settings.log_file_path,  # 日誌檔案路徑（例如：logs/app.log）
    rotation=settings.log_rotation,  # 檔案輪轉設定（例如："500 MB" 或 "1 day"）
    retention=settings.log_retention,  # 檔案保留時間（例如："10 days" 或 "1 week"）
    colorize_file=settings.log_colorize_file,  # 檔案日誌是否使用 ANSI 顏色碼
)
# 執行日誌系統初始化，配置 Loguru 記錄器
log_config.setup()

# 取得全局 logger 實例，用於記錄應用程序運行時的各種事件
logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    應用程序生命週期管理（異步上下文管理器）。

    此函數使用 Python 的異步上下文管理器協議（async context manager）來管理
    FastAPI 應用程序的整個生命週期，包括啟動和關閉階段。

    生命週期階段：
    1. 啟動階段（yield 之前）：
       - 記錄應用程序啟動資訊
       - 初始化資源（數據庫連接、快取等）
       - 執行啟動前的健康檢查

    2. 運行階段（yield 期間）：
       - 應用程序正常運行，處理 HTTP 請求

    3. 關閉階段（yield 之後）：
       - 清理資源（關閉數據庫連接、釋放快取）
       - 記錄關閉資訊
       - 執行優雅關機（graceful shutdown）

    Args:
        app (FastAPI): FastAPI 應用程序實例

    Yields:
        None: 在應用程序運行期間 yield，讓應用程序處理請求

    Note:
        - 啟動事件在服務器開始接受請求之前執行
        - 關閉事件在服務器停止接受請求之後執行
        - 使用異步上下文管理器確保即使發生異常也能正確清理資源

    Example:
        此函數會在以下時機被調用：
        - 啟動：uvicorn chatbot_rag.main:app
        - 關閉：Ctrl+C 或 SIGTERM 信號
    """
    # ========================================================================
    # 啟動階段：應用程序啟動時執行
    # ========================================================================
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Server: {settings.host}:{settings.port}")

    # ------------------------------------------------------------------------
    # LLM 並發控制初始化
    # ------------------------------------------------------------------------
    # 必須在 event loop 中初始化 Semaphore
    llm_concurrency.initialize()
    logger.info("LLM concurrency manager initialized")

    # ------------------------------------------------------------------------
    # 遙測採樣器初始化
    # ------------------------------------------------------------------------
    # 從 settings 讀取採樣率並初始化
    initialize_telemetry_sampler()

    # ------------------------------------------------------------------------
    # Langfuse Prompt Management 初始化
    # ------------------------------------------------------------------------
    # 如果啟用 Langfuse Prompt Management，在應用啟動時自動建立預設 prompts
    # 已存在的 prompts 會被跳過，不會重複建立
    if settings.langfuse_prompt_enabled:
        try:
            from langfuse import get_client
            from chatbot_rag.services.prompt_service import initialize_default_prompts

            langfuse_client = get_client()
            results = initialize_default_prompts(
                langfuse_client,
                default_label=settings.langfuse_prompt_label,
            )
            created_count = sum(1 for v in results.values() if v)
            existing_count = len(results) - created_count
            logger.info(
                f"Langfuse Prompt initialization: {created_count} created, "
                f"{existing_count} already exist"
            )
        except Exception as exc:
            # Prompt 初始化失敗不應阻止應用啟動，僅記錄警告
            logger.warning(f"Failed to initialize Langfuse prompts: {exc}")

    yield  # 應用程序運行期間，處理所有 HTTP 請求

    # ========================================================================
    # 關閉階段：應用程序關閉時執行
    # ========================================================================
    logger.info(f"Shutting down {settings.app_name}")

    # 未來可以在這裡添加：
    # - 關閉數據庫連接
    # - 清理臨時檔案
    # - 儲存快取數據
    # - 等待正在處理的請求完成


def create_app() -> FastAPI:
    """
    建立並配置 FastAPI 應用程序（工廠模式）。

    此函數採用工廠模式（Factory Pattern）建立 FastAPI 應用程序實例，
    並依序執行各項配置步驟，確保應用程序在啟動前完全就緒。

    配置步驟（執行順序）：
    1. 建立 FastAPI 實例
       - 設定應用程序標題、版本、除錯模式
       - 使用 ORJSONResponse 作為預設響應類別（比標準 JSON 快 2-3 倍）
       - 註冊生命週期管理器（lifespan）

    2. 註冊自訂中間件
       - RequestLoggingMiddleware：記錄每個請求的詳細資訊（URL、方法、耗時等）
       - PerformanceMiddleware：監控請求處理時間，警告慢速請求

    3. 配置 CORS 中間件
       - 設定允許的來源域名、HTTP 方法、HTTP 標頭
       - 啟用憑證傳遞（cookies、authorization headers）

    4. 註冊 API 路由
       - 基礎路由（/api/v1/*）：健康檢查、測試端點
       - RAG 路由（/api/v1/rag/*）：文檔向量化、問答查詢

    Returns:
        FastAPI: 完整配置的 FastAPI 應用程序實例

    Note:
        中間件註冊順序至關重要：
        - 先註冊的中間件在最外層（洋蔥模型）
        - 請求處理流程：Request → 外層中間件 → 內層中間件 → 路由處理器
        - 響應處理流程：路由處理器 → 內層中間件 → 外層中間件 → Response

        實際執行順序：
        1. RequestLoggingMiddleware (外層) - 記錄請求開始
        2. PerformanceMiddleware (中層) - 開始計時
        3. CORSMiddleware (內層) - 處理 CORS
        4. 路由處理器 - 執行業務邏輯
        5. CORSMiddleware - 添加 CORS 標頭
        6. PerformanceMiddleware - 檢查耗時
        7. RequestLoggingMiddleware - 記錄請求完成

    Example:
        >>> app = create_app()
        >>> # 使用 uvicorn 啟動
        >>> uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    # ========================================================================
    # 步驟 1：建立 FastAPI 應用程序實例
    # ========================================================================
    app = FastAPI(
        title=settings.app_name,  # API 標題，顯示在 Swagger UI
        version=settings.app_version,  # API 版本號
        debug=settings.debug,  # 除錯模式（開發時為 True，生產時為 False）
        lifespan=lifespan,  # 生命週期管理函數（啟動/關閉事件）
        default_response_class=ORJSONResponse,  # 使用 ORJSON 提升 JSON 序列化性能（2-3倍速度提升）
    )

    # ========================================================================
    # 步驟 2：註冊自訂中間件（洋蔥模型：先註冊的在最外層）
    # ========================================================================
    # Layer 1（最外層）：請求日誌中間件
    # 功能：記錄每個請求的詳細資訊（方法、URL、狀態碼、耗時等）
    # 位置：最外層，確保能記錄完整的請求-響應週期
    app.add_middleware(RequestLoggingMiddleware)

    # Layer 2（中層）：性能監控中間件
    # 功能：監控請求處理時間，當處理時間超過閾值時發出警告
    # 位置：在日誌中間件內層，專注於性能測量
    app.add_middleware(PerformanceMiddleware)

    # ========================================================================
    # 步驟 3：配置 CORS 中間件（跨域資源共享）
    # ========================================================================
    # Layer 3（內層）：CORS 中間件
    # 功能：處理跨域請求，允許前端應用從不同域名訪問 API
    # 配置：控制哪些來源、方法、標頭可以訪問 API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,  # 允許的來源域名（例如：["http://localhost:3000"]）
        allow_credentials=settings.cors_allow_credentials,  # 是否允許攜帶憑證（cookies、authorization headers）
        allow_methods=settings.cors_allow_methods,  # 允許的 HTTP 方法（例如：["GET", "POST", "PUT", "DELETE"]）
        allow_headers=settings.cors_allow_headers,  # 允許的 HTTP 標頭（例如：["Content-Type", "Authorization"]）
    )

    # ========================================================================
    # 步驟 4：註冊 API 路由
    # ========================================================================
    # 註冊基礎 API 路由（健康檢查、測試端點等）
    # 路徑前綴：/api/v1
    # 範例端點：/api/v1/health, /api/v1/test
    app.include_router(router, prefix="/api/v1")

    # 註冊 RAG API 路由（文檔向量化、問答查詢等）
    # 路徑前綴：已在 rag_router 內部定義（/api/v1/rag）
    # 範例端點：/api/v1/rag/vectorize, /api/v1/rag/query
    app.include_router(rag_router)

    # 註冊檔案下載路由（/files/*）
    # 讓專案根目錄下 `files/` 目錄中的檔案可以透過 URL 下載
    app.include_router(files_router)

    # 註冊報表 API 路由（/api/v1/reports/*）
    # 提供 Langfuse Trace 報表生成功能
    app.include_router(report_router)

    # 註冊管理 API 路由（/api/v1/admin/*）
    # 提供 Langfuse Prompts、Datasets 管理功能
    app.include_router(admin_router)

    # 註冊語意快取管理路由（/api/v1/cache/*）
    # 提供快取統計、清除、依條件清除等功能
    app.include_router(cache_router)

    # 註冊網頁爬取 API 路由（/api/v1/scraper/*）
    # 提供網頁爬取、內容轉換、向量化功能
    app.include_router(scraper_router)

    # ========================================================================
    # 步驟 5：定義根端點（應用程序入口）
    # ========================================================================
    @app.get("/")
    async def root():
        """
        根端點（Root Endpoint）- API 入口資訊。

        此端點提供 API 的基本資訊，包括應用程序名稱、版本號和 API 文檔連結。
        通常用於快速檢查 API 是否正常運行，以及獲取 API 的基本元數據。

        HTTP 方法：GET
        路徑：/
        認證：不需要

        Returns:
            dict: 包含以下欄位的 JSON 物件：
                - app (str): 應用程序名稱
                - version (str): 應用程序版本號（語義化版本）
                - docs (str): Swagger UI 文檔的相對路徑

        響應狀態碼：
            200: 成功返回 API 資訊

        Example:
            請求範例：
            ```bash
            curl http://localhost:8000/
            ```

            響應範例：
            ```json
            {
                "app": "Chatbot RAG API",
                "version": "0.1.0",
                "docs": "/docs"
            }
            ```

        Note:
            - 此端點不包含在 /api/v1 前綴下，直接掛載在根路徑
            - 可用於健康檢查（health check）和服務發現
            - Swagger UI 文檔位於 /docs，ReDoc 文檔位於 /redoc
        """
        # 記錄根端點訪問事件（除錯級別）
        logger.debug("Root endpoint accessed")

        # 返回 API 基本資訊
        return {
            "app": settings.app_name,  # 應用程序名稱（來自配置）
            "version": settings.app_version,  # 版本號（來自配置）
            "docs": "/docs",  # Swagger UI 文檔位置
        }

    # 返回完整配置的 FastAPI 應用程序實例
    return app


# ============================================================================
# 建立全局應用程序實例（模組級別）
# ============================================================================
# 調用 create_app() 工廠函數，建立並配置 FastAPI 應用程序實例。
# 此實例在模組載入時建立，成為全局單例（singleton），供 uvicorn 服務器使用。
#
# 使用方式：
#   1. 開發模式（自動重載）：
#      uvicorn chatbot_rag.main:app --host 0.0.0.0 --port 8000 --reload
#
#   2. 生產模式（多工作進程）：
#      uvicorn chatbot_rag.main:app --host 0.0.0.0 --port 8000 --workers 4
#
#   3. 使用 Gunicorn + Uvicorn Worker（生產環境推薦）：
#      gunicorn chatbot_rag.main:app -w 4 -k uvicorn.workers.UvicornWorker
#
# 注意：
#   - 此變數名稱（app）必須與 uvicorn 命令中的應用程序名稱一致
#   - 不要在此處直接執行 uvicorn.run()，應該從命令列啟動
#   - 工廠模式的好處：便於測試、配置管理、依賴注入
app = create_app()
