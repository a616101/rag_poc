"""
GraphRAG API - 主應用程式進入點

===============================================================================
模組概述 (Module Overview)
===============================================================================
此模組是 GraphRAG 服務的主要進入點，負責：
1. 初始化 FastAPI 應用程式
2. 配置中介軟體（CORS、日誌等）
3. 註冊 API 路由
4. 管理應用程式生命週期（啟動/關閉服務連線）

API 端點 (Endpoints):
- POST /api/v1/rag/vectorize      - 文件攝取與向量化
- POST /api/v1/rag/ask/stream     - 串流問答（Responses API 格式）
- POST /api/v1/rag/ask/stream_chat - 串流問答（Chat API 格式）
- POST /api/v1/rag/ask            - 非串流問答

健康檢查端點 (Health Check):
- GET /health        - 基本健康檢查
- GET /health/ready  - 就緒檢查（驗證所有服務是否可用）
- GET /health/live   - 存活檢查
- GET /health/concurrency - LLM 並行狀態
===============================================================================
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

# 匯入核心設定
from chatbot_graphrag.core.config import settings
# 匯入 API 路由器
from chatbot_graphrag.api import (
    vectorize_router,       # 文件向量化路由
    ask_stream_router,      # 串流問答路由（Responses API）
    ask_stream_chat_router, # 串流問答路由（Chat API）
)

# =============================================================================
# 日誌配置 (Logging Configuration)
# =============================================================================
def setup_logging() -> None:
    """
    配置應用程式日誌系統
    
    此函式負責：
    1. 根據設定檔設定日誌等級
    2. 配置控制台輸出處理器（如果啟用）
    3. 配置檔案輸出處理器（如果啟用）
    4. 降低外部函式庫的日誌等級以減少雜訊
    """
    # 從設定取得日誌等級，預設為 INFO
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # 日誌處理器列表
    handlers = []

    # ----- 控制台日誌處理器 -----
    if settings.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(console_handler)

    # ----- 檔案日誌處理器 -----
    if settings.log_to_file:
        from pathlib import Path

        # 確保日誌目錄存在
        log_path = Path(settings.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(settings.log_file_path)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_handler)

    # 套用日誌配置
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True,  # 強制覆蓋現有配置
    )

    # ----- 降低外部函式庫的日誌等級以減少雜訊 -----
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


# 建立此模組的日誌記錄器
logger = logging.getLogger(__name__)


# =============================================================================
# 應用程式生命週期管理 (Application Lifespan Manager)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    應用程式生命週期管理器
    
    此非同步上下文管理器負責：
    1. 啟動階段（Startup）：初始化所有必要的服務連線
    2. 關閉階段（Shutdown）：優雅地關閉所有服務連線
    
    初始化的服務包括：
    - LLM 並行管理器
    - NebulaGraph 圖資料庫客戶端
    - Qdrant 向量資料庫服務
    - MinIO 物件儲存服務
    - OpenSearch 搜尋服務
    
    Yields:
        None: 在啟動和關閉之間讓出控制權給應用程式
    """
    # =========================================================================
    # 啟動階段 (Startup Phase)
    # =========================================================================
    setup_logging()
    logger.info(f"正在啟動 {settings.app_name} v{settings.app_version}")
    logger.info(f"除錯模式: {settings.debug}")

    # ----- 初始化 LLM 並行管理器 -----
    from chatbot_graphrag.core.concurrency import llm_concurrency

    llm_concurrency.initialize()
    logger.info("LLM 並行管理器初始化完成")

    # ----- 初始化 PostgreSQL 資料庫 -----
    try:
        from chatbot_graphrag.db import init_db

        await init_db()
        logger.info("PostgreSQL 資料庫初始化完成")
    except Exception as e:
        logger.warning(f"PostgreSQL 資料庫初始化失敗: {e}")

    # ----- 初始化非同步服務 -----

    # NebulaGraph 圖資料庫客戶端
    try:
        from chatbot_graphrag.services.graph.nebula_client import nebula_client

        await nebula_client.connect()
        logger.info("NebulaGraph 客戶端初始化完成")
    except Exception as e:
        logger.warning(f"NebulaGraph 客戶端初始化失敗: {e}")

    # Qdrant 向量資料庫服務
    try:
        from chatbot_graphrag.services.vector.qdrant_service import qdrant_service

        await qdrant_service.initialize()
        logger.info("Qdrant 服務初始化完成")
    except Exception as e:
        logger.warning(f"Qdrant 服務初始化失敗: {e}")

    # MinIO 物件儲存服務
    try:
        from chatbot_graphrag.services.storage.minio_service import minio_service

        await minio_service.initialize()
        logger.info("MinIO 服務初始化完成")
    except Exception as e:
        logger.warning(f"MinIO 服務初始化失敗: {e}")

    # OpenSearch 搜尋服務
    try:
        from chatbot_graphrag.services.search.opensearch_service import opensearch_service

        await opensearch_service.initialize()
        logger.info("OpenSearch 服務初始化完成")
    except Exception as e:
        logger.warning(f"OpenSearch 服務初始化失敗: {e}")

    logger.info("應用程式啟動完成")

    # 讓出控制權，應用程式開始處理請求
    yield

    # =========================================================================
    # 關閉階段 (Shutdown Phase)
    # =========================================================================
    logger.info("正在關閉應用程式...")

    # 關閉 PostgreSQL 資料庫連線
    try:
        from chatbot_graphrag.db import close_db

        await close_db()
        logger.info("PostgreSQL 資料庫連線已關閉")
    except Exception as e:
        logger.warning(f"PostgreSQL 資料庫關閉錯誤: {e}")

    # 關閉 NebulaGraph 客戶端
    try:
        from chatbot_graphrag.services.graph.nebula_client import nebula_client

        await nebula_client.close()
        logger.info("NebulaGraph 客戶端已關閉")
    except Exception as e:
        logger.warning(f"NebulaGraph 客戶端關閉錯誤: {e}")

    # 關閉 Qdrant 服務
    try:
        from chatbot_graphrag.services.vector.qdrant_service import qdrant_service

        await qdrant_service.close()
        logger.info("Qdrant 服務已關閉")
    except Exception as e:
        logger.warning(f"Qdrant 服務關閉錯誤: {e}")

    # 關閉 MinIO 服務
    try:
        from chatbot_graphrag.services.storage.minio_service import minio_service

        await minio_service.close()
        logger.info("MinIO 服務已關閉")
    except Exception as e:
        logger.warning(f"MinIO 服務關閉錯誤: {e}")

    # 關閉 OpenSearch 服務
    try:
        from chatbot_graphrag.services.search.opensearch_service import opensearch_service

        await opensearch_service.close()
        logger.info("OpenSearch 服務已關閉")
    except Exception as e:
        logger.warning(f"OpenSearch 服務關閉錯誤: {e}")

    logger.info("應用程式關閉完成")


# =============================================================================
# 建立 FastAPI 應用程式實例
# =============================================================================
app = FastAPI(
    title=settings.app_name,           # 應用程式名稱
    version=settings.app_version,      # 版本號
    description="GraphRAG API - 具備實體-關係-事件圖儲存與多模式檢索功能",
    # 在除錯模式下啟用 API 文件介面
    docs_url="/docs" if settings.debug else None,        # Swagger UI
    redoc_url="/redoc" if settings.debug else None,      # ReDoc
    openapi_url="/openapi.json" if settings.debug else None,  # OpenAPI JSON
    default_response_class=ORJSONResponse,  # 使用 ORJSON 加速 JSON 序列化
    lifespan=lifespan,                 # 生命週期管理器
)

# =============================================================================
# 配置 CORS 中介軟體 (Cross-Origin Resource Sharing)
# =============================================================================
# 允許來自不同網域的前端應用程式存取此 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,              # 允許的來源網域列表
    allow_credentials=settings.cors_allow_credentials, # 是否允許攜帶憑證
    allow_methods=settings.cors_allow_methods,        # 允許的 HTTP 方法
    allow_headers=settings.cors_allow_headers,        # 允許的 HTTP 標頭
)


# =============================================================================
# 健康檢查端點 (Health Check Endpoints)
# =============================================================================
# 這些端點供 Kubernetes 或其他編排系統進行服務監控

@app.get("/health")
async def health_check():
    """
    基本健康檢查端點
    
    Returns:
        dict: 包含狀態和版本資訊
    """
    return {"status": "healthy", "version": settings.app_version}


@app.get("/health/ready")
async def readiness_check():
    """
    就緒檢查端點
    
    驗證所有必要的後端服務是否已就緒並可用。
    適用於 Kubernetes 的 readinessProbe。
    
    Returns:
        dict: 包含狀態、版本和各服務的健康狀態
    """
    checks = {}

    # 檢查 PostgreSQL 資料庫
    try:
        from chatbot_graphrag.db import is_db_initialized

        checks["postgres"] = is_db_initialized()
    except Exception:
        checks["postgres"] = False

    # 檢查 NebulaGraph 圖資料庫
    try:
        from chatbot_graphrag.services.graph.nebula_client import nebula_client

        checks["nebula"] = await nebula_client.is_connected()
    except Exception:
        checks["nebula"] = False

    # 檢查 Qdrant 向量資料庫
    try:
        from chatbot_graphrag.services.vector.qdrant_service import qdrant_service

        checks["qdrant"] = await qdrant_service.health_check()
    except Exception:
        checks["qdrant"] = False

    # 檢查 OpenSearch 搜尋服務
    try:
        from chatbot_graphrag.services.search.opensearch_service import opensearch_service

        checks["opensearch"] = await opensearch_service.health_check()
    except Exception:
        checks["opensearch"] = False

    # 判斷整體狀態
    all_healthy = all(checks.values())

    return {
        "status": "ready" if all_healthy else "degraded",  # ready=全部正常, degraded=部分降級
        "version": settings.app_version,
        "checks": checks,
    }


@app.get("/health/live")
async def liveness_check():
    """
    存活檢查端點
    
    適用於 Kubernetes 的 livenessProbe。
    只要應用程式進程還在運行，此端點就會回應。
    
    Returns:
        dict: 包含存活狀態
    """
    return {"status": "alive"}


@app.get("/health/concurrency")
async def concurrency_status():
    """
    LLM 並行狀態端點
    
    回傳目前 LLM 並行處理的狀態，包括：
    - 正在處理的請求數量
    - 等待中的請求數量
    - 各後端的詳細分解
    
    Returns:
        dict: 並行狀態資訊
    """
    from chatbot_graphrag.core.concurrency import llm_concurrency

    return {
        "status": llm_concurrency.get_status(),           # 詳細狀態
        "summary": llm_concurrency.get_summary(),         # 摘要資訊
        "priority_stats": llm_concurrency.get_priority_stats(),  # 優先權統計
    }


# =============================================================================
# 請求日誌中介軟體 (Request Logging Middleware)
# =============================================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    記錄傳入的 HTTP 請求
    
    在除錯模式下，會記錄每個請求的方法和路徑。
    
    Args:
        request: 傳入的 HTTP 請求物件
        call_next: 下一個中介軟體或路由處理器
    
    Returns:
        Response: HTTP 回應物件
    """
    if settings.debug:
        logger.debug(f"{request.method} {request.url.path}")
    response = await call_next(request)
    return response


# =============================================================================
# 註冊 API 路由器 (Register API Routers)
# =============================================================================
app.include_router(vectorize_router)       # 文件向量化路由
app.include_router(ask_stream_router)      # 串流問答路由（Responses API 格式）
app.include_router(ask_stream_chat_router) # 串流問答路由（Chat API 格式）


# =============================================================================
# 伺服器啟動函式 (Server Startup Functions)
# =============================================================================

def run_dev():
    """
    啟動開發伺服器（支援熱重載）
    
    此函式適用於開發階段，啟用 reload=True 讓程式碼變更時自動重新載入。
    """
    uvicorn.run(
        "chatbot_graphrag.main:app",  # 應用程式模組路徑
        host=settings.host,            # 監聽的主機位址
        port=settings.port,            # 監聽的連接埠
        reload=True,                   # 啟用熱重載
        log_level=settings.log_level.lower(),  # 日誌等級
    )


def run_prod():
    """
    啟動生產伺服器（多 Worker 模式）
    
    此函式適用於正式部署，使用多個 Worker 進程來處理並行請求。
    Worker 數量預設為 CPU 核心數 * 2 + 1。
    """
    import multiprocessing

    # 計算 Worker 數量
    workers = settings.workers or (multiprocessing.cpu_count() * 2) + 1

    uvicorn.run(
        "chatbot_graphrag.main:app",  # 應用程式模組路徑
        host=settings.host,            # 監聽的主機位址
        port=settings.port,            # 監聽的連接埠
        workers=workers,               # Worker 進程數量
        log_level=settings.log_level.lower(),  # 日誌等級
        access_log=False,              # 關閉存取日誌以提升效能
    )


# =============================================================================
# 程式進入點
# =============================================================================
if __name__ == "__main__":
    # 直接執行此檔案時，啟動開發伺服器
    run_dev()
