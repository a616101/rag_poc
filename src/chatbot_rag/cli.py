"""
命令行介面模組

此模組提供了 Chatbot RAG 應用程式的命令行工具，用於啟動不同模式的伺服器：
- dev: 開發模式，啟用自動重載功能
- start: 生產模式，使用優化的設定
- prod: 高效能生產模式，使用所有可用的 CPU 核心

主要功能：
1. 配置 UTF-8 編碼，確保中文和其他 Unicode 字元正常顯示
2. 提供三種不同的伺服器啟動模式，適應不同的使用場景
3. 使用 Uvicorn ASGI 伺服器運行 FastAPI 應用程式

使用方式：
    # 開發模式
    uv run chatbot-dev

    # 標準生產模式
    uv run chatbot-start

    # 高效能生產模式
    uv run chatbot-prod

    # 或使用 Python 模組方式
    python -m chatbot_rag.cli dev
"""

import locale
import multiprocessing
import os
import sys

import uvicorn

from chatbot_rag.core.config import settings

# 確保所有 I/O 操作使用 UTF-8 編碼
# 這對於處理中文字元特別重要
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
locale.setlocale(locale.LC_ALL, "")

# 重新配置標準輸出和標準錯誤輸出為 UTF-8 編碼
# 確保控制台輸出的中文字元能正確顯示
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def dev():
    """
    啟動開發模式伺服器，啟用自動重載功能

    此函數用於開發環境，當檔案變更時會自動重載應用程式，
    方便開發人員即時查看修改效果，無需手動重啟伺服器。

    功能特點：
    - 自動重載：檔案變更時自動重啟伺服器
    - 詳細日誌：使用 info 級別的日誌輸出
    - 單一 worker：適合開發和除錯

    使用方式：
        uv run chatbot-dev

    注意事項：
        此模式僅適用於開發環境，不應在生產環境中使用，
        因為自動重載功能會降低效能且不穩定。
    """
    # 顯示伺服器啟動資訊
    print("🚀 Starting development server with auto-reload...")
    print(f"📍 Server will be available at http://{settings.host}:{settings.port}")
    print("📝 API documentation: http://localhost:8000/docs")
    print("🔄 Auto-reload: Enabled")
    print()

    # 使用 Uvicorn 啟動開發伺服器
    # reload=True 啟用自動重載功能
    uvicorn.run(
        "chatbot_rag.main:app",  # FastAPI 應用程式的路徑
        host=settings.host,       # 綁定的主機位址
        port=settings.port,       # 監聽的埠號
        reload=True,              # 啟用自動重載
        log_level="info",         # 日誌級別：顯示詳細資訊
    )


def start():
    """
    啟動標準生產模式伺服器，使用優化的設定

    此函數用於生產環境，使用配置檔案中指定的 worker 數量，
    並啟用高效能的事件循環和 HTTP 解析器。

    功能特點：
    - 多 worker 支援：根據設定檔配置 worker 數量
    - 高效能元件：使用 uvloop (事件循環) 和 httptools (HTTP 解析)
    - 連線管理：支援最大連線數和 backlog 限制
    - Keep-alive 設定：優化長連線處理

    使用方式：
        uv run chatbot-start

    設定參數：
        - workers: 工作程序數量 (最少為 1)
        - max_connections: 最大並發連線數
        - backlog: 待處理請求的佇列大小
        - keepalive_timeout: Keep-alive 超時時間

    注意事項：
        此模式適用於中小型生產環境，如需最高效能請使用 prod() 函數。
    """
    # 確保至少有一個 worker
    workers = settings.workers if settings.workers > 1 else 1

    # 顯示伺服器配置資訊
    print("🚀 Starting production server...")
    print(f"📍 Server: http://{settings.host}:{settings.port}")
    print(f"👷 Workers: {workers}")
    print(f"🔌 Max connections: {settings.max_connections}")
    print(f"📊 Backlog: {settings.backlog}")
    print()

    # 使用 Uvicorn 啟動生產伺服器
    uvicorn.run(
        "chatbot_rag.main:app",                         # FastAPI 應用程式的路徑
        host=settings.host,                              # 綁定的主機位址
        port=settings.port,                              # 監聽的埠號
        workers=workers,                                 # worker 程序數量
        loop="uvloop",                                   # 使用 uvloop 高效能事件循環
        http="httptools",                                # 使用 httptools 快速 HTTP 解析
        backlog=settings.backlog,                        # 待處理連線的佇列大小
        limit_concurrency=settings.max_connections,      # 限制最大並發連線數
        timeout_keep_alive=settings.keepalive_timeout,   # Keep-alive 連線超時時間
        log_level="info",                                # 日誌級別：顯示一般資訊
    )


def prod():
    """
    啟動高效能生產模式伺服器，使用最大效能設定

    此函數用於高流量的生產環境，自動根據 CPU 核心數計算最佳的 worker 數量，
    充分利用所有可用的 CPU 資源，以達到最大吞吐量。

    功能特點：
    - 自動優化 worker 數量：使用公式 (CPU 核心數 * 2) + 1
    - 高效能元件：uvloop 和 httptools
    - 精簡日誌：使用 warning 級別減少日誌開銷
    - 完整的連線管理和優化設定

    Worker 數量計算原理：
        公式：(CPU 核心數 * 2) + 1
        - 乘以 2：充分利用 I/O 等待時間
        - 加 1：確保始終有可用的 worker 處理請求
        例如：4 核心 CPU → (4 * 2) + 1 = 9 個 workers

    使用方式：
        uv run chatbot-prod

    設定參數：
        - workers: 自動計算的最佳 worker 數量
        - max_connections: 最大並發連線數
        - backlog: 待處理請求的佇列大小
        - keepalive_timeout: Keep-alive 超時時間
        - log_level: warning (減少日誌輸出，提升效能)

    注意事項：
        此模式會佔用大量系統資源，適用於專用的生產伺服器。
        確保伺服器有足夠的記憶體支援所有 worker 程序。
    """
    # 計算最佳 worker 數量：(CPU 核心數 * 2) + 1
    # 這個公式能在 I/O 密集型應用中最大化吞吐量
    cpu_count = multiprocessing.cpu_count()
    workers = (cpu_count * 2) + 1

    # 顯示伺服器配置資訊
    print("🚀 Starting high-performance production server...")
    print(f"📍 Server: http://{settings.host}:{settings.port}")
    print(f"🖥️  CPU cores: {cpu_count}")
    print(f"👷 Workers: {workers} (optimized for CPU)")
    print(f"🔌 Max connections: {settings.max_connections}")
    print(f"📊 Backlog: {settings.backlog}")
    print(f"⏱️  Keep-alive: {settings.keepalive_timeout}s")
    print()

    # 使用 Uvicorn 啟動高效能生產伺服器
    uvicorn.run(
        "chatbot_rag.main:app",                         # FastAPI 應用程式的路徑
        host=settings.host,                              # 綁定的主機位址
        port=settings.port,                              # 監聽的埠號
        workers=workers,                                 # 最佳化的 worker 程序數量
        loop="uvloop",                                   # 使用 uvloop 高效能事件循環
        http="httptools",                                # 使用 httptools 快速 HTTP 解析
        backlog=settings.backlog,                        # 待處理連線的佇列大小
        limit_concurrency=settings.max_connections,      # 限制最大並發連線數
        timeout_keep_alive=settings.keepalive_timeout,   # Keep-alive 連線超時時間
        log_level="warning",                             # 日誌級別：僅警告，減少輸出開銷
    )


if __name__ == "__main__":
    """
    主程式進入點

    允許使用 Python 模組方式執行命令行工具。
    例如：python -m chatbot_rag.cli dev
    """
    # 檢查是否提供了命令參數
    if len(sys.argv) > 1:
        # 取得第一個參數作為命令
        command = sys.argv[1]

        # 根據命令執行對應的函數
        if command == "dev":
            dev()  # 開發模式
        elif command == "start":
            start()  # 標準生產模式
        elif command == "prod":
            prod()  # 高效能生產模式
        else:
            # 未知的命令，顯示錯誤訊息並退出
            print(f"Unknown command: {command}")
            print("Available commands: dev, start, prod")
            sys.exit(1)
    else:
        # 未提供命令參數，顯示使用說明並退出
        print("Usage: python -m chatbot_rag.cli [dev|start|prod]")
        sys.exit(1)
