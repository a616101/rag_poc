"""
ChatBot GraphRAG 命令列介面 (CLI)

===============================================================================
模組概述 (Module Overview)
===============================================================================
此模組提供命令列介面，用於啟動和管理 GraphRAG 應用程式。
支援開發模式和生產模式兩種執行方式。

使用方式 (Usage):
    # 開發模式 - 支援熱重載（auto-reload），適合開發階段
    python -m chatbot_graphrag.cli dev

    # 生產模式 - 使用多個 Worker 進程，適合正式部署
    python -m chatbot_graphrag.cli prod

    # 顯示說明
    python -m chatbot_graphrag.cli --help

可選參數 (Optional Arguments):
    --host     覆蓋預設主機位址
    --port     覆蓋預設連接埠
    --workers  指定 Worker 數量（僅生產模式）
===============================================================================
"""

import argparse
import sys


def main():
    """
    CLI 主要進入點
    
    建立命令列解析器，處理 'dev' 和 'prod' 子命令，
    並根據使用者輸入啟動對應的伺服器模式。
    """
    # 建立主解析器
    parser = argparse.ArgumentParser(
        prog="chatbot_graphrag",
        description="ChatBot GraphRAG - 生產級 GraphRAG API 服務",
    )

    # 建立子命令解析器
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # =========================================================================
    # 開發伺服器命令設定
    # =========================================================================
    dev_parser = subparsers.add_parser(
        "dev",
        help="啟動開發伺服器（支援熱重載）",
    )
    dev_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="覆蓋主機位址（預設：從設定檔讀取）",
    )
    dev_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="覆蓋連接埠（預設：從設定檔讀取）",
    )

    # =========================================================================
    # 生產伺服器命令設定
    # =========================================================================
    prod_parser = subparsers.add_parser(
        "prod",
        help="啟動生產伺服器（多 Worker 模式）",
    )
    prod_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="覆蓋主機位址（預設：從設定檔讀取）",
    )
    prod_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="覆蓋連接埠（預設：從設定檔讀取）",
    )
    prod_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker 進程數量（預設：CPU 核心數 * 2 + 1）",
    )

    # 解析命令列參數
    args = parser.parse_args()

    # 根據子命令執行對應的啟動函式
    if args.command == "dev":
        run_dev(host=args.host, port=args.port)
    elif args.command == "prod":
        run_prod(host=args.host, port=args.port, workers=args.workers)
    else:
        # 如果沒有指定命令，顯示說明並結束
        parser.print_help()
        sys.exit(1)


def run_dev(host: str = None, port: int = None):
    """
    啟動開發伺服器（支援熱重載）
    
    Args:
        host: 伺服器監聽的主機位址，None 則使用設定檔的值
        port: 伺服器監聽的連接埠，None 則使用設定檔的值
    
    特點:
        - 啟用 reload=True，程式碼變更時自動重新載入
        - 適合開發階段使用，不建議用於生產環境
    """
    import uvicorn
    from chatbot_graphrag.core.config import settings

    uvicorn.run(
        "chatbot_graphrag.main:app",  # 應用程式模組路徑
        host=host or settings.host,    # 主機位址
        port=port or settings.port,    # 連接埠
        reload=True,                   # 啟用熱重載
        log_level=settings.log_level.lower(),  # 日誌等級
    )


def run_prod(host: str = None, port: int = None, workers: int = None):
    """
    啟動生產伺服器（多 Worker 模式）
    
    Args:
        host: 伺服器監聽的主機位址，None 則使用設定檔的值
        port: 伺服器監聯的連接埠，None 則使用設定檔的值
        workers: Worker 進程數量，None 則使用公式 CPU*2+1 計算
    
    特點:
        - 使用多個 Worker 進程處理請求，提升並行能力
        - 關閉存取日誌（access_log=False）以提升效能
        - 適合正式部署環境使用
    """
    import multiprocessing
    import uvicorn
    from chatbot_graphrag.core.config import settings

    # 計算 Worker 數量：優先使用參數值 > 設定檔值 > 自動計算
    worker_count = workers or settings.workers or (multiprocessing.cpu_count() * 2) + 1

    uvicorn.run(
        "chatbot_graphrag.main:app",  # 應用程式模組路徑
        host=host or settings.host,    # 主機位址
        port=port or settings.port,    # 連接埠
        workers=worker_count,          # Worker 進程數量
        log_level=settings.log_level.lower(),  # 日誌等級
        access_log=False,              # 關閉存取日誌以提升效能
    )


# =============================================================================
# 程式進入點
# =============================================================================
if __name__ == "__main__":
    main()
