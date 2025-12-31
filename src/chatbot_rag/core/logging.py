"""
應用程式日誌記錄配置模組

此模組提供彈性的日誌記錄系統，支援以下功能：
- 帶顏色的控制台輸出
- 具有輪替和保留機制的檔案輸出
- 同時支援控制台和檔案輸出
- 具有上下文資訊的結構化日誌
- 效能追蹤

使用 loguru 函式庫提供強大且易用的日誌功能。
"""

import sys
from pathlib import Path
from typing import Any

from loguru import logger

# 確保標準輸出和標準錯誤輸出使用 UTF-8 編碼，以支援中文等多語言字元
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


class LogConfig:
    """
    日誌記錄器配置和設定類別

    此類別負責配置和初始化應用程式的日誌記錄系統，
    支援靈活的輸出選項和格式化設定。
    """

    def __init__(
        self,
        level: str = "INFO",
        log_to_console: bool = True,
        log_to_file: bool = False,
        log_file_path: str = "logs/app.log",
        rotation: str = "100 MB",
        retention: str = "30 days",
        colorize_file: bool = True,
    ):
        """
        初始化日誌記錄器配置

        Args:
            level: 日誌級別（DEBUG, INFO, WARNING, ERROR, CRITICAL）
            log_to_console: 是否輸出日誌到控制台
            log_to_file: 是否輸出日誌到檔案
            log_file_path: 日誌檔案路徑
            rotation: 日誌檔案輪替條件（大小或時間，例如 "100 MB" 或 "1 day"）
            retention: 舊日誌檔案保留時間（例如 "30 days"）
            colorize_file: 是否在檔案輸出中使用顏色標記
        """
        self.level = level.upper()  # 轉換為大寫以確保一致性
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file
        self.log_file_path = Path(log_file_path)
        self.rotation = rotation
        self.retention = retention
        self.colorize_file = colorize_file

        # 控制台輸出格式（帶顏色標記）
        # 格式包含：時間、日誌級別、模組名稱、函數名稱、行號、訊息、額外資訊
        self.console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level> | "
            "{extra}"
        )

        # 檔案輸出格式（可選擇是否包含顏色標記）
        if self.colorize_file:
            # 帶顏色的檔案格式，方便使用工具查看時保持可讀性
            self.file_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level> | "
                "{extra}"
            )
        else:
            # 純文字格式，適合用於日誌分析工具處理
            self.file_format = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message} | "
                "{extra}"
            )

    def setup(self) -> None:
        """
        設定日誌記錄器並配置處理器

        此方法會移除預設的處理器，然後根據配置添加控制台和/或檔案處理器。
        """
        # 移除預設處理器，以便使用自訂配置
        logger.remove()

        # 添加控制台處理器
        if self.log_to_console:
            logger.add(
                sys.stdout,  # 輸出到標準輸出
                format=self.console_format,  # 使用控制台格式
                level=self.level,  # 設定日誌級別
                colorize=True,  # 啟用顏色輸出
                backtrace=True,  # 啟用堆疊追蹤
                diagnose=True,  # 啟用診斷資訊（變數值等）
            )

        # 添加檔案處理器，使用 UTF-8 編碼
        if self.log_to_file:
            # 如果日誌目錄不存在則建立
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

            # 使用字串路徑以保留輪替功能
            # Loguru 會使用系統預設編碼，我們已在模組層級確保使用 UTF-8
            logger.add(
                str(self.log_file_path),  # 日誌檔案路徑
                format=self.file_format,  # 使用檔案格式
                level=self.level,  # 設定日誌級別
                colorize=self.colorize_file,  # 根據配置決定是否使用顏色
                backtrace=True,  # 啟用堆疊追蹤
                diagnose=True,  # 啟用診斷資訊
                rotation=self.rotation,  # 設定檔案輪替條件
                retention=self.retention,  # 設定舊檔案保留時間
                compression="zip",  # 壓縮舊日誌檔案以節省空間
                enqueue=True,  # 啟用非同步安全的日誌記錄
            )

        # 記錄日誌記錄器初始化資訊
        logger.info(
            f"Logger initialized - Console: {self.log_to_console}, "
            f"File: {self.log_to_file}, Level: {self.level}"
        )


def get_logger() -> Any:
    """
    取得已配置的日誌記錄器實例

    Returns:
        Any: 日誌記錄器實例
    """
    return logger


# 用於為日誌添加上下文資訊的上下文管理器
class LogContext:
    """
    日誌上下文管理器

    此類別提供一個上下文管理器，用於在特定範圍內為日誌添加上下文資訊。
    這對於追蹤請求 ID、使用者 ID 等資訊非常有用。

    使用範例:
        with LogContext(user_id="123", request_id="abc"):
            logger.info("Processing request")
            # 日誌將包含 user_id=123 和 request_id=abc
    """

    def __init__(self, **kwargs: Any):
        """
        初始化上下文管理器

        Args:
            **kwargs: 要添加到日誌的上下文鍵值對
        """
        self.context = kwargs
        self.logger = None

    def __enter__(self):
        """
        進入上下文時將上下文資訊綁定到日誌記錄器

        Returns:
            綁定了上下文的日誌記錄器實例
        """
        # 建立一個綁定了上下文的日誌記錄器
        self.logger = logger.bind(**self.context)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        離開上下文時清理（實際上不需要清理，因為 bind 會建立新的實例）

        Args:
            exc_type: 異常類型
            exc_val: 異常值
            exc_tb: 異常追蹤資訊

        Returns:
            bool: False 表示不抑制異常
        """
        return False  # 不抑制異常，讓異常繼續傳播
