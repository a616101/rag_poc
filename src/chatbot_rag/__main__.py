"""
應用程式命令行入口點模組

此模組提供直接執行應用程式的入口點，使用 Python 的 -m 標誌運行。
當使用 `python -m chatbot_rag` 命令時，此檔案會被執行。

使用方式:
    python -m chatbot_rag

此方式使用配置檔案中的預設設定啟動應用程式。
如需更多控制選項，請使用 cli.py 中的命令。
"""

import uvicorn

from chatbot_rag.core.config import settings


def main():
    """
    使用 Uvicorn 運行應用程式

    此函數使用從配置中讀取的設定啟動 Uvicorn 伺服器。
    會自動載入應用程式實例並在指定的主機和埠號上運行。

    配置來源：
        - host: 從 settings.host 讀取（預設: 0.0.0.0）
        - port: 從 settings.port 讀取（預設: 8000）
        - reload: 從 settings.reload 讀取（預設: False）

    注意：
        此方法適合快速啟動，但功能有限。
        生產環境建議使用 cli.py 中的 prod 命令。
    """
    uvicorn.run(
        "chatbot_rag.main:app",  # 應用程式路徑（模組:變數名）
        host=settings.host,  # 監聽的主機位址
        port=settings.port,  # 監聽的埠號
        reload=settings.reload,  # 是否啟用自動重載
    )


if __name__ == "__main__":
    # 當直接執行此模組時，啟動應用程式
    main()
