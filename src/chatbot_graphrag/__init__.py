"""
ChatBot GraphRAG - 生產級 GraphRAG 聊天機器人 API 套件

===============================================================================
模組概述 (Module Overview)
===============================================================================
此套件實現了完整的 GraphRAG（Graph-based Retrieval-Augmented Generation）架構，
專為醫院客服系統設計，提供高效能的問答服務。

核心特色 (Core Features):
- NebulaGraph 圖資料庫：儲存實體-關係-事件（Entity-Relation-Event）及社群報告
- 雙軌攝取管道（Dual Ingestion Pipelines）：
  * Curated Pipeline：處理結構化 YAML + Markdown 文件
  * Raw Pipeline：處理原始 PDF/DOCX/HTML 文件
- 三種查詢模式（Query Modes）：
  * Local Mode：局部圖搜索，適合特定實體查詢
  * Global Mode：全域圖搜索，適合跨領域綜合查詢
  * DRIFT Mode：動態檢索模式，支援即時上下文調整
- Human-in-the-Loop (HITL)：高風險查詢的人工審核機制
- Langfuse 可觀測性：完整的追蹤與迴歸測試閉環

API 端點 (Target Routes):
- POST /api/v1/rag/vectorize     - 文件向量化與攝取
- POST /api/v1/rag/ask/stream    - 串流問答（Responses API 格式）
- POST /api/v1/rag/ask/stream_chat - 串流問答（Chat API 格式）

效能目標 (SLO):
- P95 延遲：12-20 秒
- 允許迴圈次數：2-4 次
===============================================================================
"""

# 版本資訊
__version__ = "2.0.0"  # 套件版本號
__author__ = "Bruce"    # 作者


# =============================================================================
# 延遲匯入函式 (Lazy Import Functions)
# =============================================================================
# 使用延遲匯入以避免循環依賴問題（circular dependencies）
# 這些函式只在被呼叫時才會實際載入相關模組

def get_app():
    """
    取得 FastAPI 應用程式實例
    
    Returns:
        FastAPI: 主應用程式實例，包含所有已註冊的路由和中介軟體
    
    使用範例:
        >>> from chatbot_graphrag import get_app
        >>> app = get_app()
    """
    from chatbot_graphrag.main import app
    return app


def get_settings():
    """
    取得應用程式設定實例
    
    Returns:
        Settings: 包含所有環境變數和配置的設定物件
    
    使用範例:
        >>> from chatbot_graphrag import get_settings
        >>> settings = get_settings()
        >>> print(settings.debug)
    """
    from chatbot_graphrag.core.config import settings
    return settings


# =============================================================================
# 公開介面 (Public API)
# =============================================================================
# 定義此套件對外公開的符號，使用 `from chatbot_graphrag import *` 時會匯入這些
__all__ = [
    "__version__",   # 套件版本
    "__author__",    # 作者資訊
    "get_app",       # 取得應用程式實例
    "get_settings",  # 取得設定實例
]
