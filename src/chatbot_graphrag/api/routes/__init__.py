"""
API 路由處理器模組

===============================================================================
模組概述 (Module Overview)
===============================================================================
此模組定義並匯出所有 API 路由處理器。

路由器說明：
- vectorize_router: 處理文件攝取和向量化請求
  * POST /api/v1/rag/vectorize - 單一文件向量化
  * POST /api/v1/rag/vectorize/file - 檔案上傳向量化
  * POST /api/v1/rag/vectorize/directory - 批次目錄向量化
  * GET  /api/v1/rag/vectorize/status/{job_id} - 查詢工作狀態

- ask_stream_router: 處理串流問答請求（自定義 SSE 格式）
  * POST /api/v1/rag/ask/stream - 串流問答
  * POST /api/v1/rag/ask - 非串流問答
  * POST /api/v1/rag/ask/resume - 恢復 HITL 工作流程
  * GET  /api/v1/rag/ask/status/{thread_id} - 查詢工作流程狀態

- ask_stream_chat_router: 處理 OpenAI 相容的聊天請求
  * POST /api/v1/rag/ask/stream_chat - OpenAI 格式聊天

- cache_admin_router: 處理快取管理請求
  * POST /api/v1/admin/cache/invalidate - 失效快取
  * GET  /api/v1/admin/cache/stats - 快取統計
===============================================================================
"""

# 匯入各路由器
from chatbot_graphrag.api.routes.vectorize import router as vectorize_router
from chatbot_graphrag.api.routes.ask_stream import router as ask_stream_router
from chatbot_graphrag.api.routes.ask_stream_chat import router as ask_stream_chat_router
from chatbot_graphrag.api.routes.cache_admin import router as cache_admin_router

# =============================================================================
# 公開介面 (Public API)
# =============================================================================
__all__ = [
    "vectorize_router",       # 向量化路由
    "ask_stream_router",      # 串流問答路由
    "ask_stream_chat_router", # OpenAI 相容聊天路由
    "cache_admin_router",     # 快取管理路由
]
