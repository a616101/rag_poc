"""
GraphRAG API 路由模組

===============================================================================
模組概述 (Module Overview)
===============================================================================
此模組匯出所有 API 路由器，供主應用程式註冊使用。

路由器列表：
- vectorize_router: 文件向量化 API (/api/v1/rag/vectorize)
- ask_stream_router: 串流問答 API (/api/v1/rag/ask/stream)
- ask_stream_chat_router: OpenAI 相容聊天 API (/api/v1/rag/ask/stream_chat)
- cache_admin_router: 快取管理 API (/api/v1/admin/cache)

使用範例:
    from chatbot_graphrag.api import vectorize_router
    app.include_router(vectorize_router)
===============================================================================
"""

# 匯入所有路由器
from chatbot_graphrag.api.routes import (
    vectorize_router,       # 文件向量化路由
    ask_stream_router,      # 串流問答路由（Responses API 格式）
    ask_stream_chat_router, # 串流問答路由（OpenAI Chat API 格式）
    cache_admin_router,     # 快取管理路由
)

# =============================================================================
# 公開介面 (Public API)
# =============================================================================
__all__ = [
    "vectorize_router",
    "ask_stream_router",
    "ask_stream_chat_router",
    "cache_admin_router",
]
