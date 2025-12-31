"""
核心模組 (Core Module)

===============================================================================
模組概述 (Module Overview)
===============================================================================
此模組包含 GraphRAG 系統的核心配置和工具，包括：

1. 設定管理 (Configuration Management):
   - settings: 全域設定實例，包含所有環境變數和配置參數

2. 並行控制 (Concurrency Control):
   - llm_concurrency: LLM 並行管理器單例
   - with_llm_semaphore: 通用 LLM 呼叫包裝器
   - with_chat_semaphore: 聊天操作專用包裝器
   - with_embedding_semaphore: 嵌入操作專用包裝器
   - RequestContext: 請求上下文追蹤類別
   - request_context_var: 請求上下文的 ContextVar

這些工具確保系統在高並發場景下能夠穩定運行，防止 LLM 後端過載。
===============================================================================
"""

# 匯入設定實例
from .config import settings

# 匯入並行控制工具
from .concurrency import (
    llm_concurrency,        # LLM 並行管理器單例
    with_llm_semaphore,     # 通用 LLM 呼叫包裝器
    with_chat_semaphore,    # 聊天操作專用包裝器
    with_embedding_semaphore,  # 嵌入操作專用包裝器
    RequestContext,         # 請求上下文追蹤類別
    request_context_var,    # 請求上下文的 ContextVar
)

# =============================================================================
# 公開介面 (Public API)
# =============================================================================
__all__ = [
    # 設定
    "settings",
    # 並行控制
    "llm_concurrency",
    "with_llm_semaphore",
    "with_chat_semaphore",
    "with_embedding_semaphore",
    "RequestContext",
    "request_context_var",
]
