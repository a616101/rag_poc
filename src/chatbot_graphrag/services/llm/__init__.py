"""
LLM 服務與工廠模組。

提供大型語言模型（LLM）的服務和工廠類別，用於 GraphRAG 工作流程。

主要匯出：
- LLMFactory / llm_factory: LLM 實例工廠
- ResponsesChatModel: OpenAI Responses API 的 LangChain 封裝
- ResponsesAccumulator / ChannelBuffer: 串流累加器
- ConcurrentChatModel: 帶並發控制的聊天模型封裝
- concurrent_llm_invoke / concurrent_llm_stream: 並發控制的輔助函數
"""

from chatbot_graphrag.services.llm.factory import (
    LLMFactory,
    llm_factory,
    get_default_chat_model,
)
from chatbot_graphrag.services.llm.responses_chat_model import ResponsesChatModel
from chatbot_graphrag.services.llm.responses_accumulator import (
    ResponsesAccumulator,
    ChannelBuffer,
)
from chatbot_graphrag.services.llm.concurrent_llm import (
    ConcurrentChatModel,
    concurrent_llm_invoke,
    concurrent_llm_stream,
)

__all__ = [
    "LLMFactory",
    "llm_factory",
    "get_default_chat_model",
    "ResponsesChatModel",
    "ResponsesAccumulator",
    "ChannelBuffer",
    "ConcurrentChatModel",
    "concurrent_llm_invoke",
    "concurrent_llm_stream",
]
