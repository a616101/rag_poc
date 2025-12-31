"""
LLM 相關整合模組。

目前主要提供：
- 對 OpenAI Responses API 的封裝 (`ResponsesChatModel` / `ResponsesAccumulator`)
- 建立預設 Responses LLM 實例的工廠函式
- 與 LangGraph 整合用的 `llm_node` 建立工具
"""

from .responses_chat_model import ResponsesChatModel
from .responses_accumulator import ResponsesAccumulator
from .factory import (
    create_responses_llm,
    create_generation_node,
    create_chat_completion_llm,
    create_chat_generation_node,
)
from .graph_nodes import State, create_llm_node, create_chat_llm_node

__all__ = [
    "ResponsesChatModel",
    "ResponsesAccumulator",
    "create_responses_llm",
    "create_generation_node",
    "create_chat_completion_llm",
    "create_chat_generation_node",
    "State",
    "create_llm_node",
    "create_chat_llm_node",
]



