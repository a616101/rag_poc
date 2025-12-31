from typing import Optional, Callable, Dict, Any

from langchain_openai import ChatOpenAI

from chatbot_rag.core.config import settings
from chatbot_rag.llm.responses_chat_model import ResponsesChatModel
from chatbot_rag.llm.graph_nodes import (
    State,
    create_llm_node,
    create_chat_llm_node,
)


def create_responses_llm(
    *,
    model: Optional[str] = None,
    streaming: bool = True,
    reasoning_effort: Optional[str] = "medium",
    reasoning_summary: Optional[str] = "auto",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> ResponsesChatModel:
    """
    建立可配置的 ResponsesChatModel 實例（使用 OpenAI Responses API）。

    Args:
        model: 使用的模型名稱，預設為 settings.chat_model
        streaming: 是否啟用 streaming（預設 True）
        reasoning_effort: reasoning 強度（"low" | "medium" | "high" | None）
        reasoning_summary: reasoning summary 模式（"concise" | "detailed" | "auto" | None）
        temperature: 溫度參數，控制輸出的隨機性（0.0-2.0）
        max_tokens: 最大輸出 token 數
    """
    return ResponsesChatModel(
        model=model or settings.chat_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
        streaming=streaming,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def create_generation_node(
    *,
    model: Optional[str] = None,
    streaming: bool = True,
    reasoning_effort: Optional[str] = "medium",
    reasoning_summary: Optional[str] = "auto",
) -> Callable[[State], Dict[str, Any]]:
    """
    建立一個「用於最終回答」的 LangGraph generation node（Responses API 版本）。

    封裝：
    - 建立對應的 ResponsesChatModel
    - 使用 create_llm_node 轉成 LangGraph node（會發出 answer/reasoning/meta 等事件）
    """
    llm = create_responses_llm(
        model=model,
        streaming=streaming,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
    )
    return create_llm_node(llm)


def create_chat_completion_llm(
    *,
    model: Optional[str] = None,
    streaming: bool = True,
    reasoning_effort: Optional[str] = "medium",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> ChatOpenAI:
    """
    建立一個基於 OpenAI Chat Completions API 的 ChatOpenAI 實例。

    - 明確設定 use_responses_api=False，確保走 Chat Completions 路徑
    - `reasoning_effort` 僅對支援 reasoning 的 chat 模型有效
    - 啟用 `stream_usage`，讓最後一個 streaming chunk 夾帶 usage 資訊
    """

    return ChatOpenAI(
        model=model or settings.chat_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
        streaming=streaming,
        temperature=temperature if temperature is not None else settings.chat_temperature,
        max_tokens=max_tokens if max_tokens is not None else settings.chat_max_tokens,
        # Chat Completions API 僅支援 reasoning_effort（對 reasoning model 有效）
        # reasoning_effort=reasoning_effort,
        # 確保使用 Chat Completions API，而非 Responses API
        use_responses_api=False,
        # 讓 ChatOpenAI 自動在 streaming 回傳 usage（會在最後一個 chunk 的 usage_metadata 中出現）
        stream_usage=True,
    )


def create_chat_generation_node(
    *,
    model: Optional[str] = None,
    streaming: bool = True,
    reasoning_effort: Optional[str] = "medium",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Callable[[State], Dict[str, Any]]:
    """
    建立一個「用於最終回答」的 LangGraph generation node（Chat Completions 版本）。

    - 內部使用 ChatOpenAI（Chat Completions API）
    - 透過 create_chat_llm_node 轉成 LangGraph node
      （會以「先 reasoning 串流、再 answer 串流」的方式輸出事件）
    """
    llm = create_chat_completion_llm(
        model=model,
        streaming=streaming,
        # reasoning_effort=reasoning_effort,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return create_chat_llm_node(llm)


