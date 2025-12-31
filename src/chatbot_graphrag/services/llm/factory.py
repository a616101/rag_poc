"""
LLM 工廠模組

為 GraphRAG 工作流程提供聊天模型實例。
支援 Chat Completions API 和 Responses API 兩種後端。
整合並發控制以應對高負載場景。

主要功能：
- 建立標準聊天模型（ChatOpenAI）
- 建立快速模型用於意圖偵測等快速操作
- 建立串流模型
- 建立帶並發控制的模型（ConcurrentChatModel）
- 建立 Responses API 模型（ResponsesChatModel）
"""

import logging
from functools import lru_cache
from typing import Optional, Union

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from chatbot_graphrag.core.config import settings
from chatbot_graphrag.services.llm.responses_chat_model import ResponsesChatModel
from chatbot_graphrag.services.llm.concurrent_llm import ConcurrentChatModel

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    LLM 實例工廠。

    通過 OpenAI 相容 API 支援多種模型後端。
    使用單例模式快取模型實例以提高效能。
    """

    def __init__(self):
        """初始化 LLM 工廠。"""
        self._chat_model: Optional[BaseChatModel] = None
        self._fast_model: Optional[BaseChatModel] = None

    def get_chat_model(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> BaseChatModel:
        """
        取得主要聊天模型。

        Args:
            model_name: 覆蓋預設模型名稱
            temperature: 模型溫度參數
            max_tokens: 最大生成 token 數

        Returns:
            聊天模型實例
        """
        if self._chat_model is None or model_name:
            self._chat_model = self._create_chat_model(
                model_name=model_name or settings.chat_model,
                temperature=temperature,
                max_tokens=max_tokens or settings.chat_max_tokens,
            )
        return self._chat_model

    def get_fast_model(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> BaseChatModel:
        """
        取得快速模型用於快速操作（意圖偵測等）。

        Args:
            model_name: 覆蓋預設模型名稱
            temperature: 模型溫度參數
            max_tokens: 最大生成 token 數

        Returns:
            快速聊天模型實例
        """
        if self._fast_model is None or model_name:
            # 使用相同模型但較低的 max_tokens 以提高速度
            self._fast_model = self._create_chat_model(
                model_name=model_name or settings.chat_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return self._fast_model

    def _create_chat_model(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
    ) -> BaseChatModel:
        """
        建立聊天模型實例。

        Args:
            model_name: 模型名稱
            temperature: 模型溫度參數
            max_tokens: 最大 token 數

        Returns:
            聊天模型實例
        """
        logger.info(f"正在建立聊天模型: {model_name}")

        # 使用 OpenAI 相容 API
        model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_base=settings.openai_api_base,
            openai_api_key=settings.openai_api_key,
            request_timeout=settings.llm_request_timeout,
        )

        return model

    def get_streaming_model(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> BaseChatModel:
        """
        取得配置為串流的模型。

        Args:
            model_name: 覆蓋預設模型名稱
            temperature: 模型溫度參數
            max_tokens: 最大生成 token 數

        Returns:
            串流聊天模型實例
        """
        return ChatOpenAI(
            model=model_name or settings.chat_model,
            temperature=temperature,
            max_tokens=max_tokens or settings.chat_max_tokens,
            openai_api_base=settings.openai_api_base,
            openai_api_key=settings.openai_api_key,
            request_timeout=settings.llm_request_timeout,
            streaming=True,
        )

    def reset(self) -> None:
        """重置快取的模型實例。"""
        self._chat_model = None
        self._fast_model = None
        logger.debug("LLM 工廠已重置")

    def get_concurrent_chat_model(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        backend: str = "chat",
    ) -> ConcurrentChatModel:
        """
        取得帶有內建並發控制的聊天模型。

        這是在高負載場景下取得 LLM 實例的推薦方式。
        對此模型的所有呼叫都會通過 LLM 並發信號量。

        Args:
            model_name: 覆蓋預設模型名稱
            temperature: 模型溫度參數
            max_tokens: 最大生成 token 數
            backend: 信號量後端類型（chat/responses/default/embedding）

        Returns:
            ConcurrentChatModel 封裝器
        """
        base_model = self._create_chat_model(
            model_name=model_name or settings.chat_model,
            temperature=temperature,
            max_tokens=max_tokens or settings.chat_max_tokens,
        )
        return ConcurrentChatModel(model=base_model, backend=backend)

    def get_concurrent_streaming_model(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        backend: str = "chat",
    ) -> ConcurrentChatModel:
        """
        取得帶有內建並發控制的串流模型。

        Args:
            model_name: 覆蓋預設模型名稱
            temperature: 模型溫度參數
            max_tokens: 最大生成 token 數
            backend: 信號量後端類型

        Returns:
            配置為串流的 ConcurrentChatModel 封裝器
        """
        base_model = ChatOpenAI(
            model=model_name or settings.chat_model,
            temperature=temperature,
            max_tokens=max_tokens or settings.chat_max_tokens,
            openai_api_base=settings.openai_api_base,
            openai_api_key=settings.openai_api_key,
            request_timeout=settings.llm_request_timeout,
            streaming=True,
        )
        return ConcurrentChatModel(model=base_model, backend=backend)

    def create_responses_llm(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        streaming: bool = True,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = "medium",
        reasoning_summary: Optional[str] = "auto",
    ) -> ResponsesChatModel:
        """
        建立 Responses API 模型實例（支援真正的串流）。

        Args:
            model_name: 模型名稱覆蓋
            temperature: 溫度參數
            streaming: 是否啟用串流
            max_tokens: 最大生成 token 數
            reasoning_effort: 推理努力程度（"low"/"medium"/"high"）
            reasoning_summary: 推理摘要模式（"concise"/"detailed"/"auto"）

        Returns:
            ResponsesChatModel 實例
        """
        logger.info(f"正在建立 Responses LLM: {model_name or settings.chat_model}, streaming={streaming}")
        return ResponsesChatModel(
            model=model_name or settings.chat_model,
            base_url=settings.openai_api_base,
            api_key=settings.openai_api_key,
            streaming=streaming,
            temperature=temperature,
            max_tokens=max_tokens or settings.chat_max_tokens,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
        )

    def create_chat_completion_llm(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        streaming: bool = True,
        max_tokens: Optional[int] = None,
    ) -> ChatOpenAI:
        """
        建立 Chat Completions API 模型實例（支援串流）。

        Args:
            model_name: 模型名稱覆蓋
            temperature: 溫度參數
            streaming: 是否啟用串流
            max_tokens: 最大生成 token 數

        Returns:
            ChatOpenAI 實例
        """
        logger.info(f"正在建立 Chat Completion LLM: {model_name or settings.chat_model}, streaming={streaming}")
        return ChatOpenAI(
            model=model_name or settings.chat_model,
            openai_api_base=settings.openai_api_base,
            openai_api_key=settings.openai_api_key,
            temperature=temperature,
            max_tokens=max_tokens or settings.chat_max_tokens,
            streaming=streaming,
            stream_usage=True,  # 在最後一個 chunk 中包含使用量資訊
            request_timeout=settings.llm_request_timeout,
        )


# 單例實例
llm_factory = LLMFactory()


@lru_cache(maxsize=1)
def get_default_chat_model() -> BaseChatModel:
    """取得預設聊天模型（已快取）。"""
    return llm_factory.get_chat_model()
