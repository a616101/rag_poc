"""
並發 LLM 封裝器

提供帶有並發控制的 LLM 操作封裝器。
確保所有 LLM 呼叫都通過信號量機制進行背壓控制。

主要類別：
- ConcurrentChatModel: 封裝 BaseChatModel，自動進行並發控制

輔助函數：
- concurrent_llm_invoke: 帶並發控制的 LLM 呼叫
- concurrent_llm_stream: 帶並發控制的 LLM 串流
"""

from typing import (
    Any,
    AsyncIterator,
    List,
    Optional,
    Sequence,
)

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig
from loguru import logger

from chatbot_graphrag.core.concurrency import (
    llm_concurrency,
    with_llm_semaphore,
)


class ConcurrentChatModel:
    """
    帶有內建並發控制的聊天模型封裝器。

    封裝 BaseChatModel 並確保所有 invoke/ainvoke/stream/astream
    操作都通過 LLM 並發信號量。
    """

    def __init__(
        self,
        model: BaseChatModel,
        backend: str = "chat",
    ):
        """
        初始化並發聊天模型封裝器。

        Args:
            model: 要封裝的底層聊天模型
            backend: 信號量後端類型（chat/responses/default）
        """
        self._model = model
        self._backend = backend

    @property
    def model(self) -> BaseChatModel:
        """取得底層模型。"""
        return self._model

    async def ainvoke(
        self,
        input: List[BaseMessage],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """
        帶並發控制的非同步呼叫。

        Args:
            input: 輸入訊息
            config: Runnable 配置
            **kwargs: 額外參數

        Returns:
            回應訊息
        """
        async with llm_concurrency.acquire(self._backend):
            return await self._model.ainvoke(input, config, **kwargs)

    def invoke(
        self,
        input: List[BaseMessage],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """
        同步呼叫 - 委派給底層模型，不使用非同步信號量。

        注意：對於同步呼叫，無法使用 asyncio 信號量。
        請使用 ainvoke() 以獲得正確的並發控制。
        """
        logger.warning(
            "[ConcurrentChatModel] 呼叫了同步 invoke() - "
            "請使用 ainvoke() 以獲得正確的並發控制"
        )
        return self._model.invoke(input, config, **kwargs)

    async def astream(
        self,
        input: List[BaseMessage],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        帶並發控制的非同步串流。

        信號量在整個串流持續期間保持。

        Args:
            input: 輸入訊息
            config: Runnable 配置
            **kwargs: 額外參數

        Yields:
            聊天生成 chunk
        """
        async with llm_concurrency.acquire(self._backend):
            async for chunk in self._model.astream(input, config, **kwargs):
                yield chunk

    async def agenerate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        帶並發控制的非同步生成。

        Args:
            messages: 訊息列表的列表
            stop: 停止序列
            callbacks: 回調函數
            **kwargs: 額外參數

        Returns:
            聊天結果
        """
        async with llm_concurrency.acquire(self._backend):
            return await self._model.agenerate(messages, stop, callbacks, **kwargs)

    def bind(self, **kwargs: Any) -> "ConcurrentChatModel":
        """
        為模型綁定額外參數。

        Args:
            **kwargs: 要綁定的參數

        Returns:
            帶有綁定模型的新 ConcurrentChatModel
        """
        return ConcurrentChatModel(
            model=self._model.bind(**kwargs),
            backend=self._backend,
        )

    def with_config(self, config: RunnableConfig) -> "ConcurrentChatModel":
        """
        使用 runnable 配置來配置模型。

        Args:
            config: Runnable 配置

        Returns:
            帶有配置模型的新 ConcurrentChatModel
        """
        return ConcurrentChatModel(
            model=self._model.with_config(config),
            backend=self._backend,
        )

    def __getattr__(self, name: str) -> Any:
        """將屬性存取委派給底層模型。"""
        return getattr(self._model, name)


async def concurrent_llm_invoke(
    model: BaseChatModel,
    messages: List[BaseMessage],
    backend: str = "chat",
    **kwargs: Any,
) -> BaseMessage:
    """
    帶並發控制呼叫 LLM 的輔助函數。

    Args:
        model: 聊天模型
        messages: 輸入訊息
        backend: 信號量後端類型
        **kwargs: 額外參數

    Returns:
        回應訊息

    範例：
        result = await concurrent_llm_invoke(
            llm_factory.get_chat_model(),
            messages,
            backend="chat"
        )
    """
    return await with_llm_semaphore(
        lambda: model.ainvoke(messages, **kwargs),
        backend=backend,
    )


async def concurrent_llm_stream(
    model: BaseChatModel,
    messages: List[BaseMessage],
    backend: str = "chat",
    **kwargs: Any,
) -> AsyncIterator[Any]:
    """
    帶並發控制串流 LLM 的輔助函數。

    信號量在整個串流持續期間保持。

    Args:
        model: 聊天模型
        messages: 輸入訊息
        backend: 信號量後端類型
        **kwargs: 額外參數

    Yields:
        串流 chunk

    範例：
        async for chunk in concurrent_llm_stream(
            llm_factory.get_streaming_model(),
            messages,
            backend="chat"
        ):
            print(chunk.content)
    """
    async with llm_concurrency.acquire(backend):
        async for chunk in model.astream(messages, **kwargs):
            yield chunk
