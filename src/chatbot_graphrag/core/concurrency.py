"""
LLM 並行控制模組 (Concurrency Control Module)

===============================================================================
模組概述 (Module Overview)
===============================================================================
此模組提供信號量（Semaphore）機制來控制並行的 LLM 呼叫數量，
防止後端因同時處理過多請求而過載。

支援兩種排程模式：
1. FIFO 模式（預設）：先進先出，按到達順序處理
2. 優先權模式：已完成更多 LLM 呼叫的請求獲得更高優先權

設計原理：
- 使用 asyncio.Semaphore 實現非阻塞的並行控制
- 透過 ContextVar 在非同步呼叫鏈中傳遞請求上下文
- 優先權模式可防止新請求「搶占」已投入資源的舊請求

使用範例：
    # 方式一：使用上下文管理器
    async with llm_concurrency.acquire("chat"):
        result = await llm.ainvoke(messages)
    
    # 方式二：使用包裝函式
    result = await with_chat_semaphore(lambda: llm.ainvoke(messages))
===============================================================================
"""

import asyncio
import heapq
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
)

from loguru import logger

# 泛型類型變數，用於保持回傳類型一致
T = TypeVar("T")


# =============================================================================
# 請求上下文追蹤 (Request Context Tracking)
# =============================================================================


@dataclass
class RequestContext:
    """
    追蹤單一請求的 LLM 呼叫進度
    
    用於優先權排程：已完成更多 LLM 呼叫的請求獲得更高優先權。
    這確保了「已投入資源」的請求能更快完成，避免資源浪費。
    
    Attributes:
        request_id: 請求的唯一識別碼
        completed_calls: 已完成的 LLM 呼叫次數
        start_time: 請求開始時間（單調時鐘）
    
    使用範例:
        ctx = RequestContext(request_id="req-123")
        request_context_var.set(ctx)
        # ... 執行 LLM 呼叫 ...
        ctx.increment()  # 完成後遞增計數
    """

    request_id: str                                        # 請求 ID
    completed_calls: int = 0                               # 已完成的呼叫數
    start_time: float = field(default_factory=time.monotonic)  # 開始時間

    def increment(self) -> None:
        """
        遞增呼叫計數
        
        在完成一次 LLM 呼叫後呼叫此方法，
        用於更新優先權計算。
        """
        self.completed_calls += 1


# ContextVar 用於在非同步呼叫鏈中傳遞請求上下文
# 這允許在深層巢狀的非同步函式中存取當前請求的上下文
request_context_var: ContextVar[Optional[RequestContext]] = ContextVar(
    "request_context", default=None
)


@dataclass(order=True)
class PriorityItem:
    """
    優先權佇列中的等待項目
    
    使用 dataclass 的 order=True 配合 heapq，
    priority 值越小優先權越高（先被處理）。
    
    Attributes:
        priority: 排序鍵（越小優先權越高）
        timestamp: 加入佇列的時間（用於飢餓保護）
        request_id: 請求 ID（不參與排序）
        event: asyncio.Event 用於通知等待者
    """

    priority: int                              # 優先權值
    timestamp: float = field(compare=False)    # 加入時間（不比較）
    request_id: str = field(compare=False)     # 請求 ID（不比較）
    event: asyncio.Event = field(compare=False)  # 通知事件（不比較）


# =============================================================================
# LLM 並行管理器 (LLM Concurrency Manager)
# =============================================================================


class LLMConcurrencyManager:
    """
    LLM 並行管理器
    
    使用 asyncio.Semaphore 控制不同後端的並行 LLM 請求，
    實現背壓（backpressure）機制保護下游服務。
    
    支援兩種排程模式：
    1. FIFO（預設）：簡單的先進先出
    2. 優先權：根據請求進度動態調整優先權
    
    後端類型：
    - default: 預設後端
    - chat: 聊天操作（Chat Completions API）
    - responses: Responses API（串流）
    - embedding: 嵌入操作
    
    使用範例:
        # 初始化（在應用程式啟動時）
        llm_concurrency.initialize()
        
        # 取得並行鎖
        async with llm_concurrency.acquire("chat"):
            result = await llm.ainvoke(messages)
        
        # 查看狀態
        status = llm_concurrency.get_status()
    """

    def __init__(self):
        """初始化管理器（不建立信號量，等待 initialize() 呼叫）"""
        # 信號量和限制
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._limits: Dict[str, int] = {}
        self._initialized: bool = False
        
        # 統計資訊
        self._stats: Dict[str, Dict[str, int]] = {}
        self._waiting: Dict[str, int] = {}  # 等待中的請求數

        # 優先權排程相關
        self._priority_enabled: bool = False
        self._starvation_threshold: float = 5.0  # 飢餓保護閾值（秒）
        self._priority_queues: Dict[str, List[PriorityItem]] = {}
        self._queue_locks: Dict[str, asyncio.Lock] = {}

        # 請求上下文追蹤（用於優先權計算）
        self._request_contexts: Dict[str, RequestContext] = {}
        self._contexts_lock: Optional[asyncio.Lock] = None

    def initialize(self) -> None:
        """
        初始化信號量
        
        必須在事件迴圈運行時呼叫（例如在 FastAPI lifespan 中）。
        從設定檔讀取各後端的並行限制並建立對應的信號量。
        """
        from chatbot_graphrag.core.config import settings

        if self._initialized:
            logger.warning("[CONCURRENCY] 管理器已初始化，跳過")
            return

        # 從設定讀取各後端的並行限制
        self._limits = {
            "default": settings.llm_max_concurrent_default,
            "chat": settings.llm_max_concurrent_chat,
            "responses": settings.llm_max_concurrent_responses,
            "embedding": settings.llm_max_concurrent_embedding,
        }

        # 為每個後端建立信號量
        self._semaphores = {
            backend: asyncio.Semaphore(limit)
            for backend, limit in self._limits.items()
        }

        # 初始化統計資訊
        self._stats = {
            backend: {"acquired": 0, "released": 0, "timeout": 0, "priority_boosts": 0}
            for backend in self._semaphores
        }

        # 初始化等待計數
        self._waiting = {backend: 0 for backend in self._semaphores}

        # 初始化優先權排程（從設定讀取）
        self._priority_enabled = getattr(settings, "llm_priority_enabled", False)
        self._starvation_threshold = getattr(
            settings, "llm_priority_starvation_threshold", 5.0
        )

        if self._priority_enabled:
            # 為每個後端建立優先權佇列和鎖
            self._priority_queues = {backend: [] for backend in self._semaphores}
            self._queue_locks = {
                backend: asyncio.Lock() for backend in self._semaphores
            }
            self._contexts_lock = asyncio.Lock()
            logger.info(
                "[CONCURRENCY] 優先權排程已啟用，飢餓閾值=%.1f秒",
                self._starvation_threshold,
            )

        self._initialized = True
        logger.info(
            "[CONCURRENCY] LLM 並行管理器已初始化，限制: "
            "default={}, chat={}, responses={}, embedding={}",
            settings.llm_max_concurrent_default,
            settings.llm_max_concurrent_chat,
            settings.llm_max_concurrent_responses,
            settings.llm_max_concurrent_embedding,
        )

    def is_initialized(self) -> bool:
        """檢查是否已初始化"""
        return self._initialized

    def register_request(self, ctx: RequestContext) -> None:
        """
        註冊請求上下文（用於優先權計算）
        
        Args:
            ctx: 請求上下文物件
        """
        if self._priority_enabled and self._contexts_lock:
            self._request_contexts[ctx.request_id] = ctx

    def unregister_request(self, request_id: str) -> None:
        """
        取消註冊請求上下文
        
        Args:
            request_id: 請求 ID
        """
        if self._priority_enabled:
            self._request_contexts.pop(request_id, None)

    def _calculate_priority(
        self,
        completed_calls: int,
        wait_start: float,
    ) -> int:
        """
        計算優先權值（越小優先權越高）
        
        計算公式：
        - 基礎優先權 = -(completed_calls + 1)
        - 飢餓保護：超過閾值後每秒額外提升 10 點
        
        這確保：
        1. 已完成更多呼叫的請求優先處理
        2. 等待過久的請求不會被無限延遲
        
        Args:
            completed_calls: 已完成的 LLM 呼叫次數
            wait_start: 開始等待的時間
        
        Returns:
            優先權值（越小越優先）
        """
        base_priority = -(completed_calls + 1)

        wait_time = time.monotonic() - wait_start
        if wait_time > self._starvation_threshold:
            # 超過閾值，每秒提升 10 點優先權
            starvation_boost = -int((wait_time - self._starvation_threshold) * 10)
            return base_priority + starvation_boost

        return base_priority

    def _get_completed_calls(self, request_id: str) -> int:
        """取得請求的已完成 LLM 呼叫次數"""
        ctx = self._request_contexts.get(request_id)
        return ctx.completed_calls if ctx else 0

    @asynccontextmanager
    async def _acquire_fifo(
        self,
        backend: str,
        timeout: Optional[float],
    ) -> AsyncIterator[None]:
        """
        FIFO 模式取得實現（先進先出）
        
        這是原始的簡單實現，按請求到達順序處理。
        適用於不需要優先權排程的場景。
        
        Args:
            backend: 後端類型
            timeout: 逾時時間（秒）
        
        Yields:
            None: 取得信號量後讓出控制權
        
        Raises:
            asyncio.TimeoutError: 等待逾時
        """
        from chatbot_graphrag.core.config import settings

        # 確定有效的後端（不存在則使用 default）
        effective_backend = backend if backend in self._semaphores else "default"
        sem = self._semaphores[effective_backend]
        effective_timeout = (
            timeout if timeout is not None else settings.llm_request_timeout
        )

        # 標記開始等待
        self._waiting[effective_backend] += 1

        try:
            # 等待取得信號量
            await asyncio.wait_for(sem.acquire(), timeout=effective_timeout)
            # 取得信號量後，不再處於等待狀態
            self._waiting[effective_backend] -= 1
            self._stats[effective_backend]["acquired"] += 1
            try:
                yield  # 讓出控制權給呼叫者
            finally:
                # 確保釋放信號量
                sem.release()
                self._stats[effective_backend]["released"] += 1
        except asyncio.TimeoutError:
            # 逾時時遞減等待計數
            self._waiting[effective_backend] -= 1
            self._stats[effective_backend]["timeout"] += 1
            logger.warning(
                "[CONCURRENCY] 取得信號量逾時，backend=%s，等待 %.1f 秒",
                backend,
                effective_timeout,
            )
            raise
        except Exception:
            # 其他例外時遞減等待計數
            self._waiting[effective_backend] -= 1
            raise

    @asynccontextmanager
    async def _acquire_priority(
        self,
        backend: str,
        ctx: RequestContext,
        timeout: Optional[float],
    ) -> AsyncIterator[None]:
        """
        優先權模式取得實現
        
        使用優先權佇列來排程請求，已完成更多 LLM 呼叫的請求優先處理。
        這避免了「新請求搶占舊請求」的問題，確保資源有效利用。
        
        Args:
            backend: 後端類型
            ctx: 請求上下文
            timeout: 逾時時間（秒）
        
        Yields:
            None: 取得信號量後讓出控制權
        
        Raises:
            asyncio.TimeoutError: 等待逾時
        """
        from chatbot_graphrag.core.config import settings

        # 確定有效的後端
        effective_backend = backend if backend in self._semaphores else "default"
        sem = self._semaphores[effective_backend]
        effective_timeout = (
            timeout if timeout is not None else settings.llm_request_timeout
        )
        queue_lock = self._queue_locks[effective_backend]
        queue = self._priority_queues[effective_backend]

        wait_start = time.monotonic()

        # 建立等待項目
        item = PriorityItem(
            priority=self._calculate_priority(ctx.completed_calls, wait_start),
            timestamp=wait_start,
            request_id=ctx.request_id,
            event=asyncio.Event(),
        )

        # 加入優先權佇列
        async with queue_lock:
            heapq.heappush(queue, item)
            self._waiting[effective_backend] += 1

        acquired = False
        try:
            # 嘗試觸發排程
            await self._try_signal_next(effective_backend)

            # 等待輪到（或逾時）
            try:
                await asyncio.wait_for(item.event.wait(), timeout=effective_timeout)
            except asyncio.TimeoutError:
                # 從佇列移除
                async with queue_lock:
                    try:
                        queue.remove(item)
                        heapq.heapify(queue)
                    except ValueError:
                        pass  # 已被移除
                self._waiting[effective_backend] -= 1
                self._stats[effective_backend]["timeout"] += 1
                logger.warning(
                    "[CONCURRENCY] 優先權等待逾時，backend=%s，request=%s，等待 %.1f 秒",
                    backend,
                    ctx.request_id,
                    effective_timeout,
                )
                raise

            # 收到通知後，取得信號量
            await sem.acquire()
            acquired = True
            self._waiting[effective_backend] -= 1
            self._stats[effective_backend]["acquired"] += 1

            try:
                yield  # 讓出控制權給呼叫者
            finally:
                sem.release()
                self._stats[effective_backend]["released"] += 1
                # 通知下一個等待者
                await self._signal_next(effective_backend)

        except Exception:
            if not acquired:
                # 取得信號量前發生例外，需要清理
                async with queue_lock:
                    try:
                        queue.remove(item)
                        heapq.heapify(queue)
                    except ValueError:
                        pass
                if self._waiting[effective_backend] > 0:
                    self._waiting[effective_backend] -= 1
            raise

    async def _try_signal_next(self, backend: str) -> None:
        """
        嘗試通知最高優先權的等待者（如果有可用槽位）
        
        只有當信號量有可用槽位時才會通知等待者。
        """
        sem = self._semaphores[backend]

        # 檢查是否有可用槽位
        if sem._value <= 0:
            return

        await self._signal_next(backend)

    async def _signal_next(self, backend: str) -> None:
        """
        通知佇列中最高優先權的等待者
        
        此方法會：
        1. 重新計算所有等待者的優先權（考慮飢餓保護）
        2. 如果有優先權變化則重新排序
        3. 通知最高優先權的等待者
        """
        queue_lock = self._queue_locks[backend]
        queue = self._priority_queues[backend]

        async with queue_lock:
            if not queue:
                return

            # 重新計算所有等待者的優先權（考慮飢餓保護）
            now = time.monotonic()
            updated = False
            for item in queue:
                completed = self._get_completed_calls(item.request_id)
                old_priority = item.priority
                new_priority = self._calculate_priority(completed, item.timestamp)
                if new_priority != old_priority:
                    item.priority = new_priority
                    updated = True
                    if new_priority < old_priority:
                        self._stats[backend]["priority_boosts"] += 1

            # 如果有更新則重新排序
            if updated:
                heapq.heapify(queue)

            # 通知最高優先權的等待者
            if queue:
                next_item = heapq.heappop(queue)
                next_item.event.set()
                logger.debug(
                    "[CONCURRENCY] 通知 request=%s，優先權=%d，"
                    "等待時間=%.2f秒，佇列長度=%d",
                    next_item.request_id,
                    next_item.priority,
                    now - next_item.timestamp,
                    len(queue),
                )

    @asynccontextmanager
    async def acquire(
        self,
        backend: str = "default",
        timeout: Optional[float] = None,
    ) -> AsyncIterator[None]:
        """
        取得 LLM 並行鎖
        
        使用上下文管理器模式確保正確釋放。
        如果啟用優先權模式且有請求上下文，則使用優先權排程。
        
        Args:
            backend: 後端類型（default/chat/responses/embedding）
            timeout: 逾時時間（秒），None 使用設定檔的值
        
        Yields:
            None: 取得鎖後讓出控制權
        
        Raises:
            asyncio.TimeoutError: 等待逾時
        
        使用範例:
            async with llm_concurrency.acquire("chat"):
                result = await llm.ainvoke(messages)
        """
        if not self._initialized:
            # 如果未初始化，不進行限制
            logger.warning(
                "[CONCURRENCY] 管理器未初始化，不進行並行限制"
            )
            yield
            return

        # 取得請求上下文
        ctx = request_context_var.get()

        # 如果優先權未啟用或沒有上下文，使用 FIFO
        if not self._priority_enabled or ctx is None:
            async with self._acquire_fifo(backend, timeout):
                yield
            return

        # 優先權模式
        async with self._acquire_priority(backend, ctx, timeout):
            yield

        # 完成後遞增呼叫計數
        ctx.increment()

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """
        取得並行統計資訊
        
        Returns:
            Dict: 各後端的統計資訊
        """
        return dict(self._stats)

    def get_available_slots(self) -> Dict[str, int]:
        """
        取得各後端的可用並行槽位數
        
        Returns:
            Dict: 各後端的可用槽位數
        """
        if not self._initialized:
            return {}

        return {backend: sem._value for backend, sem in self._semaphores.items()}

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """
        取得各後端的完整狀態
        
        Returns:
            Dict: 各後端的狀態，包括：
                - limit: 最大並行數
                - available: 可用槽位
                - in_progress: 處理中的請求
                - waiting: 等待中的請求
                - total_acquired: 總取得次數
                - total_released: 總釋放次數
                - total_timeout: 總逾時次數
        
        使用範例:
            >>> llm_concurrency.get_status()
            {
                "chat": {
                    "limit": 20,
                    "available": 15,
                    "in_progress": 5,
                    "waiting": 3,
                    "total_acquired": 1000,
                    "total_released": 995,
                    "total_timeout": 2
                },
                ...
            }
        """
        if not self._initialized:
            return {}

        result: Dict[str, Dict[str, Any]] = {}
        for backend, sem in self._semaphores.items():
            limit = self._limits.get(backend, 0)
            available = sem._value
            in_progress = limit - available
            stats = self._stats.get(backend, {})

            result[backend] = {
                "limit": limit,
                "available": available,
                "in_progress": in_progress,
                "waiting": self._waiting.get(backend, 0),
                "total_acquired": stats.get("acquired", 0),
                "total_released": stats.get("released", 0),
                "total_timeout": stats.get("timeout", 0),
            }

        return result

    def get_summary(self) -> Dict[str, Any]:
        """
        取得所有後端的摘要狀態
        
        Returns:
            Dict: 摘要資訊，包括：
                - total_in_progress: 所有後端處理中的請求總數
                - total_waiting: 所有後端等待中的請求總數
                - by_backend: 各後端的簡要狀態
        
        使用範例:
            >>> llm_concurrency.get_summary()
            {
                "total_in_progress": 12,
                "total_waiting": 5,
                "by_backend": {
                    "chat": {"in_progress": 5, "waiting": 2},
                    "responses": {"in_progress": 7, "waiting": 3},
                    ...
                }
            }
        """
        if not self._initialized:
            return {"total_in_progress": 0, "total_waiting": 0, "by_backend": {}}

        total_in_progress = 0
        total_waiting = 0
        by_backend: Dict[str, Dict[str, int]] = {}

        for backend, sem in self._semaphores.items():
            limit = self._limits.get(backend, 0)
            available = sem._value
            in_progress = limit - available
            waiting = self._waiting.get(backend, 0)

            total_in_progress += in_progress
            total_waiting += waiting
            by_backend[backend] = {
                "in_progress": in_progress,
                "waiting": waiting,
            }

        return {
            "total_in_progress": total_in_progress,
            "total_waiting": total_waiting,
            "by_backend": by_backend,
        }

    def get_priority_stats(self) -> Dict[str, Any]:
        """
        取得優先權排程統計
        
        Returns:
            Dict: 優先權統計，包括：
                - priority_enabled: 是否啟用
                - starvation_threshold: 飢餓閾值
                - queues: 各後端的佇列狀態
                - active_requests: 活躍請求數
        
        使用範例:
            >>> llm_concurrency.get_priority_stats()
            {
                "priority_enabled": true,
                "starvation_threshold": 5.0,
                "queues": {
                    "chat": {
                        "length": 3,
                        "top_priorities": [-5, -3, -2]
                    }
                },
                "active_requests": 10
            }
        """
        if not self._priority_enabled:
            return {"priority_enabled": False}

        queues_info: Dict[str, Dict[str, Any]] = {}
        for backend, queue in self._priority_queues.items():
            priorities = sorted([item.priority for item in queue])
            queues_info[backend] = {
                "length": len(queue),
                "top_priorities": priorities[:5],
                "priority_boosts": self._stats.get(backend, {}).get(
                    "priority_boosts", 0
                ),
            }

        return {
            "priority_enabled": True,
            "starvation_threshold": self._starvation_threshold,
            "queues": queues_info,
            "active_requests": len(self._request_contexts),
        }


# =============================================================================
# 全域單例實例 (Global Singleton Instance)
# =============================================================================
# 整個應用程式共用此實例
llm_concurrency = LLMConcurrencyManager()


# =============================================================================
# 便利包裝函式 (Convenience Wrapper Functions)
# =============================================================================


async def with_llm_semaphore(
    coro_func: Callable[[], Awaitable[T]],
    backend: str = "default",
) -> T:
    """
    包裝 LLM 呼叫，自動取得和釋放信號量
    
    使用 lambda 包裝 LLM 呼叫，避免過早建立協程。
    如果啟用優先權模式且有請求上下文，則使用優先權排程。
    
    Args:
        coro_func: 回傳協程的函式（例如 lambda: llm.ainvoke(...)）
        backend: 信號量類型（default/chat/responses/embedding）
    
    Returns:
        LLM 呼叫結果
    
    使用範例:
        # 改造前
        result = await llm.ainvoke(messages)
        
        # 改造後
        result = await with_llm_semaphore(
            lambda: llm.ainvoke(messages),
            backend="chat",
        )
    """
    async with llm_concurrency.acquire(backend):
        return await coro_func()


async def with_embedding_semaphore(
    coro_func: Callable[[], Awaitable[T]],
) -> T:
    """
    嵌入操作的便利包裝器
    
    Args:
        coro_func: 回傳協程的函式
    
    Returns:
        嵌入操作結果
    
    使用範例:
        embeddings = await with_embedding_semaphore(
            lambda: embed_model.aembed_documents(texts)
        )
    """
    return await with_llm_semaphore(coro_func, backend="embedding")


async def with_chat_semaphore(
    coro_func: Callable[[], Awaitable[T]],
) -> T:
    """
    聊天操作的便利包裝器
    
    Args:
        coro_func: 回傳協程的函式
    
    Returns:
        聊天操作結果
    
    使用範例:
        response = await with_chat_semaphore(
            lambda: chat_model.ainvoke(messages)
        )
    """
    return await with_llm_semaphore(coro_func, backend="chat")
