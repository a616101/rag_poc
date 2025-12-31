"""
LLM 並發控制模組

提供 Semaphore 機制來控制 LLM 呼叫的並發數量，
避免過多同時請求造成後端過載。

支援兩種模式：
1. FIFO 模式（預設）：先到先服務
2. 優先級模式：已完成較多 LLM 呼叫的請求優先執行
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

T = TypeVar("T")


# ==============================================================================
# 請求上下文追蹤
# ==============================================================================


@dataclass
class RequestContext:
    """
    追蹤單一請求的 LLM 呼叫進度。

    用於優先級排序：已完成較多 LLM 呼叫的請求會獲得更高優先級。

    Attributes:
        request_id: 請求的唯一識別碼
        completed_calls: 已完成的 LLM 呼叫次數
        start_time: 請求開始時間（monotonic）
    """

    request_id: str
    completed_calls: int = 0
    start_time: float = field(default_factory=time.monotonic)

    def increment(self) -> None:
        """完成一個 LLM 呼叫後增加計數。"""
        self.completed_calls += 1


# ContextVar 用於跨 async 呼叫傳遞請求上下文
request_context_var: ContextVar[Optional[RequestContext]] = ContextVar(
    "request_context", default=None
)


@dataclass(order=True)
class PriorityItem:
    """
    優先級佇列中的等待項目。

    使用 dataclass 的 order=True 配合 heapq，
    優先級越小的項目越優先執行。

    Attributes:
        priority: 排序依據（越小越優先）
        timestamp: 加入佇列的時間（用於飢餓防護）
        request_id: 請求 ID（不參與排序）
        event: 用於通知等待者的 asyncio.Event
    """

    priority: int
    timestamp: float = field(compare=False)
    request_id: str = field(compare=False)
    event: asyncio.Event = field(compare=False)


# ==============================================================================
# LLM 並發管理器
# ==============================================================================


class LLMConcurrencyManager:
    """
    LLM 並發管理器

    使用 asyncio.Semaphore 控制不同後端的 LLM 請求並發數量，
    實現背壓控制（backpressure）以保護下游服務。

    支援兩種排程模式：
    1. FIFO（預設）：簡單的先到先服務
    2. Priority：根據請求進度動態調整優先級
    """

    def __init__(self):
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._limits: Dict[str, int] = {}
        self._initialized: bool = False
        self._stats: Dict[str, Dict[str, int]] = {}
        self._waiting: Dict[str, int] = {}  # 正在排隊等待的請求數

        # 優先級排序相關
        self._priority_enabled: bool = False
        self._starvation_threshold: float = 5.0
        self._priority_queues: Dict[str, List[PriorityItem]] = {}
        self._queue_locks: Dict[str, asyncio.Lock] = {}

        # 請求上下文追蹤（用於計算優先級）
        self._request_contexts: Dict[str, RequestContext] = {}
        self._contexts_lock: Optional[asyncio.Lock] = None

    def initialize(self) -> None:
        """
        初始化 Semaphore。

        必須在 event loop 執行時呼叫（例如在 FastAPI lifespan 中）。
        """
        from chatbot_rag.core.config import settings

        if self._initialized:
            logger.warning("[CONCURRENCY] Manager already initialized, skipping")
            return

        self._limits = {
            "default": settings.llm_max_concurrent_default,
            "chat": settings.llm_max_concurrent_chat,
            "responses": settings.llm_max_concurrent_responses,
            "embedding": settings.llm_max_concurrent_embedding,
        }

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

        # 初始化優先級排序
        self._priority_enabled = settings.llm_priority_enabled
        self._starvation_threshold = settings.llm_priority_starvation_threshold

        if self._priority_enabled:
            self._priority_queues = {backend: [] for backend in self._semaphores}
            self._queue_locks = {
                backend: asyncio.Lock() for backend in self._semaphores
            }
            self._contexts_lock = asyncio.Lock()
            logger.info(
                "[CONCURRENCY] Priority scheduling enabled with starvation_threshold=%.1fs",
                self._starvation_threshold,
            )

        self._initialized = True
        logger.info(
            "[CONCURRENCY] LLM concurrency manager initialized with limits: "
            "default={}, chat={}, responses={}, embedding={}",
            settings.llm_max_concurrent_default,
            settings.llm_max_concurrent_chat,
            settings.llm_max_concurrent_responses,
            settings.llm_max_concurrent_embedding,
        )

    def is_initialized(self) -> bool:
        """檢查是否已初始化。"""
        return self._initialized

    def register_request(self, ctx: RequestContext) -> None:
        """
        註冊請求上下文（用於優先級計算）。

        Args:
            ctx: 請求上下文
        """
        if self._priority_enabled and self._contexts_lock:
            self._request_contexts[ctx.request_id] = ctx

    def unregister_request(self, request_id: str) -> None:
        """
        取消註冊請求上下文。

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
        計算優先級（越小越優先）。

        基礎優先級 = -(completed_calls + 1)
        飢餓防護：等待超過門檻後額外提升

        Args:
            completed_calls: 已完成的 LLM 呼叫數
            wait_start: 開始等待的時間

        Returns:
            優先級值（越小越優先）
        """
        base_priority = -(completed_calls + 1)

        wait_time = time.monotonic() - wait_start
        if wait_time > self._starvation_threshold:
            # 超過門檻，每秒額外提升 10 點優先級
            starvation_boost = -int((wait_time - self._starvation_threshold) * 10)
            return base_priority + starvation_boost

        return base_priority

    def _get_completed_calls(self, request_id: str) -> int:
        """取得指定請求已完成的 LLM 呼叫數。"""
        ctx = self._request_contexts.get(request_id)
        return ctx.completed_calls if ctx else 0

    @asynccontextmanager
    async def _acquire_fifo(
        self,
        backend: str,
        timeout: Optional[float],
    ) -> AsyncIterator[None]:
        """FIFO 模式的 acquire 實作（原本的邏輯）。"""
        from chatbot_rag.core.config import settings

        effective_backend = backend if backend in self._semaphores else "default"
        sem = self._semaphores[effective_backend]
        effective_timeout = (
            timeout if timeout is not None else settings.llm_request_timeout
        )

        # 標記開始等待
        self._waiting[effective_backend] += 1

        try:
            await asyncio.wait_for(sem.acquire(), timeout=effective_timeout)
            # 取得 semaphore 後，不再等待
            self._waiting[effective_backend] -= 1
            self._stats[effective_backend]["acquired"] += 1
            try:
                yield
            finally:
                sem.release()
                self._stats[effective_backend]["released"] += 1
        except asyncio.TimeoutError:
            # 逾時時也要減少等待計數
            self._waiting[effective_backend] -= 1
            self._stats[effective_backend]["timeout"] += 1
            logger.warning(
                "[CONCURRENCY] Timeout acquiring semaphore for backend=%s after %.1fs",
                backend,
                effective_timeout,
            )
            raise
        except Exception:
            # 其他例外也要減少等待計數
            self._waiting[effective_backend] -= 1
            raise

    @asynccontextmanager
    async def _acquire_priority(
        self,
        backend: str,
        ctx: RequestContext,
        timeout: Optional[float],
    ) -> AsyncIterator[None]:
        """優先級模式的 acquire 實作。"""
        from chatbot_rag.core.config import settings

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

        # 加入優先級佇列
        async with queue_lock:
            heapq.heappush(queue, item)
            self._waiting[effective_backend] += 1

        acquired = False
        try:
            # 嘗試觸發排程
            await self._try_signal_next(effective_backend)

            # 等待輪到自己（或逾時）
            try:
                await asyncio.wait_for(item.event.wait(), timeout=effective_timeout)
            except asyncio.TimeoutError:
                # 從佇列移除
                async with queue_lock:
                    try:
                        queue.remove(item)
                        heapq.heapify(queue)
                    except ValueError:
                        pass  # 已經被移除了
                self._waiting[effective_backend] -= 1
                self._stats[effective_backend]["timeout"] += 1
                logger.warning(
                    "[CONCURRENCY] Priority timeout for backend=%s, request=%s after %.1fs",
                    backend,
                    ctx.request_id,
                    effective_timeout,
                )
                raise

            # 被通知後，嘗試取得 semaphore
            await sem.acquire()
            acquired = True
            self._waiting[effective_backend] -= 1
            self._stats[effective_backend]["acquired"] += 1

            try:
                yield
            finally:
                sem.release()
                self._stats[effective_backend]["released"] += 1
                # 通知下一個等待者
                await self._signal_next(effective_backend)

        except Exception:
            if not acquired:
                # 發生例外但尚未取得 semaphore，需要清理
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
        """嘗試通知佇列中優先級最高的等待者（若有可用 slot）。"""
        sem = self._semaphores[backend]

        # 檢查是否有可用 slot
        if sem._value <= 0:
            return

        await self._signal_next(backend)

    async def _signal_next(self, backend: str) -> None:
        """通知佇列中優先級最高的等待者。"""
        queue_lock = self._queue_locks[backend]
        queue = self._priority_queues[backend]

        async with queue_lock:
            if not queue:
                return

            # 重新計算所有等待者的優先級（考慮飢餓）
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

            # 如果有更新，重新排序
            if updated:
                heapq.heapify(queue)

            # 通知最高優先級的等待者
            if queue:
                next_item = heapq.heappop(queue)
                next_item.event.set()
                logger.debug(
                    "[CONCURRENCY] Signaled request=%s with priority=%d, "
                    "waited=%.2fs, queue_len=%d",
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
        取得 LLM 並發鎖。

        使用 context manager 模式確保正確釋放。
        若啟用優先級模式且有請求上下文，會使用優先級排序。

        Args:
            backend: 後端類型 (default/chat/responses/embedding)
            timeout: 逾時時間（秒），None 表示使用設定值

        Raises:
            asyncio.TimeoutError: 等待逾時時拋出

        Example:
            async with llm_concurrency.acquire("chat"):
                result = await llm.ainvoke(messages)
        """
        if not self._initialized:
            # 如果未初始化，則 fallback 為無限制
            logger.warning(
                "[CONCURRENCY] Manager not initialized, proceeding without limit"
            )
            yield
            return

        # 取得請求上下文
        ctx = request_context_var.get()

        # 如果未啟用優先級或無上下文，使用 FIFO
        if not self._priority_enabled or ctx is None:
            async with self._acquire_fifo(backend, timeout):
                yield
            return

        # 優先級模式
        async with self._acquire_priority(backend, ctx, timeout):
            yield

        # 完成後增加計數
        ctx.increment()

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """
        取得並發統計資訊。

        Returns:
            Dict: 每個後端的統計資訊
        """
        return dict(self._stats)

    def get_available_slots(self) -> Dict[str, int]:
        """
        取得各後端可用的並發槽位數。

        Returns:
            Dict: 每個後端的可用槽位數
        """
        if not self._initialized:
            return {}

        return {backend: sem._value for backend, sem in self._semaphores.items()}

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """
        取得各後端的完整狀態資訊。

        Returns:
            Dict: 每個後端的狀態，包含：
                - limit: 最大並發數
                - available: 可用槽位數
                - in_progress: 正在進行的請求數
                - waiting: 正在排隊等待的請求數
                - total_acquired: 累計取得次數
                - total_released: 累計釋放次數
                - total_timeout: 累計逾時次數

        Example:
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
        取得所有後端的彙總狀態。

        Returns:
            Dict: 彙總資訊，包含：
                - total_in_progress: 所有後端正在進行的請求總數
                - total_waiting: 所有後端正在排隊的請求總數
                - by_backend: 各後端的簡要狀態

        Example:
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
        取得優先級排序統計資訊。

        Returns:
            Dict: 優先級相關統計，包含：
                - priority_enabled: 是否啟用
                - starvation_threshold: 飢餓門檻
                - queues: 各後端佇列狀態
                - active_requests: 活躍請求數

        Example:
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


# 全域單例
llm_concurrency = LLMConcurrencyManager()


async def with_llm_semaphore(
    coro_func: Callable[[], Awaitable[T]],
    backend: str = "default",
) -> T:
    """
    包裝 LLM 呼叫，自動處理 semaphore 獲取與釋放。

    使用 lambda 包裝 LLM 呼叫，避免 coroutine 提前建立。
    若啟用優先級模式且有請求上下文，會使用優先級排序。

    Args:
        coro_func: 返回 coroutine 的函數（例如 lambda: llm.ainvoke(...)）
        backend: semaphore 類型 (default/chat/responses/embedding)

    Returns:
        LLM 呼叫結果

    Example:
        # 之前
        result = await llm.ainvoke(messages)

        # 之後
        result = await with_llm_semaphore(
            lambda: llm.ainvoke(messages),
            backend="chat",
        )
    """
    async with llm_concurrency.acquire(backend):
        return await coro_func()
