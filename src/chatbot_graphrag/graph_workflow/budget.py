"""
循環預算管理

管理檢索循環的計算預算，包含：
- 最大迭代次數
- 查詢計數限制
- Token 限制
- 牆鐘時間限制

預算管理器用於控制 Agentic RAG 的資源消耗，
確保查詢在合理的 SLO（服務水平目標）內完成。
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

from chatbot_graphrag.core.constants import DEFAULT_LOOP_BUDGET

logger = logging.getLogger(__name__)


@dataclass
class BudgetSnapshot:
    """
    特定時間點的預算狀態快照。

    用於記錄和分析預算使用情況。
    """

    loops_used: int  # 已使用的循環數
    loops_remaining: int  # 剩餘循環數
    queries_used: int  # 已使用的查詢數
    queries_remaining: int  # 剩餘查詢數
    tokens_used: int  # 已使用的 token 數
    tokens_remaining: int  # 剩餘 token 數
    time_elapsed_ms: float  # 已消耗時間（毫秒）
    time_remaining_ms: float  # 剩餘時間（毫秒）
    is_exhausted: bool  # 是否已耗盡


class BudgetManager:
    """
    管理 GraphRAG 查詢的計算預算。

    追蹤：
    - 循環迭代（最多 2-4 次）
    - 生成的子查詢（最多 8 個）
    - 累積的上下文 token（最多 12000）
    - 牆鐘時間（最多 15-20 秒）

    這確保查詢在合理的 SLO 內完成，
    同時允許 Agentic 循環獲取足夠的證據。
    """

    def __init__(
        self,
        max_loops: int = DEFAULT_LOOP_BUDGET["max_loops"],
        max_queries: int = DEFAULT_LOOP_BUDGET["max_new_queries"],
        max_tokens: int = DEFAULT_LOOP_BUDGET["max_context_tokens"],
        max_time_seconds: float = DEFAULT_LOOP_BUDGET["max_wall_time_seconds"],
    ):
        """使用限制初始化預算。"""
        self.max_loops = max_loops
        self.max_queries = max_queries
        self.max_tokens = max_tokens
        self.max_time_seconds = max_time_seconds

        # Current usage
        self._loops_used = 0
        self._queries_used = 0
        self._tokens_used = 0
        self._start_time = time.time()

        # Tracking
        self._checkpoints: list[BudgetSnapshot] = []

    def reset(self) -> None:
        """重設預算計數器。"""
        self._loops_used = 0
        self._queries_used = 0
        self._tokens_used = 0
        self._start_time = time.time()
        self._checkpoints = []
        logger.debug("預算已重設")

    def use_loop(self) -> bool:
        """
        嘗試使用一次循環迭代。

        Returns:
            True 如果循環可用，False 如果已耗盡
        """
        if self._loops_used >= self.max_loops:
            logger.warning(f"循環預算已耗盡 ({self._loops_used}/{self.max_loops})")
            return False

        self._loops_used += 1
        logger.debug(f"循環已使用: {self._loops_used}/{self.max_loops}")
        return True

    def use_queries(self, count: int = 1) -> bool:
        """
        嘗試使用查詢配額。

        Returns:
            True 如果查詢可用，False 如果會超出限制
        """
        if self._queries_used + count > self.max_queries:
            logger.warning(
                f"查詢預算將超出 ({self._queries_used + count}/{self.max_queries})"
            )
            return False

        self._queries_used += count
        logger.debug(f"查詢已使用: {self._queries_used}/{self.max_queries}")
        return True

    def add_tokens(self, count: int) -> bool:
        """
        增加 token 到預算。

        Returns:
            True 如果在預算內，False 如果超出
        """
        self._tokens_used += count
        exceeded = self._tokens_used > self.max_tokens
        if exceeded:
            logger.warning(
                f"Token 預算已超出 ({self._tokens_used}/{self.max_tokens})"
            )
        else:
            logger.debug(f"Token 已使用: {self._tokens_used}/{self.max_tokens}")
        return not exceeded

    @property
    def elapsed_time_ms(self) -> float:
        """取得已消耗時間（毫秒）。"""
        return (time.time() - self._start_time) * 1000

    @property
    def elapsed_time_seconds(self) -> float:
        """取得已消耗時間（秒）。"""
        return time.time() - self._start_time

    def is_time_exceeded(self) -> bool:
        """檢查時間預算是否超出。"""
        return self.elapsed_time_seconds >= self.max_time_seconds

    def is_exhausted(self) -> bool:
        """檢查是否有任何預算耗盡。"""
        return (
            self._loops_used >= self.max_loops
            or self._queries_used >= self.max_queries
            or self._tokens_used >= self.max_tokens
            or self.is_time_exceeded()
        )

    def can_continue(self) -> bool:
        """檢查是否有預算繼續。"""
        return not self.is_exhausted()

    def get_exhaustion_reason(self) -> str:
        """取得預算耗盡的原因。"""
        reasons = []

        if self._loops_used >= self.max_loops:
            reasons.append(f"max_loops ({self._loops_used}/{self.max_loops})")

        if self._queries_used >= self.max_queries:
            reasons.append(f"max_queries ({self._queries_used}/{self.max_queries})")

        if self._tokens_used >= self.max_tokens:
            reasons.append(f"max_tokens ({self._tokens_used}/{self.max_tokens})")

        if self.is_time_exceeded():
            reasons.append(
                f"max_time ({self.elapsed_time_seconds:.1f}s/{self.max_time_seconds}s)"
            )

        return ", ".join(reasons) if reasons else "not_exhausted"

    def snapshot(self) -> BudgetSnapshot:
        """取得當前預算狀態的快照。"""
        snapshot = BudgetSnapshot(
            loops_used=self._loops_used,
            loops_remaining=self.max_loops - self._loops_used,
            queries_used=self._queries_used,
            queries_remaining=self.max_queries - self._queries_used,
            tokens_used=self._tokens_used,
            tokens_remaining=max(0, self.max_tokens - self._tokens_used),
            time_elapsed_ms=self.elapsed_time_ms,
            time_remaining_ms=max(0, (self.max_time_seconds * 1000) - self.elapsed_time_ms),
            is_exhausted=self.is_exhausted(),
        )
        self._checkpoints.append(snapshot)
        return snapshot

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典，用於狀態序列化。"""
        return {
            "max_loops": self.max_loops,
            "max_queries": self.max_queries,
            "max_tokens": self.max_tokens,
            "max_time_seconds": self.max_time_seconds,
            "loops_used": self._loops_used,
            "queries_used": self._queries_used,
            "tokens_used": self._tokens_used,
            "elapsed_time_ms": self.elapsed_time_ms,
            "is_exhausted": self.is_exhausted(),
            "exhaustion_reason": self.get_exhaustion_reason(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BudgetManager":
        """從字典建立。"""
        manager = cls(
            max_loops=data.get("max_loops", DEFAULT_LOOP_BUDGET["max_loops"]),
            max_queries=data.get("max_queries", DEFAULT_LOOP_BUDGET["max_new_queries"]),
            max_tokens=data.get("max_tokens", DEFAULT_LOOP_BUDGET["max_context_tokens"]),
            max_time_seconds=data.get(
                "max_time_seconds", DEFAULT_LOOP_BUDGET["max_wall_time_seconds"]
            ),
        )
        manager._loops_used = data.get("loops_used", 0)
        manager._queries_used = data.get("queries_used", 0)
        manager._tokens_used = data.get("tokens_used", 0)
        return manager


def estimate_tokens(text: str) -> int:
    """
    估計文字的 token 數量。

    使用簡單的啟發式方法：
    - 英文約每 4 個字元 1 個 token
    - 中文約每 1.5 個字元 1 個 token
    """
    # 計算中文字元數
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other_chars = len(text) - chinese_chars

    # 估計 token 數
    chinese_tokens = chinese_chars / 1.5
    other_tokens = other_chars / 4

    return int(chinese_tokens + other_tokens)


def create_budget(
    max_loops: int = DEFAULT_LOOP_BUDGET["max_loops"],
    max_queries: int = DEFAULT_LOOP_BUDGET["max_new_queries"],
    max_tokens: int = DEFAULT_LOOP_BUDGET["max_context_tokens"],
    max_time_seconds: float = DEFAULT_LOOP_BUDGET["max_wall_time_seconds"],
) -> BudgetManager:
    """使用指定限制建立新的預算管理器。"""
    return BudgetManager(
        max_loops=max_loops,
        max_queries=max_queries,
        max_tokens=max_tokens,
        max_time_seconds=max_time_seconds,
    )
