"""
遙測採樣模組

提供 Langfuse 遙測資料的採樣機制，
減少高流量時的遙測開銷。
"""

import random
from typing import Optional

from loguru import logger


class TelemetrySampler:
    """
    遙測採樣器

    控制哪些請求的遙測資料會被記錄，
    以減少 Langfuse API 呼叫次數和網路開銷。
    """

    def __init__(self, sample_rate: float = 1.0):
        """
        初始化採樣器。

        Args:
            sample_rate: 採樣率（0.0 ~ 1.0），
                        1.0 = 100% 記錄，0.1 = 10% 記錄
        """
        self._sample_rate = max(0.0, min(1.0, sample_rate))
        self._total_checks = 0
        self._sampled_count = 0

    @property
    def sample_rate(self) -> float:
        """取得當前採樣率。"""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: float) -> None:
        """設定採樣率。"""
        self._sample_rate = max(0.0, min(1.0, value))
        logger.info(f"[TELEMETRY] Sample rate updated to {self._sample_rate:.2%}")

    def should_sample(self) -> bool:
        """
        判斷是否應該採樣此請求。

        Returns:
            bool: True 表示應該記錄遙測資料，False 表示跳過
        """
        self._total_checks += 1

        if self._sample_rate >= 1.0:
            self._sampled_count += 1
            return True

        if self._sample_rate <= 0.0:
            return False

        result = random.random() < self._sample_rate
        if result:
            self._sampled_count += 1
        return result

    def get_stats(self) -> dict:
        """
        取得採樣統計資訊。

        Returns:
            dict: 包含總檢查次數、採樣次數、實際採樣率
        """
        actual_rate = (
            self._sampled_count / self._total_checks
            if self._total_checks > 0
            else 0.0
        )
        return {
            "sample_rate": self._sample_rate,
            "total_checks": self._total_checks,
            "sampled_count": self._sampled_count,
            "actual_rate": actual_rate,
        }

    def reset_stats(self) -> None:
        """重置統計資訊。"""
        self._total_checks = 0
        self._sampled_count = 0


def create_sampler(sample_rate: Optional[float] = None) -> TelemetrySampler:
    """
    建立遙測採樣器。

    Args:
        sample_rate: 採樣率，None 表示使用設定值

    Returns:
        TelemetrySampler: 採樣器實例
    """
    from chatbot_rag.core.config import settings

    # 如果沒有指定，使用設定值
    rate = sample_rate if sample_rate is not None else settings.langfuse_sample_rate
    return TelemetrySampler(sample_rate=rate)


# 全域單例（預設 100% 採樣，會在 lifespan 中從 settings 更新）
telemetry_sampler = TelemetrySampler(sample_rate=1.0)


def initialize_telemetry_sampler() -> None:
    """
    從 settings 初始化遙測採樣器。

    應在 FastAPI lifespan 中呼叫，以確保使用正確的設定值。
    """
    from chatbot_rag.core.config import settings

    telemetry_sampler.sample_rate = settings.langfuse_sample_rate
    logger.info(
        f"[TELEMETRY] Sampler initialized with rate: {telemetry_sampler.sample_rate:.2%}"
    )
