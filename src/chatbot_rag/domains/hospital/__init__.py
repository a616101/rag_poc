"""
屏東基督教醫院領域模組。

包含醫院場域專屬的 prompts 和 fallback 回應。
"""

from .prompts import DOMAIN_PROMPTS
from .fallbacks import FALLBACK_RESPONSES

__all__ = [
    "DOMAIN_PROMPTS",
    "FALLBACK_RESPONSES",
]
