"""
領域模組。

提供多場域支援，每個子模組定義特定領域的 prompts 和 fallback 回應。
"""

from chatbot_rag.core.domain import (
    DomainConfig,
    get_domain_config,
    get_current_domain,
    HOSPITAL_DOMAIN,
    GENERIC_DOMAIN,
)

__all__ = [
    "DomainConfig",
    "get_domain_config",
    "get_current_domain",
    "HOSPITAL_DOMAIN",
    "GENERIC_DOMAIN",
]
