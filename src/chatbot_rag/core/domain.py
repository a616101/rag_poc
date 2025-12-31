"""
領域設定模組。

提供領域無關的架構設定，支援多場域切換（醫院、銀行、電商等）。
透過更換 prompts 即可切換場域，無需修改程式碼。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import importlib

from loguru import logger


@dataclass
class DomainConfig:
    """
    領域設定資料類別。

    用於定義特定領域（場域）的配置，包括：
    - 領域識別碼
    - Langfuse prompt namespace
    - 預設 prompts 模組路徑
    """

    domain_id: str
    """領域識別碼，如 "hospital", "bank", "ecommerce" """

    prompt_namespace: str
    """Langfuse prompt namespace 前綴，如 "ptch", "bank" """

    default_prompts_module: str = ""
    """預設 prompts 的 Python 模組路徑，如 "chatbot_rag.domains.hospital.prompts" """

    fallback_responses_module: str = ""
    """Fallback 回應的 Python 模組路徑，如 "chatbot_rag.domains.hospital.fallbacks" """

    display_name: str = ""
    """領域顯示名稱，用於 UI 和日誌"""

    description: str = ""
    """領域描述"""

    _prompts_cache: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _fallbacks_cache: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def __post_init__(self):
        """初始化後處理"""
        if not self.display_name:
            self.display_name = self.domain_id.title()

    def get_prompt_name(self, base_name: str) -> str:
        """
        取得帶有 namespace 前綴的 prompt 名稱。

        Args:
            base_name: 基礎 prompt 名稱，如 "intent-analyzer-system"

        Returns:
            帶有 namespace 的完整 prompt 名稱，如 "ptch-intent-analyzer-system"
        """
        if self.prompt_namespace:
            return f"{self.prompt_namespace}-{base_name}"
        return base_name

    def get_domain_prompts(self) -> Dict[str, Any]:
        """
        取得領域專屬的 prompts 字典。

        Returns:
            領域專屬 prompts，若無法載入則返回空字典
        """
        if self._prompts_cache is not None:
            return self._prompts_cache

        if not self.default_prompts_module:
            self._prompts_cache = {}
            return self._prompts_cache

        try:
            module = importlib.import_module(self.default_prompts_module)
            self._prompts_cache = getattr(module, "DOMAIN_PROMPTS", {})
            logger.debug(
                f"[DomainConfig] Loaded {len(self._prompts_cache)} prompts from {self.default_prompts_module}"
            )
        except ImportError as exc:
            logger.warning(
                f"[DomainConfig] Failed to load prompts module {self.default_prompts_module}: {exc}"
            )
            self._prompts_cache = {}

        return self._prompts_cache

    def get_fallback_responses(self) -> Dict[str, Any]:
        """
        取得領域專屬的 fallback 回應字典。

        Returns:
            領域專屬 fallback 回應，若無法載入則返回空字典
        """
        if self._fallbacks_cache is not None:
            return self._fallbacks_cache

        if not self.fallback_responses_module:
            self._fallbacks_cache = {}
            return self._fallbacks_cache

        try:
            module = importlib.import_module(self.fallback_responses_module)
            self._fallbacks_cache = getattr(module, "FALLBACK_RESPONSES", {})
            logger.debug(
                f"[DomainConfig] Loaded {len(self._fallbacks_cache)} fallbacks from {self.fallback_responses_module}"
            )
        except ImportError as exc:
            logger.warning(
                f"[DomainConfig] Failed to load fallbacks module {self.fallback_responses_module}: {exc}"
            )
            self._fallbacks_cache = {}

        return self._fallbacks_cache


# ============================================================================
# 預設領域設定
# ============================================================================

# 屏東基督教醫院領域設定
HOSPITAL_DOMAIN = DomainConfig(
    domain_id="hospital",
    prompt_namespace="ptch",
    default_prompts_module="chatbot_rag.domains.hospital.prompts",
    fallback_responses_module="chatbot_rag.domains.hospital.fallbacks",
    display_name="屏東基督教醫院",
    description="屏東基督教醫院智能客服",
)

# 通用領域設定（無特定場域）
GENERIC_DOMAIN = DomainConfig(
    domain_id="generic",
    prompt_namespace="",
    display_name="通用",
    description="通用 RAG 系統",
)


def get_domain_config(domain_id: str = "hospital") -> DomainConfig:
    """
    根據領域識別碼取得對應的領域設定。

    Args:
        domain_id: 領域識別碼

    Returns:
        對應的 DomainConfig 實例
    """
    domain_registry = {
        "hospital": HOSPITAL_DOMAIN,
        "generic": GENERIC_DOMAIN,
    }

    if domain_id in domain_registry:
        return domain_registry[domain_id]

    logger.warning(f"[DomainConfig] Unknown domain_id: {domain_id}, using generic")
    return GENERIC_DOMAIN


def get_current_domain() -> DomainConfig:
    """
    取得當前領域設定（從環境變數讀取）。

    環境變數 DOMAIN_ID 控制使用的領域，預設為 "hospital"。

    Returns:
        當前的 DomainConfig 實例
    """
    import os
    domain_id = os.environ.get("DOMAIN_ID", "hospital")
    return get_domain_config(domain_id)
