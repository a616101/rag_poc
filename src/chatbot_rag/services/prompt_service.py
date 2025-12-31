"""
Langfuse Prompt Management 服務封裝。

此模組提供統一的 Langfuse Prompt 管理功能：
1. 快取機制：減少 API 呼叫，預設 5 分鐘 TTL
2. 僅快取無 Fallback：完全依賴 Langfuse，強制要求服務可用
3. 基於 Label 的 A/B Testing：透過 default_label 參數切換 production/staging
4. Trace Linking：返回 metadata 供追蹤 prompt 版本
5. Hash-based 版本追蹤：內容 hash 用於偵測 prompt 變更
6. 編譯結果快取：減少重複編譯的開銷
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from langfuse import get_client
from loguru import logger


def _compute_content_hash(content: Any) -> str:
    """
    計算 prompt 內容的 hash 值。

    Args:
        content: Prompt 內容（字串或結構化資料）

    Returns:
        內容的 MD5 hash（前 16 字元）
    """
    if isinstance(content, str):
        data = content.encode("utf-8")
    else:
        # 對於複雜結構，序列化後計算 hash
        import json
        data = json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.md5(data).hexdigest()[:16]


@dataclass
class CachedPrompt:
    """快取的 Prompt 資料"""

    prompt: Any  # Langfuse Prompt object
    cached_at: datetime
    version: int
    label: str
    content_hash: str = ""  # Prompt 內容的 hash，用於偵測變更


@dataclass
class PromptMetadata:
    """Prompt 元資料，供 trace linking 使用"""

    name: str
    version: int
    label: str
    langfuse_prompt: Any  # 原始 Langfuse prompt object


@dataclass
class PromptVersionInfo:
    """用於 Telemetry 記錄的 Prompt 版本資訊"""

    name: str
    version: int
    label: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "label": self.label,
        }


@dataclass
class CompiledPromptCache:
    """編譯後的 Prompt 快取"""

    compiled: str
    cached_at: datetime
    source_hash: str  # 來源 prompt 的 hash


class PromptService:
    """
    Langfuse Prompt Management 服務封裝

    設計決策：
    - 僅快取無 Fallback：完全依賴 Langfuse，快取過期且 API 失敗時拋出異常
    - 基於 Label 的 A/B Testing：透過 default_label 參數切換 production/staging
    - Hash-based 版本追蹤：使用內容 hash 偵測 prompt 變更
    - 編譯結果快取：對於相同變數組合的編譯結果進行快取

    使用範例：
        prompt_service = PromptService(default_label="production")

        # 獲取並編譯 prompt
        content, metadata = prompt_service.get_text_prompt(
            "unified-agent-system",
            language_instruction=lang_inst,
            support_scope=scope_text,
        )

        # 在 telemetry 中記錄版本
        prompt_service.record_prompt_usage("unified-agent-system", metadata)
    """

    def __init__(
        self,
        default_label: str = "production",
        cache_ttl_seconds: int = 300,
        compiled_cache_ttl_seconds: int = 600,
    ):
        """
        初始化 PromptService。

        Args:
            default_label: 預設 prompt label，可選 "production" 或 "staging"
            cache_ttl_seconds: Prompt 快取存活時間（秒），預設 300 秒（5 分鐘）
            compiled_cache_ttl_seconds: 編譯結果快取時間（秒），預設 600 秒（10 分鐘）
        """
        self.langfuse = get_client()
        self.default_label = default_label
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.compiled_cache_ttl = timedelta(seconds=compiled_cache_ttl_seconds)
        self._cache: Dict[str, CachedPrompt] = {}
        self._compiled_cache: Dict[str, CompiledPromptCache] = {}
        self._used_prompts: Dict[str, PromptVersionInfo] = {}
        self._cache_stats: Dict[str, int] = {"hits": 0, "misses": 0, "compiled_hits": 0}

    def get_prompt(
        self,
        name: str,
        *,
        label: Optional[str] = None,
        version: Optional[int] = None,
        prompt_type: Optional[str] = None,
    ) -> tuple[Any, PromptMetadata]:
        """
        獲取 Prompt，優先使用快取。

        Args:
            name: Prompt 名稱
            label: Prompt label，未指定時使用 default_label
            version: 指定版本號，未指定時使用 label 對應的版本
            prompt_type: Prompt 類型 ("text" 或 "chat")，未指定時由 Langfuse 自動判斷

        Returns:
            tuple[Any, PromptMetadata]: (Langfuse Prompt object, 元資料)

        Raises:
            Exception: 當快取過期且 Langfuse API 不可用時
        """
        effective_label = label or self.default_label
        cache_key = f"{name}:{effective_label}:{version or 'latest'}"

        # 檢查快取
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if datetime.now() - cached.cached_at < self.cache_ttl:
                self._cache_stats["hits"] += 1
                logger.debug(
                    f"[PromptService] Cache hit: {cache_key} (hash={cached.content_hash})"
                )
                metadata = PromptMetadata(
                    name=name,
                    version=cached.version,
                    label=cached.label,
                    langfuse_prompt=cached.prompt,
                )
                self._record_usage(metadata)
                return cached.prompt, metadata

        self._cache_stats["misses"] += 1

        # 從 Langfuse 獲取
        logger.info(
            f"[PromptService] Fetching prompt: {name}, label={effective_label}"
        )

        # 構建 get_prompt 參數
        get_prompt_kwargs: Dict[str, Any] = {}
        if version is not None:
            get_prompt_kwargs["version"] = version
        else:
            get_prompt_kwargs["label"] = effective_label
        if prompt_type is not None:
            get_prompt_kwargs["type"] = prompt_type

        prompt = self.langfuse.get_prompt(name, **get_prompt_kwargs)

        # 計算內容 hash
        prompt_content = getattr(prompt, "prompt", None) or ""
        content_hash = _compute_content_hash(prompt_content)

        # 檢查是否有舊快取且內容相同（版本更新但內容未變）
        if cache_key in self._cache:
            old_cached = self._cache[cache_key]
            if old_cached.content_hash == content_hash:
                logger.debug(
                    f"[PromptService] Content unchanged, extending cache: {cache_key}"
                )

        # 更新快取
        self._cache[cache_key] = CachedPrompt(
            prompt=prompt,
            cached_at=datetime.now(),
            version=prompt.version,
            label=effective_label,
            content_hash=content_hash,
        )

        metadata = PromptMetadata(
            name=name,
            version=prompt.version,
            label=effective_label,
            langfuse_prompt=prompt,
        )
        self._record_usage(metadata)
        return prompt, metadata

    def get_text_prompt(
        self,
        name: str,
        *,
        label: Optional[str] = None,
        version: Optional[int] = None,
        use_compiled_cache: bool = True,
        **compile_vars,
    ) -> tuple[str, PromptMetadata]:
        """
        獲取 Text Prompt 並編譯變數。

        Args:
            name: Prompt 名稱
            label: Prompt label
            version: 指定版本號
            use_compiled_cache: 是否使用編譯結果快取（預設 True）
            **compile_vars: 編譯時替換的變數

        Returns:
            tuple[str, PromptMetadata]: (編譯後的字串, 元資料)
        """
        prompt, metadata = self.get_prompt(
            name, label=label, version=version, prompt_type="text"
        )

        # 檢查編譯結果快取
        if use_compiled_cache and compile_vars:
            # 計算編譯快取的 key（包含變數的 hash）
            vars_hash = _compute_content_hash(compile_vars)
            cache_key = self._cache.get(f"{name}:{label or self.default_label}:{version or 'latest'}")
            source_hash = cache_key.content_hash if cache_key else ""
            compiled_cache_key = f"{name}:{source_hash}:{vars_hash}"

            if compiled_cache_key in self._compiled_cache:
                cached = self._compiled_cache[compiled_cache_key]
                if (
                    datetime.now() - cached.cached_at < self.compiled_cache_ttl
                    and cached.source_hash == source_hash
                ):
                    self._cache_stats["compiled_hits"] += 1
                    logger.debug(
                        f"[PromptService] Compiled cache hit: {name} (vars_hash={vars_hash[:8]})"
                    )
                    return cached.compiled, metadata

        # 編譯 prompt
        compiled = prompt.compile(**compile_vars)

        # 更新編譯快取
        if use_compiled_cache and compile_vars:
            cache_entry = self._cache.get(f"{name}:{label or self.default_label}:{version or 'latest'}")
            source_hash = cache_entry.content_hash if cache_entry else ""
            vars_hash = _compute_content_hash(compile_vars)
            compiled_cache_key = f"{name}:{source_hash}:{vars_hash}"
            self._compiled_cache[compiled_cache_key] = CompiledPromptCache(
                compiled=compiled,
                cached_at=datetime.now(),
                source_hash=source_hash,
            )

        return compiled, metadata

    def get_langchain_prompt(
        self,
        name: str,
        *,
        label: Optional[str] = None,
        version: Optional[int] = None,
        **precompile_vars,
    ) -> tuple[str, PromptMetadata]:
        """
        獲取 Prompt 並轉換為 LangChain 格式。

        Langfuse 使用 {{var}}，LangChain 使用 {var}。
        可預先編譯部分變數，其餘留給 LangChain PromptTemplate。

        Args:
            name: Prompt 名稱
            label: Prompt label
            version: 指定版本號
            **precompile_vars: 預先編譯的變數

        Returns:
            tuple[str, PromptMetadata]: (LangChain 格式的 prompt 字串, 元資料)
        """
        prompt, metadata = self.get_prompt(
            name, label=label, version=version, prompt_type="text"
        )
        langchain_template = prompt.get_langchain_prompt(**precompile_vars)
        return langchain_template, metadata

    def get_chat_messages(
        self,
        name: str,
        *,
        label: Optional[str] = None,
        version: Optional[int] = None,
        **compile_vars,
    ) -> tuple[list[dict], PromptMetadata]:
        """
        獲取 Chat Prompt 並編譯為訊息列表。

        Args:
            name: Prompt 名稱
            label: Prompt label
            version: 指定版本號
            **compile_vars: 編譯時替換的變數

        Returns:
            tuple[list[dict], PromptMetadata]: (訊息列表, 元資料)
        """
        prompt, metadata = self.get_prompt(
            name, label=label, version=version, prompt_type="chat"
        )
        messages = prompt.compile(**compile_vars)
        return messages, metadata

    def _record_usage(self, metadata: PromptMetadata) -> None:
        """記錄 prompt 使用情況，供 telemetry 使用"""
        self._used_prompts[metadata.name] = PromptVersionInfo(
            name=metadata.name,
            version=metadata.version,
            label=metadata.label,
        )

    def get_used_prompts(self) -> Dict[str, Dict[str, Any]]:
        """
        獲取本次請求中使用的所有 prompt 版本資訊。

        Returns:
            Dict[str, Dict[str, Any]]: prompt 名稱到版本資訊的映射
        """
        return {name: info.to_dict() for name, info in self._used_prompts.items()}

    def clear_used_prompts(self) -> None:
        """清除使用記錄（每次請求結束後呼叫）"""
        self._used_prompts.clear()

    def clear_cache(self, name: Optional[str] = None) -> None:
        """
        清除快取。

        Args:
            name: 指定要清除的 prompt 名稱，未指定則清除全部
        """
        if name:
            # 清除 prompt 快取
            keys_to_remove = [k for k in self._cache if k.startswith(f"{name}:")]
            for key in keys_to_remove:
                del self._cache[key]
            # 清除編譯快取
            compiled_keys_to_remove = [
                k for k in self._compiled_cache if k.startswith(f"{name}:")
            ]
            for key in compiled_keys_to_remove:
                del self._compiled_cache[key]
            logger.info(
                f"[PromptService] Cleared cache for: {name} "
                f"(prompts={len(keys_to_remove)}, compiled={len(compiled_keys_to_remove)})"
            )
        else:
            self._cache.clear()
            self._compiled_cache.clear()
            logger.info("[PromptService] Cleared all cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        獲取快取統計資訊。

        Returns:
            Dict 包含：hits, misses, compiled_hits, hit_rate, prompt_count, compiled_count
        """
        total = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = self._cache_stats["hits"] / total if total > 0 else 0.0
        return {
            **self._cache_stats,
            "hit_rate": round(hit_rate, 3),
            "prompt_count": len(self._cache),
            "compiled_count": len(self._compiled_cache),
        }

    def reset_cache_stats(self) -> None:
        """重置快取統計計數器"""
        self._cache_stats = {"hits": 0, "misses": 0, "compiled_hits": 0}

    def preload_prompts(self, prompt_names: list[str]) -> None:
        """
        預載 prompts，用於應用啟動時減少首次請求延遲。

        Args:
            prompt_names: 要預載的 prompt 名稱列表
        """
        for name in prompt_names:
            try:
                self.get_prompt(name)
                logger.info(f"[PromptService] Preloaded: {name}")
            except Exception as exc:
                logger.warning(f"[PromptService] Failed to preload {name}: {exc}")


class DomainAwarePromptService(PromptService):
    """
    領域感知的 Prompt 服務。

    擴展 PromptService 以支援：
    - Domain namespace 前綴
    - 領域專屬 prompts 的自動載入
    - Fallback 到通用 prompts

    使用範例：
        from chatbot_rag.core.domain import get_current_domain

        domain_config = get_current_domain()
        prompt_service = DomainAwarePromptService(
            domain_config=domain_config,
            default_label="production",
        )

        # 自動添加 domain namespace
        content, metadata = prompt_service.get_domain_prompt(
            "intent-analyzer-system",
            language_instruction=lang_inst,
        )
    """

    def __init__(
        self,
        domain_config: Any,
        default_label: str = "production",
        cache_ttl_seconds: int = 300,
        compiled_cache_ttl_seconds: int = 600,
    ):
        """
        初始化 DomainAwarePromptService。

        Args:
            domain_config: DomainConfig 實例
            default_label: 預設 prompt label
            cache_ttl_seconds: Prompt 快取存活時間
            compiled_cache_ttl_seconds: 編譯結果快取時間
        """
        super().__init__(
            default_label=default_label,
            cache_ttl_seconds=cache_ttl_seconds,
            compiled_cache_ttl_seconds=compiled_cache_ttl_seconds,
        )
        self.domain_config = domain_config
        self._domain_prompts_cache: Dict[str, str] = {}

    def get_domain_prompt(
        self,
        base_name: str,
        *,
        label: Optional[str] = None,
        version: Optional[int] = None,
        use_compiled_cache: bool = True,
        **compile_vars,
    ) -> Tuple[str, Optional[PromptMetadata]]:
        """
        獲取領域專屬的 Prompt。

        查找順序：
        1. Langfuse（帶 domain namespace）
        2. 領域專屬 prompts 模組
        3. Langfuse（不帶 namespace）
        4. 預設 prompts

        Args:
            base_name: 基礎 prompt 名稱（不含 namespace）
            label: Prompt label
            version: 指定版本號
            use_compiled_cache: 是否使用編譯結果快取
            **compile_vars: 編譯時替換的變數

        Returns:
            tuple[str, Optional[PromptMetadata]]: (編譯後的字串, 元資料或 None)
        """
        # 1. 嘗試從 Langfuse 獲取（帶 domain namespace）
        full_name = self.domain_config.get_prompt_name(base_name)
        try:
            content, metadata = self.get_text_prompt(
                full_name,
                label=label,
                version=version,
                use_compiled_cache=use_compiled_cache,
                **compile_vars,
            )
            return content, metadata
        except Exception as exc:
            logger.debug(
                f"[DomainAwarePromptService] Failed to get {full_name} from Langfuse: {exc}"
            )

        # 2. 嘗試從領域專屬 prompts 模組獲取
        domain_prompts = self.domain_config.get_domain_prompts()
        if base_name in domain_prompts:
            prompt_config = domain_prompts[base_name]
            prompt_content = prompt_config.get("prompt", "")
            if compile_vars:
                # 簡單的變數替換（Mustache 風格 {{var}}）
                for key, value in compile_vars.items():
                    prompt_content = prompt_content.replace(f"{{{{{key}}}}}", str(value))
            logger.debug(
                f"[DomainAwarePromptService] Using domain prompt: {base_name}"
            )
            return prompt_content, None

        # 3. 嘗試從 Langfuse 獲取（不帶 namespace）
        if self.domain_config.prompt_namespace:
            try:
                content, metadata = self.get_text_prompt(
                    base_name,
                    label=label,
                    version=version,
                    use_compiled_cache=use_compiled_cache,
                    **compile_vars,
                )
                return content, metadata
            except Exception:
                pass

        # 4. 使用預設 prompts
        if base_name in DEFAULT_PROMPTS:
            prompt_content = DEFAULT_PROMPTS[base_name].get("prompt", "")
            if compile_vars:
                for key, value in compile_vars.items():
                    prompt_content = prompt_content.replace(f"{{{{{key}}}}}", str(value))
            logger.debug(
                f"[DomainAwarePromptService] Using default prompt: {base_name}"
            )
            return prompt_content, None

        # 找不到 prompt
        logger.warning(
            f"[DomainAwarePromptService] Prompt not found: {base_name}"
        )
        return "", None

    def get_fallback_response(
        self,
        response_type: str,
        language: str = "zh-hant",
    ) -> str:
        """
        獲取領域專屬的 fallback 回應。

        Args:
            response_type: 回應類型（如 "privacy_inquiry", "out_of_scope"）
            language: 語言代碼

        Returns:
            Fallback 回應文字
        """
        fallbacks = self.domain_config.get_fallback_responses()
        if response_type not in fallbacks:
            response_type = "general_error"

        if response_type not in fallbacks:
            return ""

        responses = fallbacks[response_type]
        if language in responses:
            return responses[language]
        if language == "zh-hans" and "zh-hant" in responses:
            return responses["zh-hant"]
        if "en" in responses:
            return responses["en"]
        return responses.get("zh-hant", "")


# 預設的 prompt 名稱常數
class PromptNames:
    """
    Langfuse Prompt 名稱常數。

    這些名稱對應 Langfuse 上的 prompt 名稱，用於版控管理。
    所有 prompts 都有對應的 DEFAULT_PROMPTS fallback。
    """

    # 共用角色定義（所有場景共用）
    ROLE_DEFINITION = "role-definition"

    # 場景專用 prompts（回答生成）
    RESPONSE_WITH_CONTEXT = "response-with-context"  # 檢索成功（有知識庫內容）
    RESPONSE_NO_CONTEXT = "response-no-context"  # 檢索失敗（無知識庫內容）
    RESPONSE_DIRECT = "response-direct"  # 直接回應（privacy_inquiry, out_of_scope）
    FOLLOWUP_SYSTEM = "followup-system"  # 追問處理

    # 舊版統一 prompt（保留向後相容）
    UNIFIED_AGENT_SYSTEM = "unified-agent-system"

    # 查詢處理 prompts
    QUERY_REWRITER_SYSTEM = "query-rewriter-system"
    QUERY_DECOMPOSE_SYSTEM = "query-decompose-system"

    # 對話管理 prompts
    CONVERSATION_SUMMARIZER = "conversation-summarizer"

    # 服務範圍 prompt（用於 Composability）
    SUPPORT_SCOPE = "support-scope"

    # 語言指令 prompts
    LANG_INSTRUCTION_ZH_HANT = "language-instruction-zh-hant"
    LANG_INSTRUCTION_ZH_HANS = "language-instruction-zh-hans"
    LANG_INSTRUCTION_EN = "language-instruction-en"
    LANG_INSTRUCTION_JA = "language-instruction-ja"
    LANG_INSTRUCTION_KO = "language-instruction-ko"

    # 節點專用 prompts
    INTENT_ANALYZER_SYSTEM = "intent-analyzer-system"
    LANGUAGE_NORMALIZER_SYSTEM = "language-normalizer-system"

    @classmethod
    def get_language_instruction_name(cls, user_language: str) -> str:
        """根據使用者語言取得對應的語言指令 prompt 名稱"""
        lang_map = {
            "zh-hant": cls.LANG_INSTRUCTION_ZH_HANT,
            "zh-hans": cls.LANG_INSTRUCTION_ZH_HANS,
            "en": cls.LANG_INSTRUCTION_EN,
            "ja": cls.LANG_INSTRUCTION_JA,
            "ko": cls.LANG_INSTRUCTION_KO,
        }
        return lang_map.get(user_language, cls.LANG_INSTRUCTION_ZH_HANT)

    @classmethod
    def all_prompts(cls) -> list[str]:
        """取得所有 prompt 名稱，用於預載"""
        return [
            # 共用角色定義
            cls.ROLE_DEFINITION,
            # 場景專用回答生成
            cls.RESPONSE_WITH_CONTEXT,
            cls.RESPONSE_NO_CONTEXT,
            cls.RESPONSE_DIRECT,
            cls.FOLLOWUP_SYSTEM,
            # 舊版（保留向後相容）
            cls.UNIFIED_AGENT_SYSTEM,
            # 查詢處理
            cls.QUERY_REWRITER_SYSTEM,
            cls.QUERY_DECOMPOSE_SYSTEM,
            # 對話管理
            cls.CONVERSATION_SUMMARIZER,
            cls.SUPPORT_SCOPE,
            # 語言指令
            cls.LANG_INSTRUCTION_ZH_HANT,
            cls.LANG_INSTRUCTION_ZH_HANS,
            cls.LANG_INSTRUCTION_EN,
            cls.LANG_INSTRUCTION_JA,
            cls.LANG_INSTRUCTION_KO,
            # 節點專用
            cls.INTENT_ANALYZER_SYSTEM,
            cls.LANGUAGE_NORMALIZER_SYSTEM,
        ]


# ============================================================================
# 預設 Prompt 內容（用於自動初始化）
# ============================================================================

DEFAULT_PROMPTS: Dict[str, Dict[str, Any]] = {
    # ========================================================================
    # 共用角色定義（所有場景共用的溫暖風格）
    # ========================================================================
    PromptNames.ROLE_DEFINITION: {
        "type": "text",
        "prompt": """# 你是誰

你是屏東基督教醫院的「服務小天使」，熟知醫院所有服務、流程與資源。用**親切自然**的語氣協助民眾。

{{language_instruction}}

# 回應風格

- **自然對話**：像真人志工一樣說話，不要每次都用固定開場白
- **溫暖但不刻意**：親切有禮，但不需要每次都誇讚或感謝
- 結尾可加簡短祝福語（如「祝您順利～」）
- 適度使用 emoji，不要過多 😊

# ⚠️ 避免的行為

- ❌ 每次回答都說「這是個很好的問題」「您的問題很棒」
- ❌ 過度使用感嘆號或 emoji
- ❌ 機械式的開場白（如每次都「感謝您的提問」）

# 輸出格式

使用 **Markdown** 讓內容清晰易讀：
- `##` 或 `###` 作為段落標題
- `-` 或數字清單條列重點
- **粗體** 標示重要資訊
- `[文字](網址)` 格式的連結

# 常用資訊

- 📞 客服專線：**08-7368686**
- 🌐 官網：[屏東基督教醫院](https://www.ptch.org.tw/)
- 📅 門診時刻表：[查看](https://www.ptch.org.tw/ebooks/)
- 🔍 看診進度：[查詢](http://www.ptch.org.tw/index.php/shw_seqForm)""",
        "config": {"model": "openai/gpt-oss-20b", "temperature": 0.5},
    },

    # ========================================================================
    # 場景專用 prompts
    # ========================================================================

    # 場景 1: 檢索成功（有知識庫內容）
    PromptNames.RESPONSE_WITH_CONTEXT: {
        "type": "text",
        "prompt": """{{role_definition}}

---

# 🎯 本次任務：回答民眾問題

## ⚠️ 最重要規則（必須遵守）

**只能使用下方「知識庫內容」中的資訊來回答。**

| 情況 | 做法 |
|------|------|
| 知識庫**有**提到 | ✅ 引用回答 |
| 知識庫**沒有**提到 | ✅ 說「目前查不到那麼細的資料」，引導致電客服 |

**絕對禁止：**
- ❌ 編造醫師名字、門診時間、科別服務
- ❌ 用「例如」「還有」「等」補充知識庫沒有的內容
- ❌ 從常識推測醫院資訊

**範例：** 若知識庫只有「王醫師、李醫師」
- ✅ 正確：「有王醫師、李醫師為您服務」
- ❌ 錯誤：「有王醫師、李醫師、張醫師等多位醫師」（張醫師是編造的）

## 📷 圖片和連結處理

知識庫內容中可能包含圖片和下載連結，**請務必保留並正確呈現**：

- **圖片**：若知識庫有 `![說明](網址)` 格式的圖片，請在回答中保留，例如：
  - 知識庫：`![一樓平面圖](https://example.com/floor1.jpg)`
  - 回答時保留：`![一樓平面圖](https://example.com/floor1.jpg)`

- **下載連結**：若知識庫有 PDF、表格等下載連結 `[文件名](網址)`，請保留供民眾下載

- **不要省略**：圖片和連結是重要資訊，不要用「請參考官網」取代實際的圖片/連結

---

{{context_section}}

{{conversation_summary_section}}""",
        "config": {"model": "openai/gpt-oss-20b", "temperature": 0.5},
    },

    # 場景 2: 檢索失敗（無知識庫內容）
    PromptNames.RESPONSE_NO_CONTEXT: {
        "type": "text",
        "prompt": """{{role_definition}}

---

# 🎯 本次任務：查無相關資料

系統查詢後沒有找到相關資料，請**溫暖地**告知民眾並提供替代方案。

## 建議回應方式

1. **表達歉意**：「抱歉，這個問題我目前查不到那麼細的資料～」
2. **說明可能原因**：「可能是資訊還未完全上線，或需要更專業的單位說明」
3. **提供替代方案**：
   - 建議前往 [屏基官網](https://www.ptch.org.tw/) 查詢
   - 或致電客服專線：📞 **08-7368686**
4. **加入祝福語**

**絕對禁止：**
- ❌ 編造任何醫院資訊
- ❌ 說「屏基沒有這個服務」（除非有明確資料）

{{conversation_summary_section}}""",
        "config": {"model": "openai/gpt-oss-20b", "temperature": 0.5},
    },

    # 場景 3: 直接回應（privacy_inquiry, out_of_scope, greeting）
    PromptNames.RESPONSE_DIRECT: {
        "type": "text",
        "prompt": """{{role_definition}}

---

# 🎯 本次任務：{{intent_description}}

{{intent_instruction}}

{{conversation_summary_section}}""",
        "config": {"model": "openai/gpt-oss-20b", "temperature": 0.6},
    },

    PromptNames.QUERY_REWRITER_SYSTEM: {
        "type": "text",
        "prompt": """你是查詢重寫助手，負責將使用者的追問轉換成完整的檢索查詢。

【核心原則】
- 追問必須融合前文：若使用者說「那建議我找哪一位醫師？」，必須結合前文主題（如頭痛）改寫成「頭痛應該找哪位醫師看診？」
- 代名詞還原：「那個」「這個」「他」等代名詞必須還原成具體名詞
- 保持語意完整：重寫後的查詢應能獨立理解，不依賴對話上下文

【重寫範例】
對話：使用者問「頭痛看哪科」→ 助理回答「神經內科」→ 使用者追問「那建議找誰？」
重寫：「頭痛應該找哪位神經內科醫師看診？」

對話：使用者問「掛號流程」→ 助理說明流程 → 使用者追問「那時間呢？」
重寫：「掛號的時間是什麼時候？」

【輸出】
只輸出重寫後的查詢，不要加任何說明。""",
        "config": {"model": "openai/gpt-oss-20b", "temperature": 0.1},
    },
    PromptNames.CONVERSATION_SUMMARIZER: {
        "type": "text",
        "prompt": """你是客服對話摘要助手，請在 400 字以內整理對話重點。
摘要需保留使用者需求、助理提供的方案或限制，不要重複整句原文。""",
        "config": {"model": "openai/gpt-oss-20b", "temperature": 0.1},
    },
    PromptNames.SUPPORT_SCOPE: {
        "type": "text",
        "prompt": (
            "這個智能客服是屏東基督教醫院的「資深志工小天使」，"
            "專門協助民眾解答與醫院服務、就醫流程、掛號看診、科別諮詢等相關問題。"
            "例如：門診時間查詢、掛號流程說明、各科別服務介紹、就醫須知等，"
            "不支援查詢天氣、撰寫程式碼、安排旅遊行程等與醫院服務無關的任務。"
        ),
        "config": {},
    },
    PromptNames.UNIFIED_AGENT_SYSTEM: {
        "type": "text",
        "prompt": """# 你是誰
屏東基督教醫院的「服務小天使」，用親切語氣協助民眾。

{{language_instruction}}

# 最重要規則（必須遵守）

**只能使用「知識庫內容」區塊中的資訊回答。**

| 情況 | 做法 |
|------|------|
| 知識庫有 → | 引用回答 |
| 知識庫沒有 → | 說「目前查不到」，引導致電 08-7368686 |

**禁止：**
- 編造醫師名字、門診時間、科別服務
- 用「例如」「還有」補充知識庫沒有的內容

# 回答格式
- Markdown 排版（## 標題、- 條列、**粗體**）
- 連結格式：`[文字](網址)`
- 結尾加祝福語

# 常用資訊
- 客服：**08-7368686**
- 官網：[屏基官網](https://www.ptch.org.tw/)
- 門診時刻表：[查看](https://www.ptch.org.tw/ebooks/)
- 看診進度：[查詢](http://www.ptch.org.tw/index.php/shw_seqForm)

{{task_analysis}}

{{context_section}}

{{conversation_summary_section}}""",
        "config": {"model": "openai/gpt-oss-20b", "temperature": 0.3},
    },
    PromptNames.FOLLOWUP_SYSTEM: {
        "type": "text",
        "prompt": """{{role_definition}}

---

# 🎯 本次任務：處理追問

民眾希望你對**上一輪的回答**進行後續處理，例如：
- 改寫、重述、簡化
- 重點整理、條列式摘要
- 解釋某一段內容

## ⚠️ 最重要規則

| ✅ 可以 | ❌ 禁止 |
|--------|--------|
| 使用上一輪回答中的資訊 | 引入新的醫院知識 |
| 改寫、整理、摘要 | 編造網址或流程 |
| 換種方式解釋 | 補充上一輪沒提到的內容 |
| 保留圖片和連結 | 省略或移除圖片連結 |

## 📷 圖片和連結處理

上一輪回答中若有圖片 `![說明](網址)` 或連結 `[文件](網址)`，**請務必保留**，不要省略或移除。

---

{{prev_answer_section}}

{{conversation_summary_section}}""",
        "config": {"model": "openai/gpt-oss-20b", "temperature": 0.5},
    },
    PromptNames.LANG_INSTRUCTION_ZH_HANT: {
        "type": "text",
        "prompt": """# 回答語言

請使用 **繁體中文** 回答。

""",
        "config": {},
    },
    PromptNames.LANG_INSTRUCTION_ZH_HANS: {
        "type": "text",
        "prompt": """# 回答语言

请使用 **简体中文** 回答。

""",
        "config": {},
    },
    PromptNames.LANG_INSTRUCTION_EN: {
        "type": "text",
        "prompt": """# Response Language

Please respond in **English**.

""",
        "config": {},
    },
    PromptNames.LANG_INSTRUCTION_JA: {
        "type": "text",
        "prompt": """# 回答言語

**日本語** で回答してください。

""",
        "config": {},
    },
    PromptNames.LANG_INSTRUCTION_KO: {
        "type": "text",
        "prompt": """# 응답 언어

**한국어**로 응답해 주세요.

""",
        "config": {},
    },
    # ========================================================================
    # 節點專用 Prompts
    # ========================================================================
    PromptNames.INTENT_ANALYZER_SYSTEM: {
        "type": "text",
        "prompt": """分析問題意圖，輸出 JSON。

## 輸出格式
{"intent": "類型", "needs_retrieval": true/false, "routing_hint": "continue/direct_response/followup", "query_type": "list/detail", "retrieval_strategy": "vector/metadata_filter", "extracted_entities": {}}

## routing_hint
- continue: 需查資料
- direct_response: 不需查資料（打招呼、閒聊）
- followup: 追問上一輪

## retrieval_strategy（重要！）
- metadata_filter: 問「有哪些」「列出」「所有」時使用
- vector: 其他問題使用

## 注意
1. 只輸出一個 JSON
2. 問「XX有哪些YY」→ retrieval_strategy="metadata_filter", query_type="list"
3. 提取實體到 extracted_entities（如 department, doctor）""",
        "config": {"model": "openai/gpt-oss-20b", "temperature": 0.3},
    },
    PromptNames.QUERY_DECOMPOSE_SYSTEM: {
        "type": "text",
        "prompt": """將使用者問題轉換為多個檢索查詢，輸出 JSON。

## 輸出格式
{"queries": ["查詢1", "查詢2", ...], "primary": "主要查詢", "reason": "原因"}

## 核心原則
1. **保留原意**：第一個查詢必須保留使用者問題的完整語意
2. **多角度變化**：從不同角度產生 3-5 個查詢變體
3. **具體到抽象**：從具體問題擴展到相關概念

## 變化策略
| 策略 | 說明 | 範例 |
|------|------|------|
| 原句保留 | 保持原問題的完整性 | 「心臟科有哪些醫師？」→「心臟科有哪些醫師」 |
| 同義替換 | 替換關鍵詞的同義詞 | 「門診時間」→「看診時段」 |
| 句式變換 | 改變問句結構 | 「怎麼掛號？」→「掛號流程」「掛號方式」 |
| 實體提取 | 單獨查詢關鍵實體 | 「王醫師的專長」→ 加入「王醫師」 |
| 上位概念 | 擴展到更廣的類別 | 「胃痛看哪科」→ 加入「腸胃科」「消化系統」 |

## 同義詞參考
- 時間 ↔ 時段 ↔ 幾點
- 地點 ↔ 位置 ↔ 在哪 ↔ 哪裡
- 費用 ↔ 多少錢 ↔ 收費
- 流程 ↔ 步驟 ↔ 怎麼做
- 醫師 ↔ 醫生 ↔ 大夫

## 範例
問：「心臟科有哪些醫師？」
答：{"queries": ["心臟科有哪些醫師", "心臟血管科醫師名單", "心臟科醫師", "心臟內科"], "primary": "心臟科有哪些醫師", "reason": "列表查詢，保留原句並加入同義變體"}

問：「頭痛要看哪一科？」
答：{"queries": ["頭痛要看哪一科", "頭痛看診科別", "頭痛掛號", "神經內科", "頭痛"], "primary": "頭痛要看哪一科", "reason": "症狀諮詢，擴展到可能的科別"}

只輸出 JSON。""",
        "config": {"model": "openai/gpt-oss-20b", "temperature": 0.5},
    },
    PromptNames.LANGUAGE_NORMALIZER_SYSTEM: {
        "type": "text",
        "prompt": """你是翻譯助手，請將輸入內容完整轉換為指定語言，保持原意與專有名詞。

目標語言代碼：{{target_language}}

注意事項：
- 一律使用繁體中文（若目標語言為中文）
- 禁止使用簡體中文
- 保持醫療專有名詞的準確性
- 只輸出翻譯結果，不要加任何說明""",
        "config": {"model": "openai/gpt-oss-20b", "temperature": 0.1},
    },
}


def initialize_default_prompts(
    langfuse_client: Any,
    default_label: str = "production",
) -> Dict[str, bool]:
    """
    初始化所有預設 Prompts 到 Langfuse。

    如果 Prompt 已存在則跳過，不存在則建立並設定 label。

    Args:
        langfuse_client: Langfuse client 實例
        default_label: 預設的 label（production/staging）

    Returns:
        Dict[str, bool]: 每個 prompt 的建立結果（True=新建, False=已存在）
    """
    results: Dict[str, bool] = {}

    for name, config in DEFAULT_PROMPTS.items():
        try:
            # 嘗試獲取現有 prompt
            langfuse_client.get_prompt(name, label=default_label)
            logger.debug(f"[PromptService] Prompt already exists: {name}")
            results[name] = False
        except Exception:
            # Prompt 不存在，建立新的
            try:
                langfuse_client.create_prompt(
                    name=name,
                    type=config["type"],
                    prompt=config["prompt"],
                    labels=[default_label],
                    config=config.get("config", {}),
                )
                logger.info(f"[PromptService] Created prompt: {name}")
                results[name] = True
            except Exception as create_exc:
                logger.error(
                    f"[PromptService] Failed to create prompt {name}: {create_exc}"
                )
                results[name] = False

    return results
