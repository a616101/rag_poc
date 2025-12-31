"""
語言判斷、查詢翻譯與 LLM usage 擷取的共用工具。
"""

from __future__ import annotations

from typing import Any, Optional
import time

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig
from loguru import logger

from chatbot_rag.core.config import settings
from chatbot_rag.core.concurrency import with_llm_semaphore
from chatbot_rag.llm.factory import create_responses_llm, create_chat_completion_llm


def detect_preferred_language(
    question: str, conversation_history: Optional[list] = None
) -> str:
    """
    根據問題內容（以及必要時的對話歷史）簡單推斷主要語言。

    回傳值：
    - "zh-hant"：以「繁體中文」為主（偏向 zh-TW / zh-Hant）
    - "zh-hans"：以「簡體中文」為主（偏向 zh-CN / zh-Hans）
    - "ja"：以「日文」為主（含平假名、片假名）
    - "ko"：以「韓文」為主（含韓文字母）
    - "en"：預設為英文（或其他非 CJK 語言統一視為英文回答風格）

    檢測優先順序：
    1. 先檢測日文假名（平假名、片假名）→ 如果有，判定為日文
    2. 再檢測韓文字母 → 如果有，判定為韓文
    3. 最後檢測中文（繁體/簡體）
    4. 如果都沒有 CJK 字元，判定為英文
    """

    # 少量代表性「只在簡體常見」與「只在繁體常見」的字元集合，用來粗略判斷變體。
    simplified_only = set("们国这这还没来问业车里时间进广发体东门龙云电气学后师")
    traditional_only = set("們國這還沒來問業車裡時間進廣發體東門龍雲電氣學後師")

    def _detect_language(text: str) -> str:
        """根據文字內容判斷語言"""
        has_hiragana = False
        has_katakana = False
        has_hangul = False
        has_cjk = False
        simp_count = 0
        trad_count = 0

        for ch in text:
            # 平假名範圍: U+3040 - U+309F
            if "\u3040" <= ch <= "\u309f":
                has_hiragana = True
            # 片假名範圍: U+30A0 - U+30FF
            elif "\u30a0" <= ch <= "\u30ff":
                has_katakana = True
            # 韓文字母範圍: U+AC00 - U+D7AF (常用), U+1100 - U+11FF (Jamo)
            elif "\uac00" <= ch <= "\ud7af" or "\u1100" <= ch <= "\u11ff":
                has_hangul = True
            # CJK 統一漢字範圍
            elif "\u4e00" <= ch <= "\u9fff":
                has_cjk = True
                if ch in simplified_only:
                    simp_count += 1
                if ch in traditional_only:
                    trad_count += 1

        # 優先順序：日文假名 > 韓文 > 中文 > 英文
        # 日文：有平假名或片假名就判定為日文（即使有漢字也是日文）
        if has_hiragana or has_katakana:
            return "ja"

        # 韓文
        if has_hangul:
            return "ko"

        # 中文：根據簡體/繁體字元判斷
        if has_cjk:
            if simp_count > trad_count:
                return "zh-hans"
            if trad_count > simp_count:
                return "zh-hant"
            # 若無明顯偏向，預設使用繁體中文（本專案主要面向 zh-TW）
            return "zh-hant"

        # 預設英文
        return "en"

    # 1) 先看本輪問題
    detected = _detect_language(question)
    if detected != "en":
        return detected

    # 2) 若問題本身沒有 CJK，可以視需求再參考對話歷史最後一則 user 訊息
    if conversation_history:
        for msg in reversed(conversation_history):
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", "") or ""
            if role == "user" and content:
                detected = _detect_language(content)
                if detected != "en":
                    return detected
                break

    # 3) 預設英文
    return "en"


def normalize_usage_payload(payload: Any) -> Optional[dict[str, Any]]:
    """統一整理 LLM 回傳的 usage 結構。"""

    if not payload:
        return None

    def _dict_get(key: str) -> Any:
        return payload.get(key)  # type: ignore[union-attr]

    def _attr_get(key: str) -> Any:
        return getattr(payload, key, None)

    getter: Any
    if isinstance(payload, dict):
        getter = _dict_get
    else:
        getter = _attr_get

    input_tokens = getter("input_tokens")
    if input_tokens is None:
        input_tokens = getter("prompt_tokens")

    output_tokens = getter("output_tokens")
    if output_tokens is None:
        output_tokens = getter("completion_tokens")

    total_tokens = getter("total_tokens")
    if total_tokens is None and (
        isinstance(input_tokens, (int, float)) or isinstance(output_tokens, (int, float))
    ):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)  # type: ignore[operator]

    if (
        total_tokens is None
        and input_tokens is None
        and output_tokens is None
    ):
        return None

    return {
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def extract_usage_from_llm_output(message: Any) -> Optional[dict[str, Any]]:
    """自 LLM 回傳的 Message/Chunk 中萃取 usage 統計。"""

    if not message:
        return None

    usage = getattr(message, "usage_metadata", None)
    normalized = normalize_usage_payload(usage)
    if normalized:
        return normalized

    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        for key in ("token_usage", "usage", "usage_metadata"):
            normalized = normalize_usage_payload(response_metadata.get(key))
            if normalized:
                return normalized

    additional_kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        normalized = normalize_usage_payload(additional_kwargs.get("usage"))
        if normalized:
            return normalized

    return None


def _build_translator_llm(
    *,
    base_llm_params: Optional[dict],
    agent_backend: str,
) -> BaseChatModel:
    params = base_llm_params or {}
    if agent_backend == "chat":
        return create_chat_completion_llm(
            streaming=False,
            model=params.get("model"),
        )
    return create_responses_llm(
        streaming=False,
        reasoning_effort=params.get("reasoning_effort", "low"),
        reasoning_summary=None,
        model=params.get("model", settings.chat_model),
    )


def translate_query_to_zh_hant_for_retrieval_with_metrics(
    query: str,
    *,
    base_llm_params: Optional[dict] = None,
    agent_backend: str = "responses",
    callbacks: Optional[list[Any]] = None,
) -> tuple[str, Optional[dict[str, Any]], Optional[float]]:
    """
    進階翻譯函式：除翻譯結果外，還會回傳 usage 與耗時（毫秒）。
    """

    start = time.monotonic()
    translator = _build_translator_llm(
        base_llm_params=base_llm_params,
        agent_backend=agent_backend,
    )
    try:
        system_msg = SystemMessage(
            content=(
                "你是一個翻譯助手，只負責將使用者的問題翻譯成適合在「繁體中文」文件中檢索的查詢句。\n"
                "請注意：\n"
                "- 只需要輸出一行繁體中文查詢句，不要加入任何說明文字或標註。\n"
                "- 請保留使用者問題中的關鍵名詞（例如系統名稱、課程名稱等），並轉換成自然的繁體中文用語。\n"
            )
        )
        human_msg = HumanMessage(
            content=(
                f"原始問題：{query}\n\n"
                "請將上面的問題翻譯成一行適合在繁體中文文件中做語義檢索的查詢句，"
                "只輸出該行繁體中文文字，不要加其他解釋。"
            )
        )
        config: RunnableConfig | None = {"callbacks": callbacks} if callbacks else None
        raw = translator.invoke([system_msg, human_msg], config=config)
        duration_ms = (time.monotonic() - start) * 1000.0
        usage = extract_usage_from_llm_output(raw)
        text = str(raw).strip() if raw is not None else ""
        translated = text or query
        logger.info(
            "[LANG_UTILS] Translated query: '{}' -> '{}'",
            query[:50],
            translated[:50],
        )
        return translated, usage, duration_ms
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[LANG_UTILS] Query translation to zh-Hant failed, fallback to original. error={}",
            exc,
        )
        return query, None, (time.monotonic() - start) * 1000.0


async def translate_query_to_zh_hant_for_retrieval_with_metrics_async(
    query: str,
    *,
    base_llm_params: Optional[dict] = None,
    agent_backend: str = "responses",
    callbacks: Optional[list[Any]] = None,
) -> tuple[str, Optional[dict[str, Any]], Optional[float]]:
    """
    進階翻譯函式（非同步版本）：除翻譯結果外，還會回傳 usage 與耗時（毫秒）。
    """

    start = time.monotonic()
    translator = _build_translator_llm(
        base_llm_params=base_llm_params,
        agent_backend=agent_backend,
    )
    try:
        system_msg = SystemMessage(
            content=(
                "你是一個翻譯助手，只負責將使用者的問題翻譯成適合在「繁體中文」文件中檢索的查詢句。\n"
                "請注意：\n"
                "- 只需要輸出一行繁體中文查詢句，不要加入任何說明文字或標註。\n"
                "- 請保留使用者問題中的關鍵名詞（例如系統名稱、課程名稱等），並轉換成自然的繁體中文用語。\n"
            )
        )
        human_msg = HumanMessage(
            content=(
                f"原始問題：{query}\n\n"
                "請將上面的問題翻譯成一行適合在繁體中文文件中做語義檢索的查詢句，"
                "只輸出該行繁體中文文字，不要加其他解釋。"
            )
        )
        config: RunnableConfig | None = {"callbacks": callbacks} if callbacks else None
        raw = await with_llm_semaphore(
            lambda: translator.ainvoke([system_msg, human_msg], config=config),
            backend=agent_backend,
        )
        duration_ms = (time.monotonic() - start) * 1000.0
        usage = extract_usage_from_llm_output(raw)
        text = str(raw).strip() if raw is not None else ""
        translated = text or query
        logger.info(
            "[LANG_UTILS] Translated query (async): '{}' -> '{}'",
            query[:50],
            translated[:50],
        )
        return translated, usage, duration_ms
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[LANG_UTILS] Query translation to zh-Hant failed (async), fallback to original. error={}",
            exc,
        )
        return query, None, (time.monotonic() - start) * 1000.0
