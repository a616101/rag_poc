"""
ask_stream 模組共用工具函數。

文字處理：
- maybe_extract_from_repr_string: 從 LLM repr 字串提取內容
- looks_like_llm_repr: 判斷是否像 LLM repr 字串
- extract_text_from_payload: 從 payload 提取純文字
- message_to_text: 將 langchain message 轉為文字
- strip_code_fence: 去除程式碼區塊包覆
- is_single_cjk: 判斷是否為單一 CJK 字元
- maybe_fix_character_spaced_query: 修正逐字拆解的查詢

JSON 處理：
- extract_first_json_object: 從文字中提取第一個完整 JSON 物件

對話處理：
- summarize_recent_messages: 擷取近期對話片段
- select_conversation_history: 擷取要帶入 LLM 的歷史訊息
- fallback_conversation_summary: 簡易串接摘要
- looks_like_followup_request: 判斷是否是追問請求

Chunk Expansion：
- calculate_needed_chunk_indices: 計算需要取得的 chunk 索引
"""

from collections import defaultdict
from typing import Any, Dict, Optional, List, Set
import re

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from .constants import RESPONSE_HISTORY_LIMIT, CONVERSATION_SUMMARY_MAX_CHARS


_LLM_REPR_FIELD_PATTERNS = [
    re.compile(r"content=['\"](.+?)['\"](?=\s|$)", re.DOTALL),
    re.compile(r"[\"']text[\"']:\s*[\"'](.+?)[\"']", re.DOTALL),
]
_LLM_REPR_KEYWORDS = ("response_metadata", "usage_metadata", "raw_response", "additional_kwargs")


def maybe_extract_from_repr_string(text: str) -> Optional[str]:
    """嘗試從 LLM repr 字串中提取內容。"""
    for pattern in _LLM_REPR_FIELD_PATTERNS:
        match = pattern.search(text)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate
    return None


def looks_like_llm_repr(text: str) -> bool:
    """判斷是否像 LLM repr 字串。"""
    return any(keyword in text for keyword in _LLM_REPR_KEYWORDS)


def extract_text_from_payload(payload: Any) -> Optional[str]:
    """
    從 Responses 風格 payload 中萃取純文字，盡可能過濾結構化欄位。
    """
    if payload is None:
        return None
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return None
        extracted = maybe_extract_from_repr_string(text)
        if extracted:
            return extracted
        if looks_like_llm_repr(text):
            return None
        return text
    if isinstance(payload, dict):
        # Responses API 常見欄位：content / text / value / output_text / message
        for key in ("content", "text", "value", "output_text", "message"):
            if key in payload:
                extracted = extract_text_from_payload(payload[key])
                if extracted:
                    return extracted
        return None
    if isinstance(payload, list):
        parts: list[str] = []
        for item in payload:
            extracted = extract_text_from_payload(item)
            if extracted:
                parts.append(extracted)
        if parts:
            return "\n".join(parts)
        return None
    return None


def message_to_text(message: BaseMessage | Any) -> str:
    """將 langchain message 或 payload 轉為文字。"""
    if isinstance(message, BaseMessage):
        payload = getattr(message, "content", "")
    else:
        payload = getattr(message, "content", None) if hasattr(message, "content") else message

    if isinstance(payload, str):
        return payload

    if isinstance(payload, list):
        parts: list[str] = []
        for item in payload:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "\n".join(parts)
        return ""

    if payload is None:
        return ""

    if isinstance(payload, dict):
        return extract_text_from_payload(payload) or ""

    return str(payload)


def strip_code_fence(text: str) -> str:
    """去除簡單的程式碼區塊包覆。"""
    trimmed = text.strip()
    if trimmed.startswith("```") and trimmed.endswith("```"):
        inner = trimmed.strip("`")
        lines = inner.splitlines()
        if lines and not lines[0].strip():
            lines = lines[1:]
        if lines and ":" in lines[0] and lines[0].strip().isalpha():
            lines = lines[1:]
        return "\n".join(lines).strip()
    return trimmed


def is_single_cjk(char: str) -> bool:
    """判斷是否為單一 CJK 字元。"""
    return len(char) == 1 and "\u4e00" <= char <= "\u9fff"


def maybe_fix_character_spaced_query(query: str) -> tuple[str, bool]:
    """
    偵測 LLM 把句子拆成逐字輸出（例：'課 程 查 詢'）的情況，並嘗試合併。
    回傳：修正後的字串與是否有修正。
    """
    tokens = query.strip().split()
    if len(tokens) < 3:
        return query, False

    # 所有 token 為單一 CJK 或單一字母時，判定為逐字拆解。
    if all(len(token) == 1 and (token.isalpha() or is_single_cjk(token)) for token in tokens):
        return "".join(tokens), True

    return query, False


def summarize_recent_messages(messages: List[BaseMessage], limit: int = 4) -> str:
    """擷取近期對話片段供 query rewrite 使用。"""
    snippet_lines: list[str] = []
    recent_messages = messages[-limit:] if limit > 0 else messages
    for msg in recent_messages:
        text = message_to_text(msg).strip()
        if not text:
            continue
        if len(text) > 400:
            text = text[:400] + "..."
        if isinstance(msg, HumanMessage):
            role = "使用者"
        elif isinstance(msg, AIMessage):
            role = "助理"
        elif isinstance(msg, SystemMessage):
            role = "系統"
        else:
            role = msg.__class__.__name__
        snippet_lines.append(f"{role}：{text}")
    return "\n".join(snippet_lines)


def select_conversation_history(
    messages: List[BaseMessage], limit: int = RESPONSE_HISTORY_LIMIT
) -> List[BaseMessage]:
    """
    擷取要帶入回答 LLM 的歷史訊息，避免遺失上下文。
    預設會移除最新一則使用者輸入，因為稍後會以 normalized_question 重新提供。
    """
    if not messages:
        return []

    history = list(messages)
    if history and isinstance(history[-1], HumanMessage):
        history = history[:-1]

    if limit > 0 and len(history) > limit:
        history = history[-limit:]

    filtered: List[BaseMessage] = []
    for msg in history:
        if isinstance(msg, (HumanMessage, AIMessage)):
            filtered.append(msg)
    return filtered


def fallback_conversation_summary(
    prev_summary: str,
    latest_user: str,
    latest_answer: str,
    max_chars: int = CONVERSATION_SUMMARY_MAX_CHARS,
) -> str:
    """簡易串接摘要，供 LLM 摘要失敗時使用。"""
    segments: list[str] = []
    if prev_summary:
        segments.append(prev_summary.strip())
    if latest_user:
        segments.append(f"使用者：{latest_user.strip()}")
    if latest_answer:
        segments.append(f"助理：{latest_answer.strip()}")
    combined = "\n".join(segment for segment in segments if segment).strip()
    if not combined:
        return ""
    if len(combined) <= max_chars:
        return combined
    return combined[-max_chars:]


def looks_like_followup_request(question: str, user_language: str) -> bool:
    """以關鍵字快速判斷是否是針對上一輪回答的翻譯/改寫需求。"""
    if not question:
        return False
    text = question.lower()
    zh_keywords = ["再說明", "再解釋", "再整理", "重點整理", "翻譯", "轉成", "用中文", "用英文", "改寫", "摘要", "重述"]
    en_keywords = [
        "translate",
        "rewrite",
        "summarize",
        "summarise",
        "in chinese",
        "in english",
        "rephrase",
        "simplify",
        "explain again",
    ]
    ja_keywords = ["翻訳", "言い換え", "説明し直", "日本語で", "まとめて"]
    ko_keywords = ["번역", "다시", "정리해", "한국어로", "요약"]

    if user_language.startswith("zh"):
        return any(keyword in question for keyword in zh_keywords)
    if user_language == "ja":
        return any(keyword in question for keyword in ja_keywords)
    if user_language == "ko":
        return any(keyword in question for keyword in ko_keywords)
    return any(keyword in text for keyword in en_keywords)


# ============================================================================
# JSON 處理
# ============================================================================

def extract_first_json_object(text: str) -> Optional[str]:
    """
    從文字中提取第一個完整的 JSON 物件。

    使用括號深度追蹤來正確處理巢狀結構，比正則表達式更可靠。

    Args:
        text: 包含 JSON 的文字

    Returns:
        Optional[str]: 提取的 JSON 字串，如果找不到則返回 None
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


# ============================================================================
# Adaptive Chunk Expansion
# ============================================================================

# 閾值設定
SMALL_DOC_THRESHOLD = 3   # 小文件閾值（chunks 數量）
MEDIUM_DOC_THRESHOLD = 8  # 中型文件閾值


def calculate_needed_chunk_indices(
    hit_indices: List[int],
    total_chunks: int,
) -> Set[int]:
    """
    根據命中的 chunk 索引和文件總 chunks 數計算需要取得的 indices。

    策略：
    - 小文件（≤3 chunks）：取全部
    - 中型文件（≤8 chunks）：每個命中 ±1
    - 大型文件（>8 chunks）：智慧窗口
        - 開頭命中：往後擴展
        - 結尾命中：往前擴展
        - 中間命中：±1

    Args:
        hit_indices: 命中的 chunk 索引列表
        total_chunks: 文件總 chunks 數

    Returns:
        Set[int]: 需要取得的 chunk 索引集合
    """
    if total_chunks <= SMALL_DOC_THRESHOLD:
        # 小文件：取全部
        return set(range(total_chunks))

    needed: Set[int] = set()

    if total_chunks <= MEDIUM_DOC_THRESHOLD:
        # 中型文件：每個命中 ±1
        for idx in hit_indices:
            needed.update(range(max(0, idx - 1), min(total_chunks, idx + 2)))
    else:
        # 大型文件：智慧窗口
        for idx in hit_indices:
            if idx < 2:
                # 開頭：往後擴展
                needed.update(range(0, min(4, total_chunks)))
            elif idx >= total_chunks - 2:
                # 結尾：往前擴展
                needed.update(range(max(0, total_chunks - 4), total_chunks))
            else:
                # 中間：±1
                needed.update(range(idx - 1, idx + 2))

    return needed
