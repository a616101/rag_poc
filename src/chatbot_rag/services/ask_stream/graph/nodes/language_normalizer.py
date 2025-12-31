"""
Language Normalizer 節點：語言偵測與問題標準化。

功能：
- 偵測使用者語言（detect_preferred_language）
- 將問題翻譯成偵測到的語言（如需要）
- 將上一輪回答翻譯成偵測到的語言（如需要）

輸出：
- normalized_question: 標準化後的問題
- prev_answer_normalized: 標準化後的上一輪回答
- user_language: 偵測到的語言

前置節點：guard
後續節點：cache_lookup
"""

from typing import Any, Callable, Optional, cast
import time

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.config import get_stream_writer
from loguru import logger

from chatbot_rag.core.config import settings
from chatbot_rag.core.concurrency import with_llm_semaphore
from chatbot_rag.llm import State, create_responses_llm
from chatbot_rag.services.language_utils import (
    detect_preferred_language,
    extract_usage_from_llm_output,
)
from chatbot_rag.services.prompt_service import PromptService, PromptNames, DEFAULT_PROMPTS
from ...types import StateDict, extract_latest_human_message, extract_last_ai_message
from ...constants import AskStreamStages
from ...events import emit_node_event, emit_llm_meta_event


def _get_language_normalizer_prompt(
    prompt_service: Optional[PromptService],
    target_language: str,
) -> str:
    """取得語言轉換 prompt，優先從 Langfuse 獲取。"""
    prompt_name = PromptNames.LANGUAGE_NORMALIZER_SYSTEM

    if prompt_service:
        try:
            prompt_text, _ = prompt_service.get_text_prompt(
                prompt_name,
                target_language=target_language,
            )
            if prompt_text:
                return prompt_text
        except Exception:
            pass

    # Fallback: 使用 DEFAULT_PROMPTS 並手動替換變數
    base_prompt = DEFAULT_PROMPTS[prompt_name]["prompt"]
    return base_prompt.replace("{{target_language}}", target_language)


async def _ensure_text_in_language(
    text: str,
    target_language: str,
    prompt_service: Optional[PromptService] = None,
) -> tuple[str, Optional[dict[str, Any]], Optional[float]]:
    """必要時將文字轉換為指定語言，以降低語言飄移（非同步版本）。"""
    if not text:
        return text, None, None
    detected = detect_preferred_language(text)
    if detected == target_language:
        return text, None, None

    start = time.monotonic()
    try:
        converter = create_responses_llm(
            streaming=False,
            reasoning_effort="low",
            reasoning_summary=None,
            model=settings.chat_model,
        )
        system_prompt = _get_language_normalizer_prompt(prompt_service, target_language)
        system_msg = SystemMessage(content=system_prompt)
        human_msg = HumanMessage(
            content=(
                "請將以下內容完整轉換為指定語言，保持原意與專有名詞：\n"
                f"{text}"
            )
        )
        raw = await with_llm_semaphore(
            lambda: converter.ainvoke([system_msg, human_msg]),
            backend="responses",
        )
        duration_ms = (time.monotonic() - start) * 1000.0
        usage = extract_usage_from_llm_output(raw)
        converted = str(raw).strip() if raw is not None else ""
        return (converted or text, usage, duration_ms)
    except Exception as exc:  # noqa: BLE001 pylint: disable=broad-exception-caught
        logger.warning("[ASK_STREAM] ensure text language failed, fallback. error={}", exc)
        return text, None, (time.monotonic() - start) * 1000.0


def build_language_normalizer_node(
    prompt_service: Optional[PromptService] = None,
) -> Callable[[State], State]:
    """
    將最新問題與上一輪回答統一成 user_language。

    Args:
        prompt_service: Langfuse Prompt 服務（用於從 Langfuse 獲取 prompt）
    """

    async def language_normalizer(state: State) -> State:
        writer = get_stream_writer()
        emit_node_event(
            writer,
            node="language_normalizer",
            phase="preprocess",
            payload={
                "channel": "status",
                "stage": AskStreamStages.LANGUAGE_NORMALIZER_START,
            },
        )
        state_dict = cast(StateDict, state)
        messages_list = state_dict.get("messages", [])
        user_question = extract_latest_human_message(messages_list)
        detected_language = (
            detect_preferred_language(user_question)
            or state_dict.get("user_language")
            or "zh-hant"
        )
        (
            normalized_question,
            question_usage,
            _,
        ) = await _ensure_text_in_language(user_question, detected_language, prompt_service)
        prev_answer_raw = extract_last_ai_message(messages_list)
        (
            normalized_prev_answer,
            prev_answer_usage,
            _,
        ) = await _ensure_text_in_language(prev_answer_raw, detected_language, prompt_service)

        new_state = cast(State, dict(state))
        new_state["user_language"] = detected_language
        new_state["normalized_question"] = normalized_question
        new_state["latest_question"] = user_question
        new_state["prev_answer_normalized"] = normalized_prev_answer

        if question_usage:
            emit_llm_meta_event(
                writer,
                node="language_normalizer",
                phase="preprocess",
                component="normalize_latest_question",
                usage=question_usage,
            )

        if prev_answer_usage:
            emit_llm_meta_event(
                writer,
                node="language_normalizer",
                phase="preprocess",
                component="normalize_prev_answer",
                usage=prev_answer_usage,
            )

        emit_node_event(
            writer,
            node="language_normalizer",
            phase="preprocess",
            payload={
                "channel": "status",
                "stage": AskStreamStages.LANGUAGE_NORMALIZER_DONE,
                "user_language": detected_language,
            },
        )
        return new_state

    return language_normalizer
