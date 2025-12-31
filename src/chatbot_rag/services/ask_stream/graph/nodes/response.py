"""
回答生成節點：產生最終回答並透過 SSE 串流輸出。

功能：
- 根據 intent 和 context 組合 LLM prompt
- SSE 串流輸出回答內容（answer 事件）
- 產生對話摘要（供下一輪對話使用）
- 處理 fallback 回應（當 LLM 失敗或無結果時）

SSE 事件：
- answer: delta 串流內容
- meta: usage/duration 統計

前置節點：result_evaluator / followup_transform / intent_analyzer
後續節點：cache_store
"""

from typing import Any, Callable, Optional, List, Dict, cast
import time

from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.config import get_stream_writer
from loguru import logger

from chatbot_rag.core.concurrency import llm_concurrency, with_llm_semaphore
from chatbot_rag.llm import State, create_chat_completion_llm, create_responses_llm
from chatbot_rag.services.language_utils import (
    normalize_usage_payload,
    extract_usage_from_llm_output,
)
from chatbot_rag.services.prompt_service import PromptService, PromptNames, DEFAULT_PROMPTS
from ...types import StateDict, generation_inputs_from_state
from ...constants import (
    AskStreamStages,
    RESPONSE_HISTORY_LIMIT,
    CONVERSATION_SUMMARY_MAX_CHARS,
    SUPPORT_SCOPE_TEXT,
)
from ...events import emit_node_event, emit_llm_meta_event
from ...utils import (
    message_to_text,
    select_conversation_history,
    fallback_conversation_summary,
)


# Fallback 回應模板（按 intent 分類）- 屏東基督教醫院風格
FALLBACK_RESPONSES = {
    "simple_faq": """抱歉，您的問題我目前查不到那麼細的資料，
有可能是資訊還未完全上線，也可能您的問題需要更專業的單位說明～
建議您前往屏基官網查詢：<a href="https://www.ptch.org.tw/index.php/index" target="_blank">屏東基督教醫院官網</a>
或致電客服專線：☎️ 08-7368686

您真的很關心健康耶！謝謝您的耐心，也歡迎隨時再回來問我唷！""",
    "symptom_inquiry": """抱歉，關於您描述的症狀，我目前查不到那麼細的資料～
建議您可以先參考我們的門診時刻表：<a href="https://www.ptch.org.tw/ebooks/" target="_blank">門診時刻表</a>
或直接致電客服專線：☎️ 08-7368686 詢問適合的科別

身體健康最重要，希望您早日康復！""",
    "privacy_inquiry": """非常抱歉，您詢問的問題涉及到個人資料的部分，
基於個資保護的規定，這類資訊無法在此查詢喔～
如果需要查詢您自己的就醫紀錄，建議您：
1. 親自至醫院的服務台洽詢
2. 或致電客服專線：☎️ 08-7368686

感謝您的理解，也祝您健康平安！""",
    "conversation_followup": """抱歉，我剛才的回答可能沒有完全滿足您的需求～
能否請您再說明一下想了解的部分呢？我會盡力幫您解答！

祝您一切順利，有需要我一直都在這裡喔～""",
    "out_of_scope": """這個問題我可能不是很理解，不過沒關係～
如果您對健康或就醫有任何需要，我都很願意幫忙喔！
例如：門診時間、掛號流程、科別諮詢等問題，都可以問我～

祝您健康平安，有需要隨時找我喔！""",
    "default": """抱歉，系統暫時遇到一些狀況，無法回答您的問題～
建議您可以：
1. 稍後再試一次
2. 前往屏基官網查詢：<a href="https://www.ptch.org.tw/index.php/index" target="_blank">屏東基督教醫院官網</a>
3. 致電客服專線：☎️ 08-7368686

感謝您的耐心，祝您健康平安！""",
}


def _generate_fallback_response(
    *,
    intent: str,
    user_language: str,
    error: Optional[Exception] = None,
    include_error_hint: bool = False,
) -> str:
    """
    產生 fallback 回應內容。

    Args:
        intent: 任務意圖（用於選擇回應模板）
        user_language: 使用者語言
        error: 觸發 fallback 的錯誤（可選）
        include_error_hint: 是否包含錯誤提示（僅用於 debug）

    Returns:
        Fallback 回應文字（屏東基督教醫院風格）
    """
    base_response = FALLBACK_RESPONSES.get(intent, FALLBACK_RESPONSES["default"])

    # 屏東基督教醫院：強制使用繁體中文，不提供英文版本
    # 即使使用者使用其他語言，也使用繁體中文回應（符合客戶要求：禁止簡體中文）

    if include_error_hint and error:
        error_type = type(error).__name__
        base_response += f"\n\n（技術參考：{error_type}）"

    return base_response


async def _summarize_conversation_history(
    *,
    prev_summary: str,
    latest_user: str,
    latest_answer: str,
    base_llm_params: dict,
    prompt_service: Optional[PromptService] = None,
) -> tuple[str, Optional[dict[str, Any]], Optional[float]]:
    """
    透過小型 LLM 產生對話摘要，提供長期記憶；失敗時退回簡易串接（非同步版本）。
    """
    from chatbot_rag.core.config import settings

    if not latest_user and not latest_answer:
        return prev_summary, None, None

    summarizer_model = base_llm_params.get("model") or settings.chat_model
    summary_llm = create_responses_llm(
        streaming=False,
        reasoning_effort=base_llm_params.get("reasoning_effort", "low"),
        reasoning_summary=None,
        model=summarizer_model,
    )

    # 從 Langfuse 獲取 summarizer prompt（fallback 使用 DEFAULT_PROMPTS）
    fallback_prompt = DEFAULT_PROMPTS[PromptNames.CONVERSATION_SUMMARIZER]["prompt"]
    if prompt_service:
        try:
            system_prompt, _ = prompt_service.get_text_prompt(
                PromptNames.CONVERSATION_SUMMARIZER
            )
        except Exception as exc:
            logger.warning(f"[SUMMARY] Failed to fetch prompt from Langfuse: {exc}")
            system_prompt = fallback_prompt
    else:
        system_prompt = fallback_prompt
    content_sections = [
        f"【既有摘要】\n{prev_summary.strip() or '（無）'}",
        f"【最新對話】\n使用者：{latest_user.strip() or '（無）'}\n助理：{latest_answer.strip() or '（無）'}",
        "請產出新的綜合摘要，若資訊重複可略，維持繁體中文。",
    ]

    start = time.monotonic()
    try:
        raw = await with_llm_semaphore(
            lambda: summary_llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content="\n\n".join(content_sections)),
                ],
            ),
            backend="responses",
        )
        duration_ms = (time.monotonic() - start) * 1000.0
        usage = extract_usage_from_llm_output(raw)
        summary = message_to_text(raw).strip()
        if not summary:
            summary = fallback_conversation_summary(
                prev_summary, latest_user, latest_answer
            )
        if len(summary) > CONVERSATION_SUMMARY_MAX_CHARS:
            summary = summary[:CONVERSATION_SUMMARY_MAX_CHARS]
        return summary, usage, duration_ms
    except Exception as exc:  # noqa: BLE001 pylint: disable=broad-exception-caught
        logger.warning("[SUMMARY] conversation summarization failed: %s", exc)
        return (
            fallback_conversation_summary(prev_summary, latest_user, latest_answer),
            None,
            (time.monotonic() - start) * 1000.0,
        )


def _build_task_analysis_section(
    *,
    task_type: str,
    intent: str,
    used_tools: Optional[List[str]] = None,
    context_text: str = "",
) -> str:
    """
    建立任務分析區段（Non-followup 專用）。

    針對小模型優化：精簡、結構化、關鍵約束前置。

    Args:
        task_type: 任務類型
        intent: 使用者意圖
        used_tools: 已使用工具
        context_text: 知識庫內容

    Returns:
        str: 任務分析區段文字（Markdown 格式）
    """
    has_context = bool(context_text and context_text.strip())

    # 精簡版任務資訊（小模型友好）
    lines = [
        "# 任務",
        f"意圖：{intent}",
    ]

    if has_context:
        # 有知識庫內容：強調「只用知識庫」
        lines.extend([
            "",
            "# ⚠️ 回答規則",
            "1. **只用**下方「知識庫內容」回答",
            "2. 知識庫沒有的 → 說「查不到」，引導致電客服",
            "3. **禁止**編造醫師、時間、科別",
            "",
            "範例：知識庫只有「王醫師、李醫師」",
            "- ✅「有王醫師、李醫師」",
            "- ❌「有王醫師、李醫師、張醫師等」（張醫師是編造的）",
        ])
    else:
        # 無知識庫內容：引導使用標準回應
        lines.extend([
            "",
            "# ⚠️ 狀態：查無資料",
            "請使用標準回應：",
            "「目前查不到相關資料，建議致電客服 08-7368686 或查詢官網」",
        ])

    return "\n".join(lines)


def _build_context_section(context_text: str) -> str:
    """
    建立知識庫內容區段。

    Args:
        context_text: 知識庫內容

    Returns:
        str: 知識庫區段文字（Markdown 格式，空字串表示無內容）
    """
    if not context_text:
        return ""
    lines = [
        "# 知識庫內容",
        "",
        context_text,
    ]
    return "\n".join(lines)


def _build_prev_answer_section(prev_answer: str) -> str:
    """
    建立上一輪回答區段（Followup 專用）。

    Args:
        prev_answer: 上一輪回答

    Returns:
        str: 上一輪回答區段文字（Markdown 格式）
    """
    if not prev_answer:
        return ""
    lines = [
        "# 上一輪回答",
        "",
        "> 以下是你上一輪的回答，請基於此內容進行後續處理。",
        "",
        "---",
        "",
        prev_answer,
    ]
    return "\n".join(lines)


def _build_conversation_summary_section(conversation_summary: str) -> str:
    """
    建立對話摘要區段。

    Args:
        conversation_summary: 對話摘要

    Returns:
        str: 對話摘要區段文字（Markdown 格式，空字串表示無內容）
    """
    if not conversation_summary:
        return ""
    lines = [
        "# 對話摘要（僅供參考）",
        "",
        "> 💡 以下是先前對話的摘要，供你參考上下文，請勿逐字輸出。",
        "",
        conversation_summary,
    ]
    return "\n".join(lines)


def _get_role_definition(
    user_language: str,
    prompt_service: Optional[PromptService] = None,
) -> str:
    """
    取得共用角色定義 prompt。

    Args:
        user_language: 使用者語言
        prompt_service: Prompt 服務

    Returns:
        str: 編譯後的角色定義
    """
    lang_instruction_name = PromptNames.get_language_instruction_name(user_language)
    if prompt_service:
        try:
            lang_instruction, _ = prompt_service.get_text_prompt(lang_instruction_name)
        except Exception:
            lang_instruction = DEFAULT_PROMPTS[lang_instruction_name]["prompt"]
    else:
        lang_instruction = DEFAULT_PROMPTS[lang_instruction_name]["prompt"]

    # 取得角色定義
    if prompt_service:
        try:
            role_def, _ = prompt_service.get_text_prompt(
                PromptNames.ROLE_DEFINITION,
                language_instruction=lang_instruction,
            )
            return role_def
        except Exception:
            pass

    # Fallback
    role_template = DEFAULT_PROMPTS[PromptNames.ROLE_DEFINITION]["prompt"]
    return role_template.replace("{{language_instruction}}", lang_instruction)


def _get_intent_instruction(intent: str) -> tuple[str, str]:
    """
    根據 intent 取得對應的描述和指令。

    Args:
        intent: 使用者意圖

    Returns:
        tuple: (intent_description, intent_instruction)
    """
    intent_configs = {
        "privacy_inquiry": (
            "婉拒個資查詢",
            """民眾詢問的是個人醫療資訊（如病歷、看診記錄、費用明細等）。

## 回應要點

1. **感謝並表達理解**：「感謝您的提問！」
2. **說明原因**：為保護隱私權益，這些資訊需透過正式管道查詢
3. **提供替代方案**：
   - 親自至醫院服務台申請
   - 致電客服專線 **08-7368686**
4. **引導其他問題**：詢問是否有其他關於醫院服務的問題可以協助

**範例開頭**：「感謝您的提問！😊 關於您詢問的個人醫療資訊...」""",
        ),
        "service_capability": (
            "說明服務能力",
            """民眾在詢問你能做什麼、能不能幫忙某件事。

## 回應要點

1. **親切回應**：先肯定民眾的提問
2. **說明能力範圍**：
   - ✅ **可以協助**：查詢門診時間、掛號流程、科別諮詢、醫師資訊、就醫須知等
   - ❌ **無法協助**：直接幫民眾掛號、查詢個人病歷、預約手術等需要身份驗證的操作
3. **提供替代方案**：
   - 線上掛號：[我要掛號](https://www.ptch.org.tw/index.php/reg_listForm01)
   - 電話預約：📞 **08-7368686**
4. **主動引導**：詢問民眾想了解哪方面的資訊

## 範例

若民眾問「你能幫我掛號嗎？」：
「您好！😊 很抱歉，我目前無法直接幫您完成掛號，但我可以協助您了解掛號流程喔！

您可以透過以下方式掛號：
- 📱 **線上掛號**：[我要掛號](https://www.ptch.org.tw/index.php/reg_listForm01)
- 📞 **電話預約**：撥打 **08-7368686** 客服專線

如果您想了解哪個科別適合您，或是想知道某位醫師的門診時間，都可以問我喔！✨」""",
        ),
        "out_of_scope": (
            "引導回醫院相關問題",
            """民眾的問題與醫院服務無關（如天氣、旅遊、程式等）。

## 回應要點

1. **親切回應**：不要讓民眾覺得被拒絕
2. **說明服務範圍**：介紹你可以協助的問題類型
   - 📋 掛號流程、門診時間
   - 🩺 各科別服務諮詢
   - 🏥 就醫須知、院內設施
3. **溫暖邀請**：歡迎民眾詢問健康/就醫相關問題

**範例開頭**：「謝謝您的提問！😊 我是屏基的服務小天使，專門協助...」""",
        ),
        "greeting": (
            "回應打招呼",
            """民眾只是打招呼或寒暄。

## 回應要點

1. **熱情回應**：用溫暖的方式打招呼
2. **自我介紹**：簡單介紹你是屏基的服務小天使
3. **主動詢問**：詢問民眾今天有什麼可以幫忙的

**範例**：「您好！😊 我是屏東基督教醫院的服務小天使，很高興為您服務～請問今天有什麼我可以幫您的呢？」""",
        ),
    }

    return intent_configs.get(
        intent,
        ("回應民眾問題", "請用親切的方式回應民眾的問題，如有需要可引導至客服專線。"),
    )


def _build_complete_system_prompt(
    *,
    is_followup: bool,
    user_language: str,
    prompt_service: Optional[PromptService] = None,
    # Followup 專用參數
    prev_answer: str = "",
    # Non-followup 專用參數
    task_type: str = "",
    intent: str = "",
    used_tools: Optional[List[str]] = None,
    context_text: str = "",
    # 共用參數
    conversation_summary: str = "",
) -> str:
    """
    建立完整的系統 prompt，根據場景選擇對應的 prompt 模板。

    場景切割：
    1. Followup（追問）→ FOLLOWUP_SYSTEM
    2. 有知識庫內容 → RESPONSE_WITH_CONTEXT
    3. 無知識庫內容 → RESPONSE_NO_CONTEXT
    4. 直接回應（privacy_inquiry, out_of_scope）→ RESPONSE_DIRECT

    Args:
        is_followup: 是否為追問場景
        user_language: 使用者語言
        prompt_service: Prompt 服務
        prev_answer: 上一輪回答（followup 用）
        task_type: 任務類型（non-followup 用）
        intent: 使用者意圖（non-followup 用）
        used_tools: 已使用工具（non-followup 用）
        context_text: 知識庫內容（non-followup 用）
        conversation_summary: 對話摘要（共用）

    Returns:
        str: 完整的系統 prompt
    """
    # 取得共用角色定義
    role_definition = _get_role_definition(user_language, prompt_service)

    # 取得對話摘要區段
    conversation_summary_section = _build_conversation_summary_section(conversation_summary)

    # 根據場景選擇 prompt
    if is_followup:
        # 場景：追問處理
        prev_answer_section = _build_prev_answer_section(prev_answer)
        compile_vars = {
            "role_definition": role_definition,
            "prev_answer_section": prev_answer_section,
            "conversation_summary_section": conversation_summary_section,
        }
        prompt_name = PromptNames.FOLLOWUP_SYSTEM

    elif intent in ("privacy_inquiry", "out_of_scope", "greeting", "service_capability"):
        # 場景：直接回應（不需要知識庫）
        intent_description, intent_instruction = _get_intent_instruction(intent)
        compile_vars = {
            "role_definition": role_definition,
            "intent_description": intent_description,
            "intent_instruction": intent_instruction,
            "conversation_summary_section": conversation_summary_section,
        }
        prompt_name = PromptNames.RESPONSE_DIRECT

    elif context_text and context_text.strip():
        # 場景：檢索成功（有知識庫內容）
        context_section = _build_context_section(context_text)
        compile_vars = {
            "role_definition": role_definition,
            "context_section": context_section,
            "conversation_summary_section": conversation_summary_section,
        }
        prompt_name = PromptNames.RESPONSE_WITH_CONTEXT

    else:
        # 場景：檢索失敗（無知識庫內容）
        compile_vars = {
            "role_definition": role_definition,
            "conversation_summary_section": conversation_summary_section,
        }
        prompt_name = PromptNames.RESPONSE_NO_CONTEXT

    # 嘗試從 Langfuse 取得並編譯
    if prompt_service:
        try:
            compiled_prompt, _ = prompt_service.get_text_prompt(
                prompt_name,
                **compile_vars,
            )
            return compiled_prompt
        except Exception as exc:
            logger.warning(f"[RESPONSE] Failed to fetch {prompt_name}: {exc}")

    # Fallback: 使用 DEFAULT_PROMPTS
    if prompt_name in DEFAULT_PROMPTS:
        fallback_template = DEFAULT_PROMPTS[prompt_name]["prompt"]
    else:
        # 最後備援：使用舊的 UNIFIED_AGENT_SYSTEM
        fallback_template = DEFAULT_PROMPTS[PromptNames.UNIFIED_AGENT_SYSTEM]["prompt"]

    for var_name, var_value in compile_vars.items():
        fallback_template = fallback_template.replace(f"{{{{{var_name}}}}}", var_value)

    return fallback_template


def build_response_node(
    base_llm_params: dict,
    *,
    agent_backend: str = "chat",
    prompt_service: Optional[PromptService] = None,
) -> Callable[[State], State]:
    """回答產生節點，統一處理 streaming 與 meta。"""
    from chatbot_rag.core.config import settings

    # 回應生成的溫度設定：
    # - 0.5 讓回應更自然、溫暖
    # - 防止幻覺主要靠 prompt 中的約束，而非低溫度
    RESPONSE_TEMPERATURE = 0.5

    def _create_streaming_llm(task_type: str) -> BaseChatModel:
        """根據任務類型動態建立 LLM。"""
        task_params = settings.get_llm_params_for_task(task_type)
        # 使用固定的回應溫度，讓回應更自然
        temperature = task_params.get("temperature", RESPONSE_TEMPERATURE)
        reasoning_effort = str(task_params.get("reasoning_effort", "medium"))

        if agent_backend == "chat":
            return create_chat_completion_llm(
                streaming=True,
                model=base_llm_params.get("model"),
                temperature=float(temperature),
            )
        else:
            return cast(
                BaseChatModel,
                create_responses_llm(
                    streaming=True,
                    reasoning_effort=reasoning_effort,
                    reasoning_summary=base_llm_params.get("reasoning_summary", "auto"),
                    model=base_llm_params.get("model"),
                    temperature=float(temperature),
                ),
            )

    async def response_node(state: State) -> State:
        writer = get_stream_writer()
        state_dict = cast(StateDict, state)
        state_messages = cast(List[BaseMessage], state_dict.get("messages") or [])
        retrieval_state = cast(dict[str, Any], state_dict.get("retrieval") or {})
        history_messages = select_conversation_history(
            state_messages, limit=RESPONSE_HISTORY_LIMIT
        )
        summary_enabled = bool(state_dict.get("conversation_summary_enabled", True))
        conversation_summary = (
            state_dict.get("conversation_summary") or "" if summary_enabled else ""
        )
        rewrite_msg = cast(
            Optional[BaseMessage], retrieval_state.get("rewritten_query_message")
        )
        rewritten_query = ""
        if rewrite_msg is not None:
            msg_content = getattr(rewrite_msg, "content", "") or ""
            if isinstance(msg_content, str):
                rewritten_query = msg_content
            else:
                rewritten_query = message_to_text(rewrite_msg)
        if not rewritten_query:
            summary_query = state_dict.get("summary_search_query")
            if isinstance(summary_query, BaseMessage):
                msg_content = getattr(summary_query, "content", "") or ""
                if isinstance(msg_content, str):
                    rewritten_query = msg_content
                else:
                    rewritten_query = message_to_text(summary_query)
            elif isinstance(summary_query, str):
                rewritten_query = summary_query
        gen_inputs = generation_inputs_from_state(state_dict)
        task_type = gen_inputs["task_type"]
        intent = gen_inputs["intent"]
        user_language = gen_inputs["user_language"]
        normalized_question = gen_inputs["normalized_question"]
        followup_instruction = gen_inputs["followup_instruction"]
        prev_answer = gen_inputs["prev_answer"]
        context_text = gen_inputs["context_text"]
        used_tools = gen_inputs["used_tools"]
        loop_count = gen_inputs["loop_count"]
        is_followup = gen_inputs["is_followup"]
        is_out_of_scope = gen_inputs["is_out_of_scope"]

        emit_node_event(
            writer,
            node="response_synth",
            phase="generation",
            payload={
                "channel": "status",
                "stage": AskStreamStages.RESPONSE_GENERATING,
                "intent": intent,
                "used_tools": used_tools,
                "loop": loop_count,
                "is_out_of_scope": is_out_of_scope,
                "followup": is_followup,
            },
        )

        # 使用統一的 system prompt 建構函數
        complete_system_prompt = _build_complete_system_prompt(
            is_followup=is_followup,
            user_language=user_language,
            prompt_service=prompt_service,
            # Followup 專用
            prev_answer=prev_answer,
            # Non-followup 專用
            task_type=task_type,
            intent=intent,
            used_tools=used_tools,
            context_text=context_text,
            # 共用
            conversation_summary=conversation_summary,
        )

        final_messages: List[BaseMessage] = [SystemMessage(content=complete_system_prompt)]
        if history_messages:
            final_messages.extend(history_messages)
        final_messages.append(
            HumanMessage(content=followup_instruction if is_followup else normalized_question)
        )

        # 動態建立 LLM，根據 task_type 調整參數
        streaming_llm = _create_streaming_llm(task_type)

        answer_tokens: List[str] = []
        reasoning_tokens: List[str] = []
        final_usage: Optional[dict[str, Any]] = None
        response_id: Optional[str] = None
        reasoning_started = False
        first_token_at: Optional[float] = None
        is_fallback = False
        fallback_error: Optional[Exception] = None

        try:
            async with llm_concurrency.acquire(agent_backend):
                async for chunk in streaming_llm.astream(final_messages):
                    now = time.monotonic()
                    if first_token_at is None:
                        first_token_at = now

                    add_kwargs = getattr(chunk, "additional_kwargs", {}) or {}
                    channel = add_kwargs.get("channel")
                    delta = add_kwargs.get("delta") or ""
                    content = getattr(chunk, "content", None)
                    if isinstance(content, str) and content and not delta:
                        delta = content

                    if channel in ("reasoning", "reasoning_summary"):
                        if delta:
                            reasoning_tokens.append(delta)
                            if not reasoning_started:
                                reasoning_started = True
                                emit_node_event(
                                    writer,
                                    node="response_synth",
                                    phase="generation",
                                    payload={
                                        "channel": "status",
                                        "stage": AskStreamStages.RESPONSE_REASONING,
                                    },
                                )
                            writer(
                                {
                                    "source": "response_synth",
                                    "node": "response_synth",
                                    "phase": "generation",
                                    "channel": "reasoning",
                                    "delta": delta,
                                }
                            )
                    elif channel == "meta":
                        responses_meta = add_kwargs.get("responses_meta")
                        if isinstance(responses_meta, dict):
                            usage_dict = normalize_usage_payload(responses_meta.get("usage"))
                            if usage_dict:
                                final_usage = usage_dict
                            resp_id = responses_meta.get("response_id") or responses_meta.get("id")
                            if resp_id and not response_id:
                                response_id = str(resp_id)
                    elif delta and channel not in ("done",):
                        answer_tokens.append(delta)
                        writer(
                            {
                                "source": "response_synth",
                                "node": "response_synth",
                                "phase": "generation",
                                "channel": "answer",
                                "delta": delta,
                            }
                        )

                    usage_from_chunk = normalize_usage_payload(getattr(chunk, "usage_metadata", None))
                    if usage_from_chunk:
                        final_usage = usage_from_chunk
                    else:
                        response_metadata = getattr(chunk, "response_metadata", None)
                        if isinstance(response_metadata, dict):
                            usage_from_response_meta = normalize_usage_payload(
                                response_metadata.get("token_usage")
                            )
                            if usage_from_response_meta:
                                final_usage = usage_from_response_meta

        except Exception as stream_error:  # noqa: BLE001 pylint: disable=broad-exception-caught
            # LLM streaming 失敗，使用 fallback 回應
            is_fallback = True
            fallback_error = stream_error
            logger.error(
                "[RESPONSE] LLM streaming failed, using fallback: %s",
                stream_error,
            )

            # 產生 fallback 回應
            from chatbot_rag.core.config import settings
            fallback_answer = _generate_fallback_response(
                intent=intent,
                user_language=user_language,
                error=stream_error,
                include_error_hint=settings.debug,
            )

            # 發送 fallback 事件
            emit_node_event(
                writer,
                node="response_synth",
                phase="generation",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.RESPONSE_FALLBACK,
                    "error": str(stream_error),
                    "error_type": type(stream_error).__name__,
                },
            )

            # 寫入 fallback 回應
            writer(
                {
                    "source": "response_synth",
                    "node": "response_synth",
                    "phase": "generation",
                    "channel": "answer",
                    "delta": fallback_answer,
                }
            )
            answer_tokens = [fallback_answer]

        final_answer = "".join(answer_tokens)
        latest_question = state_dict.get("latest_question") or normalized_question
        if summary_enabled:
            (
                updated_summary,
                summary_usage,
                _,
            ) = await _summarize_conversation_history(
                prev_summary=conversation_summary,
                latest_user=latest_question,
                latest_answer=final_answer,
                base_llm_params=base_llm_params,
                prompt_service=prompt_service,
            )
        else:
            updated_summary = ""
            summary_usage = None

        meta_payload: Dict[str, Any] = {
            "response_id": response_id,
            "usage": final_usage,
            "loops": loop_count,
            "used_tools": used_tools,
            "eval_query_rewrite": rewritten_query,
            "channels": {
                "output_text": {
                    "text": final_answer,
                    "char_count": len(final_answer),
                }
            },
        }
        if is_fallback:
            meta_payload["is_fallback"] = True
            meta_payload["fallback_error_type"] = (
                type(fallback_error).__name__ if fallback_error else "unknown"
            )
        if summary_enabled:
            meta_payload["conversation_summary"] = updated_summary

        writer(
            {
                "source": "response_synth",
                "node": "response_synth",
                "phase": "generation",
                "channel": "meta",
                "meta": meta_payload,
            }
        )

        emit_node_event(
            writer,
            node="response_synth",
            phase="generation",
            payload={
                "channel": "status",
                "stage": AskStreamStages.RESPONSE_DONE,
                "intent": intent,
                "loops": loop_count,
                "used_tools": used_tools,
                "is_out_of_scope": is_out_of_scope,
            },
        )

        final_msg = AIMessage(
            content=final_answer,
            additional_kwargs={
                "reasoning_text": "".join(reasoning_tokens),
                "responses_meta": meta_payload,
            },
        )

        new_state = cast(State, dict(state))
        new_state["messages"] = state_dict.get("messages", []) + [final_msg]
        new_state["final_answer"] = final_answer
        new_state["response_meta"] = meta_payload
        new_state["intent"] = intent
        new_state["is_out_of_scope"] = is_out_of_scope
        new_state["eval_question"] = latest_question
        new_state["eval_context"] = context_text
        new_state["eval_answer"] = final_answer
        cast(StateDict, new_state)["eval_query_rewrite"] = rewritten_query
        cast(StateDict, new_state)["conversation_summary"] = (
            updated_summary if summary_enabled else ""
        )
        cast(StateDict, new_state)["conversation_summary_enabled"] = summary_enabled
        if summary_enabled and summary_usage:
            emit_llm_meta_event(
                writer,
                node="response_synth",
                phase="generation",
                component="conversation_summary",
                usage=summary_usage,
            )
        return new_state

    return response_node
