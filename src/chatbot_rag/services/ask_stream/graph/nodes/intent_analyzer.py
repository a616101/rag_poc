"""
意圖分析節點：分析使用者意圖並決定路由。

功能：
- 透過 LLM 分析使用者問題的意圖
- 決定路由方向：direct_response / followup / query_builder
- 輸出 IntentAnalyzerOutput 結構化資料

設計原則：
- 意圖類型由 prompt 定義，非程式碼硬編碼
- LLM 輸出直接驅動路由決策
- 支援領域 prompts 和 Langfuse prompt management
"""

from typing import Any, Callable, Optional, cast
import json
import time

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.config import get_stream_writer
from loguru import logger
from pydantic import ValidationError

from chatbot_rag.core.concurrency import with_llm_semaphore
from chatbot_rag.core.domain import DomainConfig, get_current_domain
from chatbot_rag.llm import State, create_chat_completion_llm, create_responses_llm
from chatbot_rag.models.llm_outputs import IntentAnalyzerOutput
from chatbot_rag.services.language_utils import extract_usage_from_llm_output
from chatbot_rag.services.prompt_service import PromptService, PromptNames, DEFAULT_PROMPTS
from ...types import StateDict, intent_analyzer_inputs_from_state, extract_latest_human_message
from ...constants import AskStreamStages
from ...events import emit_node_event, emit_llm_meta_event
from ...utils import extract_first_json_object


def _parse_and_validate_intent(
    data: dict[str, Any],
) -> IntentAnalyzerOutput:
    """
    使用 Pydantic 驗證並轉換 LLM 輸出的 JSON。

    Args:
        data: 解析後的 JSON 字典

    Returns:
        驗證後的 IntentAnalyzerOutput
    """
    try:
        output = IntentAnalyzerOutput.model_validate(data)
        logger.debug("[INTENT_ANALYZER] Pydantic validation successful")
        return output
    except ValidationError as ve:
        logger.debug(
            "[INTENT_ANALYZER] Pydantic validation failed, using loose parsing: {}",
            ve
        )

        # 寬鬆解析
        intent = str(data.get("intent", "general_inquiry")) or "general_inquiry"
        needs_retrieval = bool(data.get("needs_retrieval", True))
        needs_followup = bool(data.get("needs_followup", False))

        # 解析 routing_hint
        routing_hint_raw = str(data.get("routing_hint", "continue")) or "continue"
        valid_hints = {"continue", "direct_response", "followup"}
        routing_hint = routing_hint_raw if routing_hint_raw in valid_hints else "continue"

        # 解析 confidence
        confidence_raw = data.get("confidence", 0.8)
        try:
            confidence = float(confidence_raw)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.8

        reasoning = str(data.get("reasoning", "")) or ""

        # 解析新欄位：query_type
        query_type_raw = str(data.get("query_type", "detail")) or "detail"
        valid_query_types = {"list", "detail", "hybrid"}
        query_type = query_type_raw if query_type_raw in valid_query_types else "detail"

        # 解析新欄位：retrieval_strategy
        retrieval_strategy_raw = str(data.get("retrieval_strategy", "vector")) or "vector"
        valid_strategies = {"vector", "metadata_filter", "hybrid"}
        retrieval_strategy = (
            retrieval_strategy_raw if retrieval_strategy_raw in valid_strategies else "vector"
        )

        # 解析新欄位：extracted_entities
        extracted_entities_raw = data.get("extracted_entities", {})
        if isinstance(extracted_entities_raw, dict):
            extracted_entities = {
                str(k): str(v) for k, v in extracted_entities_raw.items() if v
            }
        else:
            extracted_entities = {}

        return IntentAnalyzerOutput(
            intent=intent,
            needs_retrieval=needs_retrieval,
            needs_followup=needs_followup,
            routing_hint=routing_hint,  # type: ignore
            confidence=confidence,
            reasoning=reasoning,
            query_type=query_type,  # type: ignore
            retrieval_strategy=retrieval_strategy,  # type: ignore
            extracted_entities=extracted_entities,
        )


async def _analyze_intent(
    *,
    llm: BaseChatModel,
    user_question: str,
    last_ai_content: str,
    user_language: str,
    agent_backend: str = "chat",
    prompt_service: Optional[PromptService] = None,
    domain_config: Optional[DomainConfig] = None,
) -> tuple[IntentAnalyzerOutput, Optional[dict[str, Any]], Optional[float]]:
    """
    透過 LLM 分析使用者意圖。

    Args:
        llm: LLM 實例
        user_question: 使用者問題
        last_ai_content: 上一輪助理回答
        user_language: 使用者語言
        agent_backend: Agent 後端類型
        prompt_service: Prompt 服務
        domain_config: 領域設定

    Returns:
        tuple: (IntentAnalyzerOutput, usage, duration_ms)
    """
    default_output = IntentAnalyzerOutput(
        intent="general_inquiry",
        needs_retrieval=True,
        needs_followup=False,
        routing_hint="continue",
        confidence=0.5,
        reasoning="Default fallback",
    )

    # 取得領域設定
    if domain_config is None:
        domain_config = get_current_domain()

    # 從 Langfuse 或 DEFAULT_PROMPTS 取得 system prompt
    prompt_name = PromptNames.INTENT_ANALYZER_SYSTEM
    base_prompt = ""

    # 1. 優先使用領域 prompts
    domain_prompts = domain_config.get_domain_prompts()
    if prompt_name in domain_prompts:
        base_prompt = domain_prompts[prompt_name].get("prompt", "")

    # 2. 嘗試從 Langfuse 取得（帶 namespace）
    if not base_prompt and prompt_service:
        try:
            full_prompt_name = domain_config.get_prompt_name(prompt_name)
            base_prompt, _ = prompt_service.get_text_prompt(full_prompt_name)
        except Exception:
            pass

    # 3. 嘗試從 Langfuse 取得（不帶 namespace）
    if not base_prompt and prompt_service:
        try:
            base_prompt, _ = prompt_service.get_text_prompt(prompt_name)
        except Exception as exc:
            logger.debug(
                f"[INTENT_ANALYZER] Failed to fetch prompt from Langfuse: {exc}"
            )

    # 4. Fallback: 使用 DEFAULT_PROMPTS
    if not base_prompt:
        base_prompt = DEFAULT_PROMPTS[prompt_name]["prompt"]

    previous_answer_hint = last_ai_content[:600] if last_ai_content else ""

    system_msg = SystemMessage(content=base_prompt)
    human_parts = [
        f"使用者語言：{user_language}",
        f"最新問題：{user_question}",
    ]
    if previous_answer_hint:
        human_parts.append("上一輪助理回覆（可做後續處理）：")
        human_parts.append(previous_answer_hint)
    human_parts.append("\n請輸出單一 JSON，不要加額外文字或程式碼區塊。")

    human_msg = HumanMessage(content="\n".join(human_parts))

    start = time.monotonic()
    usage: Optional[dict[str, Any]] = None
    text: str = ""

    try:
        raw = await with_llm_semaphore(
            lambda: llm.ainvoke([system_msg, human_msg]),
            backend=agent_backend,
        )
        duration_ms = (time.monotonic() - start) * 1000.0
        usage = extract_usage_from_llm_output(raw)
        text = str(raw).strip() if raw is not None else ""

        # 使用括號深度追蹤提取第一個完整 JSON 物件
        json_str = extract_first_json_object(text)
        if not json_str:
            logger.debug(
                "[INTENT_ANALYZER] Raw LLM output (no JSON found): {}", text[:500]
            )
            raise ValueError("Intent analyzer 沒有輸出 JSON")

        # 嘗試解析 JSON，處理可能的 escape 問題
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as je:
            # 嘗試修復常見的 escape 問題
            # 1. 處理無效的 escape 序列（如 \n 在不應該出現的地方）
            logger.debug(
                "[INTENT_ANALYZER] JSON decode error: {}, attempting fix", je
            )
            # 將無效的 backslash 替換為空字串或有效的 escape
            import re
            # 修復無效的 \escape（保留有效的 \n, \t, \r, \\, \", \/ 等）
            fixed_str = re.sub(
                r'\\(?![nrtbf\\/"\u])',  # 匹配不是有效 escape 的 backslash
                r'\\\\',  # 替換為雙 backslash
                json_str
            )
            try:
                data = json.loads(fixed_str)
                logger.debug("[INTENT_ANALYZER] JSON fixed and parsed successfully")
            except json.JSONDecodeError:
                # 最後嘗試：移除所有非標準 escape
                cleaned_str = re.sub(r'\\(?![nrtbf\\/"u])', '', json_str)
                data = json.loads(cleaned_str)

        logger.debug("[INTENT_ANALYZER] Extracted JSON: {}", json_str[:300])

        # 使用 Pydantic 驗證（含 fallback 寬鬆解析）
        output = _parse_and_validate_intent(data)
        return output, usage, duration_ms

    except Exception as exc:
        logger.warning("[INTENT_ANALYZER] analysis fallback. error={}", exc)
        if "Extra data" in str(exc) or "JSON" in str(exc):
            logger.debug(
                "[INTENT_ANALYZER] Raw text that caused error: {}",
                text[:500] if text else "(empty)"
            )
        return default_output, usage, (time.monotonic() - start) * 1000.0


def build_intent_analyzer_node(
    base_llm_params: dict,
    *,
    agent_backend: str = "chat",
    prompt_service: Optional[PromptService] = None,
    domain_config: Optional[DomainConfig] = None,
) -> Callable[[State], State]:
    """
    建立意圖分析節點。

    Args:
        base_llm_params: LLM 參數
        agent_backend: Agent 後端類型
        prompt_service: Prompt 服務
        domain_config: 領域設定

    Returns:
        意圖分析節點函數
    """
    # Intent Analyzer 需要適度的 temperature 來靈活判斷意圖
    intent_analyzer_temperature = 0.3

    analyzer_llm: BaseChatModel
    if agent_backend == "chat":
        analyzer_llm = create_chat_completion_llm(
            streaming=False,
            model=base_llm_params.get("model"),
            temperature=intent_analyzer_temperature,
        )
    else:
        analyzer_llm = cast(
            BaseChatModel,
            create_responses_llm(
                streaming=False,
                reasoning_effort=base_llm_params.get("reasoning_effort", "low"),
                reasoning_summary=None,
                model=base_llm_params.get("model"),
                temperature=intent_analyzer_temperature,
            ),
        )

    async def intent_analyzer_node(state: State) -> State:
        writer = get_stream_writer()
        emit_node_event(
            writer,
            node="intent_analyzer",
            phase="planner",
            payload={
                "channel": "status",
                "stage": AskStreamStages.PLANNER_START,  # 使用相同的 stage 以保持向後相容
            },
        )

        state_dict = cast(StateDict, state)
        inputs = intent_analyzer_inputs_from_state(state_dict)
        user_question = inputs["question"] or extract_latest_human_message(
            state_dict.get("messages", [])
        )
        normalized_prev = inputs["prev_answer"]
        user_language = inputs["user_language"]

        # 檢查 guard 是否已經檢測到特定意圖
        guard_intent = state_dict.get("guard_intent")
        if guard_intent:
            # 直接使用 guard 檢測的意圖，不需要 LLM 分析
            logger.info(f"[INTENT_ANALYZER] Using guard-detected intent: {guard_intent}")

            # 根據 guard_intent 決定 routing_hint
            routing_hint = "direct_response"
            if guard_intent == "conversation_followup":
                routing_hint = "followup"

            intent_output = IntentAnalyzerOutput(
                intent=guard_intent,
                needs_retrieval=False,
                needs_followup=guard_intent == "conversation_followup",
                routing_hint=routing_hint,  # type: ignore
                confidence=1.0,
                reasoning="Detected by guard node",
            )

            new_state = cast(State, dict(state))
            new_state["intent_output"] = intent_output.to_dict()
            new_state["intent"] = guard_intent
            new_state["should_retrieve"] = False
            new_state["routing_hint"] = routing_hint

            # 向後相容：保留舊的 plan 結構
            new_state["plan"] = {
                "task_type": guard_intent,
                "should_retrieve": False,
                "tool_calls": [],
                "transform_instruction": None,
            }

            emit_node_event(
                writer,
                node="intent_analyzer",
                phase="planner",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.PLANNER_DONE,
                    "intent": guard_intent,
                    "routing_hint": routing_hint,
                    "should_retrieve": False,
                    "guard_detected": True,
                },
            )
            return new_state

        try:
            intent_output, usage, _ = await _analyze_intent(
                llm=analyzer_llm,
                user_question=user_question,
                last_ai_content=normalized_prev,
                user_language=user_language,
                agent_backend=agent_backend,
                prompt_service=prompt_service,
                domain_config=domain_config,
            )

            intent = intent_output.intent
            needs_retrieval = intent_output.needs_retrieval
            routing_hint = intent_output.routing_hint

            new_state = cast(State, dict(state))
            new_state["intent_output"] = intent_output.to_dict()
            new_state["intent"] = intent
            new_state["should_retrieve"] = needs_retrieval
            new_state["routing_hint"] = routing_hint

            # 新增：傳遞 query_type、retrieval_strategy 和 extracted_entities 到 state
            new_state["query_type"] = intent_output.query_type
            new_state["retrieval_strategy"] = intent_output.retrieval_strategy
            new_state["extracted_entities"] = intent_output.extracted_entities

            # 向後相容：保留舊的 plan 結構
            new_state["plan"] = {
                "task_type": intent,  # 使用 intent 作為 task_type
                "should_retrieve": needs_retrieval,
                "tool_calls": [],
                "transform_instruction": None,
            }

            # 如果是追問，設置 followup_instruction
            if intent_output.needs_followup:
                new_state["followup_instruction"] = user_question

            logger.info(
                "[INTENT_ANALYZER] Analysis result: intent={}, query_type={}, "
                "retrieval_strategy={}, entities={}",
                intent,
                intent_output.query_type,
                intent_output.retrieval_strategy,
                intent_output.extracted_entities,
            )

            emit_node_event(
                writer,
                node="intent_analyzer",
                phase="planner",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.PLANNER_DONE,
                    "intent": intent,
                    "routing_hint": routing_hint,
                    "should_retrieve": needs_retrieval,
                    "confidence": intent_output.confidence,
                    "query_type": intent_output.query_type,
                    "retrieval_strategy": intent_output.retrieval_strategy,
                },
            )

            if usage:
                emit_llm_meta_event(
                    writer,
                    node="intent_analyzer",
                    phase="planner",
                    component="intent_analyzer_llm",
                    usage=usage,
                )

            return new_state

        except Exception as exc:
            logger.error("[INTENT_ANALYZER] Failed to analyze intent: {}", exc)
            emit_node_event(
                writer,
                node="intent_analyzer",
                phase="planner",
                payload={
                    "channel": "status",
                    "stage": AskStreamStages.PLANNER_ERROR,
                    "error": str(exc),
                },
            )
            return state

    return intent_analyzer_node
