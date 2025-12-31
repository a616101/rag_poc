"""
Query builder 節點：將使用者問題轉換為檢索查詢。

功能：
- 透過 LLM 產生多個查詢變體（Multi-Query 策略）
- 將查詢翻譯為繁體中文以提高檢索命中率
- 重試時使用 QueryVariationStrategy 進行查詢變異
- 建立 tool_calls 供 tool_executor 執行

設計原則：
- 使用 Langfuse prompt management
- 支援 chat 和 responses 後端
"""

from typing import Any, Callable, Optional, List, cast
import json
import time

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.config import get_stream_writer
from loguru import logger

from chatbot_rag.core.concurrency import with_llm_semaphore
from chatbot_rag.llm import State, create_chat_completion_llm, create_responses_llm
from chatbot_rag.llm.graph_nodes import retrieval_top_k_var
from chatbot_rag.services.language_utils import (
    detect_preferred_language,
    extract_usage_from_llm_output,
    translate_query_to_zh_hant_for_retrieval_with_metrics_async,
)
from chatbot_rag.services.prompt_service import PromptService, PromptNames, DEFAULT_PROMPTS
from chatbot_rag.services.query_variation import (
    QueryVariationStrategy,
    generate_variation_async,
)
from ...types import (
    StateDict,
    TaskPlan,
    query_builder_inputs_from_state,
    extract_latest_human_message,
)
from ...constants import AskStreamStages, UNIFIED_AGENT_TOOLS
from ...events import emit_node_event, emit_llm_meta_event
from ...utils import (
    message_to_text,
    strip_code_fence,
    maybe_fix_character_spaced_query,
    summarize_recent_messages,
    extract_first_json_object,
)


# 取得 retrieve_documents_tool 的名稱
retrieve_documents_tool = UNIFIED_AGENT_TOOLS[0]

# Multi-Query 設定：不限制查詢數量，由 LLM 決定
MAX_DECOMPOSED_QUERIES = 10  # 安全上限，防止過多查詢


def _get_query_decompose_prompt(prompt_service: Optional[PromptService]) -> str:
    """
    取得 Query Decompose prompt。

    優先順序：Langfuse → DEFAULT_PROMPTS
    """
    prompt_name = PromptNames.QUERY_DECOMPOSE_SYSTEM

    if prompt_service:
        try:
            prompt_text, _ = prompt_service.get_text_prompt(prompt_name)
            if prompt_text:
                return prompt_text
        except Exception as exc:
            logger.debug(f"[QUERY_BUILDER] Failed to fetch prompt from Langfuse: {exc}")

    return DEFAULT_PROMPTS[prompt_name]["prompt"]


async def _rewrite_and_decompose_query(
    *,
    llm: BaseChatModel,
    question: str,
    user_language: str,
    prev_answer: str = "",
    conversation_summary: str = "",
    messages: Optional[List[BaseMessage]] = None,
    agent_backend: str = "chat",
    prompt_service: Optional[PromptService] = None,
) -> tuple[str, List[str], str, Optional[dict[str, Any]], float]:
    """
    透過 LLM 產生多個查詢變體（類似 ChatGPT 的多查詢策略）。

    結合對話上下文，讓追問類型的問題可以正確融合前文。

    Args:
        llm: LLM 實例
        question: 使用者問題
        user_language: 使用者語言
        prev_answer: 上一輪助理回答（用於追問融合）
        conversation_summary: 對話摘要
        messages: 對話歷史
        agent_backend: Agent 後端類型
        prompt_service: Langfuse Prompt 服務

    Returns:
        tuple: (主要查詢, 所有查詢列表, 分解原因, usage, 耗時毫秒)
    """
    start = time.monotonic()
    default_queries = [question]
    default_primary = question
    default_reason = "使用原始查詢"

    try:
        # 從 Langfuse 或 DEFAULT_PROMPTS 取得 prompt
        decompose_prompt = _get_query_decompose_prompt(prompt_service)
        system_msg = SystemMessage(content=decompose_prompt)

        # 建構 human message，包含對話上下文
        human_parts = [
            f"使用者語言：{user_language}",
            f"問題：{question}",
        ]

        # 如果有上一輪回答，提供給 LLM 以便融合追問
        if prev_answer:
            prev_snippet = prev_answer[:500] + "..." if len(prev_answer) > 500 else prev_answer
            human_parts.append(f"\n上一輪助理回答：\n{prev_snippet}")

        # 如果有對話摘要
        if conversation_summary:
            human_parts.append(f"\n對話摘要：{conversation_summary}")

        # 如果有對話歷史，提取近期上下文
        if messages:
            recent_context = summarize_recent_messages(messages, limit=3)
            if recent_context:
                human_parts.append(f"\n近期對話：\n{recent_context}")

        human_parts.append("\n請輸出 JSON 格式的查詢結果。如果是追問，請將追問與前文主題融合。")
        human_msg = HumanMessage(content="\n".join(human_parts))

        raw = await with_llm_semaphore(
            lambda: llm.ainvoke([system_msg, human_msg]),
            backend=agent_backend,
        )
        duration_ms = (time.monotonic() - start) * 1000.0
        usage = extract_usage_from_llm_output(raw)
        text = str(raw.content).strip() if hasattr(raw, "content") else str(raw).strip()

        # 解析 JSON
        json_str = extract_first_json_object(text)
        if not json_str:
            logger.warning("[QUERY_BUILDER] No JSON found in decompose output: %s", text[:200])
            return default_primary, default_queries, default_reason, usage, duration_ms

        data = json.loads(json_str)
        queries = data.get("queries", [])
        primary = data.get("primary", "")
        reason = data.get("reason", "")

        # 驗證結果
        if not isinstance(queries, list) or len(queries) == 0:
            logger.warning("[QUERY_BUILDER] Invalid queries format in decompose")
            return default_primary, default_queries, default_reason, usage, duration_ms

        # 過濾空字串並限制數量
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        if len(queries) == 0:
            return default_primary, default_queries, default_reason, usage, duration_ms

        queries = queries[:MAX_DECOMPOSED_QUERIES]
        primary = primary.strip() if primary else queries[0]

        logger.info(
            "[QUERY_BUILDER] Decomposed into {} queries: {}",
            len(queries),
            queries,
        )

        return primary, queries, reason, usage, duration_ms

    except json.JSONDecodeError as exc:
        logger.warning("[QUERY_BUILDER] JSON parse error in decompose: %s", exc)
        return default_primary, default_queries, default_reason, None, (time.monotonic() - start) * 1000.0
    except Exception as exc:
        logger.warning("[QUERY_BUILDER] Query decomposition failed: %s", exc)
        return default_primary, default_queries, default_reason, None, (time.monotonic() - start) * 1000.0


async def _rewrite_query_with_context(
    *,
    llm: BaseChatModel,
    seed_query: str,
    plan: TaskPlan,
    latest_question: str,
    normalized_question: str,
    prev_answer: str,
    followup_instruction: str,
    user_language: str,
    loop: int,
    retrieval_status: Optional[str],
    messages: List[BaseMessage],
    agent_backend: str = "chat",
    prompt_service: Optional[PromptService] = None,
) -> tuple[
    Optional[str], Optional[dict[str, Any]], Optional[float], Optional[BaseMessage]
]:
    """利用 LLM 參考前後文重寫檢索 Query（非同步版本）。"""
    candidate = seed_query.strip()
    if not candidate:
        return None, None, None, None

    start = time.monotonic()
    raw_message: Optional[BaseMessage] = None
    try:
        plan_task = plan.get("task_type", "simple_faq")
        tool_names = [call.get("name", "") for call in plan.get("tool_calls", []) or []]
        tool_summary = ", ".join(filter(None, tool_names)) or "無"
        conversation_snippet = summarize_recent_messages(messages, limit=4)
        prev_answer_snippet = prev_answer[:600] + "..." if prev_answer and len(prev_answer) > 600 else prev_answer

        loop_hint = ""
        if loop > 1:
            loop_hint = f"目前為第 {loop} 次嘗試，上一輪檢索狀態：{retrieval_status or '無資料'}。"

        instructions = [
            f"使用者語言：{user_language}",
            f"最新原問題：{latest_question or normalized_question or candidate}",
            f"規劃後的標準問題：{normalized_question or candidate}",
            f"任務類型：{plan_task}",
            f"預計使用工具：{tool_summary}",
        ]
        if followup_instruction:
            instructions.append(f"額外指示：{followup_instruction}")
        if prev_answer_snippet:
            instructions.append("上一輪助理回答摘要：")
            instructions.append(prev_answer_snippet)
        if conversation_snippet:
            instructions.append("近期對話片段：")
            instructions.append(conversation_snippet)
        if loop_hint:
            instructions.append(loop_hint)
        instructions.append(
            "【重要】請根據以上資訊，重寫成**一段自然句**的查詢描述。"
            "若這是追問，必須將追問與前文主題融合（如「找誰」要改成「頭痛找哪位醫師」）。"
            "重寫後的查詢必須能獨立理解，不依賴對話上下文。"
            "只需輸出重寫後的查詢，不要加任何說明。"
        )

        # 從 Langfuse 獲取 query rewriter prompt（fallback 使用 DEFAULT_PROMPTS）
        fallback_prompt = DEFAULT_PROMPTS[PromptNames.QUERY_REWRITER_SYSTEM]["prompt"]
        if prompt_service:
            try:
                system_prompt, _ = prompt_service.get_text_prompt(
                    PromptNames.QUERY_REWRITER_SYSTEM
                )
            except Exception as exc:
                logger.warning(
                    f"[QUERY_BUILDER] Failed to fetch prompt from Langfuse: {exc}"
                )
                system_prompt = fallback_prompt
        else:
            system_prompt = fallback_prompt

        system_msg = SystemMessage(content=system_prompt)
        human_msg = HumanMessage(content="\n".join(instructions))
        raw = await with_llm_semaphore(
            lambda: llm.ainvoke([system_msg, human_msg]),
            backend=agent_backend,
        )
        duration_ms = (time.monotonic() - start) * 1000.0
        usage = extract_usage_from_llm_output(raw)
        raw_message = cast(Optional[BaseMessage], raw)
        text = message_to_text(raw).strip()
        text = strip_code_fence(text)
        if not text:
            return None, usage, duration_ms, raw_message
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        rewritten = first_line or text
        rewritten, fixed = maybe_fix_character_spaced_query(rewritten)
        if fixed:
            logger.debug("[QUERY_BUILDER] Fixed character-spaced query: %s", rewritten)
        return rewritten, usage, duration_ms, raw_message
    except Exception as exc:  # noqa: BLE001 pylint: disable=broad-exception-caught
        logger.warning("[QUERY_BUILDER] contextual rewrite failed: {}", exc)
        return None, None, (time.monotonic() - start) * 1000.0, raw_message


def build_query_builder_node(
    base_llm_params: dict,
    *,
    agent_backend: str = "chat",
    prompt_service: Optional[PromptService] = None,
) -> Callable[[State], State]:
    """Query builder：統一準備檢索 query 與工具計畫。"""

    # Query Decompose 需要較高的 temperature 來產生多樣化的查詢變體
    query_decompose_temperature = 0.5

    query_rewriter_llm: BaseChatModel
    if agent_backend == "chat":
        query_rewriter_llm = create_chat_completion_llm(
            streaming=False,
            model=base_llm_params.get("model"),
            temperature=query_decompose_temperature,
        )
    else:
        query_rewriter_llm = cast(
            BaseChatModel,
            create_responses_llm(
                streaming=False,
                reasoning_effort=base_llm_params.get("reasoning_effort", "low"),
                reasoning_summary=None,
                model=base_llm_params.get("model"),
                temperature=query_decompose_temperature,
            ),
        )

    async def query_builder(state: State) -> State:
        writer = get_stream_writer()
        emit_node_event(
            writer,
            node="query_builder",
            phase="retrieval",
            payload={
                "channel": "status",
                "stage": AskStreamStages.QUERY_BUILDER_START,
            },
        )
        state_dict = cast(StateDict, state)
        qb_inputs = query_builder_inputs_from_state(state_dict)
        plan = dict(qb_inputs["plan"])
        normalized_question = qb_inputs["normalized_question"]
        latest_question = state_dict.get("latest_question") or extract_latest_human_message(
            state_dict.get("messages", [])
        )
        prev_answer = state_dict.get("prev_answer_normalized") or ""
        followup_instruction = state_dict.get("followup_instruction") or ""
        user_language = state_dict.get("user_language", "zh-hant")
        messages_list: List[BaseMessage] = state_dict.get("messages", [])
        retrieval_state = dict(qb_inputs["retrieval"])
        loop = int(retrieval_state.get("loop", 0)) + 1
        retrieval_state["loop"] = loop
        plan_tool_calls_raw = plan.get("tool_calls") or []
        plan_tool_calls: List[dict[str, Any]] = list(
            cast(List[dict[str, Any]], plan_tool_calls_raw)
        )
        should_retrieve = bool(plan.get("should_retrieve", True))

        # 取得檢索策略（從 intent_analyzer 傳遞）
        retrieval_strategy = state_dict.get("retrieval_strategy", "vector")
        extracted_entities = state_dict.get("extracted_entities", {})

        # 如果是 metadata_filter 策略，使用 list_by_metadata_tool
        if retrieval_strategy == "metadata_filter" and extracted_entities:
            logger.info(
                "[QUERY_BUILDER] Using metadata_filter strategy with entities: {}",
                extracted_entities,
            )

            # 建構 list_by_metadata_tool 的參數
            # 優先使用 department，其次使用 entry_type
            filter_field = "department" if "department" in extracted_entities else "entry_type"
            filter_value = extracted_entities.get(filter_field, "")

            # 注意：科別名稱的模糊匹配由 agent_tools.py 中的 list_by_fuzzy_filter 處理
            # 不需要在這裡進行硬編碼的正規化

            if filter_value:
                # 構建 metadata filter 工具呼叫
                metadata_tool_call = {
                    "name": "list_by_metadata_tool",
                    "args": {
                        "filter_field": filter_field,
                        "filter_value": filter_value,
                    },
                }

                # 如果有額外的 entry_type 過濾，加入參數
                if filter_field == "department" and "entry_type" in extracted_entities:
                    metadata_tool_call["args"]["entry_type"] = extracted_entities["entry_type"]

                plan_tool_calls = [metadata_tool_call]
                retrieval_state["strategy"] = "metadata_filter"
                retrieval_state["filter_params"] = extracted_entities

                # 更新 state 並返回
                new_state = cast(State, dict(state))
                new_state["retrieval"] = retrieval_state
                new_state["active_tool_calls"] = plan_tool_calls
                new_state["retrieval_strategy"] = retrieval_strategy
                new_state["extracted_entities"] = extracted_entities

                emit_node_event(
                    writer,
                    node="query_builder",
                    phase="retrieval",
                    payload={
                        "channel": "status",
                        "stage": AskStreamStages.QUERY_BUILDER_DONE,
                        "retrieval_strategy": "metadata_filter",
                        "filter_field": filter_field,
                        "filter_value": filter_value,
                        "loop": loop,
                    },
                )
                return new_state

        if not plan_tool_calls and should_retrieve:
            plan_tool_calls = [
                {"name": retrieve_documents_tool.name, "args": {"query": normalized_question}}
            ]

        if loop > 1:
            plan_tool_calls = [
                call
                for call in plan_tool_calls
                if call.get("name") == retrieve_documents_tool.name
            ] or [
                {"name": retrieve_documents_tool.name, "args": {"query": normalized_question}}
            ]

        base_query = normalized_question
        for call in plan_tool_calls:
            if call.get("name") == retrieve_documents_tool.name:
                base_query = call.get("args", {}).get("query") or normalized_question
                break

        decomposed_queries: List[str] = []
        decompose_reason: str = ""
        decompose_usage: Optional[dict[str, Any]] = None
        multi_query_enabled = False

        if loop == 1:
            # 第一次查詢：永遠使用 Multi-Query 分解（類似 ChatGPT）
            # 傳入對話上下文，讓追問類問題可以正確融合前文
            conversation_summary = state_dict.get("conversation_summary") or ""
            (
                primary_query,
                decomposed_queries,
                decompose_reason,
                decompose_usage,
                _,
            ) = await _rewrite_and_decompose_query(
                llm=query_rewriter_llm,
                question=base_query,
                user_language=user_language,
                prev_answer=prev_answer,
                conversation_summary=conversation_summary,
                messages=messages_list,
                agent_backend=agent_backend,
                prompt_service=prompt_service,
            )
            base_query = primary_query
            multi_query_enabled = len(decomposed_queries) > 1
            retrieval_state["decomposed_queries"] = decomposed_queries
            retrieval_state["query_decompose_reason"] = decompose_reason
            retrieval_state["variation_strategy_used"] = "multi_query"

            logger.info(
                "[QUERY_BUILDER] Multi-query with {} queries: {}",
                len(decomposed_queries),
                decomposed_queries,
            )
        else:
            # 重試時：使用 Smart Retry 策略
            variation_strategy = QueryVariationStrategy.NONE
            variation_strategy_str = retrieval_state.get("next_strategy")
            if variation_strategy_str:
                try:
                    variation_strategy = QueryVariationStrategy(variation_strategy_str)
                except ValueError:
                    variation_strategy = QueryVariationStrategy.NONE

            if variation_strategy != QueryVariationStrategy.NONE and query_rewriter_llm:
                logger.info(
                    "[QUERY_BUILDER] Applying variation strategy=%s for loop=%d",
                    variation_strategy.value,
                    loop,
                )
                varied_result = await generate_variation_async(
                    base_query,
                    variation_strategy,
                    query_rewriter_llm,
                )
                # 如果是 DECOMPOSE 策略，取第一個子查詢作為主查詢
                if isinstance(varied_result, list):
                    base_query = varied_result[0] if varied_result else base_query
                    retrieval_state["decomposed_queries"] = varied_result
                    multi_query_enabled = len(varied_result) > 1
                else:
                    base_query = varied_result
                retrieval_state["variation_strategy_used"] = variation_strategy.value
            else:
                retrieval_state["variation_strategy_used"] = QueryVariationStrategy.NONE.value

        rewritten_query = base_query
        rewrite_usage: Optional[dict[str, Any]] = None
        was_rewritten = False
        rewrite_message: Optional[BaseMessage] = None

        # 第一次查詢已經透過 Multi-Query 分解處理，不需要額外的 contextual rewrite
        # 只有在重試時（loop > 1）且使用 variation strategy 後，才可能需要 contextual rewrite
        # 但目前 variation strategy 已經處理了查詢變化，所以跳過 contextual rewrite
        skip_contextual_rewrite = loop == 1  # 第一次查詢永遠跳過（已有 decompose）

        if should_retrieve and query_rewriter_llm and base_query.strip() and not skip_contextual_rewrite:
            (
                contextual_query,
                rewrite_usage,
                _,
                rewrite_message,
            ) = await _rewrite_query_with_context(
                llm=query_rewriter_llm,
                seed_query=base_query,
                plan=cast(TaskPlan, plan),
                latest_question=latest_question,
                normalized_question=normalized_question,
                prev_answer=prev_answer,
                followup_instruction=followup_instruction,
                user_language=user_language,
                loop=loop,
                retrieval_status=retrieval_state.get("status"),
                messages=messages_list,
                agent_backend=agent_backend,
                prompt_service=prompt_service,
            )
            if contextual_query:
                rewritten_query = contextual_query
                was_rewritten = rewritten_query != base_query
                base_query = rewritten_query

        query_lang = detect_preferred_language(base_query)
        translation_usage: Optional[dict[str, Any]] = None
        if query_lang not in ("zh-hant", "zh-hans"):
            (
                effective_query,
                translation_usage,
                _,
            ) = await translate_query_to_zh_hant_for_retrieval_with_metrics_async(
                base_query,
                base_llm_params=base_llm_params,
                agent_backend=agent_backend,
            )
        else:
            effective_query = base_query

        for call in plan_tool_calls:
            if call.get("name") == retrieve_documents_tool.name:
                call_args = dict(call.get("args") or {})
                call_args["query"] = effective_query
                call["args"] = call_args

        retrieval_state.update(
            {
                "query": effective_query,
                "raw_chunks": [],
                "status": "pending",
                "rewritten_query_message": rewrite_message,
            }
        )

        top_k = qb_inputs["top_k"]
        retrieval_top_k_var.set(top_k)

        new_state = cast(State, dict(state))
        new_state["retrieval"] = retrieval_state
        new_state["active_tool_calls"] = plan_tool_calls
        new_state["summary_search_query"] = effective_query

        # 更新 state 中的 multi-query 欄位
        new_state["decomposed_queries"] = decomposed_queries if decomposed_queries else [effective_query]
        new_state["query_decompose_reason"] = decompose_reason

        emit_node_event(
            writer,
            node="query_builder",
            phase="retrieval",
            payload={
                "channel": "status",
                "stage": AskStreamStages.QUERY_BUILDER_DONE,
                "query": effective_query,
                "contextual_query": rewritten_query,
                "loop": loop,
                "rewritten": was_rewritten,
                "variation_strategy": retrieval_state.get("variation_strategy_used"),
                "decomposed_queries": decomposed_queries,
                "decompose_reason": decompose_reason,
            },
        )
        if decompose_usage:
            emit_llm_meta_event(
                writer,
                node="query_builder",
                phase="retrieval",
                component="query_decompose",
                usage=decompose_usage,
            )
        if rewrite_usage:
            emit_llm_meta_event(
                writer,
                node="query_builder",
                phase="retrieval",
                component="contextual_query_rewrite",
                usage=rewrite_usage,
            )
        if translation_usage:
            emit_llm_meta_event(
                writer,
                node="query_builder",
                phase="retrieval",
                component="query_translation",
                usage=translation_usage,
            )
        return new_state

    return query_builder
