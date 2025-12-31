"""
Telemetry 節點：彙整 Langfuse 追蹤與 SSE 狀態。

功能：
- 更新 Langfuse trace（tags、metadata、output）
- 使用採樣機制減少高流量時的遙測開銷
- 非同步執行避免阻塞 event loop

前置節點：cache_store | cache_response
後續節點：END
"""

from functools import partial
from typing import Any, Callable, Dict, List, Optional, cast

import anyio
from langchain_core.messages import HumanMessage, AIMessage
from langfuse import get_client as get_langfuse_client
from langgraph.config import get_stream_writer
from loguru import logger

from chatbot_rag.core.telemetry_sampler import telemetry_sampler
from chatbot_rag.llm import State
from chatbot_rag.services.prompt_service import PromptService
from ...types import StateDict, telemetry_inputs_from_state
from ...constants import AskStreamStages
from ...events import emit_node_event


async def _update_langfuse_trace_async(
    tags: List[str],
    metadata: Dict[str, Any],
    output: Dict[str, Any],
) -> None:
    """
    非同步更新 Langfuse trace。

    使用 anyio.to_thread 將同步的 Langfuse API 呼叫
    移至 thread pool，避免阻塞 event loop。
    """
    def _update_trace():
        langfuse = get_langfuse_client()
        langfuse.update_current_trace(
            tags=tags,
            metadata=metadata,
            output=output,
        )

    await anyio.to_thread.run_sync(_update_trace)


def build_telemetry_node(
    *,
    prompt_service: Optional[PromptService] = None,
) -> Callable[[State], State]:
    """
    Telemetry 節點：彙整 Langfuse 與 SSE 狀態。

    包含採樣機制，根據 langfuse_sample_rate 設定決定是否記錄遙測。
    使用 async 避免阻塞 event loop。
    """

    async def telemetry_node(state: State) -> State:
        writer = get_stream_writer()
        telemetry_inputs = telemetry_inputs_from_state(cast(StateDict, state))
        intent = telemetry_inputs["intent"]
        used_tools = telemetry_inputs["used_tools"]
        user_language = telemetry_inputs["user_language"]
        is_out_of_scope = telemetry_inputs["is_out_of_scope"]
        final_answer = telemetry_inputs["final_answer"]
        eval_question = telemetry_inputs["eval_question"]
        eval_context = telemetry_inputs["eval_context"]
        eval_query_rewrite = telemetry_inputs["eval_query_rewrite"]

        conversation_history_for_eval: List[Dict[str, str]] = []
        for msg in telemetry_inputs["messages"]:
            if isinstance(msg, HumanMessage):
                conversation_history_for_eval.append(
                    {"role": "user", "content": getattr(msg, "content", "") or ""}
                )
            elif isinstance(msg, AIMessage):
                content = getattr(msg, "content", "") or ""
                if content:
                    conversation_history_for_eval.append(
                        {"role": "assistant", "content": content}
                    )

        conversation_history_text = ""
        if len(conversation_history_for_eval) > 1:
            history_lines = []
            for history_entry in conversation_history_for_eval[:-1]:
                role_label = "User" if history_entry["role"] == "user" else "Assistant"
                content = history_entry["content"]
                if len(content) > 500:
                    content = content[:500] + "..."
                history_lines.append(f"[{role_label}]: {content}")
            conversation_history_text = "\n\n".join(history_lines)

        # 獲取 prompt 版本資訊
        prompts_used: Dict[str, Any] = {}
        if prompt_service:
            prompts_used = prompt_service.get_used_prompts()
            # 清除使用記錄，為下次請求做準備
            prompt_service.clear_used_prompts()

        # 採樣判斷：只有被採樣的請求才會更新 Langfuse
        if telemetry_sampler.should_sample():
            try:
                tags = ["unified_agent", f"intent:{intent}"]
                metadata = {
                    "flow_type": "unified_agent",
                    "intent": intent,
                    "is_out_of_scope": is_out_of_scope,
                    "user_language": user_language,
                    "used_tools": used_tools,
                    "eval_question": eval_question,
                    "eval_context": eval_context,
                    "eval_query_rewrite": eval_query_rewrite,
                    "eval_answer": final_answer,
                    "prompts_used": prompts_used,
                }
                output = {
                    "eval_question": eval_question,
                    "eval_context": eval_context,
                    "eval_answer": final_answer,
                    "eval_conversation_history": conversation_history_text,
                    "eval_query_rewrite": eval_query_rewrite,
                    "eval_intent": intent,
                    "eval_is_followup": intent == "conversation_followup",
                    "eval_used_tools": used_tools,
                }
                # 非同步更新，避免阻塞 event loop
                await _update_langfuse_trace_async(tags, metadata, output)
            except Exception as exc:  # noqa: BLE001 pylint: disable=broad-exception-caught
                logger.warning("[TELEMETRY] Failed to update Langfuse: {}", exc)
        else:
            logger.debug("[TELEMETRY] Request not sampled, skipping Langfuse update")

        emit_node_event(
            writer,
            node="telemetry",
            phase="telemetry",
            payload={
                "channel": "status",
                "stage": AskStreamStages.TELEMETRY_SUMMARY,
                "intent": intent,
                "used_tools": used_tools,
            },
        )
        return state

    return telemetry_node
