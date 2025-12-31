"""
Followup Transform 節點：追問任務前處理。

功能：
- 驗證是否有上一輪回答可供追問
- 無上一輪回答時 fallback 到一般檢索

前置節點：intent_analyzer（routing_hint="followup"）
後續節點：response_synth
"""

from typing import Callable, cast

from langgraph.config import get_stream_writer

from chatbot_rag.llm import State
from ...types import StateDict
from ...constants import AskStreamStages
from ...events import emit_node_event


def build_followup_transform_node() -> Callable[[State], State]:
    """追問任務資料前處理。"""

    async def followup_node(state: State) -> State:
        writer = get_stream_writer()
        emit_node_event(
            writer,
            node="followup_transform",
            phase="followup",
            payload={
                "channel": "status",
                "stage": AskStreamStages.FOLLOWUP_START,
            },
        )
        state_dict = cast(StateDict, state)
        new_state = cast(State, dict(state))
        prev_answer = state_dict.get("prev_answer_normalized") or ""
        fallback = False
        if not prev_answer:
            plan = dict(new_state.get("plan") or {})
            plan["task_type"] = "simple_faq"
            plan["should_retrieve"] = True
            plan.setdefault("tool_calls", [])
            new_state["plan"] = plan
            new_state["should_retrieve"] = True
            fallback = True
        else:
            new_state["followup_ready"] = True

        emit_node_event(
            writer,
            node="followup_transform",
            phase="followup",
            payload={
                "channel": "status",
                "stage": AskStreamStages.FOLLOWUP_DONE,
                "fallback_to_retrieval": fallback,
            },
        )
        return new_state

    return followup_node
