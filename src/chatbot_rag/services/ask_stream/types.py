"""
ask_stream 模組的類型定義。

TypedDict 定義：
- AskStreamEvent: SSE 事件結構
- TaskPlan: 任務規劃產物（相容舊架構）
- IntentAnalyzerInputs: 意圖分析器輸入
- QueryBuilderInputs: 查詢建構器輸入
- ToolExecutorInputs: 工具執行器輸入
- GenerationInputs: 回應生成器輸入
- TelemetryInputs: 遙測輸入

狀態轉換函數：
- intent_analyzer_inputs_from_state: 提取意圖分析器輸入
- query_builder_inputs_from_state: 提取查詢建構器輸入
- tool_executor_inputs_from_state: 提取工具執行器輸入
- generation_inputs_from_state: 提取回應生成器輸入
- telemetry_inputs_from_state: 提取遙測輸入
"""

from typing import Any, Optional, List, Dict, TypedDict, cast

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from chatbot_rag.llm import State


class AskStreamEvent(TypedDict, total=False):
    """/ask/stream 事件的標準結構，用於 SSE payload。"""

    source: str
    node: str
    phase: str
    channel: str
    stage: str
    node_stage: str
    search_query: str
    is_out_of_scope: bool
    documents_count: int
    error: str
    guard_reason: str


class TaskPlan(TypedDict, total=False):
    """Unified agent 任務規劃產物。"""

    task_type: str
    should_retrieve: bool
    tool_calls: List[dict[str, Any]]
    transform_instruction: Optional[str]


class IntentAnalyzerInputs(TypedDict):
    """意圖分析器輸入結構。"""

    question: str
    normalized_question: str
    prev_answer: str
    user_language: str


# 向後相容別名
PlannerInputs = IntentAnalyzerInputs


class QueryBuilderInputs(TypedDict):
    normalized_question: str
    plan: TaskPlan
    retrieval: Dict[str, Any]
    top_k: int


class ToolExecutorInputs(TypedDict):
    tool_calls: List[dict[str, Any]]
    user_question: str
    retrieval_query: str
    used_tools: List[str]


class GenerationInputs(TypedDict):
    task_type: str
    intent: str
    user_language: str
    normalized_question: str
    followup_instruction: str
    prev_answer: str
    context_text: str
    used_tools: List[str]
    loop_count: int
    is_followup: bool
    is_out_of_scope: bool


class TelemetryInputs(TypedDict):
    intent: Optional[str]
    used_tools: List[str]
    user_language: str
    is_out_of_scope: Optional[bool]
    final_answer: str
    eval_question: str
    eval_context: str
    eval_query_rewrite: str
    messages: List[BaseMessage]


# Type alias for state dict operations
StateDict = Dict[str, Any]


def intent_analyzer_inputs_from_state(state_dict: StateDict) -> IntentAnalyzerInputs:
    """從 State 中提取意圖分析器所需的輸入。"""
    question = state_dict.get("latest_question") or ""
    normalized_question = state_dict.get("normalized_question") or question
    return {
        "question": question,
        "normalized_question": normalized_question,
        "prev_answer": state_dict.get("prev_answer_normalized") or "",
        "user_language": state_dict.get("user_language", "zh-hant"),
    }


# 向後相容別名
planner_inputs_from_state = intent_analyzer_inputs_from_state


def query_builder_inputs_from_state(state_dict: StateDict) -> QueryBuilderInputs:
    """從 State 中提取 QueryBuilder 所需的輸入。"""
    plan = dict(state_dict.get("plan") or {})
    retrieval = dict(state_dict.get("retrieval") or {})
    normalized_question = state_dict.get("normalized_question") or state_dict.get(
        "latest_question", ""
    )
    return {
        "normalized_question": normalized_question,
        "plan": cast(TaskPlan, plan),
        "retrieval": retrieval,
        "top_k": state_dict.get("top_k", 3),
    }


def tool_executor_inputs_from_state(state_dict: StateDict) -> ToolExecutorInputs:
    """從 State 中提取 ToolExecutor 所需的輸入。"""
    retrieval_state = dict(state_dict.get("retrieval") or {})
    return {
        "tool_calls": list(state_dict.get("active_tool_calls") or []),
        "user_question": state_dict.get("latest_question") or "",
        "retrieval_query": retrieval_state.get("query") or "",
        "used_tools": list(state_dict.get("used_tools") or []),
    }


def generation_inputs_from_state(state_dict: StateDict) -> GenerationInputs:
    """從 State 中提取 Generation 所需的輸入。"""
    plan = cast(TaskPlan, state_dict.get("plan") or {})
    task_type = plan.get("task_type", "simple_faq")
    intent = state_dict.get("intent", task_type)
    user_language = state_dict.get("user_language", "zh-hant")
    normalized_question = state_dict.get("normalized_question") or state_dict.get(
        "latest_question", ""
    )
    followup_instruction = state_dict.get("followup_instruction") or ""
    prev_answer = state_dict.get("prev_answer_normalized") or ""
    context_text = state_dict.get("context") or ""
    used_tools = state_dict.get("used_tools") or []
    retrieval_state = state_dict.get("retrieval") or {}
    loop_count = retrieval_state.get("loop", 0) or 0
    # is_followup 判斷：
    # 1. task_type 為 conversation_followup 且有 prev_answer（從 intent_analyzer）
    # 2. 或者經由 followup_transform 設定了 followup_ready 標記
    is_followup = (
        (task_type == "conversation_followup" and bool(prev_answer))
        or bool(state_dict.get("followup_ready", False))
    )
    is_out_of_scope = intent == "out_of_scope"
    return {
        "task_type": task_type,
        "intent": intent,
        "user_language": user_language,
        "normalized_question": normalized_question,
        "followup_instruction": followup_instruction,
        "prev_answer": prev_answer,
        "context_text": context_text,
        "used_tools": used_tools,
        "loop_count": loop_count,
        "is_followup": is_followup,
        "is_out_of_scope": is_out_of_scope,
    }


def telemetry_inputs_from_state(state_dict: StateDict) -> TelemetryInputs:
    """從 State 中提取 Telemetry 所需的輸入。"""
    return {
        "intent": state_dict.get("intent"),
        "used_tools": state_dict.get("used_tools") or [],
        "user_language": state_dict.get("user_language", "zh-hant"),
        "is_out_of_scope": state_dict.get("is_out_of_scope"),
        "final_answer": state_dict.get("final_answer", ""),
        "eval_question": state_dict.get("eval_question", ""),
        "eval_context": state_dict.get("eval_context", ""),
        "eval_query_rewrite": state_dict.get("eval_query_rewrite", ""),
        "messages": state_dict.get("messages", []),
    }


def extract_latest_human_message(messages: List[BaseMessage]) -> str:
    """取得最新一則使用者訊息內容。"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return getattr(msg, "content", "") or ""
    return ""


def extract_last_ai_message(messages: List[BaseMessage]) -> str:
    """取得上一輪助理回答內容。"""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = getattr(msg, "content", "") or ""
            if content:
                return content
    return ""
