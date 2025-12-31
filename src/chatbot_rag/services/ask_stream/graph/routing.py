"""
LangGraph 條件路由函數。

定義圖中各節點的條件分支邏輯：
- route_after_guard: guard → end | language_normalizer
- route_after_cache_lookup: cache_lookup → cache_response | intent_analyzer
- route_after_intent_analyzer: intent_analyzer → response_synth | followup_transform | query_builder
- route_after_result_evaluator: result_evaluator → query_builder (retry) | response_synth

設計原則：
- 路由決策基於 state 中的欄位（routing_hint、needs_retrieval 等）
- 不硬編碼特定領域邏輯，保持通用性
"""

from typing import cast

from chatbot_rag.llm import State
from ..types import StateDict


def route_after_guard(state: State) -> str:
    """Guard 節點後的路由：被擋則結束，否則進入語言正規化。"""
    state_dict = cast(StateDict, state)
    if state_dict.get("guard_blocked", False):
        return "end"
    return "language_normalizer"


def route_after_cache_lookup(state: State) -> str:
    """Cache lookup 節點後的路由：命中則返回快取，否則進入意圖分析。"""
    state_dict = cast(StateDict, state)
    if state_dict.get("cache_hit", False):
        return "cache_response"
    return "intent_analyzer"


def route_after_intent_analyzer(state: State) -> str:
    """
    Intent Analyzer 節點後的路由。

    領域無關設計：
    - 基於 LLM 輸出的 routing_hint 和 needs_retrieval 決定路由
    - 不硬編碼特定任務類型

    路由邏輯：
    1. routing_hint == "direct_response" → response_synth
    2. routing_hint == "followup" 且有上一輪回答 → followup_transform
    3. needs_retrieval == True → query_builder
    4. 其他 → response_synth
    """
    state_dict = cast(StateDict, state)

    # 讀取 intent_output
    intent_output = state_dict.get("intent_output") or {}
    routing_hint = intent_output.get("routing_hint") or state_dict.get("routing_hint", "continue")
    needs_retrieval = intent_output.get("needs_retrieval", True)
    needs_followup = intent_output.get("needs_followup", False)

    # 1. 直接回應（如個資問題、離題問題）
    if routing_hint == "direct_response":
        return "response_synth"

    # 2. 追問處理
    prev_answer = state_dict.get("prev_answer_normalized") or ""
    if routing_hint == "followup" and prev_answer:
        return "followup_transform"
    if needs_followup and prev_answer:
        return "followup_transform"

    # 3. 需要檢索
    if needs_retrieval:
        return "query_builder"

    # 4. 預設進入回應生成
    return "response_synth"


def route_after_result_evaluator(state: State) -> str:
    """
    Result Evaluator 節點後的路由。

    路由邏輯：
    1. status == "retry" → query_builder（重試檢索）
    2. 其他 → response_synth（進入回應生成）
    """
    state_dict = cast(StateDict, state)

    # 檢查是否需要重試檢索
    retrieval_state = state_dict.get("retrieval") or {}
    status = retrieval_state.get("status")
    if status == "retry":
        return "query_builder"

    return "response_synth"
