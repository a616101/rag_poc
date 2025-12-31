"""
意圖路由節點

分析查詢意圖以決定路由（direct/local/global/drift）。

路由模式：
- direct: 直接回答（問候語、感謝等）
- local: 基於實體的檢索（具體問題）
- global: 基於社群的檢索（概覽問題）
- drift: 動態探索（全面性問題）
"""

import logging
import re
from typing import Any

from chatbot_graphrag.graph_workflow.types import GraphRAGState, QueryMode

logger = logging.getLogger(__name__)

# 直接回答模式（無需檢索）
DIRECT_PATTERNS = [
    r"^(hi|hello|hey|嗨|你好|哈囉)",
    r"^(thanks?|thank\s*you|謝謝|感謝)",
    r"^(bye|goodbye|再見|掰掰)",
    r"(who|what)\s+(are|is)\s+you",
    r"你是誰",
    r"^(ok|okay|好的?|沒問題)",
]

# 建議全局模式的模式（社群級查詢）
GLOBAL_PATTERNS = [
    r"(overview|summary|summarize|概述|摘要|總結)",
    r"(all|every|各種|所有|全部)\s*(types?|kinds?|services?|類型|種類|服務)",
    r"(compare|comparison|比較|對比)",
    r"(list|列出|羅列)\s*(all|所有)",
    r"how\s+many\s+(types?|kinds?)",
    r"有(哪些|幾種|什麼)",
]

# 建議 drift 模式的模式（探索性查詢）
DRIFT_PATTERNS = [
    r"(explore|exploration|deep\s*dive|深入|探索)",
    r"(comprehensive|thorough|detailed|完整|詳盡|全面)",
    r"(everything|all\s*about|關於.*一切)",
    r"(research|investigate|研究|調查)",
]

COMPILED_DIRECT = [re.compile(p, re.IGNORECASE) for p in DIRECT_PATTERNS]
COMPILED_GLOBAL = [re.compile(p, re.IGNORECASE) for p in GLOBAL_PATTERNS]
COMPILED_DRIFT = [re.compile(p, re.IGNORECASE) for p in DRIFT_PATTERNS]


def classify_intent(question: str) -> tuple[QueryMode, str]:
    """
    根據模式分類查詢意圖。

    Returns:
        (query_mode, reasoning)
    """
    # 檢查直接回答模式
    for pattern in COMPILED_DIRECT:
        if pattern.search(question):
            return QueryMode.DIRECT, "Greeting or conversational - no retrieval needed"

    # 檢查 drift 模式（探索性）
    for pattern in COMPILED_DRIFT:
        if pattern.search(question):
            return QueryMode.DRIFT, "Exploratory query - using DRIFT mode for comprehensive search"

    # 檢查全局模式（社群級）
    for pattern in COMPILED_GLOBAL:
        if pattern.search(question):
            return QueryMode.GLOBAL, "Overview query - using global mode for community-level context"

    # 預設為本地模式（基於實體）
    return QueryMode.LOCAL, "Specific query - using local mode for entity-based retrieval"


async def intent_router_node(state: GraphRAGState) -> dict[str, Any]:
    """
    用於查詢分類的意圖路由節點。

    決定是否：
    - Direct: 無需檢索直接回答（問候語、感謝等）
    - Local: 基於實體的檢索用於具體問題
    - Global: 基於社群的檢索用於概覽問題
    - Drift: 動態探索用於全面性問題

    Returns:
        更新後的狀態，包含 query_mode 和 intent_reasoning
    """
    import time
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("intent_router", "START")

    start_time = time.time()
    question = state.get("normalized_question") or state.get("question", "")
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))

    # 分類意圖
    query_mode, reasoning = classify_intent(question)
    logger.info(f"Intent: {query_mode.value} - {reasoning}")

    emit_status("intent_router", "DONE")
    return {
        "query_mode": query_mode.value,
        "intent_reasoning": reasoning,
        "retrieval_path": retrieval_path + [f"intent:{query_mode.value}"],
        "timing": {**timing, "intent_ms": (time.time() - start_time) * 1000},
    }


async def direct_answer_node(state: GraphRAGState) -> dict[str, Any]:
    """
    用於非檢索回應的直接回答節點。

    為問候語、感謝等生成直接回應。

    Returns:
        更新後的狀態，包含 final_answer
    """
    import time
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("direct_answer", "START")

    start_time = time.time()
    question = state.get("normalized_question") or state.get("question", "")
    user_language = state.get("user_language", "en")
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))

    # 根據問題類型生成適當的回應
    question_lower = question.lower()

    if any(g in question_lower for g in ["hi", "hello", "hey", "嗨", "你好", "哈囉"]):
        if user_language.startswith("zh"):
            response = "您好！我是智慧助理，有什麼可以幫助您的嗎？"
        else:
            response = "Hello! I'm your assistant. How can I help you today?"

    elif any(t in question_lower for t in ["thank", "謝謝", "感謝"]):
        if user_language.startswith("zh"):
            response = "不客氣！如果還有其他問題，請隨時詢問。"
        else:
            response = "You're welcome! Feel free to ask if you have other questions."

    elif any(b in question_lower for b in ["bye", "goodbye", "再見", "掰掰"]):
        if user_language.startswith("zh"):
            response = "再見！祝您有美好的一天！"
        else:
            response = "Goodbye! Have a great day!"

    elif "你是誰" in question_lower or "who are you" in question_lower:
        if user_language.startswith("zh"):
            response = "我是一個智慧問答助理，可以回答您關於醫院服務、流程和相關資訊的問題。"
        else:
            response = "I'm an intelligent Q&A assistant that can help answer questions about hospital services, procedures, and related information."

    else:
        if user_language.startswith("zh"):
            response = "好的，我明白了。還有什麼可以幫助您的嗎？"
        else:
            response = "Got it. Is there anything else I can help you with?"

    emit_status("direct_answer", "DONE")
    return {
        "final_answer": response,
        "confidence": 1.0,
        "retrieval_path": retrieval_path + ["direct_answer"],
        "timing": {**timing, "direct_answer_ms": (time.time() - start_time) * 1000},
    }
