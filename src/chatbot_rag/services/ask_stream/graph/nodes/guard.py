"""
Guard 節點：輸入驗證與安全檢查。

功能：
- 個資相關問題檢測（設置 guard_blocked=True, guard_intent="privacy_inquiry"）
- 非醫療相關問題檢測（設置 guard_intent="out_of_scope"）
- 保留擴充空間（可新增更多檢測規則）

路由：
- guard_blocked=True → END（直接結束）
- guard_blocked=False → language_normalizer（繼續流程）
"""

import re
from typing import Callable, cast, Optional, List

from langgraph.config import get_stream_writer
from loguru import logger

from chatbot_rag.llm import State
from ...constants import (
    AskStreamStages,
    PTCH_PRIVACY_RESPONSE,
    PTCH_OFF_TOPIC_RESPONSE,
)
from ...events import emit_node_event


# 個資相關關鍵字模式
PRIVACY_PATTERNS = [
    r"我的.*(?:看診|就醫|就診|病歷|紀錄|記錄)",
    r"(?:查詢|查看|調閱).*(?:病歷|就醫|就診|看診).*(?:紀錄|記錄)",
    r"(?:我|他|她|某人).*(?:上次|之前|過去).*(?:看診|就醫|就診)",
    r"(?:誰|哪位|什麼人).*(?:看診|就醫|就診)",
    r"(?:我的|某人的).*(?:醫療費用|掛號費|住院費)",
    r"(?:查|看|找).*(?:我的|某人的).*(?:報告|檢驗|檢查)",
]

# 非醫療相關問題模式（用於檢測完全離題的問題）
OFF_TOPIC_PATTERNS = [
    r"(?:天氣|氣象|溫度|下雨)",
    r"(?:股票|股市|投資|理財)",
    r"(?:寫|幫我寫|產生).*(?:程式|代碼|code)",
    r"(?:旅遊|旅行|行程|景點)",
    r"(?:美食|餐廳|小吃|推薦吃)",
    r"(?:電影|影集|追劇)",
    r"(?:遊戲|game|打電動)",
]


def _check_privacy_inquiry(question: str) -> bool:
    """
    檢查問題是否涉及個資查詢。

    Args:
        question: 使用者問題

    Returns:
        True 表示涉及個資，應該攔截
    """
    if not question:
        return False

    for pattern in PRIVACY_PATTERNS:
        if re.search(pattern, question, re.IGNORECASE):
            logger.debug(f"[GUARD] Privacy pattern matched: {pattern}")
            return True
    return False


def _check_off_topic(question: str) -> bool:
    """
    檢查問題是否完全與醫療無關。

    Args:
        question: 使用者問題

    Returns:
        True 表示完全離題
    """
    if not question:
        return False

    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, question, re.IGNORECASE):
            logger.debug(f"[GUARD] Off-topic pattern matched: {pattern}")
            return True
    return False


def build_guard_node() -> Callable[[State], State]:
    """
    Guard 節點：進行基本檢查。

    屏東基督教醫院版本：
    - 檢測個資相關問題
    - 檢測完全離題問題
    - 設置對應的攔截標記和預設回應
    """

    async def guard_node(state: State) -> State:
        writer = get_stream_writer()
        emit_node_event(
            writer,
            node="guard",
            phase="guard",
            payload={
                "channel": "status",
                "stage": AskStreamStages.GUARD_START,
            },
        )

        # 取得使用者問題
        messages = state.get("messages", [])
        user_question = ""
        if messages:
            last_message = messages[-1]
            content = getattr(last_message, "content", "")
            if isinstance(content, str):
                user_question = content
            elif isinstance(content, list):
                # 處理多模態訊息
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        user_question = item.get("text", "")
                        break

        blocked = False
        block_reason: Optional[str] = None
        guard_response: Optional[str] = None
        guard_intent: Optional[str] = None

        # 檢查個資問題（優先級最高）
        if _check_privacy_inquiry(user_question):
            blocked = True
            block_reason = "privacy_inquiry"
            guard_response = PTCH_PRIVACY_RESPONSE
            guard_intent = "privacy_inquiry"
            logger.info(f"[GUARD] Blocked privacy inquiry: {user_question[:50]}...")

        # 檢查完全離題問題
        elif _check_off_topic(user_question):
            # 離題問題不攔截，但標記意圖讓 planner 處理
            guard_intent = "out_of_scope"
            logger.info(f"[GUARD] Detected off-topic: {user_question[:50]}...")

        emit_node_event(
            writer,
            node="guard",
            phase="guard",
            payload={
                "channel": "status",
                "stage": AskStreamStages.GUARD_END,
                "blocked": blocked,
                "block_reason": block_reason,
            },
        )

        new_state = cast(State, dict(state))
        new_state["guard_blocked"] = blocked

        # 如果被攔截，設置預設回應
        if blocked and guard_response:
            new_state["guard_response"] = guard_response

        # 設置 guard 檢測到的意圖（供 planner 參考）
        if guard_intent:
            new_state["guard_intent"] = guard_intent

        return new_state

    return guard_node
