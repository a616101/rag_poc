"""
Query variation 服務：提供多種 query 變異策略來改善檢索結果。

支援三種策略：
1. SYNONYM: 同義詞替換
2. DECOMPOSE: 問題分解
3. KEYWORD_EXPAND: 關鍵字擴展
"""

from enum import Enum
from typing import Any, Optional, List, Union

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger

from chatbot_rag.core.concurrency import with_llm_semaphore


class QueryVariationStrategy(Enum):
    """Query 變異策略枚舉。"""

    SYNONYM = "synonym"  # 同義詞替換
    DECOMPOSE = "decompose"  # 問題分解
    KEYWORD_EXPAND = "keyword"  # 關鍵字擴展
    NONE = "none"  # 不進行變異


# 策略輪換順序（根據 loop 次數）
RETRY_STRATEGY_ORDER: List[QueryVariationStrategy] = [
    QueryVariationStrategy.SYNONYM,  # Loop 1 retry
    QueryVariationStrategy.DECOMPOSE,  # Loop 2 retry
    QueryVariationStrategy.KEYWORD_EXPAND,  # Loop 3+ retry
]


def get_strategy_for_loop(loop: int) -> QueryVariationStrategy:
    """
    根據 loop 次數返回對應的變異策略。

    Args:
        loop: 當前的 loop 次數（1-based）

    Returns:
        對應的變異策略
    """
    if loop <= 1:
        return QueryVariationStrategy.NONE
    # loop 2 -> index 0, loop 3 -> index 1, etc.
    index = min(loop - 2, len(RETRY_STRATEGY_ORDER) - 1)
    return RETRY_STRATEGY_ORDER[index]


# Prompt 模板
SYNONYM_PROMPT = """你是一個查詢優化助手。請將以下查詢改寫成使用不同的同義詞或近義詞表達，但保持原意不變。

規則：
1. 使用繁體中文
2. 保持查詢的核心意圖
3. 用不同的詞彙表達相同概念
4. 只輸出改寫後的查詢，不要加任何說明

原始查詢：{query}

請輸出改寫後的查詢："""

DECOMPOSE_PROMPT = """你是一個查詢分析助手。請將以下複合查詢分解成更簡單、更具體的子查詢。

規則：
1. 使用繁體中文
2. 每個子查詢應該聚焦於原問題的一個面向
3. 子查詢之間用換行分隔
4. 最多產生 3 個子查詢
5. 只輸出子查詢，不要加任何說明或編號

原始查詢：{query}

請輸出分解後的子查詢："""

KEYWORD_EXPAND_PROMPT = """你是一個關鍵字擴展助手。請將以下查詢擴展為包含更多相關關鍵字的查詢描述。

規則：
1. 使用繁體中文
2. 加入相關的專業術語、別名或常用說法
3. 保持查詢的自然流暢
4. 只輸出擴展後的查詢，不要加任何說明

原始查詢：{query}

請輸出擴展後的查詢："""


async def generate_variation_async(
    original_query: str,
    strategy: QueryVariationStrategy,
    llm: BaseChatModel,
) -> Union[str, List[str]]:
    """
    非同步版本：根據指定策略生成查詢變異。

    Args:
        original_query: 原始查詢
        strategy: 變異策略
        llm: 用於生成的 LLM

    Returns:
        變異後的查詢（DECOMPOSE 策略返回 List，其他返回 str）
    """
    if strategy == QueryVariationStrategy.NONE:
        return original_query

    prompt_template = _get_prompt_for_strategy(strategy)
    if not prompt_template:
        return original_query

    try:
        prompt = prompt_template.format(query=original_query)
        messages = [
            SystemMessage(content="你是一個專業的查詢優化助手。"),
            HumanMessage(content=prompt),
        ]

        response = await with_llm_semaphore(
            lambda: llm.ainvoke(messages),
            backend="default",
        )
        result_text = str(response.content).strip() if response.content else ""

        if not result_text:
            logger.warning(
                "[QUERY_VARIATION] Empty response for strategy=%s", strategy.value
            )
            return original_query

        if strategy == QueryVariationStrategy.DECOMPOSE:
            # 分解策略返回多個子查詢
            queries = [q.strip() for q in result_text.split("\n") if q.strip()]
            return queries[:3] if queries else [original_query]

        return result_text

    except Exception as exc:
        logger.warning(
            "[QUERY_VARIATION] Failed to generate variation for strategy=%s: %s",
            strategy.value,
            exc,
        )
        return original_query


def generate_variation(
    original_query: str,
    strategy: QueryVariationStrategy,
    llm: BaseChatModel,
) -> Union[str, List[str]]:
    """
    同步版本：根據指定策略生成查詢變異。

    Args:
        original_query: 原始查詢
        strategy: 變異策略
        llm: 用於生成的 LLM

    Returns:
        變異後的查詢（DECOMPOSE 策略返回 List，其他返回 str）
    """
    if strategy == QueryVariationStrategy.NONE:
        return original_query

    prompt_template = _get_prompt_for_strategy(strategy)
    if not prompt_template:
        return original_query

    try:
        prompt = prompt_template.format(query=original_query)
        messages = [
            SystemMessage(content="你是一個專業的查詢優化助手。"),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)
        result_text = str(response.content).strip() if response.content else ""

        if not result_text:
            logger.warning(
                "[QUERY_VARIATION] Empty response for strategy=%s", strategy.value
            )
            return original_query

        if strategy == QueryVariationStrategy.DECOMPOSE:
            # 分解策略返回多個子查詢
            queries = [q.strip() for q in result_text.split("\n") if q.strip()]
            return queries[:3] if queries else [original_query]

        return result_text

    except Exception as exc:
        logger.warning(
            "[QUERY_VARIATION] Failed to generate variation for strategy=%s: %s",
            strategy.value,
            exc,
        )
        return original_query


def _get_prompt_for_strategy(strategy: QueryVariationStrategy) -> Optional[str]:
    """根據策略返回對應的 prompt 模板。"""
    prompts = {
        QueryVariationStrategy.SYNONYM: SYNONYM_PROMPT,
        QueryVariationStrategy.DECOMPOSE: DECOMPOSE_PROMPT,
        QueryVariationStrategy.KEYWORD_EXPAND: KEYWORD_EXPAND_PROMPT,
    }
    return prompts.get(strategy)


class QueryVariationResult:
    """Query 變異結果封裝。"""

    def __init__(
        self,
        original_query: str,
        varied_queries: Union[str, List[str]],
        strategy: QueryVariationStrategy,
        success: bool = True,
    ):
        self.original_query = original_query
        self.strategy = strategy
        self.success = success

        if isinstance(varied_queries, str):
            self.queries = [varied_queries]
        else:
            self.queries = varied_queries

    @property
    def primary_query(self) -> str:
        """返回主要的變異查詢（第一個）。"""
        return self.queries[0] if self.queries else self.original_query

    @property
    def is_multi_query(self) -> bool:
        """是否為多查詢結果（DECOMPOSE 策略）。"""
        return len(self.queries) > 1

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典格式。"""
        return {
            "original_query": self.original_query,
            "queries": self.queries,
            "strategy": self.strategy.value,
            "success": self.success,
            "is_multi_query": self.is_multi_query,
        }
