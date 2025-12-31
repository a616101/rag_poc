"""
查詢感知檢索的查詢分類器

第 5 階段：為動態 RRF 權重調整分類查詢類型。

查詢類型：
- FACTUAL（事實型）: 特定事實、定義、日期（「X 是什麼？」「Y 何時發生？」）
- CONCEPTUAL（概念型）: 解釋、比較、運作原理（「X 如何運作？」「為什麼 Y 很重要？」）
- NAVIGATIONAL（導航型）: 尋找特定實體或文件（「找到文件 X」「顯示 Y 政策」）
- PROCEDURAL（程序型）: 步驟式流程、說明（「如何做 X？」「Y 的步驟」）
- COMPARATIVE（比較型）: 比較多個事物（「X 和 Y 有什麼區別？」）
- AGGREGATE（聚合型）: 摘要或概述問題（「X 的概述」「Y 的摘要」）
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """用於權重調整的查詢類型分類。"""

    FACTUAL = "factual"  # 密集檢索效果最好
    CONCEPTUAL = "conceptual"  # 平衡檢索
    NAVIGATIONAL = "navigational"  # 關鍵字/稀疏檢索效果最好
    PROCEDURAL = "procedural"  # 密集 + 關鍵字平衡
    COMPARATIVE = "comparative"  # 多面向的密集檢索
    AGGREGATE = "aggregate"  # 偏好社區/全域檢索


@dataclass
class QueryClassification:
    """查詢分類結果。"""

    query_type: QueryType  # 查詢類型
    confidence: float  # 信心分數
    signals: dict[str, Any]  # 偵測到的訊號

    # 基於查詢類型的推薦 RRF 權重
    # (dense, sparse, fts)
    recommended_weights: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典。"""
        return {
            "query_type": self.query_type.value,
            "confidence": self.confidence,
            "signals": self.signals,
            "recommended_weights": list(self.recommended_weights),
        }


# 基於模式的查詢類型偵測
QUERY_TYPE_PATTERNS: dict[QueryType, list[re.Pattern]] = {
    QueryType.FACTUAL: [
        re.compile(r"^(什麼|何時|誰|哪裡|哪個|多少|幾)", re.IGNORECASE),
        re.compile(r"^(what|when|who|where|which|how\s+many|how\s+much)\b", re.IGNORECASE),
        re.compile(r"(是什麼|是誰|在哪裡|什麼時候)", re.IGNORECASE),
        re.compile(r"(定義|意思|意義|名稱)", re.IGNORECASE),
    ],
    QueryType.CONCEPTUAL: [
        re.compile(r"^(為什麼|如何|怎麼)", re.IGNORECASE),
        re.compile(r"^(why|how)\b", re.IGNORECASE),
        re.compile(r"(原理|原因|機制|運作|工作)", re.IGNORECASE),
        re.compile(r"(explain|describe|understanding)", re.IGNORECASE),
    ],
    QueryType.NAVIGATIONAL: [
        re.compile(r"(找|查詢|搜尋|看|顯示).*(文件|資料|表單|文檔)", re.IGNORECASE),
        re.compile(r"(find|show|search|locate|get)\s+(the|a)?\s*(document|file|form|policy)", re.IGNORECASE),
        re.compile(r"(政策|規定|規範|辦法|要點|準則|細則)", re.IGNORECASE),
    ],
    QueryType.PROCEDURAL: [
        re.compile(r"^(如何|怎麼|怎樣)", re.IGNORECASE),
        re.compile(r"^(how\s+to|how\s+do|steps\s+to|process\s+for)", re.IGNORECASE),
        re.compile(r"(步驟|流程|程序|方法|做法|指南)", re.IGNORECASE),
        re.compile(r"(申請|辦理|提交|處理)", re.IGNORECASE),
    ],
    QueryType.COMPARATIVE: [
        re.compile(r"(比較|差異|不同|區別|對比)", re.IGNORECASE),
        re.compile(r"(compare|difference|versus|vs\.?|differ)", re.IGNORECASE),
        re.compile(r"(哪個.*更|哪種.*好)", re.IGNORECASE),
        re.compile(r"(which\s+is\s+better|what's\s+the\s+difference)", re.IGNORECASE),
    ],
    QueryType.AGGREGATE: [
        re.compile(r"(概述|總結|摘要|整體|概覽|綜述)", re.IGNORECASE),
        re.compile(r"(summary|overview|summarize|main\s+points|key\s+aspects)", re.IGNORECASE),
        re.compile(r"(列出|列表|清單|所有)", re.IGNORECASE),
        re.compile(r"(list\s+all|enumerate|outline)", re.IGNORECASE),
    ],
}

# 按查詢類型的推薦 RRF 權重：(dense, sparse, fts)
QUERY_TYPE_WEIGHTS: dict[QueryType, tuple[float, float, float]] = {
    # 事實型：密集檢索擅長語意匹配
    QueryType.FACTUAL: (0.5, 0.25, 0.25),
    # 概念型：平衡方法用於細微理解
    QueryType.CONCEPTUAL: (0.45, 0.3, 0.25),
    # 導航型：關鍵字/稀疏匹配用於特定文件
    QueryType.NAVIGATIONAL: (0.25, 0.35, 0.4),
    # 程序型：密集用於理解 + 關鍵字用於細節
    QueryType.PROCEDURAL: (0.4, 0.3, 0.3),
    # 比較型：密集用於跨概念的語意相似度
    QueryType.COMPARATIVE: (0.5, 0.25, 0.25),
    # 聚合型：更廣泛的搜尋，偏好社區級別
    QueryType.AGGREGATE: (0.35, 0.35, 0.3),
}


class QueryClassifier:
    """
    為查詢感知檢索最佳化分類查詢。

    使用模式匹配，對於模糊情況可選擇使用 LLM 後備。
    """

    def __init__(self, use_llm_fallback: bool = False):
        """
        初始化查詢分類器。

        Args:
            use_llm_fallback: 對模糊分類使用 LLM
        """
        self.use_llm_fallback = use_llm_fallback

    def classify(self, query: str) -> QueryClassification:
        """
        基於模式和啟發式分類查詢。

        Args:
            query: 要分類的查詢文字

        Returns:
            包含類型、信心度和推薦權重的 QueryClassification
        """
        signals: dict[str, Any] = {}
        query_lower = query.lower().strip()

        # 計算每種類型的模式匹配數
        type_scores: dict[QueryType, float] = {}

        for query_type, patterns in QUERY_TYPE_PATTERNS.items():
            match_count = 0
            matched_patterns = []

            for pattern in patterns:
                if pattern.search(query):
                    match_count += 1
                    matched_patterns.append(pattern.pattern[:30])

            if match_count > 0:
                type_scores[query_type] = match_count / len(patterns)
                signals[query_type.value] = {
                    "matches": match_count,
                    "patterns": matched_patterns,
                }

        # 決定最佳匹配
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            best_score = type_scores[best_type]

            # 基於分數和差距計算信心度
            second_best = sorted(type_scores.values(), reverse=True)[1] if len(type_scores) > 1 else 0
            margin = best_score - second_best
            confidence = min(1.0, best_score + margin * 0.5)
        else:
            # 對於未匹配的查詢預設為概念型
            best_type = QueryType.CONCEPTUAL
            confidence = 0.3
            signals["default"] = True

        # 取得推薦權重
        weights = QUERY_TYPE_WEIGHTS.get(best_type, (0.4, 0.3, 0.3))

        logger.debug(
            f"Query classified as {best_type.value} "
            f"(confidence={confidence:.2f}, weights={weights})"
        )

        return QueryClassification(
            query_type=best_type,
            confidence=confidence,
            signals=signals,
            recommended_weights=weights,
        )

    async def classify_with_llm(
        self,
        query: str,
        pattern_result: Optional[QueryClassification] = None,
    ) -> QueryClassification:
        """
        對模糊情況使用 LLM 分類。

        Args:
            query: 要分類的查詢
            pattern_result: 可選的基於模式的分類結果以進行精煉

        Returns:
            來自 LLM 的 QueryClassification
        """
        # 如果模式分類信心度高，則使用它
        if pattern_result and pattern_result.confidence >= 0.7:
            return pattern_result

        # 目前返回模式結果或預設值
        # 在生產環境中，會呼叫 LLM 進行分類
        if pattern_result:
            return pattern_result

        return QueryClassification(
            query_type=QueryType.CONCEPTUAL,
            confidence=0.5,
            signals={"llm_fallback": True},
            recommended_weights=(0.4, 0.3, 0.3),
        )


# 單例實例
_classifier_instance: Optional[QueryClassifier] = None


def get_query_classifier() -> QueryClassifier:
    """取得或建立單例查詢分類器實例。"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = QueryClassifier()
    return _classifier_instance


def classify_query(query: str) -> QueryClassification:
    """
    分類查詢的便捷函數。

    Args:
        query: 要分類的查詢文字

    Returns:
        QueryClassification 結果
    """
    classifier = get_query_classifier()
    return classifier.classify(query)


def get_weights_for_query(query: str) -> tuple[float, float, float]:
    """
    取得查詢的推薦 RRF 權重。

    Args:
        query: 查詢文字

    Returns:
        (dense_weight, sparse_weight, fts_weight) 元組
    """
    classification = classify_query(query)
    return classification.recommended_weights
