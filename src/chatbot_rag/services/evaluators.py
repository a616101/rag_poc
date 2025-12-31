"""
Langfuse Experiment 評估器模組。

此模組提供用於 Prompt Experiments 的評估函數：

## Rule-based Evaluators（客觀、確定性）
1. tool_selection: 評估工具選擇是否正確
2. language_compliance: 評估回答語言是否正確
3. out_of_scope_handling: 評估超出範圍問題的處理
4. response_completeness: 評估回答是否完整（有內容、不是錯誤訊息）

## LLM-as-a-Judge Evaluators（主觀、需 LLM）
5. response_relevance: 評估回答是否與問題相關
6. criteria_satisfaction: 根據自定義準則評估回答

評估器可用於：
- SDK-based Experiments（程式碼中呼叫）
- 產生 Langfuse Scores
"""

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from loguru import logger


@dataclass
class EvaluationResult:
    """評估結果"""

    score: float  # 0.0 - 1.0
    comment: str
    details: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        result = {"score": self.score, "comment": self.comment}
        if self.details:
            result["details"] = self.details
        return result


# ==============================================================================
# Tool Selection Accuracy（P0 - Rule-based）
# ==============================================================================


def extract_tool_calls_from_response(response: Any) -> list[str]:
    """
    從 LLM 回應中提取工具呼叫名稱。

    支援多種回應格式：
    - LangChain AIMessage with tool_calls
    - Dict with tool_calls key
    - Raw string (嘗試解析)

    Args:
        response: LLM 回應物件

    Returns:
        工具名稱列表
    """
    tool_names: list[str] = []

    # LangChain AIMessage
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            if isinstance(tc, dict) and "name" in tc:
                tool_names.append(tc["name"])
            elif hasattr(tc, "name"):
                tool_names.append(tc.name)

    # Dict format
    elif isinstance(response, dict):
        if "tool_calls" in response:
            for tc in response["tool_calls"]:
                if isinstance(tc, dict) and "name" in tc:
                    tool_names.append(tc["name"])
                elif isinstance(tc, dict) and "function" in tc:
                    tool_names.append(tc["function"].get("name", ""))

    # Additional content (for OpenAI response format)
    if hasattr(response, "additional_kwargs"):
        additional = response.additional_kwargs
        if "tool_calls" in additional:
            for tc in additional["tool_calls"]:
                if isinstance(tc, dict) and "function" in tc:
                    name = tc["function"].get("name", "")
                    if name and name not in tool_names:
                        tool_names.append(name)

    return [name for name in tool_names if name]


def evaluate_tool_selection(
    response: Any,
    expected_output: dict[str, Any],
) -> EvaluationResult:
    """
    評估工具選擇是否正確。

    Args:
        response: LLM 回應（AIMessage 或 dict）
        expected_output: Dataset item 的 expected_output，包含 expected_tool

    Returns:
        EvaluationResult
    """
    expected_tool = expected_output.get("expected_tool")
    actual_tools = extract_tool_calls_from_response(response)

    details = {
        "expected_tool": expected_tool,
        "actual_tools": actual_tools,
    }

    if expected_tool is None:
        # 預期不應呼叫任何工具
        if not actual_tools:
            return EvaluationResult(
                score=1.0,
                comment="正確：未呼叫工具（符合預期）",
                details=details,
            )
        else:
            return EvaluationResult(
                score=0.0,
                comment=f"錯誤：不應呼叫工具，但呼叫了 {actual_tools}",
                details=details,
            )
    else:
        # 預期應呼叫特定工具
        if expected_tool in actual_tools:
            return EvaluationResult(
                score=1.0,
                comment=f"正確：呼叫了預期的工具 {expected_tool}",
                details=details,
            )
        elif not actual_tools:
            return EvaluationResult(
                score=0.0,
                comment=f"錯誤：應呼叫 {expected_tool}，但未呼叫任何工具",
                details=details,
            )
        else:
            return EvaluationResult(
                score=0.0,
                comment=f"錯誤：應呼叫 {expected_tool}，實際呼叫 {actual_tools}",
                details=details,
            )


# ==============================================================================
# Language Compliance（P1）
# ==============================================================================

# 語言特徵正則表達式
LANGUAGE_PATTERNS = {
    "zh-hant": {
        "positive": re.compile(r"[繁體這裡時數認證學習]"),
        "negative": re.compile(r"[这里学习认证时数]"),  # 簡體字
    },
    "zh-hans": {
        "positive": re.compile(r"[简体这里学习认证时数]"),
        "negative": re.compile(r"[這裡學習認證時數]"),  # 繁體字
    },
    "en": {
        "positive": re.compile(r"\b(the|is|are|you|can|how|to)\b", re.IGNORECASE),
        "negative": None,
    },
    "ja": {
        "positive": re.compile(r"[ぁ-んァ-ンー]"),  # 平假名、片假名
        "negative": None,
    },
    "ko": {
        "positive": re.compile(r"[가-힣]"),  # 韓文字元
        "negative": None,
    },
}


def evaluate_language_compliance(
    response_text: str,
    expected_language: str,
) -> EvaluationResult:
    """
    評估回答語言是否符合預期。

    Args:
        response_text: LLM 回答文字
        expected_language: 預期語言代碼（zh-hant, zh-hans, en, ja, ko）

    Returns:
        EvaluationResult
    """
    if not response_text.strip():
        return EvaluationResult(
            score=0.0,
            comment="回答為空",
            details={"expected_language": expected_language},
        )

    patterns = LANGUAGE_PATTERNS.get(expected_language)
    if not patterns:
        # 未知語言，預設通過
        return EvaluationResult(
            score=1.0,
            comment=f"未知語言代碼 {expected_language}，跳過檢查",
            details={"expected_language": expected_language},
        )

    positive_pattern = patterns["positive"]
    negative_pattern = patterns.get("negative")

    # 計算正向特徵匹配
    positive_matches = len(positive_pattern.findall(response_text))

    # 計算負向特徵匹配（如果有定義）
    negative_matches = 0
    if negative_pattern:
        negative_matches = len(negative_pattern.findall(response_text))

    # 評分邏輯
    details = {
        "expected_language": expected_language,
        "positive_matches": positive_matches,
        "negative_matches": negative_matches,
        "response_length": len(response_text),
    }

    # 特殊處理：繁體/簡體中文
    if expected_language in ("zh-hant", "zh-hans"):
        if negative_matches > positive_matches * 0.3:
            # 負向特徵過多
            return EvaluationResult(
                score=0.5,
                comment=f"語言可能不正確：發現 {negative_matches} 個非預期字元",
                details=details,
            )

    # 正向特徵足夠
    if positive_matches >= 3:
        return EvaluationResult(
            score=1.0,
            comment=f"語言符合預期 ({expected_language})",
            details=details,
        )
    elif positive_matches > 0:
        return EvaluationResult(
            score=0.8,
            comment=f"語言大致符合，但特徵較少 ({positive_matches} 個)",
            details=details,
        )
    else:
        return EvaluationResult(
            score=0.3,
            comment=f"未偵測到預期語言 ({expected_language}) 的特徵",
            details=details,
        )


# ==============================================================================
# Out of Scope Handling（P2）
# ==============================================================================

# 婉拒相關關鍵字
REFUSAL_KEYWORDS = {
    "zh-hant": ["抱歉", "無法", "不支援", "服務範圍", "無法協助", "不在"],
    "zh-hans": ["抱歉", "无法", "不支持", "服务范围", "无法协助", "不在"],
    "en": ["sorry", "cannot", "unable", "outside", "scope", "not support"],
    "ja": ["申し訳", "できません", "対応できない", "範囲外"],
    "ko": ["죄송", "할 수 없", "지원하지", "범위"],
}


def evaluate_out_of_scope_handling(
    response_text: str,
    expected_output: dict[str, Any],
    input_language: str = "zh-hant",
) -> EvaluationResult:
    """
    評估超出範圍問題的處理是否正確。

    Args:
        response_text: LLM 回答文字
        expected_output: Dataset item 的 expected_output，包含 should_refuse
        input_language: 輸入語言

    Returns:
        EvaluationResult
    """
    should_refuse = expected_output.get("should_refuse", False)
    response_lower = response_text.lower()

    # 檢查是否包含婉拒關鍵字
    keywords = REFUSAL_KEYWORDS.get(input_language, REFUSAL_KEYWORDS["zh-hant"])
    found_keywords = [kw for kw in keywords if kw.lower() in response_lower]
    is_refusing = len(found_keywords) > 0

    details = {
        "should_refuse": should_refuse,
        "is_refusing": is_refusing,
        "found_keywords": found_keywords,
    }

    if should_refuse:
        # 預期應婉拒
        if is_refusing:
            return EvaluationResult(
                score=1.0,
                comment="正確：適當地婉拒了超出範圍的問題",
                details=details,
            )
        else:
            return EvaluationResult(
                score=0.0,
                comment="錯誤：應婉拒但未婉拒，可能回答了不該回答的問題",
                details=details,
            )
    else:
        # 預期不應婉拒（應該回答）
        if is_refusing:
            # 誤判為超出範圍
            return EvaluationResult(
                score=0.3,
                comment="警告：不應婉拒但包含婉拒語句，可能誤判",
                details=details,
            )
        else:
            return EvaluationResult(
                score=1.0,
                comment="正確：正常回答問題",
                details=details,
            )


# ==============================================================================
# Answer Relevancy（RAGAS-style，取代 Response Completeness）
# ==============================================================================

# 錯誤訊息模式（用於前置檢查）
ERROR_PATTERNS = [
    re.compile(r"(發生錯誤|出錯了|error occurred)", re.IGNORECASE),
    re.compile(r"(無法處理|cannot process)", re.IGNORECASE),
    re.compile(r"(系統錯誤|system error)", re.IGNORECASE),
    re.compile(r"(請稍後再試|try again later)", re.IGNORECASE),
]

# 空洞回答模式（用於前置檢查）
EMPTY_RESPONSE_PATTERNS = [
    re.compile(r"^(好的|OK|了解|收到)[。！]?$", re.IGNORECASE),
    re.compile(r"^(我不知道|I don't know)[。！]?$", re.IGNORECASE),
]


def evaluate_response_completeness(
    response_text: str,
    question: str,
    llm_call: Optional[Callable[[str], str]] = None,
) -> EvaluationResult:
    """
    評估答案相關性（Answer Relevancy）。

    基於 RAGAS Answer Relevancy 指標設計：
    - 評估回答是否針對問題本身
    - 評估回答是否涵蓋問題的所有面向
    - 評估回答中是否包含不相關或冗餘的資訊

    RAGAS 原理：
    Answer Relevancy 透過生成「可能產生此回答的問題」，
    再計算這些問題與原始問題的相似度來評估相關性。
    此實作簡化為直接由 LLM 評估相關性。

    Args:
        response_text: LLM 回答文字
        question: 原始問題
        llm_call: 可選的 LLM 呼叫函數

    Returns:
        EvaluationResult
    """
    response_stripped = response_text.strip()
    response_length = len(response_stripped)

    details = {
        "response_length": response_length,
        "question_length": len(question),
        "metric": "answer_relevancy",
    }

    # 前置檢查：回答是否為空
    if not response_stripped:
        return EvaluationResult(
            score=0.0,
            comment="回答為空，無法評估相關性",
            details=details,
        )

    # 前置檢查：是否為錯誤訊息
    for pattern in ERROR_PATTERNS:
        if pattern.search(response_stripped):
            return EvaluationResult(
                score=0.0,
                comment="回答包含錯誤訊息，視為無效回答",
                details={**details, "error_detected": True},
            )

    # 前置檢查：是否為空洞回答
    if response_length < 10:
        for pattern in EMPTY_RESPONSE_PATTERNS:
            if pattern.match(response_stripped):
                return EvaluationResult(
                    score=0.1,
                    comment="回答過於簡短或空洞，無實質內容",
                    details=details,
                )

    # 使用 LLM 評估 Answer Relevancy
    if llm_call:
        prompt = f"""你是一個專業的答案相關性評估員。請根據 RAGAS Answer Relevancy 指標評估以下回答。

## 評估任務
評估「回答」對於「問題」的相關性和完整性。

## 問題
{question}

## 回答
{response_text}

## 評估準則（RAGAS Answer Relevancy）

### 1. 直接相關性（Direct Relevance）
- 回答是否直接針對問題
- 回答是否解答了問題的核心意圖

### 2. 完整性（Completeness）
- 回答是否涵蓋問題的所有面向
- 是否有遺漏問題中隱含的子問題

### 3. 簡潔性（Conciseness）
- 回答是否包含過多不相關的資訊
- 是否有冗餘或偏題的內容

### 4. 資訊品質（Information Quality）
- 回答是否提供了有用的資訊
- 資訊是否具體而非模糊籠統

## 評分標準
- 1.0：完全相關，全面回答問題，無冗餘資訊
- 0.8：高度相關，回答大部分問題要點，可能略有冗餘
- 0.6：部分相關，回答了問題但有遺漏或包含不相關資訊
- 0.4：相關性有限，僅回答部分問題或有較多偏題內容
- 0.2：大部分不相關，答非所問或過於籠統
- 0.0：完全不相關或無實質內容

## 輸出格式
請以 JSON 格式回覆：
{{"score": <0.0-1.0的分數>, "reason": "<繁體中文評分理由>"}}
"""

        try:
            result = llm_call(prompt)
            import json
            json_match = re.search(r'\{[^}]+\}', result)
            if json_match:
                parsed = json.loads(json_match.group())
                score = float(parsed.get("score", 0.5))
                reason = parsed.get("reason", "評估完成")
                return EvaluationResult(
                    score=min(max(score, 0.0), 1.0),
                    comment=reason,
                    details={
                        **details,
                        "ragas_style": True,
                        "raw_response": result[:300],
                    },
                )
        except Exception as e:
            logger.warning(f"Answer Relevancy 評估失敗: {e}")
            # 繼續使用 fallback 邏輯

    # Fallback：基於規則的簡單評估（無 LLM 時使用）
    # 這只是一個粗略的啟發式評估
    length_ratio = response_length / max(len(question), 1)

    if response_length >= 100:
        return EvaluationResult(
            score=0.8,
            comment=f"回答內容豐富（{response_length} 字），但未經 LLM 評估相關性",
            details={**details, "llm_judge": False, "length_ratio": round(length_ratio, 2)},
        )
    elif response_length >= 50:
        return EvaluationResult(
            score=0.7,
            comment=f"回答有一定內容（{response_length} 字），但未經 LLM 評估相關性",
            details={**details, "llm_judge": False, "length_ratio": round(length_ratio, 2)},
        )
    elif response_length >= 20:
        return EvaluationResult(
            score=0.6,
            comment=f"回答簡潔（{response_length} 字），相關性待確認",
            details={**details, "llm_judge": False, "length_ratio": round(length_ratio, 2)},
        )
    else:
        return EvaluationResult(
            score=0.4,
            comment=f"回答過短（{response_length} 字），可能不夠完整",
            details={**details, "llm_judge": False, "length_ratio": round(length_ratio, 2)},
        )


# ==============================================================================
# LLM-as-a-Judge Evaluators
# ==============================================================================


def evaluate_with_llm_judge(
    question: str,
    response_text: str,
    criteria: str,
    llm_call: Callable[[str], str],
) -> EvaluationResult:
    """
    使用 LLM 作為評審來評估回答。

    Args:
        question: 原始問題
        response_text: LLM 回答
        criteria: 評估準則（描述好的回答應該具備什麼）
        llm_call: LLM 呼叫函數，接受 prompt 返回回答

    Returns:
        EvaluationResult
    """
    prompt = f"""你是一個評估助手。請根據以下準則評估回答的品質。

## 問題
{question}

## 回答
{response_text}

## 評估準則
{criteria}

## 評分指引
- 1.0: 完全符合準則，回答優秀
- 0.8: 大致符合，有小缺點
- 0.6: 部分符合，有改進空間
- 0.4: 勉強及格，有明顯問題
- 0.2: 不符合準則，回答有問題
- 0.0: 完全不符合或有嚴重錯誤

請只回答一個 JSON 物件，格式如下：
{{"score": <0.0-1.0>, "reason": "<評分理由>"}}
"""

    try:
        result = llm_call(prompt)
        # 嘗試解析 JSON
        import json
        # 提取 JSON 部分
        json_match = re.search(r'\{[^}]+\}', result)
        if json_match:
            parsed = json.loads(json_match.group())
            score = float(parsed.get("score", 0.5))
            reason = parsed.get("reason", "LLM 評估完成")
            return EvaluationResult(
                score=min(max(score, 0.0), 1.0),  # 確保在 0-1 範圍內
                comment=reason,
                details={"llm_judge": True, "raw_response": result[:200]},
            )
    except Exception as e:
        logger.warning(f"LLM-as-a-Judge 評估失敗: {e}")

    return EvaluationResult(
        score=0.5,
        comment="LLM 評估失敗，使用預設分數",
        details={"error": str(e) if 'e' in dir() else "unknown"},
    )


def evaluate_response_relevance(
    question: str,
    response_text: str,
    llm_call: Optional[Callable[[str], str]] = None,
) -> EvaluationResult:
    """
    評估回答是否與問題相關。

    如果提供 llm_call，使用 LLM-as-a-Judge；
    否則使用簡單的規則檢查。

    Args:
        question: 原始問題
        response_text: LLM 回答
        llm_call: 可選的 LLM 呼叫函數

    Returns:
        EvaluationResult
    """
    if llm_call:
        criteria = """
評估回答是否與問題相關：
1. 回答是否針對問題本身
2. 回答是否提供了有用的資訊
3. 回答是否偏題或答非所問
"""
        return evaluate_with_llm_judge(question, response_text, criteria, llm_call)

    # 簡單規則檢查（fallback）
    if not response_text.strip():
        return EvaluationResult(score=0.0, comment="回答為空")

    # 至少需要一些內容
    if len(response_text) < 20:
        return EvaluationResult(
            score=0.5,
            comment="回答過短，無法判斷相關性",
        )

    return EvaluationResult(
        score=0.7,
        comment="回答有內容，但未使用 LLM 判斷相關性",
        details={"llm_judge": False},
    )


def evaluate_criteria_satisfaction(
    question: str,
    response_text: str,
    criteria: str,
    llm_call: Optional[Callable[[str], str]] = None,
) -> EvaluationResult:
    """
    根據自定義準則評估回答。

    Args:
        question: 原始問題
        response_text: LLM 回答
        criteria: 自定義評估準則
        llm_call: 可選的 LLM 呼叫函數

    Returns:
        EvaluationResult
    """
    if not criteria:
        return EvaluationResult(
            score=1.0,
            comment="未指定評估準則，跳過",
        )

    if llm_call:
        return evaluate_with_llm_judge(question, response_text, criteria, llm_call)

    return EvaluationResult(
        score=0.5,
        comment="有準則但無 LLM，無法評估",
        details={"criteria": criteria, "llm_judge": False},
    )


# ==============================================================================
# 綜合評估
# ==============================================================================


def run_all_evaluations(
    response: Any,
    response_text: str,
    dataset_item: dict[str, Any],
    llm_call: Optional[Callable[[str], str]] = None,
) -> dict[str, EvaluationResult]:
    """
    執行所有評估。

    Args:
        response: 原始 LLM 回應物件（用於提取 tool_calls）
        response_text: LLM 回答文字
        dataset_item: Dataset item（包含 input, expected_output, metadata）
        llm_call: 可選的 LLM 呼叫函數（用於 LLM-as-a-Judge）

    Returns:
        評估名稱到結果的映射
    """
    input_data = dataset_item.get("input", {})
    expected_output = dataset_item.get("expected_output", {})
    metadata = dataset_item.get("metadata", {})

    question = input_data.get("question", "")
    input_language = input_data.get("language", "zh-hant")
    category = metadata.get("category", "")

    results: dict[str, EvaluationResult] = {}

    # 1. Tool Selection（客觀評估）
    results["tool_selection"] = evaluate_tool_selection(response, expected_output)

    # 2. Language Compliance（客觀評估）
    results["language_compliance"] = evaluate_language_compliance(
        response_text, input_language
    )

    # 3. Response Completeness（RAGAS Answer Relevancy 評估）
    results["response_completeness"] = evaluate_response_completeness(
        response_text, question, llm_call
    )

    # 4. Out of Scope Handling（僅對需要拒絕的問題評估）
    if category in ("out_of_scope", "security_boundary") or expected_output.get("should_refuse"):
        results["out_of_scope_handling"] = evaluate_out_of_scope_handling(
            response_text, expected_output, input_language
        )

    # 5. Criteria Satisfaction（如果有自定義評估準則）
    evaluation_criteria = expected_output.get("evaluation_criteria")
    if evaluation_criteria:
        results["criteria_satisfaction"] = evaluate_criteria_satisfaction(
            question, response_text, evaluation_criteria, llm_call
        )

    return results


def evaluate_answer_similarity(
    question: str,
    actual_answer: str,
    expected_answer: str,
    llm_call: Optional[Callable[[str], str]] = None,
) -> EvaluationResult:
    """
    評估答案正確性（Answer Correctness）。

    基於 RAGAS Answer Correctness 指標設計：
    - 評估實際回答與預期答案之間的語義相似度
    - 評估事實正確性（Factual Correctness）
    - 綜合考量真陽性（TP）、假陽性（FP）、假陰性（FN）

    Args:
        question: 原始問題
        actual_answer: 實際回答
        expected_answer: 預期回答（Ground Truth）
        llm_call: LLM 呼叫函數

    Returns:
        EvaluationResult
    """
    if not expected_answer:
        return EvaluationResult(
            score=1.0,
            comment="未指定預期答案，跳過評估",
        )

    if not actual_answer.strip():
        return EvaluationResult(
            score=0.0,
            comment="實際回答為空",
            details={"expected_answer": expected_answer[:100]},
        )

    if llm_call:
        # RAGAS-style Answer Correctness prompt (繁體中文)
        prompt = f"""你是一個專業的答案正確性評估員。請根據 RAGAS Answer Correctness 指標評估以下回答。

## 評估任務
比較「實際回答」與「標準答案」，評估答案的正確性和完整性。

## 問題
{question}

## 標準答案（Ground Truth）
{expected_answer}

## 實際回答
{actual_answer}

## 評估準則（RAGAS Answer Correctness）

### 第一步：識別關鍵陳述
從標準答案中提取所有關鍵事實陳述（Key Statements）。

### 第二步：分類比對
將實際回答中的陳述分為三類：
1. **真陽性（TP）**：實際回答中正確涵蓋標準答案的陳述
2. **假陽性（FP）**：實際回答中存在但與標準答案矛盾或錯誤的陳述
3. **假陰性（FN）**：標準答案中有但實際回答遺漏的重要陳述

### 第三步：計算分數
- 考量語義相似度和事實正確性
- 正確涵蓋越多關鍵資訊，分數越高
- 錯誤陳述會降低分數
- 遺漏重要資訊會降低分數

## 評分標準
- 1.0：完全正確，涵蓋所有關鍵資訊，無錯誤陳述
- 0.8：大致正確，涵蓋主要資訊，可能有小遺漏但無錯誤
- 0.6：部分正確，涵蓋部分關鍵資訊，有遺漏但無重大錯誤
- 0.4：正確性有限，遺漏較多關鍵資訊或有輕微錯誤
- 0.2：大部分不正確或遺漏大量關鍵資訊
- 0.0：完全錯誤或答非所問

## 輸出格式
請以 JSON 格式回覆：
{{"score": <0.0-1.0的分數>, "reason": "<繁體中文評分理由，說明 TP/FP/FN 情況>"}}
"""

        try:
            result = llm_call(prompt)
            import json
            json_match = re.search(r'\{[^}]+\}', result)
            if json_match:
                parsed = json.loads(json_match.group())
                score = float(parsed.get("score", 0.5))
                reason = parsed.get("reason", "評估完成")
                return EvaluationResult(
                    score=min(max(score, 0.0), 1.0),
                    comment=reason,
                    details={
                        "metric": "answer_correctness",
                        "ragas_style": True,
                        "raw_response": result[:300],
                    },
                )
        except Exception as e:
            logger.warning(f"Answer Correctness 評估失敗: {e}")
            return EvaluationResult(
                score=0.5,
                comment="LLM 評估失敗，使用預設分數",
                details={"error": str(e)},
            )

    # 簡單的 fallback：檢查是否有足夠長度的回答
    if len(actual_answer) >= 20:
        return EvaluationResult(
            score=0.5,
            comment="有回答內容，但未使用 LLM 評估答案正確性",
            details={"llm_judge": False},
        )

    return EvaluationResult(
        score=0.3,
        comment="回答過短，無法評估答案正確性",
        details={"llm_judge": False},
    )


def run_e2e_evaluations(
    question: str,
    response_text: str,
    used_tools: list[str],
    expected_output: dict[str, Any],
    metadata: dict[str, Any],
    llm_call: Optional[Callable[[str], str]] = None,
) -> dict[str, EvaluationResult]:
    """
    執行 E2E 實驗的評估。

    專門為 E2E 測試設計，不需要 LLM response 物件。

    Args:
        question: 原始問題
        response_text: 回答文字
        used_tools: 使用的工具列表
        expected_output: 預期輸出
        metadata: 元資料
        llm_call: 可選的 LLM 呼叫函數（用於 answer_similarity 評估）

    Returns:
        評估結果
    """
    category = metadata.get("category", "")
    input_language = metadata.get("language", "zh-hant")
    results: dict[str, EvaluationResult] = {}

    # 1. Tool Selection（客觀評估）
    expected_tool = expected_output.get("expected_tool")
    if expected_tool is None:
        # 預期不應呼叫任何工具
        if not used_tools:
            results["tool_selection"] = EvaluationResult(
                score=1.0,
                comment="正確：未呼叫工具（符合預期）",
                details={"expected_tool": None, "actual_tools": used_tools},
            )
        else:
            results["tool_selection"] = EvaluationResult(
                score=0.0,
                comment=f"錯誤：不應呼叫工具，但呼叫了 {used_tools}",
                details={"expected_tool": None, "actual_tools": used_tools},
            )
    else:
        if expected_tool in used_tools:
            results["tool_selection"] = EvaluationResult(
                score=1.0,
                comment=f"正確：呼叫了預期的工具 {expected_tool}",
                details={"expected_tool": expected_tool, "actual_tools": used_tools},
            )
        else:
            results["tool_selection"] = EvaluationResult(
                score=0.0,
                comment=f"錯誤：應呼叫 {expected_tool}，實際呼叫 {used_tools}",
                details={"expected_tool": expected_tool, "actual_tools": used_tools},
            )

    # 2. Response Completeness（RAGAS Answer Relevancy 評估）
    results["response_completeness"] = evaluate_response_completeness(
        response_text, question, llm_call
    )

    # 3. Out of Scope Handling（客觀評估）
    should_refuse = expected_output.get("should_refuse", False)
    if category in ("out_of_scope", "security_boundary") or should_refuse:
        results["out_of_scope_handling"] = evaluate_out_of_scope_handling(
            response_text,
            expected_output,
            input_language,
        )

    # 4. Answer Similarity（LLM-as-a-Judge 評估）
    expected_answer = expected_output.get("expected_answer")
    if expected_answer and not should_refuse:
        results["answer_similarity"] = evaluate_answer_similarity(
            question, response_text, expected_answer, llm_call
        )

    return results
