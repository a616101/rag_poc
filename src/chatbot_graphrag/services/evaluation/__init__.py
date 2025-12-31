"""
評估服務

使用 Ragas 指標進行品質評估，支援抽樣。

指標：
- faithfulness: 答案是否基於提供的上下文
- answer_relevancy: 答案是否回答了問題
- context_precision: 檢索的上下文是否相關
"""

from chatbot_graphrag.services.evaluation.ragas_evaluator import (
    RagasEvaluator,
    EvaluationResult,
    EvaluationSample,
    should_sample_for_evaluation,
    get_ragas_evaluator,
)

__all__ = [
    "RagasEvaluator",
    "EvaluationResult",
    "EvaluationSample",
    "should_sample_for_evaluation",
    "get_ragas_evaluator",
]
