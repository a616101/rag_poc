"""
Ragas 評估器服務

使用 Ragas 指標進行品質評估，支援抽樣。
實現階段 4：GraphRAG 管線的品質指標。

指標：
- faithfulness: 答案是否基於提供的上下文
- answer_relevancy: 答案是否回答了問題
- context_precision: 檢索的上下文是否相關
"""

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from chatbot_graphrag.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    """Ragas 評估的輸入樣本。"""

    question: str                        # 問題
    answer: str                          # 答案
    contexts: list[str]                  # 上下文列表
    ground_truth: Optional[str] = None   # 用於基於參考的指標


@dataclass
class EvaluationResult:
    """Ragas 評估結果。"""

    faithfulness: Optional[float] = None       # 忠實度
    answer_relevancy: Optional[float] = None   # 答案相關性
    context_precision: Optional[float] = None  # 上下文精確度
    context_recall: Optional[float] = None     # 上下文召回率

    # 聚合分數
    overall_score: float = 0.0

    # 元資料
    evaluated: bool = False                     # 是否已評估
    sampled: bool = False                       # 是否被抽樣
    evaluation_time_ms: float = 0.0             # 評估時間（毫秒）
    error: Optional[str] = None                 # 錯誤訊息
    metrics_computed: list[str] = field(default_factory=list)  # 已計算的指標

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典以供序列化。"""
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "overall_score": self.overall_score,
            "evaluated": self.evaluated,
            "sampled": self.sampled,
            "evaluation_time_ms": self.evaluation_time_ms,
            "error": self.error,
            "metrics_computed": self.metrics_computed,
        }


def should_sample_for_evaluation() -> bool:
    """
    判斷此請求是否應該被抽樣進行 Ragas 評估。

    使用 settings.ragas_sample_rate（預設 10%）隨機選擇請求。

    Returns:
        如果請求應該使用 Ragas 評估則為 True
    """
    if not settings.ragas_enabled:
        return False

    sample_rate = settings.ragas_sample_rate
    if sample_rate <= 0:
        return False
    if sample_rate >= 1.0:
        return True

    return random.random() < sample_rate


class RagasEvaluator:
    """
    基於 Ragas 的品質評估器，支援抽樣。

    使用 Ragas 指標評估 RAG 管線品質：
    - faithfulness: 答案是否基於上下文？
    - answer_relevancy: 答案是否回答了問題？
    - context_precision: 檢索的上下文是否相關？

    用法：
        evaluator = RagasEvaluator()
        sample = EvaluationSample(
            question="What is GraphRAG?",
            answer="GraphRAG is...",
            contexts=["GraphRAG combines..."]
        )
        result = await evaluator.evaluate(sample)
    """

    def __init__(self):
        """初始化 Ragas 評估器。"""
        self._ragas_available = False
        self._metrics_to_compute: list[str] = []
        self._evaluator = None

        self._initialize_ragas()

    def _initialize_ragas(self) -> None:
        """初始化 Ragas 函式庫和指標。"""
        try:
            # 解析配置的指標
            metrics_str = settings.ragas_metrics
            self._metrics_to_compute = [m.strip() for m in metrics_str.split(",")]

            # 嘗試導入 Ragas
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
            )

            self._ragas_available = True
            self._metric_objects = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
            }

            logger.info(
                f"Ragas evaluator initialized with metrics: {self._metrics_to_compute}"
            )

        except ImportError as e:
            logger.warning(
                f"Ragas not available: {e}. "
                "Install with: pip install ragas. "
                "Evaluation will use fallback heuristics."
            )
            self._ragas_available = False

        except Exception as e:
            logger.error(f"Failed to initialize Ragas: {e}")
            self._ragas_available = False

    @property
    def is_available(self) -> bool:
        """檢查 Ragas 是否可用。"""
        return self._ragas_available

    async def evaluate(
        self,
        sample: EvaluationSample,
        use_fallback: bool = True,
    ) -> EvaluationResult:
        """
        使用 Ragas 指標評估樣本。

        Args:
            sample: 包含問題、答案和上下文的輸入樣本
            use_fallback: 如果 Ragas 不可用則使用啟發式回退

        Returns:
            包含計算指標的 EvaluationResult
        """
        start_time = time.time()

        # 檢查是否應該評估此樣本
        should_eval = should_sample_for_evaluation()

        if not should_eval:
            return EvaluationResult(
                sampled=False,
                evaluated=False,
                evaluation_time_ms=(time.time() - start_time) * 1000,
            )

        # 首先嘗試 Ragas 評估
        if self._ragas_available:
            try:
                result = await self._evaluate_with_ragas(sample)
                result.sampled = True
                result.evaluated = True
                result.evaluation_time_ms = (time.time() - start_time) * 1000
                return result
            except Exception as e:
                logger.warning(f"Ragas evaluation failed: {e}")
                if not use_fallback:
                    return EvaluationResult(
                        sampled=True,
                        evaluated=False,
                        error=str(e),
                        evaluation_time_ms=(time.time() - start_time) * 1000,
                    )

        # 回退到啟發式評估
        if use_fallback:
            result = self._evaluate_with_heuristics(sample)
            result.sampled = True
            result.evaluated = True
            result.evaluation_time_ms = (time.time() - start_time) * 1000
            return result

        return EvaluationResult(
            sampled=True,
            evaluated=False,
            error="Ragas not available and fallback disabled",
            evaluation_time_ms=(time.time() - start_time) * 1000,
        )

    async def _evaluate_with_ragas(self, sample: EvaluationSample) -> EvaluationResult:
        """
        使用 Ragas 函式庫評估。

        Args:
            sample: 輸入樣本

        Returns:
            包含 Ragas 指標的 EvaluationResult
        """
        from datasets import Dataset
        from ragas import evaluate

        # 為 Ragas 準備資料集
        data = {
            "question": [sample.question],
            "answer": [sample.answer],
            "contexts": [sample.contexts],
        }

        if sample.ground_truth:
            data["ground_truth"] = [sample.ground_truth]

        dataset = Dataset.from_dict(data)

        # 獲取要評估的指標物件
        metrics = []
        for metric_name in self._metrics_to_compute:
            if metric_name in self._metric_objects:
                metrics.append(self._metric_objects[metric_name])

        if not metrics:
            raise ValueError(f"No valid metrics configured: {self._metrics_to_compute}")

        # 執行 Ragas 評估
        # 注意：這可能在內部使用 LLM 呼叫
        result = evaluate(
            dataset,
            metrics=metrics,
        )

        # 提取分數
        scores = result.to_pandas().iloc[0].to_dict()

        eval_result = EvaluationResult(
            faithfulness=scores.get("faithfulness"),
            answer_relevancy=scores.get("answer_relevancy"),
            context_precision=scores.get("context_precision"),
            context_recall=scores.get("context_recall"),
            metrics_computed=list(scores.keys()),
        )

        # 計算整體分數（可用指標的平均值）
        available_scores = [
            s for s in [
                eval_result.faithfulness,
                eval_result.answer_relevancy,
                eval_result.context_precision,
            ] if s is not None
        ]

        if available_scores:
            eval_result.overall_score = sum(available_scores) / len(available_scores)

        return eval_result

    def _evaluate_with_heuristics(self, sample: EvaluationSample) -> EvaluationResult:
        """
        當 Ragas 不可用時使用簡單啟發式評估。

        提供不需要 LLM 呼叫的基本品質信號。

        Args:
            sample: 輸入樣本

        Returns:
            包含啟發式分數的 EvaluationResult
        """
        metrics_computed = []

        # 啟發式 1：上下文覆蓋率（偽忠實度）
        # 檢查答案中有多少出現在上下文中
        answer_tokens = set(sample.answer.lower().split())
        context_text = " ".join(sample.contexts).lower()
        context_tokens = set(context_text.split())

        if answer_tokens:
            overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
            faithfulness = min(1.0, overlap * 1.5)  # Boost slightly
        else:
            faithfulness = 0.0
        metrics_computed.append("faithfulness")

        # 啟發式 2：答案相關性（與問題的關鍵字重疊）
        question_tokens = set(sample.question.lower().split())
        if question_tokens:
            q_overlap = len(question_tokens & answer_tokens) / len(question_tokens)
            answer_relevancy = min(1.0, q_overlap * 2.0)  # Boost more
        else:
            answer_relevancy = 0.0
        metrics_computed.append("answer_relevancy")

        # 啟發式 3：上下文精確度（問題-上下文重疊）
        if question_tokens and context_tokens:
            c_overlap = len(question_tokens & context_tokens) / len(question_tokens)
            context_precision = min(1.0, c_overlap * 1.8)
        else:
            context_precision = 0.0
        metrics_computed.append("context_precision")

        # 整體分數
        overall = (faithfulness + answer_relevancy + context_precision) / 3

        logger.debug(
            f"Heuristic evaluation: faithfulness={faithfulness:.2f}, "
            f"relevancy={answer_relevancy:.2f}, precision={context_precision:.2f}"
        )

        return EvaluationResult(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            overall_score=overall,
            metrics_computed=metrics_computed,
        )

    async def evaluate_batch(
        self,
        samples: list[EvaluationSample],
    ) -> list[EvaluationResult]:
        """
        批次評估樣本。

        Args:
            samples: 要評估的樣本列表

        Returns:
            EvaluationResult 列表
        """
        results = []
        batch_size = settings.ragas_batch_size

        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            for sample in batch:
                result = await self.evaluate(sample)
                results.append(result)

        return results


# 單例實例
_evaluator_instance: Optional[RagasEvaluator] = None


def get_ragas_evaluator() -> RagasEvaluator:
    """獲取或建立單例 Ragas 評估器實例。"""
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = RagasEvaluator()
    return _evaluator_instance
