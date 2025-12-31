"""
執行 reference-free regression（依賴本地 Qdrant/LLM/Langfuse 設定）。

用法（範例）：
  uv run python scripts/run_regression_eval.py --test-set tests/fixtures/eval_test_set.json --limit 20 --backend chat
"""

import argparse
import time
from typing import Any, Dict

from loguru import logger

from chatbot_rag.evaluation.regression_runner import RegressionRunner
from chatbot_rag.models.rag import QuestionRequest
from chatbot_rag.services.ask_stream.service import run_stream_graph


async def _ask_once(question: str, *, backend: str) -> Dict[str, Any]:
    req = QuestionRequest(question=question, top_k=20)

    # 收集 /ask/stream 的 meta_summary
    trace_id = None
    final_answer = ""
    citations_count = 0
    groundedness_decision = ""
    groundedness_confidence = 0.0

    start = time.monotonic()

    async def _is_disconnected() -> bool:
        return False

    async for ev in run_stream_graph(req, _is_disconnected, agent_backend=backend):
        if ev.get("channel") == "meta_summary":
            summary = ev.get("summary") or {}
            trace_id = ev.get("trace_id") or summary.get("trace_id")
        if ev.get("node") == "response_synth" and ev.get("channel") == "meta":
            meta = ev.get("meta") or {}
            channels = meta.get("channels") or {}
            out = channels.get("output_text") or {}
            final_answer = out.get("text") or final_answer
            citations_count = int(meta.get("citation_count") or 0)
        if ev.get("node") == "groundedness_check" and ev.get("channel") == "status":
            groundedness_decision = ev.get("decision") or groundedness_decision
            groundedness_confidence = float(ev.get("confidence") or groundedness_confidence)

    latency_ms = (time.monotonic() - start) * 1000.0
    return {
        "answer": final_answer,
        "citations_count": citations_count,
        "groundedness_decision": groundedness_decision,
        "groundedness_confidence": groundedness_confidence,
        "trace_id": trace_id,
        "latency_ms": latency_ms,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", default="tests/fixtures/eval_test_set.json")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--backend", choices=["chat", "responses"], default="chat")
    parser.add_argument("--out", default="files/regression_report.json")
    args = parser.parse_args()

    runner = RegressionRunner(args.test_set)
    results = await runner.run(
        ask_stream_fn=lambda q: _ask_once(q, backend=args.backend),
        limit=args.limit,
    )
    runner.export_json(results, args.out)
    logger.info("summary={}", runner.summarize(results))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())




