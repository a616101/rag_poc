"""
對 regression_report.json 做快速 error analysis（把低分 case 拉出來看）。

用法：
  python scripts/analyze_regression_report.py --report files/regression_report.json --top 20
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--report", default="files/regression_report.json")
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()

    data = json.loads(Path(args.report).read_text(encoding="utf-8"))
    results = data.get("results") or []
    results = [r for r in results if isinstance(r, dict)]
    results.sort(key=lambda r: (r.get("answer_relevance", 0.0), r.get("citations_count", 0)))

    for r in results[: args.top]:
        print("===")
        print("id:", r.get("case_id"))
        print("answer_relevance:", r.get("answer_relevance"))
        print("citations_count:", r.get("citations_count"))
        print("groundedness:", r.get("groundedness_decision"), r.get("groundedness_confidence"))
        print("latency_ms:", r.get("latency_ms"))
        print("trace_id:", r.get("trace_id"))
        print("Q:", r.get("question"))
        print("A:", (r.get("answer") or "")[:600])


if __name__ == "__main__":
    main()




