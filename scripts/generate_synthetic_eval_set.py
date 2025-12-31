"""
產生 synthetic test set（沒有題庫也能先跑回歸）。

注意：
- 這是「產生問題」為主（reference-free），不要求人工標註答案
- 會把 doc_id/title/doc_type_hint 帶入，方便後續做分桶與 error analysis

用法（範例）：
  uv run python scripts/generate_synthetic_eval_set.py --source-dir rag_test_data/cus --out files/synth_eval_set.json --max-files 50 --qpf 2 --backend chat
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage

from chatbot_rag.core.config import settings
from chatbot_rag.llm import create_chat_completion_llm, create_responses_llm


def _infer_category(path: Path) -> str:
    p = str(path)
    if "門診" in p or "時刻表" in p:
        return "schedule"
    if "衛教" in p:
        return "education"
    if "流程" in p or "就醫" in p:
        return "process"
    return "general"


def _first_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="rag_test_data/cus")
    parser.add_argument("--out", default="files/synth_eval_set.json")
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--qpf", type=int, default=2, help="questions per file")
    parser.add_argument("--backend", choices=["chat", "responses"], default="chat")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    files = sorted([p for p in source_dir.rglob("*.md") if p.is_file()])[: args.max_files]
    if not files:
        raise SystemExit(f"No markdown files under {source_dir}")

    if args.backend == "chat":
        llm = create_chat_completion_llm(streaming=False, model=settings.chat_model, temperature=0.6)
    else:
        llm = create_responses_llm(streaming=False, model=settings.chat_model, temperature=0.6, reasoning_effort="low", reasoning_summary=None)

    questions: List[Dict[str, Any]] = []
    qid = 1
    for p in files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        title = _first_title(text) or p.stem
        category = _infer_category(p)
        excerpt = text[:2000]

        system = SystemMessage(
            content=(
                "你是 RAG 測試題生成器。請根據提供的文件片段，產生可用於醫院客服的測試問題。\n"
                "請輸出單一 JSON，格式：\n"
                "{ \"questions\": [\"...\", \"...\"], \"expected_doc_type\": \"schedule|education|process|general\" }\n"
                "規則：\n"
                "- 一律繁體中文\n"
                "- 問題要可由文件回答，不要憑空延伸\n"
                "- 盡量涵蓋：流程、時間、地點、注意事項、例外條件\n"
            )
        )
        human = HumanMessage(
            content=(
                f"檔名：{p.name}\n"
                f"推測類別：{category}\n"
                f"標題：{title}\n"
                f"文件片段：\n{excerpt}\n\n"
                f"請產生 {args.qpf} 個問題。"
            )
        )

        raw = await llm.ainvoke([system, human])
        s = str(raw).strip()
        # 粗略取 JSON（保守：找第一個 { 到最後一個 }）
        start = s.find("{")
        end = s.rfind("}")
        data: Dict[str, Any] = {}
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(s[start : end + 1])
            except Exception:
                data = {}

        qs = data.get("questions") or []
        expected_doc_type = data.get("expected_doc_type") or category
        if not isinstance(qs, list) or not qs:
            # fallback：模板題
            qs = [
                f"請問「{title}」的重點是什麼？",
                f"依據文件內容，請整理「{title}」的注意事項。",
            ][: args.qpf]

        for q in qs[: args.qpf]:
            questions.append(
                {
                    "id": f"synth_{qid:04d}",
                    "category": category,
                    "question": str(q),
                    "expected_doc_type": expected_doc_type,
                    "expected_chunks": [],
                    "expected_slots": {},
                    "expected_answer_contains": [],
                    "difficulty": "medium",
                    "source_file": str(p.relative_to(source_dir)),
                }
            )
            qid += 1

        logger.info("[SYNTH] {} -> {} questions", p.name, args.qpf)

    out = {
        "version": "synth-1.0",
        "description": "synthetic test set (reference-free)",
        "source_dir": str(source_dir),
        "questions": questions,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[SYNTH] wrote {}", args.out)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())




