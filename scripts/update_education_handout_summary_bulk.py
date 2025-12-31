"""
批次重建 education.handout Markdown 檔案的「統整內容」（主旨/重點整理/流程/注意事項/表格/版本資訊）。

使用情境：
- 部分大檔在生成統整內容時因 ReadTimeout 等例外，導致只剩 `> 統整失敗：ReadTimeout...` + `## 全文`
- 想重新產生統整章節，但保留 YAML front-matter 與 `## 全文` 原文

重要保證：
- 只會改動 YAML front-matter 之後、`## 全文` 之前的區塊（也就是統整章節區塊）
- `## 全文` 及其後所有原文內容保持原樣（位元級不變）

LLM：
- 使用 OpenAI 相容 Chat Completions API（讀取 OPENAI_API_BASE / OPENAI_API_KEY / CHAT_MODEL）
- 支援 tokens/耗時統計
- 支援 chunked + carry（Refine / Running Summary）以涵蓋超長文件
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


FRONT_MATTER_RE = re.compile(r"\A---\n(?P<yaml>.*?\n)---\n", re.DOTALL)

# allow importing src/ if needed in future; keep consistent with other scripts
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if _SRC_DIR.exists():
    sys.path.insert(0, str(_SRC_DIR))


@dataclasses.dataclass
class UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "UsageStats") -> None:
        self.prompt_tokens += int(getattr(other, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(other, "completion_tokens", 0) or 0)
        self.total_tokens += int(getattr(other, "total_tokens", 0) or 0)


@dataclasses.dataclass
class FileRunResult:
    changed: bool
    status: str
    elapsed_s: float
    usage: UsageStats


@dataclasses.dataclass
class LLMConfig:
    api_base: str
    api_key: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 1400
    timeout_s: float = 180.0
    retries: int = 2
    retry_backoff_s: float = 2.0


@dataclasses.dataclass
class ChunkingConfig:
    chunk_chars: int = 6000
    overlap_chars: int = 300
    chunk_max: int = 0
    carry: bool = True
    carry_max_chars: int = 3500
    carry_summary_window: int = 3
    final_max_chars: int = 14000
    # Adaptive splitting when a chunk fails (timeout / invalid json)
    adaptive_split: bool = True
    adaptive_min_chunk_chars: int = 1200
    adaptive_max_depth: int = 4


@dataclasses.dataclass
class PreprocessConfig:
    compact_tables: bool = True
    table_max_rows: int = 30
    table_keep_rows: int = 6
    # Prevent extremely long runs of whitespace from bloating prompts
    collapse_blank_lines: bool = True
    blank_lines_max: int = 2


@dataclasses.dataclass
class SignalsCapConfig:
    # Hard caps to stop merged signals from exploding on huge documents
    per_key_max: int = 60
    # Keep version/title signals more aggressively (these are small but useful)
    version_lines_max: int = 60
    title_candidates_max: int = 30


def _load_llm_config(model_override: Optional[str] = None) -> LLMConfig:
    env_base = os.getenv("OPENAI_API_BASE") or os.getenv("openai_api_base")
    env_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
    env_model = os.getenv("CHAT_MODEL") or os.getenv("chat_model")
    if env_base and env_key:
        return LLMConfig(
            api_base=env_base.rstrip("/"),
            api_key=env_key,
            model=model_override or env_model or "gpt-4o-mini",
        )

    # Fallback: try import project settings (loads .env via pydantic-settings)
    try:
        from chatbot_rag.core.config import settings  # type: ignore

        return LLMConfig(
            api_base=str(settings.openai_api_base).rstrip("/"),
            api_key=str(settings.openai_api_key),
            model=model_override or str(getattr(settings, "chat_model", "")) or "gpt-4o-mini",
            temperature=float(getattr(settings, "chat_temperature", 0.1)),
            max_tokens=int(getattr(settings, "chat_max_tokens", 1400)),
            timeout_s=float(getattr(settings, "llm_request_timeout", 180.0)),
        )
    except Exception as e:
        raise RuntimeError(
            "找不到 LLM 設定。請設定環境變數 OPENAI_API_BASE/OPENAI_API_KEY（必要），"
            "或確保可 import chatbot_rag.core.config.settings（會從 .env 讀取）。"
        ) from e


def _usage_from_payload(payload: Dict[str, Any]) -> UsageStats:
    usage = payload.get("usage") or {}
    try:
        return UsageStats(
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
        )
    except Exception:
        return UsageStats()


def _extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _json_sanitize_common_issues(text: str) -> str:
    s = (text or "").strip()
    # remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def _call_chat_completions(
    cfg: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens_override: Optional[int] = None,
) -> Tuple[str, UsageStats, float]:
    url = cfg.api_base.rstrip("/") + "/chat/completions"
    body = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": cfg.temperature,
        "max_tokens": int(max_tokens_override if max_tokens_override is not None else cfg.max_tokens),
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"},
        method="POST",
    )

    last_err: Optional[Exception] = None
    for attempt in range(cfg.retries + 1):
        try:
            started = time.monotonic()
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                resp_bytes = resp.read()
                payload = json.loads(resp_bytes.decode("utf-8"))
                content = payload["choices"][0]["message"]["content"]
                usage = _usage_from_payload(payload)
                elapsed = time.monotonic() - started
                return content, usage, elapsed
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            raise RuntimeError(f"LLM API HTTPError: {e.code} {e.reason} {detail}") from e
        except Exception as e:
            last_err = e
            msg = str(e)
            # retry for common timeout-ish failures
            if attempt < cfg.retries and ("timed out" in msg or "ReadTimeout" in msg or "timeout" in msg.lower()):
                time.sleep(cfg.retry_backoff_s * (2**attempt))
                continue
            raise RuntimeError(f"LLM API error: {e}") from e
    raise RuntimeError(f"LLM API error: {last_err}")


def _split_front_matter(text: str) -> Tuple[Optional[str], str]:
    m = FRONT_MATTER_RE.search(text)
    if not m:
        return None, text
    yaml_text = m.group("yaml")
    rest = text[m.end() :]
    return yaml_text, rest


def _find_h1_block(body: str) -> Tuple[str, str]:
    """
    回傳：(h1_block, rest_after_h1_block)
    - h1_block：從第一個 '# ' 開始，到下一個 '## ' 或 EOF 前的區段（包含中間空行/blockquote）
    - 若找不到 H1，回傳 ("", body)
    """
    m = re.search(r"(?m)^(#\s+.+)\s*$", body)
    if not m:
        return "", body
    start = m.start()
    # 找下一個二級標題（## ）
    m2 = re.search(r"(?m)^##\s+\S+", body[m.end() :])
    end = m.end() + (m2.start() if m2 else len(body[m.end() :]))
    return body[start:end].rstrip() + "\n\n", body[end:]


def _extract_section(body: str, heading: str) -> Optional[str]:
    pat = re.compile(rf"(?m)^\#\#\s+{re.escape(heading)}\s*$")
    m = pat.search(body)
    if not m:
        return None
    start = m.end()
    m2 = re.search(r"(?m)^\#\#\s+\S+", body[start:])
    end = start + (m2.start() if m2 else len(body[start:]))
    return body[start:end].lstrip("\n").rstrip()


def _split_before_fulltext(body: str) -> Tuple[str, str]:
    """
    回傳：(prefix_before_fulltext_heading, from_fulltext_heading_to_end)
    若找不到 '## 全文'，則 (body, "")。
    """
    m = re.search(r"(?m)^##\s+全文\s*$", body)
    if not m:
        return body, ""
    return body[: m.start()], body[m.start() :]


def _split_text_into_chunks(text: str, *, chunk_chars: int, overlap_chars: int) -> List[str]:
    if chunk_chars <= 0:
        return [text]
    if not text:
        return [""]
    ov = max(0, int(overlap_chars))
    cc = max(1, int(chunk_chars))
    if len(text) <= cc:
        return [text]
    out: List[str] = []
    i = 0
    while i < len(text):
        j = min(len(text), i + cc)
        out.append(text[i:j])
        if j >= len(text):
            break
        i = max(0, j - ov)
        if len(out) > 2000:
            break
    return out


def _unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        s = (it or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _cap_list(items: List[str], max_n: int) -> List[str]:
    if max_n <= 0:
        return items
    return items[: max_n]


def _cap_signals(signals: Dict[str, Any], caps: SignalsCapConfig) -> Dict[str, Any]:
    s = dict(signals or {})
    for k, v in list(s.items()):
        if not isinstance(v, list):
            continue
        vv = _unique_keep_order([str(x) for x in v])
        if k == "version_lines":
            s[k] = _cap_list(vv, caps.version_lines_max)
        elif k == "title_candidates":
            s[k] = _cap_list(vv, caps.title_candidates_max)
        else:
            s[k] = _cap_list(vv, caps.per_key_max)
    return s


def _merge_signals(base: Dict[str, Any], inc: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for k, v in (inc or {}).items():
        if isinstance(v, list):
            cur = out.get(k)
            cur_list = cur if isinstance(cur, list) else []
            out[k] = _unique_keep_order(cur_list + [str(x) for x in v])
        elif isinstance(v, str):
            if not out.get(k) and v.strip():
                out[k] = v.strip()
        elif isinstance(v, dict):
            cur = out.get(k)
            out[k] = _merge_signals(cur if isinstance(cur, dict) else {}, v)
        else:
            if k not in out:
                out[k] = v
    return out


def _collapse_blank_lines(text: str, *, max_blank_lines: int) -> str:
    if max_blank_lines < 0:
        return text
    # Replace runs of >= (max_blank_lines + 1) blank lines with exactly max_blank_lines blank lines.
    # "blank line" here means a line containing only whitespace.
    # We use a conservative regex to avoid accidentally touching YAML blocks.
    if max_blank_lines == 0:
        return re.sub(r"(?m)^\s*\n+", "", text)
    # e.g. max_blank_lines=2 => allow 2 blank lines => collapse \n{3,} (with whitespace lines) to \n\n
    pat = re.compile(rf"(?:\n[ \t]*\n){{{max_blank_lines + 1},}}", re.MULTILINE)
    repl = "\n" + ("\n" * max_blank_lines)
    return pat.sub(repl, text)


def _compact_large_markdown_tables(
    text: str, *, max_rows: int, keep_rows: int
) -> str:
    """
    將超大型 Markdown table 壓縮，避免把大量數值/參考值塞進 LLM。
    規則：
    - 偵測連續以 '|' 開頭的行視為 table block
    - 若 rows > max_rows：保留前 keep_rows 行，其餘以摘要提示取代
    """
    if not text:
        return text
    lines = text.splitlines()
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.lstrip().startswith("|"):
            out.append(line)
            i += 1
            continue
        # collect table block
        j = i
        block: List[str] = []
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            block.append(lines[j])
            j += 1
        rows = len(block)
        if rows > max_rows:
            keep = max(2, min(rows, keep_rows))
            kept = block[:keep]
            # Extract header if present
            header = kept[0].strip()
            # Try to extract column names from header
            col_names = [c.strip() for c in header.strip("|").split("|")]
            col_names = [c for c in col_names if c]
            col_hint = "、".join(col_names[:10]) if col_names else "（欄位不明）"
            out.extend(kept)
            out.append(
                f"〔大型表格已省略 {rows - keep} 行；欄位：{col_hint}〕"
            )
        else:
            out.extend(block)
        i = j
    return "\n".join(out)


def _preprocess_fulltext_for_llm(text: str, cfg: PreprocessConfig) -> str:
    s = text or ""
    if cfg.compact_tables:
        s = _compact_large_markdown_tables(
            s, max_rows=int(cfg.table_max_rows), keep_rows=int(cfg.table_keep_rows)
        )
    if cfg.collapse_blank_lines:
        s = _collapse_blank_lines(s, max_blank_lines=int(cfg.blank_lines_max))
    return s.strip()


def _parse_llm_json_with_one_repair(
    cfg: LLMConfig,
    *,
    llm_text: str,
    schema_hint: Optional[Dict[str, Any]] = None,
    usage_total: UsageStats,
    elapsed_total: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], float]:
    json_text = (llm_text or "").strip()
    if not json_text.startswith("{"):
        extracted = _extract_first_json_object(json_text)
        if extracted:
            json_text = extracted
    json_text = _json_sanitize_common_issues(json_text)
    try:
        return json.loads(json_text), None, elapsed_total
    except Exception:
        try:
            schema_text = json.dumps(schema_hint, ensure_ascii=False) if schema_hint else ""
            repaired, u2, e2 = _call_chat_completions(
                cfg,
                system_prompt=(
                    "你是嚴格的 JSON 修復器。\n"
                    "任務：把輸入轉成「可被 json.loads 解析」的 JSON 物件。\n"
                    "輸出規則（非常重要）：\n"
                    "- 只輸出 JSON（以 { 開頭，以 } 結尾），不得包含任何說明文字。\n"
                    "- 不得包含 code fence。\n"
                    "- 字串內的換行請用 \\n 轉義（不得出現未轉義的實際換行）。\n"
                    "- 只保留 schema 中允許的欄位；若原始內容過長/截斷，允許刪減 list 項目以保證 JSON 合法。\n"
                    "- 所有 list 請保守：最多保留前 6 項；過長字串可截斷到 100 字。\n"
                    "- 若你無法確定內容，請回傳最小合法 JSON（可用空字串/空陣列），但 key 結構要完整。\n"
                ),
                user_prompt=(
                    "下面這段應該是一個 JSON 物件，但目前無法被解析。請修復成有效 JSON，並只輸出 JSON：\n\n"
                    + (f"### schema_hint\n{schema_text}\n\n" if schema_text else "")
                    + f"{llm_text}\n"
                ),
                max_tokens_override=1400,
            )
            usage_total.add(u2)
            elapsed_total += float(e2 or 0.0)
            repaired = repaired.strip()
            if not repaired.startswith("{"):
                extracted = _extract_first_json_object(repaired)
                if extracted:
                    repaired = extracted
            repaired = _json_sanitize_common_issues(repaired)
            return json.loads(repaired), None, elapsed_total
        except Exception:
            snippet = (llm_text or "").strip().replace("\n", "\\n")[:400]
            return None, f"llm_json_parse_failed: {snippet}", elapsed_total


def _render_bullets(items: List[str]) -> str:
    items = _unique_keep_order([str(x) for x in (items or [])])
    if not items:
        return "〔無〕\n"
    return "\n".join([f"- {it}" for it in items]) + "\n"


def _render_steps(items: List[str]) -> str:
    items = [str(x).strip() for x in (items or []) if str(x).strip()]
    if not items:
        return "〔無明確流程〕\n"
    return "\n".join([f"{i}. {it}" for i, it in enumerate(items, start=1)]) + "\n"


def _build_chunked_prompts(template_md: str) -> Tuple[str, Dict[str, Any]]:
    system = (
        "你是「衛教文件分段統整器」。\n"
        "任務：對單一片段抽取可用於重建『統整內容』的資訊。\n"
        "輸出規則（非常重要）：\n"
        "- 只輸出嚴格 JSON 物件，不得包含 code fence，不得包含任何說明文字。\n"
        "- JSON 必須『完全符合 schema 的 key 結構』，不得輸出 schema 以外的欄位。\n"
        "- 不要輸出 YAML front-matter、不要輸出整份 metadata、不要輸出文件內容原文。\n"
        "- 請輸出『單行 JSON』（不要包含實際換行字元）；字串內若需要換行請用 \\n。\n"
        "- signals 內的每個字串都必須是『單行』，不得包含換行。\n"
        "- 使用繁體中文（zh-Hant）。\n"
        "- 只萃取你在片段中能支持的資訊；不確定就留空或省略。\n"
        "- key_points / key_actions / red_flags 以條列句（每項一句）。\n"
        "- 若偵測到版本資訊（編號/制定/修訂），請盡量保留原字串。\n"
        "- 若片段中有大型表格：不要輸出完整表格/大量 Markdown；只要輸出『表格主題』或『欄位名稱』即可（每項 <= 80 字）。\n"
        "- signals 陣列請盡量精煉：key_points/steps/warnings/table_topics/version_lines/title_candidates 各最多 6 項。\n"
        "- 每個 signals 字串請控制在 100 字以內。\n"
        "\n"
        "統整內容風格參考（只參考，不要輸出整份範本）：\n"
        + template_md[:2500]
    )
    schema = {
        "chunk_summary_zh_hant": "string (<=5 sentences)",
        "signals": {
            "purpose": ["string", "..."],
            "key_points": ["string", "..."],
            "steps": ["string", "..."],
            "warnings": ["string", "..."],
            "table_topics": ["string", "..."],  # do NOT output full markdown tables
            "version_lines": ["string", "..."],
            "title_candidates": ["string", "..."],
        },
    }
    return system, schema


def _build_final_prompts(template_md: str, digest_text: str) -> Tuple[str, str]:
    system = (
        "你是「衛教文件統整內容產生器」。\n"
        "任務：根據我提供的『全文分段萃取統整資訊』，產出一份可直接貼回 Markdown 的統整章節內容。\n"
        "輸出規則（非常重要）：\n"
        "- 只輸出嚴格 JSON 物件，不得包含 code fence，不得包含任何說明文字。\n"
        "- 使用繁體中文（zh-Hant）。\n"
        "- 不要杜撰；若未知請留空或輸出〔無〕。\n"
        "- 內容要能對應既有資料集慣例：主旨、重點整理、流程/步驟、注意事項/警示、表格/比較、版本資訊。\n"
        "- 請控制篇幅：key_points 最多 10 項、steps 最多 10 項、warnings 最多 10 項、version_info 最多 8 項。\n"
        "- tables_markdown 若內容很大，請輸出 '〔無法辨識〕' 或 '無'，不要輸出超長表格。\n"
        "\n"
        "範本風格參考（只參考，不要輸出整份範本）：\n"
        + template_md[:2500]
    )
    schema = {
        "purpose": "string",
        "key_points": ["string", "..."],
        "steps": ["string", "..."],
        "warnings": ["string", "..."],
        "tables_markdown": "string (若表格過大/過多：請輸出 '〔無法辨識〕' 或 '無'；不要輸出超長表格)",
        "version_info": ["string", "..."],
    }
    user = (
        "請依照 schema 輸出 JSON。\n\n"
        f"### schema\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"### content\n{digest_text}\n"
    )
    return system, user


def _rebuild_summary_block_with_chunked(
    *,
    cfg: LLMConfig,
    fulltext: str,
    template_md: str,
    chunk_cfg: ChunkingConfig,
    preprocess_cfg: PreprocessConfig,
    signals_caps: SignalsCapConfig,
    usage_total: UsageStats,
    elapsed_total: float,
) -> Tuple[Optional[str], Optional[str], float]:
    # Preprocess once (table compaction etc.) to reduce noise + timeouts.
    fulltext = _preprocess_fulltext_for_llm(fulltext, preprocess_cfg)

    chunks = _split_text_into_chunks(
        fulltext, chunk_chars=chunk_cfg.chunk_chars, overlap_chars=chunk_cfg.overlap_chars
    )
    if chunk_cfg.chunk_max and chunk_cfg.chunk_max > 0:
        chunks = chunks[: chunk_cfg.chunk_max]

    chunk_system, chunk_schema = _build_chunked_prompts(template_md)
    chunk_summaries: List[str] = []
    merged: Dict[str, Any] = {}

    def _build_carry_text() -> str:
        if not chunk_cfg.carry:
            return ""
        win = max(0, int(chunk_cfg.carry_summary_window))
        recent = chunk_summaries[-win:] if win > 0 else []
        carry_lines: List[str] = []
        if recent:
            carry_lines.append("recent_summaries:")
            for i, s in enumerate(recent, start=max(1, len(chunk_summaries) - len(recent) + 1)):
                carry_lines.append(f"- ({i}) {s}")
        if merged:
            carry_lines.append("")
            carry_lines.append("running_signals_json (capped):")
            capped = _cap_signals(merged, signals_caps)
            carry_lines.append(json.dumps(capped, ensure_ascii=False))
        carry_text = "\n".join(carry_lines).strip()
        if chunk_cfg.carry_max_chars and chunk_cfg.carry_max_chars > 0:
            carry_text = carry_text[: chunk_cfg.carry_max_chars]
        return carry_text

    def _run_one_chunk(chunk_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        carry_text = _build_carry_text()
        user = (
            f"### chunk\n{idx}/{len(chunks)}\n"
            f"### schema\n{json.dumps(chunk_schema, ensure_ascii=False)}\n"
            + (f"### running_context\n{carry_text}\n" if carry_text else "")
            + f"### content\n{chunk_text}\n"
        )

        obj = None
        err = None
        # Per-chunk retry: if parse/repair fails, re-ask with stricter constraints to avoid truncated JSON.
        for attempt in range(2):
            attempt_system = chunk_system
            if attempt == 1:
                attempt_system = (
                    chunk_system
                    + "\n\n"
                    + "【緊急簡化模式】\n"
                    + "- 請輸出最小必要資訊：chunk_summary_zh_hant <= 3 句。\n"
                    + "- signals 中所有清單各最多 3 項，且每項 <= 60 字。\n"
                    + "- 若表格很多，table_topics 請留空陣列。\n"
                )
            try:
                out_text, u, e = _call_chat_completions(
                    cfg,
                    system_prompt=attempt_system,
                    user_prompt=user,
                    max_tokens_override=700,
                )
                usage_total.add(u)
                elapsed_total_local = float(e or 0.0)
            except Exception as ex:
                # bubble up as a structured error (so we can adaptive-split)
                return None, f"llm_call_failed: {ex}"

            # update outer elapsed_total via closure trick: return elapsed delta in err string is ugly; instead we mutate via nonlocal
            nonlocal elapsed_total
            elapsed_total += elapsed_total_local

            obj, err, elapsed_total = _parse_llm_json_with_one_repair(
                cfg,
                llm_text=out_text,
                schema_hint=chunk_schema,
                usage_total=usage_total,
                elapsed_total=elapsed_total,
            )
            if not err and isinstance(obj, dict):
                return obj, None
        return None, err or "invalid_json"

    def _process_chunk_adaptive(chunk_text: str, depth: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        obj, err = _run_one_chunk(chunk_text)
        if not err and isinstance(obj, dict):
            return obj, None
        if not chunk_cfg.adaptive_split:
            return None, err
        if depth >= int(chunk_cfg.adaptive_max_depth):
            return None, err
        if len(chunk_text) <= int(chunk_cfg.adaptive_min_chunk_chars):
            return None, err
        # If it failed (timeout / invalid json), split in half and try both; merge their signals locally.
        mid = len(chunk_text) // 2
        left = chunk_text[:mid]
        right = chunk_text[mid:]
        o1, e1 = _process_chunk_adaptive(left, depth + 1)
        o2, e2 = _process_chunk_adaptive(right, depth + 1)
        if e1 and e2:
            return None, f"adaptive_split_failed: left={e1} right={e2}"
        merged_local: Dict[str, Any] = {}
        summary_parts: List[str] = []
        for o in [o1, o2]:
            if not isinstance(o, dict):
                continue
            ss = str(o.get("chunk_summary_zh_hant") or "").strip()
            if ss:
                summary_parts.append(ss)
            sig = o.get("signals") if isinstance(o.get("signals"), dict) else {}
            merged_local = _merge_signals(merged_local, sig if isinstance(sig, dict) else {})
        merged_local = _cap_signals(merged_local, signals_caps)
        return {"chunk_summary_zh_hant": " / ".join(summary_parts[:2]), "signals": merged_local}, None

    for idx, chunk in enumerate(chunks, start=1):
        obj, err = _process_chunk_adaptive(chunk, 0)
        if err or not isinstance(obj, dict):
            return None, f"chunked_failed[{idx}/{len(chunks)}]: {err or 'invalid_json'}", elapsed_total

        summ = str(obj.get("chunk_summary_zh_hant") or "").strip()
        if summ:
            chunk_summaries.append(summ)
        sig = obj.get("signals") if isinstance(obj.get("signals"), dict) else {}
        merged = _merge_signals(merged, sig if isinstance(sig, dict) else {})
        merged = _cap_signals(merged, signals_caps)

    def _build_digest_text(max_chars: int) -> str:
        # Prefer merged signals (structured + stable) over long narrative chunk summaries.
        capped = _cap_signals(merged, signals_caps)
        merged_json = json.dumps(capped, ensure_ascii=False)
        # Keep only the last N chunk summaries; older ones tend to be redundant and bloat prompts.
        keep_n = 18
        kept_summaries = chunk_summaries[-keep_n:]
        digest_lines: List[str] = []
        digest_lines.append("merged_signals_json:")
        digest_lines.append(merged_json)
        if kept_summaries:
            digest_lines.append("")
            digest_lines.append("recent_chunk_summaries:")
            for i, s in enumerate(kept_summaries, start=max(1, len(chunk_summaries) - len(kept_summaries) + 1)):
                digest_lines.append(f"- ({i}) {s}")
        digest = "\n".join(digest_lines)
        if max_chars <= 0:
            return digest
        if len(digest) <= max_chars:
            return digest
        # If still too long, drop summaries first.
        digest = "\n".join(["merged_signals_json:", merged_json])
        if len(digest) <= max_chars:
            return digest
        # If still too long, aggressively shrink caps.
        tiny_caps = SignalsCapConfig(per_key_max=25, version_lines_max=40, title_candidates_max=15)
        merged_json2 = json.dumps(_cap_signals(merged, tiny_caps), ensure_ascii=False)
        digest2 = "\n".join(["merged_signals_json:", merged_json2])
        if len(digest2) <= max_chars:
            return digest2
        # Last resort: keep only high-signal fields
        minimal = {}
        for kk in ["purpose", "version_lines", "title_candidates"]:
            if kk in merged:
                minimal[kk] = merged.get(kk)
        merged_json3 = json.dumps(_cap_signals(minimal, tiny_caps), ensure_ascii=False)
        return "\n".join(["merged_signals_json:", merged_json3])[:max_chars]

    digest_text = _build_digest_text(int(chunk_cfg.final_max_chars))

    final_system, final_user = _build_final_prompts(template_md, digest_text)
    final_text, u2, e2 = _call_chat_completions(
        cfg, system_prompt=final_system, user_prompt=final_user, max_tokens_override=1200
    )
    usage_total.add(u2)
    elapsed_total += float(e2 or 0.0)

    final_obj, err2, elapsed_total = _parse_llm_json_with_one_repair(
        cfg,
        llm_text=final_text,
        schema_hint={"purpose": "", "key_points": [], "steps": [], "warnings": [], "tables_markdown": "", "version_info": []},
        usage_total=usage_total,
        elapsed_total=elapsed_total,
    )
    if err2 or not isinstance(final_obj, dict):
        return None, err2 or "final_json_parse_failed", elapsed_total

    purpose = str(final_obj.get("purpose") or "").strip() or "〔無〕"
    key_points = final_obj.get("key_points") if isinstance(final_obj.get("key_points"), list) else []
    steps = final_obj.get("steps") if isinstance(final_obj.get("steps"), list) else []
    warnings = final_obj.get("warnings") if isinstance(final_obj.get("warnings"), list) else []
    tables_md = str(final_obj.get("tables_markdown") or "").strip()
    version_info = final_obj.get("version_info") if isinstance(final_obj.get("version_info"), list) else []

    out = []
    out.append("## 主旨\n")
    out.append(purpose.strip() + "\n\n")
    out.append("## 重點整理\n")
    out.append(_render_bullets([str(x) for x in key_points]))
    out.append("\n")
    out.append("## 流程/步驟\n")
    out.append(_render_steps([str(x) for x in steps]))
    out.append("\n")
    out.append("## 注意事項/警示\n")
    out.append(_render_bullets([str(x) for x in warnings]))
    out.append("\n")
    out.append("## 表格/比較\n")
    out.append((tables_md + "\n") if tables_md else "〔無〕\n")
    out.append("\n")
    out.append("## 版本資訊\n")
    out.append(_render_bullets([str(x) for x in version_info]))
    out.append("\n")
    return "".join(out), None, elapsed_total


def _apply_summary_update(path: Path, dry_run: bool) -> FileRunResult:
    original = path.read_text(encoding="utf-8")
    yaml_text, body = _split_front_matter(original)
    if yaml_text is None:
        return FileRunResult(False, "no_front_matter", 0.0, UsageStats())

    # keep `## 全文` and below exactly
    before_fulltext, fulltext_block = _split_before_fulltext(body)
    if not fulltext_block:
        return FileRunResult(False, "no_fulltext_section", 0.0, UsageStats())

    h1_block, _ = _find_h1_block(before_fulltext)
    if not h1_block:
        # still proceed; we will keep empty h1 and just place summary
        h1_block = ""
    else:
        # 清掉「統整失敗」的舊占位提示（避免重建後仍殘留）
        h1_block = re.sub(r"(?m)^\>\s*統整失敗.*\n?", "", h1_block).strip() + "\n\n"

    # Extract the actual fulltext content (for LLM input) but preserve block in output
    fulltext_content = _extract_section(fulltext_block, "全文") or ""
    if not fulltext_content.strip():
        # fallback: use the fulltext block as-is (after heading)
        parts = fulltext_block.splitlines()
        fulltext_content = "\n".join(parts[1:]).strip()

    # Load template
    template_path = _PROJECT_ROOT / "rag_test_data" / "cus" / "範本" / "education.handout.md"
    template_md = template_path.read_text(encoding="utf-8") if template_path.exists() else ""

    model_override = getattr(_apply_summary_update, "_model", None)
    cfg = _load_llm_config(model_override=model_override)
    usage_total = UsageStats()
    elapsed_total = 0.0

    # runtime flags (set by main via setattr)
    chunked = bool(getattr(_apply_summary_update, "_chunked", True))
    chunk_cfg = ChunkingConfig(
        chunk_chars=int(getattr(_apply_summary_update, "_chunk_chars", 6000) or 6000),
        overlap_chars=int(getattr(_apply_summary_update, "_chunk_overlap", 300) or 300),
        chunk_max=int(getattr(_apply_summary_update, "_chunk_max", 0) or 0),
        carry=bool(getattr(_apply_summary_update, "_chunk_carry", True)),
        carry_max_chars=int(getattr(_apply_summary_update, "_chunk_carry_max_chars", 3500) or 3500),
        carry_summary_window=int(getattr(_apply_summary_update, "_chunk_carry_summary_window", 3) or 3),
        final_max_chars=int(getattr(_apply_summary_update, "_chunk_final_max_chars", 14000) or 14000),
        adaptive_split=bool(getattr(_apply_summary_update, "_chunk_adaptive_split", True)),
        adaptive_min_chunk_chars=int(getattr(_apply_summary_update, "_chunk_adaptive_min_chars", 1200) or 1200),
        adaptive_max_depth=int(getattr(_apply_summary_update, "_chunk_adaptive_max_depth", 4) or 4),
    )
    preprocess_cfg = PreprocessConfig(
        compact_tables=bool(getattr(_apply_summary_update, "_preprocess_compact_tables", True)),
        table_max_rows=int(getattr(_apply_summary_update, "_preprocess_table_max_rows", 30) or 30),
        table_keep_rows=int(getattr(_apply_summary_update, "_preprocess_table_keep_rows", 6) or 6),
        collapse_blank_lines=bool(getattr(_apply_summary_update, "_preprocess_collapse_blank_lines", True)),
        blank_lines_max=int(getattr(_apply_summary_update, "_preprocess_blank_lines_max", 2) or 2),
    )
    signals_caps = SignalsCapConfig(
        per_key_max=int(getattr(_apply_summary_update, "_signals_per_key_max", 60) or 60),
        version_lines_max=int(getattr(_apply_summary_update, "_signals_version_lines_max", 60) or 60),
        title_candidates_max=int(getattr(_apply_summary_update, "_signals_title_candidates_max", 30) or 30),
    )

    if not chunked:
        # direct mode: a single call with bounded max chars
        max_chars = int(getattr(_apply_summary_update, "_max_chars", 16000) or 16000)
        text = _preprocess_fulltext_for_llm(fulltext_content, preprocess_cfg)
        if max_chars and max_chars > 0:
            text = text[:max_chars]
        system, user = _build_final_prompts(template_md, text)
        out_text, u, e = _call_chat_completions(cfg, system_prompt=system, user_prompt=user)
        usage_total.add(u)
        elapsed_total += float(e or 0.0)
        obj, err, elapsed_total = _parse_llm_json_with_one_repair(
            cfg, llm_text=out_text, usage_total=usage_total, elapsed_total=elapsed_total
        )
        if err or not isinstance(obj, dict):
            return FileRunResult(False, err or "direct_failed", elapsed_total, usage_total)
        purpose = str(obj.get("purpose") or "").strip() or "〔無〕"
        key_points = obj.get("key_points") if isinstance(obj.get("key_points"), list) else []
        steps = obj.get("steps") if isinstance(obj.get("steps"), list) else []
        warnings = obj.get("warnings") if isinstance(obj.get("warnings"), list) else []
        tables_md = str(obj.get("tables_markdown") or "").strip()
        version_info = obj.get("version_info") if isinstance(obj.get("version_info"), list) else []
        summary_block = (
            "## 主旨\n\n"
            + purpose
            + "\n\n## 重點整理\n"
            + _render_bullets([str(x) for x in key_points])
            + "\n## 流程/步驟\n"
            + _render_steps([str(x) for x in steps])
            + "\n## 注意事項/警示\n"
            + _render_bullets([str(x) for x in warnings])
            + "\n## 表格/比較\n"
            + ((tables_md + "\n") if tables_md else "〔無〕\n")
            + "\n## 版本資訊\n"
            + _render_bullets([str(x) for x in version_info])
            + "\n"
        )
    else:
        summary_block, err, elapsed_total = _rebuild_summary_block_with_chunked(
            cfg=cfg,
            fulltext=fulltext_content,
            template_md=template_md,
            chunk_cfg=chunk_cfg,
            preprocess_cfg=preprocess_cfg,
            signals_caps=signals_caps,
            usage_total=usage_total,
            elapsed_total=elapsed_total,
        )
        if err or not summary_block:
            return FileRunResult(False, err or "chunked_failed", float(elapsed_total), usage_total)

    new_body = (h1_block + summary_block + fulltext_block.lstrip("\n")).rstrip() + "\n"
    new_text = "---\n" + (yaml_text.rstrip() + "\n") + "---\n" + new_body

    changed = new_text != original
    if changed and not dry_run:
        path.write_text(new_text, encoding="utf-8")
    status = ("updated" if changed else "no_change") + (
        f" ({elapsed_total:.2f}s, tokens_in={usage_total.prompt_tokens} tokens_out={usage_total.completion_tokens})"
    )
    return FileRunResult(changed, status, float(elapsed_total), usage_total)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="目標資料夾（包含 *.md）")
    ap.add_argument("--file", action="append", default=[], help="只處理指定檔案（可重複指定）")
    ap.add_argument("--file-list", default="", help="只處理指定檔案清單（每行一個路徑，可含 # 註解）")
    ap.add_argument("--exclude", action="append", default=[], help="排除檔名（可重複指定）")
    ap.add_argument("--limit", type=int, default=0, help="只處理前 N 個檔案（0=不限制）")
    ap.add_argument("--dry-run", action="store_true", help="不寫檔，只輸出統計/失敗清單")
    ap.add_argument("--no-progress", action="store_true", help="不輸出逐檔進度")
    ap.add_argument("--only-failed", action="store_true", help="只處理含『統整失敗』字樣的檔案")
    ap.add_argument("--model", default="", help="覆寫使用的模型（預設用 CHAT_MODEL 或 gpt-4o-mini）")

    # modes
    ap.add_argument("--chunked", action="store_true", help="使用 chunked+final 統整（適合超長文件）")
    ap.add_argument("--no-chunked", action="store_true", help="強制不用 chunked（單次直出，可能爆掉）")
    ap.add_argument("--max-chars", type=int, default=16000, help="非 chunked：提供給 LLM 的最大字元數（0=不截斷）")
    ap.add_argument("--chunk-chars", type=int, default=6000, help="chunked：每片最大字元數")
    ap.add_argument("--chunk-overlap", type=int, default=300, help="chunked：相鄰片段重疊字元數")
    ap.add_argument("--chunk-max", type=int, default=0, help="chunked：最多處理前 N 片（0=不限制）")
    ap.add_argument("--chunk-final-max-chars", type=int, default=16000, help="chunked：最終統整那次輸入最大字元數")
    ap.add_argument("--chunk-carry", action="store_true", help="chunked：帶入前次摘要/線索（降低失焦）")
    ap.add_argument("--chunk-carry-max-chars", type=int, default=4000, help="chunked：carry 上限字元數")
    ap.add_argument("--chunk-carry-summary-window", type=int, default=3, help="chunked：帶入最近幾段摘要")
    ap.add_argument("--chunk-adaptive-split", action="store_true", help="chunked：chunk 失敗時自動再切小重試（更抗 timeout/壞 JSON）")
    ap.add_argument("--chunk-adaptive-min-chars", type=int, default=1200, help="chunked：adaptive split 最小 chunk 字元數")
    ap.add_argument("--chunk-adaptive-max-depth", type=int, default=4, help="chunked：adaptive split 最大遞迴深度")

    # preprocess
    ap.add_argument("--no-preprocess-compact-tables", action="store_true", help="不要壓縮大型 Markdown 表格（不建議，容易失真/timeout）")
    ap.add_argument("--preprocess-table-max-rows", type=int, default=30, help="表格超過此列數視為大型表格")
    ap.add_argument("--preprocess-table-keep-rows", type=int, default=6, help="大型表格保留前幾列，其餘省略")
    ap.add_argument("--no-preprocess-collapse-blank-lines", action="store_true", help="不要壓縮多餘空行")
    ap.add_argument("--preprocess-blank-lines-max", type=int, default=2, help="最多允許連續空行數")

    # signals caps
    ap.add_argument("--signals-per-key-max", type=int, default=60, help="merged signals 每個清單欄位最多保留幾項（避免爆長）")
    ap.add_argument("--signals-version-lines-max", type=int, default=60, help="版本資訊最多保留幾項")
    ap.add_argument("--signals-title-candidates-max", type=int, default=30, help="標題候選最多保留幾項")

    args = ap.parse_args()

    target_dir = Path(args.dir)
    if not target_dir.exists():
        raise SystemExit(f"dir not found: {target_dir}")

    excludes = set(args.exclude or [])

    # Build file list
    explicit: List[Path] = []
    if args.file_list:
        file_list_path = Path(args.file_list)
        if not file_list_path.exists():
            raise SystemExit(f"file-list not found: {file_list_path}")
        for raw in file_list_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            explicit.append(Path(line))
    if args.file:
        explicit.extend([Path(x) for x in args.file])

    if explicit:
        md_files: List[Path] = []
        for p in explicit:
            if not p.is_absolute():
                p = (_PROJECT_ROOT / p).resolve()
            if p.name in excludes:
                continue
            if not p.exists():
                raise SystemExit(f"file not found: {p}")
            md_files.append(p)
        # de-dup
        seen = set()
        uniq: List[Path] = []
        for p in md_files:
            s = str(p)
            if s in seen:
                continue
            seen.add(s)
            uniq.append(p)
        md_files = uniq
    else:
        md_files = sorted([p for p in target_dir.glob("*.md") if p.is_file()])

    # pre-load config to fail-fast
    _ = _load_llm_config(model_override=args.model or None)

    total = 0
    changed_count = 0
    skipped = 0
    failed = 0
    failed_items: List[Tuple[str, str]] = []
    usage_sum = UsageStats()
    elapsed_sum = 0.0
    planned_total = len(md_files)
    started_all = time.monotonic()

    for p in md_files:
        if p.name in excludes:
            skipped += 1
            continue
        if args.limit and total >= args.limit:
            break
        if args.only_failed:
            txt = p.read_text(encoding="utf-8", errors="replace")
            if "統整失敗" not in txt:
                skipped += 1
                continue
        total += 1
        idx = total

        # stash runtime flags
        setattr(_apply_summary_update, "_model", (args.model or None))
        setattr(_apply_summary_update, "_chunked", bool(args.chunked) or not bool(args.no_chunked))
        setattr(_apply_summary_update, "_max_chars", int(args.max_chars))
        setattr(_apply_summary_update, "_chunk_chars", int(args.chunk_chars))
        setattr(_apply_summary_update, "_chunk_overlap", int(args.chunk_overlap))
        setattr(_apply_summary_update, "_chunk_max", int(args.chunk_max))
        setattr(_apply_summary_update, "_chunk_final_max_chars", int(args.chunk_final_max_chars))
        setattr(_apply_summary_update, "_chunk_carry", bool(args.chunk_carry))
        setattr(_apply_summary_update, "_chunk_carry_max_chars", int(args.chunk_carry_max_chars))
        setattr(_apply_summary_update, "_chunk_carry_summary_window", int(args.chunk_carry_summary_window))
        setattr(_apply_summary_update, "_chunk_adaptive_split", bool(args.chunk_adaptive_split))
        setattr(_apply_summary_update, "_chunk_adaptive_min_chars", int(args.chunk_adaptive_min_chars))
        setattr(_apply_summary_update, "_chunk_adaptive_max_depth", int(args.chunk_adaptive_max_depth))

        setattr(_apply_summary_update, "_preprocess_compact_tables", not bool(args.no_preprocess_compact_tables))
        setattr(_apply_summary_update, "_preprocess_table_max_rows", int(args.preprocess_table_max_rows))
        setattr(_apply_summary_update, "_preprocess_table_keep_rows", int(args.preprocess_table_keep_rows))
        setattr(_apply_summary_update, "_preprocess_collapse_blank_lines", not bool(args.no_preprocess_collapse_blank_lines))
        setattr(_apply_summary_update, "_preprocess_blank_lines_max", int(args.preprocess_blank_lines_max))

        setattr(_apply_summary_update, "_signals_per_key_max", int(args.signals_per_key_max))
        setattr(_apply_summary_update, "_signals_version_lines_max", int(args.signals_version_lines_max))
        setattr(_apply_summary_update, "_signals_title_candidates_max", int(args.signals_title_candidates_max))

        # run
        per_started = time.monotonic()
        try:
            res = _apply_summary_update(p, dry_run=args.dry_run)
        except Exception as e:
            res = FileRunResult(False, f"exception: {e}", 0.0, UsageStats())
        per_elapsed = time.monotonic() - per_started

        usage_sum.add(res.usage)
        elapsed_sum += float(res.elapsed_s or per_elapsed)

        if res.changed:
            changed_count += 1
        if (
            "failed" in res.status
            or "exception:" in res.status
            or "LLM API" in res.status
            or "no_fulltext_section" in res.status
        ):
            failed += 1
            failed_items.append((p.name, res.status))
            print(f"[FAIL] {p.name}: {res.status}")

        if not args.no_progress:
            avg = elapsed_sum / max(1, idx)
            remaining = max(0, planned_total - idx)
            eta_s = remaining * avg
            print(
                f"[{idx}/{planned_total}] {p.name} | "
                f"{'CHANGED' if res.changed else 'OK'} | "
                f"t={res.elapsed_s:.2f}s | in={res.usage.prompt_tokens} out={res.usage.completion_tokens} | "
                f"{res.status} | ETA~{eta_s:.0f}s"
            )

    elapsed_all = time.monotonic() - started_all
    print(
        "done. "
        f"scanned={len(md_files)} processed={total} changed={changed_count} skipped={skipped} failed={failed} "
        f"dry_run={bool(args.dry_run)} elapsed_total={elapsed_all:.2f}s "
        f"tokens_in_total={usage_sum.prompt_tokens} tokens_out_total={usage_sum.completion_tokens}"
    )
    if failed_items:
        print("\nfailed_files:")
        for name, reason in failed_items:
            print(f"- {name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


