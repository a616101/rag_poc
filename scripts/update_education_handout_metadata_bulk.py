"""
批次更新 rag_test_data/cus/衛教 內的 education.handout Markdown 檔案 metadata。

此版本以「LLM 萃取」為主：
- 會把文件的主旨/重點整理/流程/注意事項/版本資訊/全文標題等整理後送到 LLM。
- LLM 必須輸出「嚴格 JSON」（無 code fence、無多餘文字），腳本再轉成 YAML front matter。

重要保證：
- 只會改動檔案開頭的 YAML front matter（第一個 '---' 到第二個 '---'）。
- 第二個 '---' 之後的「原文內容」會以原樣保留（位元級不變）。

LLM 設定來源（優先順序）：
1) 直接讀取環境變數：
   - OPENAI_API_BASE（例如：http://127.0.0.1:1234/v1）
   - OPENAI_API_KEY
   - CHAT_MODEL（可選，預設用專案 settings.chat_model）
2) 若可 import，則使用 chatbot_rag.core.config.settings 內的：
   - settings.openai_api_base / settings.openai_api_key / settings.chat_model

使用方式（建議先 dry-run + limit 抽查）：
  python3 scripts/update_education_handout_metadata_bulk.py \
    --dir rag_test_data/cus/衛教 \
    --exclude 1701504308_705.md \
    --limit 3 \
    --dry-run

只處理指定檔案（可重複 --file 或用 --file-list）：
  python3 scripts/update_education_handout_metadata_bulk.py \
    --file rag_test_data/cus/衛教/1701571949_705.md \
    --file rag_test_data/cus/衛教/1701571983_300.md \
    --force

正式套用：
  python3 scripts/update_education_handout_metadata_bulk.py \
    --dir rag_test_data/cus/衛教 \
    --exclude 1701504308_705.md

覆寫策略：
- 預設只會「補齊空白/佔位值」（例如 title=檔名、keywords=[]）。
- 若要強制以 LLM 結果覆寫現有 metadata，請加 --force。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import time
import urllib.request
import urllib.error
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


FRONT_MATTER_RE = re.compile(r"\A---\n(?P<yaml>.*?\n)---\n", re.DOTALL)

# 讓 scripts/ 內的腳本可直接 import src/chatbot_rag
# （避免使用者必須自行設定 PYTHONPATH）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if _SRC_DIR.exists():
    sys.path.insert(0, str(_SRC_DIR))


def _strip(s: str) -> str:
    return s.strip()


def _is_placeholder_title(title: str, file_stem: str) -> bool:
    t = (title or "").strip()
    if not t:
        return True
    if t == file_stem:
        return True
    # 僅數字/底線（常見為檔名）
    if re.fullmatch(r"[0-9_]+", t):
        return True
    return False


def _parse_inline_list(value: str) -> List[str]:
    v = value.strip()
    if not (v.startswith("[") and v.endswith("]")):
        return []
    inner = v[1:-1].strip()
    if not inner:
        return []
    parts = [p.strip() for p in inner.split(",")]
    return [p for p in parts if p]


def _yaml_quote_if_needed(value: str) -> str:
    """
    盡量保持 YAML 乾淨：大多數中文不需要引號。
    只有遇到冒號/井字等可能造成歧義時才加引號。
    """
    if value is None:
        return ""
    v = str(value)
    if v == "":
        return ""
    if v.startswith((" ", "-")) or v.endswith(" "):
        return '"' + v.replace('"', '\\"') + '"'
    if any(ch in v for ch in [":", "#", "{", "}", "[", "]"]):
        return '"' + v.replace('"', '\\"') + '"'
    return v


def _emit_yaml(data: Dict[str, Any]) -> str:
    """
    極簡 YAML 輸出器（支援 dict / list / str / None）。
    - 不依賴 PyYAML
    - 依 insertion order 輸出
    """

    def emit_node(node: Any, indent: int) -> List[str]:
        sp = " " * indent
        if node is None:
            return [sp + ""]
        if isinstance(node, dict):
            lines: List[str] = []
            for k, v in node.items():
                if isinstance(v, dict):
                    lines.append(f"{sp}{k}:")
                    lines.extend(emit_node(v, indent + 2))
                elif isinstance(v, list):
                    if len(v) == 0:
                        lines.append(f"{sp}{k}: []")
                    else:
                        lines.append(f"{sp}{k}:")
                        for item in v:
                            if isinstance(item, dict):
                                # list of dicts: "- key: val" style isn't used here; we keep multi-line dict
                                lines.append(f"{sp}  -")
                                # Emit dict with extra indentation and without leading key line
                                for dk, dv in item.items():
                                    if isinstance(dv, (dict, list)):
                                        lines.append(f"{sp}    {dk}:")
                                        lines.extend(emit_node(dv, indent + 6))
                                    else:
                                        lines.append(f"{sp}    {dk}: {_yaml_quote_if_needed(dv)}")
                            else:
                                lines.append(f"{sp}  - {_yaml_quote_if_needed(item)}")
                else:
                    lines.append(f"{sp}{k}: {_yaml_quote_if_needed(v)}")
            return lines
        if isinstance(node, list):
            if len(node) == 0:
                return [sp + "[]"]
            lines = []
            for item in node:
                lines.append(f"{sp}- {_yaml_quote_if_needed(item)}")
            return lines
        # scalar
        return [sp + _yaml_quote_if_needed(node)]

    out_lines = emit_node(data, 0)
    # 確保每行都有內容（YAML 允許空值，但我們在 key: 後面留空即可）
    return "\n".join(out_lines).rstrip() + "\n"


def _split_front_matter(text: str) -> Tuple[Optional[str], str]:
    m = FRONT_MATTER_RE.search(text)
    if not m:
        return None, text
    yaml_text = m.group("yaml")
    rest = text[m.end() :]  # includes everything after the second ---\n
    return yaml_text, rest


def _extract_section(body: str, heading: str) -> Optional[str]:
    """
    擷取 markdown 某個二級 heading (## xxx) 下面直到下一個 '## ' heading 之前的文字。
    """
    # 用 \n## 確保不是檔首
    pat = re.compile(rf"(?m)^\#\#\s+{re.escape(heading)}\s*$")
    m = pat.search(body)
    if not m:
        return None
    start = m.end()
    # 找下一個二級 heading
    m2 = re.search(r"(?m)^\#\#\s+\S+", body[start:])
    end = start + (m2.start() if m2 else len(body[start:]))
    return body[start:end].strip("\n")


def _looks_like_hospital_line(line: str) -> bool:
    l = line.strip()
    if not l:
        return True
    if "屏東基督教醫院" in l or "PINGTUNG" in l or "CHRISTIAN" in l or "HOSPITAL" in l:
        return True
    if l in {"財團法人", "財團法人事業機構", "屏東基督教醫院", "屏基醫療財團法人屏東基督教醫院"}:
        return True
    return False


def _extract_title_from_fulltext(body: str) -> Optional[str]:
    sec = _extract_section(body, "全文")
    if not sec:
        return None
    lines = [ln.rstrip() for ln in sec.splitlines()]
    # 跳過空行/醫院抬頭/英文抬頭
    i = 0
    while i < len(lines) and _looks_like_hospital_line(lines[i]):
        i += 1
    # 有些檔案在全文前有多一個空行/分隔
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    if i >= len(lines):
        return None

    title_lines: List[str] = []
    stop_re = re.compile(r"^(文／|文/|修訂日期|製訂日期|制定日期|單位/作者|衛教編號|文件編號|編號)\s*[:：]")
    while i < len(lines):
        l = lines[i].strip()
        if not l:
            break
        if l.startswith("|"):  # 版本表格
            break
        if stop_re.search(l):
            break
        # 避免把「20190051」這類尾碼當標題
        if re.fullmatch(r"\d{6,}", l):
            break
        title_lines.append(l)
        i += 1
        # 標題通常不會超過 3 行
        if len(title_lines) >= 3:
            break

    title = "".join(title_lines).strip()
    return title or None


@dataclasses.dataclass
class VersionInfo:
    code: Optional[str] = None  # 文件編號 / 衛教編號 / 編號
    created_ym: Optional[str] = None  # 製訂日期/制定日期 (YYYY-MM)
    revision_ym: Optional[str] = None  # 修訂日期/xx修日期 (YYYY-MM)
    revision_note: Optional[str] = None  # 七修/九修...
    department: Optional[str] = None  # 單位/作者 的單位


def _normalize_year(year_str: str) -> Optional[int]:
    y = year_str.strip()
    if not y.isdigit():
        return None
    # 修正 OCR 常見錯誤：20013 -> 2013
    if len(y) == 5 and y.startswith("20") and y[2] == "0":
        y = "20" + y[3:]
    if len(y) != 4:
        return None
    return int(y)


def _parse_ym(text: str) -> Optional[str]:
    """
    嘗試從文本中解析 YYYY-MM。
    支援：
    - 2003.11 / 2003-11 / 2003/11
    - 西元2019年08月
    - 2015 年 01 月
    """
    t = text.strip()
    # 先抓 YYYY + MM
    m = re.search(r"(\d{4,5})\s*[.\-/年]\s*(\d{1,2})", t)
    if not m:
        return None
    year_raw, mon_raw = m.group(1), m.group(2)
    year = _normalize_year(year_raw)
    if year is None:
        return None
    mon = int(mon_raw)
    if mon < 1 or mon > 12:
        return None
    return f"{year:04d}-{mon:02d}"


def _parse_version_info(body: str) -> VersionInfo:
    sec = _extract_section(body, "版本資訊")
    if not sec:
        return VersionInfo()

    info = VersionInfo()
    lines = [ln.strip() for ln in sec.splitlines() if ln.strip()]

    # 表格 key/value：| key | value |
    for ln in lines:
        if ln.startswith("|") and ln.count("|") >= 3:
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if len(cells) < 2:
                continue
            k = re.sub(r"\*\*(.*?)\*\*", r"\1", cells[0]).strip()
            v = re.sub(r"\*\*(.*?)\*\*", r"\1", cells[1]).strip()
            if not k or not v:
                continue
            # 統一 key
            if k in {"文件編號", "衛教編號", "編號"} and not info.code:
                info.code = v
            if k in {"製訂日期", "制定日期"} and not info.created_ym:
                info.created_ym = _parse_ym(v)
            if k == "修訂日期" and not info.revision_ym:
                info.revision_ym = _parse_ym(v)
            mnote = re.match(r"^([一二三四五六七八九十]修)日期$", k)
            if mnote and not info.revision_ym:
                info.revision_note = info.revision_note or mnote.group(1)
                info.revision_ym = _parse_ym(v)
            if k in {"單位/作者", "單位", "作者"} and not info.department:
                # 只取 '/' 前面的單位
                dep = v.split("/")[0].split("／")[0].strip()
                info.department = dep or None

    # bullet：- **key**：value
    for ln in lines:
        m = re.match(r"^-?\s*(?:\*\*(.*?)\*\*|([^：:]+))\s*[:：]\s*(.+)$", ln)
        if not m:
            continue
        k = (m.group(1) or m.group(2) or "").strip()
        v = (m.group(3) or "").strip()
        if not k or not v:
            continue
        if k in {"文件編號", "衛教編號", "編號"} and not info.code:
            info.code = v
        if k in {"製訂日期", "制定日期"} and not info.created_ym:
            info.created_ym = _parse_ym(v)
        if k == "修訂日期" and not info.revision_ym:
            info.revision_ym = _parse_ym(v)
        mnote = re.match(r"^([一二三四五六七八九十]修)日期$", k)
        if mnote and not info.revision_ym:
            info.revision_note = info.revision_note or mnote.group(1)
            info.revision_ym = _parse_ym(v)
        if k in {"單位/作者", "單位", "作者"} and not info.department:
            dep = v.split("/")[0].split("／")[0].strip()
            info.department = dep or None

    return info


def _infer_condition_from_title(title: str) -> Optional[str]:
    t = (title or "").strip()
    if not t:
        return None
    # 常見「X患者...」
    for token in ["患者", "病人", "病患"]:
        if token in t:
            base = t.split(token)[0].strip()
            if 2 <= len(base) <= 12:
                return base
    # 常見「X的...」
    if "的" in t:
        base = t.split("的")[0].strip()
        if 2 <= len(base) <= 12:
            return base
    # 常見「X篩檢」
    if "篩檢" in t:
        base = t.split("篩檢")[0].strip()
        if 2 <= len(base) <= 12:
            return base
    # fallback：整個標題（上限 20）
    return t[:20]


def _unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        s = (it or "").strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _split_text_into_chunks(text: str, *, chunk_chars: int, overlap_chars: int) -> List[str]:
    """
    以字元數切分文本，並保留 overlap。
    - chunk_chars: 每片最大字元數（<=0 則不切，回傳整段）
    - overlap_chars: 相鄰片段重疊字元數（>=0）
    """
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
        # next start with overlap
        i = max(0, j - ov)
        if len(out) > 2000:
            # safety fuse
            break
    return out


def _merge_signals(base: Dict[str, Any], inc: Dict[str, Any]) -> Dict[str, Any]:
    """
    合併 chunk signals（以 list 為主），保留順序並去重。
    """
    out = dict(base or {})
    for k, v in (inc or {}).items():
        if isinstance(v, list):
            cur = out.get(k)
            cur_list = cur if isinstance(cur, list) else []
            out[k] = _unique_keep_order(cur_list + [str(x) for x in v])
        elif isinstance(v, str):
            # prefer first non-empty
            if not out.get(k) and v.strip():
                out[k] = v.strip()
        elif isinstance(v, dict):
            cur = out.get(k)
            if isinstance(cur, dict):
                out[k] = _merge_signals(cur, v)
            else:
                out[k] = v
        else:
            if k not in out:
                out[k] = v
    return out


def _parse_llm_json_with_one_repair(
    cfg: LLMConfig,
    *,
    llm_text: str,
    usage_total: UsageStats,
    elapsed_total: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], float]:
    """
    將 LLM 輸出解析為 JSON dict；失敗時最多做一次 repair。
    回傳：(obj, error_status, elapsed_total_updated)
    """
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
            repaired, usage_repair, elapsed_repair = _call_chat_completions(
                cfg,
                system_prompt=(
                    "你是嚴格的 JSON 修復器。\n"
                    "任務：把輸入轉成「可被 json.loads 解析」的 JSON 物件。\n"
                    "輸出規則（非常重要）：\n"
                    "- 只輸出 JSON（以 { 開頭，以 } 結尾），不得包含任何說明文字。\n"
                    "- 不得包含 code fence。\n"
                    "- 字串內的換行請用 \\n 轉義（不得出現未轉義的實際換行）。\n"
                    "- 內容保持原意，不要自行新增欄位。\n"
                ),
                user_prompt=(
                    "下面這段應該是一個 JSON 物件，但目前無法被解析。請修復成有效 JSON，並只輸出 JSON：\n\n"
                    f"{llm_text}\n"
                ),
            )
            usage_total.add(usage_repair)
            elapsed_total += float(elapsed_repair or 0.0)
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


def _build_final_metadata_prompts_from_text(
    *,
    file_name: str,
    title_hint: Optional[str],
    version_hint: VersionInfo,
    text: str,
    max_chars: int,
) -> Tuple[str, str]:
    """
    將「chunk 摘要/線索統整」餵給 LLM，產出最終 metadata（schema 與原本相同）。
    """
    content = (text or "").strip()
    if max_chars and max_chars > 0:
        content = content[:max_chars]

    system_prompt = (
        "你是「醫院衛教文件 metadata 萃取器」。\n"
        "任務：根據我提供的文件統整資訊，產出可用於 RAG 檢索的 metadata。\n"
        "輸出規則（非常重要）：\n"
        "- 只輸出『嚴格 JSON 物件』，不得包含 code fence、不得包含任何說明文字。\n"
        "- 使用繁體中文（zh-Hant）。\n"
        "- key_actions / red_flags 請用條列句（每項一句）。\n"
        "- dates 使用 YYYY-MM（若未知可回傳空字串）。\n"
        "- tags 必須包含「衛教」。\n"
    )

    hints = {
        "file_name": file_name,
        "title_hint": title_hint or "",
        "version_info_hint": {
            "code": version_hint.code or "",
            "created_at_ym": version_hint.created_ym or "",
            "revision_date_ym": version_hint.revision_ym or "",
            "revision_note": version_hint.revision_note or "",
            "department": version_hint.department or "",
        },
        "org_name_zh": "屏東基督教醫院",
        "lang": "zh-Hant",
        "audience": ["patient", "family"],
    }

    schema = {
        "title_zh_hant": "string",
        "summary_zh_hant": "string (1-2 sentences)",
        "tags": ["衛教", "科別(若可推)", "疾病/主題(若可推)"],
        "education": {
            "category": "discharge|self_care|medication|pre_exam|post_exam|pre_op|post_op|''",
            "code": "string",
            "created_at": "YYYY-MM or ''",
            "revisions": [{"date": "YYYY-MM or ''", "note": "string or ''"}],
            "owner_department_zh": "string or ''",
            "conditions": ["string", "..."],
            "key_actions": ["string", "..."],
            "red_flags": ["string", "..."],
            "followup_department_zh": "string or ''",
            "followup_note": "string or ''",
        },
        "retrieval": {
            "aliases": ["string", "..."],
            "keywords": ["string", "..."],
        },
    }

    user_prompt = (
        "請根據下列內容萃取 metadata，並依 schema 輸出 JSON。\n\n"
        f"### hints\n{json.dumps(hints, ensure_ascii=False)}\n\n"
        f"### schema\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"### content\n{content}\n"
    )
    return system_prompt, user_prompt


def _get_nested(d: Dict[str, Any], keys: List[str], default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def _set_nested(d: Dict[str, Any], keys: List[str], value: Any) -> None:
    cur: Any = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _parse_front_matter_yaml(yaml_text: str) -> Dict[str, Any]:
    """
    針對本資料集固定格式的 YAML 做「足夠用」的解析。
    限制：
    - 不支援 multi-line scalar
    - 不支援複雜型別（anchors、引用...）
    """
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Any]] = [(0, root)]  # (indent, container)

    def current_container() -> Any:
        return stack[-1][1]

    lines = yaml_text.splitlines()
    for raw in lines:
        if raw.strip() == "":
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()

        # pop stack based on indent
        while len(stack) > 1 and indent < stack[-1][0]:
            stack.pop()

        cur = current_container()

        # list item
        if line.startswith("- "):
            item = line[2:].strip()
            # list container must exist
            if not isinstance(cur, list):
                # 這代表 YAML 不符合預期；跳過
                continue
            # dict item (e.g., "- date: 2022-10" in our dataset is split as two lines, so here mostly scalar)
            cur.append(item)
            continue

        # key: value
        m = re.match(r"^([^:]+):\s*(.*)$", line)
        if not m:
            continue
        key = m.group(1).strip()
        val = m.group(2)

        # decide container based on val
        if val == "":
            # create nested container (dict by default)
            # but some keys are list containers in this dataset
            if key in {"audience", "tags"}:
                # can be inline list usually; if empty treat as list
                nxt: Any = []
            elif key in {"conditions", "key_actions", "red_flags"}:
                nxt = []
            elif key in {"aliases", "keywords"}:
                nxt = []
            elif key == "revisions":
                nxt = []  # list of dicts (we'll keep as list of scalars if not parsed)
            else:
                nxt = {}
            if isinstance(cur, dict):
                cur[key] = nxt
            # push
            stack.append((indent + 2, nxt))
            continue

        # inline list
        inline_list = _parse_inline_list(val)
        if inline_list:
            if isinstance(cur, dict):
                cur[key] = inline_list
            continue

        # remove surrounding quotes
        v = val.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]

        if isinstance(cur, dict):
            cur[key] = v
        # special handling: revisions list items in dataset are multi-line; we won't parse deeply here.

    # 修正：revisions 在資料中是 list of dicts，我們用 regex 再補一個簡單解析
    # 目標：把
    # revisions:
    #   - date: 2022-10
    #     note: 七修
    # 解析成 [{"date": "...", "note": "..."}]
    rev_block = _get_nested(root, ["education", "revisions"])
    if isinstance(rev_block, list):
        # our minimal parser would have appended scalars only; rebuild from yaml_text lines
        mrev = re.search(r"(?m)^\s*revisions:\s*$", yaml_text)
        if mrev:
            # capture until next top-level (2-space) key under education or until dedent
            after = yaml_text[mrev.end() :]
            # take a reasonable slice
            slice_lines = after.splitlines()
            collected: List[str] = []
            for ln in slice_lines:
                if ln.startswith("  owner:") or ln.startswith("retrieval:") or ln.startswith("source:") or ln.startswith("updated_at:") or ln.startswith("last_reviewed:"):
                    break
                collected.append(ln)
            # parse list items
            revs: List[Dict[str, str]] = []
            cur_rev: Dict[str, str] = {}
            for ln in collected:
                if re.match(r"^\s*-\s*$", ln):
                    if cur_rev:
                        revs.append(cur_rev)
                    cur_rev = {}
                    continue
                m1 = re.match(r"^\s*-\s*date:\s*(.*)$", ln)
                if m1:
                    if cur_rev:
                        revs.append(cur_rev)
                    cur_rev = {"date": m1.group(1).strip()}
                    continue
                m2 = re.match(r"^\s*note:\s*(.*)$", ln.strip())
                if m2 and cur_rev is not None:
                    cur_rev["note"] = m2.group(1).strip()
                    continue
                m3 = re.match(r"^\s*date:\s*(.*)$", ln.strip())
                if m3 and cur_rev is not None:
                    cur_rev["date"] = m3.group(1).strip()
                    continue
                m4 = re.match(r"^\s*note:\s*(.*)$", ln.strip())
                if m4 and cur_rev is not None:
                    cur_rev["note"] = m4.group(1).strip()
                    continue
            if cur_rev:
                revs.append(cur_rev)
            if revs:
                _set_nested(root, ["education", "revisions"], revs)

    return root


@dataclasses.dataclass
class LLMConfig:
    api_base: str
    api_key: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 1200
    timeout_s: float = 120.0


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


def _load_llm_config(model_override: Optional[str] = None) -> LLMConfig:
    """
    讀取 LLM 連線設定。
    - 優先使用環境變數 OPENAI_API_BASE/OPENAI_API_KEY/CHAT_MODEL
    - 否則嘗試 import chatbot_rag.core.config.settings
    """
    env_base = os.getenv("OPENAI_API_BASE") or os.getenv("openai_api_base")
    env_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
    env_model = os.getenv("CHAT_MODEL") or os.getenv("chat_model")

    if env_base and env_key:
        return LLMConfig(
            api_base=env_base.rstrip("/"),
            api_key=env_key,
            model=model_override or env_model or "gpt-4o-mini",
        )

    try:
        from chatbot_rag.core.config import settings  # type: ignore

        return LLMConfig(
            api_base=str(settings.openai_api_base).rstrip("/"),
            api_key=str(settings.openai_api_key),
            model=model_override or str(getattr(settings, "chat_model", "")) or "gpt-4o-mini",
            temperature=float(getattr(settings, "chat_temperature", 0.1)),
            max_tokens=int(getattr(settings, "chat_max_tokens", 1200)),
            timeout_s=float(getattr(settings, "llm_request_timeout", 120.0)),
        )
    except Exception as e:
        raise RuntimeError(
            "找不到 LLM 設定。請設定環境變數 OPENAI_API_BASE/OPENAI_API_KEY，"
            "或確保可 import chatbot_rag.core.config.settings。"
        ) from e


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    從文字中抽出第一個完整 JSON object（{...}）。
    用來處理模型偶爾加上的前後說明字。
    """
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


def _call_chat_completions(
    cfg: LLMConfig, system_prompt: str, user_prompt: str
) -> Tuple[str, UsageStats, float]:
    """
    以 OpenAI 相容 Chat Completions API 呼叫 LLM。
    不依賴第三方套件（httpx/requests）。
    """
    url = cfg.api_base.rstrip("/") + "/chat/completions"
    body = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

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
        raise RuntimeError(f"LLM API error: {e}") from e


def _json_sanitize_common_issues(text: str) -> str:
    """
    嘗試修補常見的「幾乎是 JSON」輸出：
    - 移除 } 或 ] 前的 trailing comma
    注意：此函式不嘗試修補字串中的未跳脫換行等更複雜問題。
    """
    s = (text or "").strip()
    # remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def _repair_json_via_llm(cfg: LLMConfig, bad_text: str) -> str:
    """
    當模型輸出不是有效 JSON 時，請模型「只回傳可解析的 JSON」。
    這通常能解決：未跳脫換行、漏引號、尾逗號等問題。
    """
    system_prompt = (
        "你是嚴格的 JSON 修復器。\n"
        "任務：把輸入轉成「可被 json.loads 解析」的 JSON 物件。\n"
        "輸出規則（非常重要）：\n"
        "- 只輸出 JSON（以 { 開頭，以 } 結尾），不得包含任何說明文字。\n"
        "- 不得包含 code fence。\n"
        "- 字串內的換行請用 \\n 轉義（不得出現未轉義的實際換行）。\n"
        "- 內容保持原意，不要自行新增欄位。\n"
    )
    user_prompt = (
        "下面這段應該是一個 JSON 物件，但目前無法被解析。請修復成有效 JSON，並只輸出 JSON：\n\n"
        f"{bad_text}\n"
    )
    repaired, _, _ = _call_chat_completions(cfg, system_prompt=system_prompt, user_prompt=user_prompt)
    return repaired


def _build_llm_prompts(
    *,
    file_name: str,
    title_hint: Optional[str],
    version_hint: VersionInfo,
    body: str,
    max_chars: int,
    full_doc: bool = False,
    tail_chars: int = 0,
) -> Tuple[str, str]:
    """
    把原文切出「足夠」資訊給 LLM，避免超長文件爆 token。
    """
    if full_doc:
        # 直接提供整份 Markdown 主體（不含 front-matter）
        content = body
        if max_chars and max_chars > 0 and len(content) > max_chars:
            tc = max(0, int(tail_chars or 0))
            tc = min(tc, max_chars)
            if tc > 0:
                head_n = max_chars - tc
                head_part = content[:head_n]
                tail_part = content[-tc:]
                content = (
                    head_part
                    + "\n\n...[TRUNCATED: middle omitted for context safety]...\n\n"
                    + tail_part
                )
            else:
                content = content[:max_chars]
    else:
        # 優先提供結構化區塊
        sections = []
        for h in ["主旨", "重點整理", "流程/步驟", "注意事項/警示", "版本資訊"]:
            sec = _extract_section(body, h)
            if sec:
                sections.append(f"## {h}\n{sec}")

        fulltext = _extract_section(body, "全文")
        title_from_fulltext = _extract_title_from_fulltext(body)
        if fulltext:
            # 只取全文前 80 行（通常含抬頭/標題/日期/編號）
            ft_lines = fulltext.splitlines()[:80]
            sections.append("## 全文（節錄）\n" + "\n".join(ft_lines).strip())

        content = "\n\n".join(sections).strip()
        if not content:
            # fallback：取 body 前 max_chars
            content = body if not (max_chars and max_chars > 0) else body[:max_chars]
        else:
            if max_chars and max_chars > 0:
                content = content[:max_chars]

    system_prompt = (
        "你是「醫院衛教文件 metadata 萃取器」。\n"
        "任務：根據我提供的文件內容，產出可用於 RAG 檢索的 metadata。\n"
        "輸出規則（非常重要）：\n"
        "- 只輸出『嚴格 JSON 物件』，不得包含 code fence、不得包含任何說明文字。\n"
        "- 使用繁體中文（zh-Hant）。\n"
        "- key_actions / red_flags 請用條列句（每項一句）。\n"
        "- dates 使用 YYYY-MM（若未知可回傳空字串）。\n"
        "- tags 必須包含「衛教」。\n"
    )

    # 把我們用規則抽到的提示一併給 LLM（減少 OCR/格式歧義）
    hints = {
        "file_name": file_name,
        "title_hint": title_hint or title_from_fulltext or "",
        "version_info_hint": {
            "code": version_hint.code or "",
            "created_at_ym": version_hint.created_ym or "",
            "revision_date_ym": version_hint.revision_ym or "",
            "revision_note": version_hint.revision_note or "",
            "department": version_hint.department or "",
        },
        "org_name_zh": "屏東基督教醫院",
        "lang": "zh-Hant",
        "audience": ["patient", "family"],
    }

    schema = {
        "title_zh_hant": "string",
        "summary_zh_hant": "string (1-2 sentences)",
        "tags": ["衛教", "科別(若可推)", "疾病/主題(若可推)"],
        "education": {
            "category": "discharge|self_care|medication|pre_exam|post_exam|pre_op|post_op|''",
            "code": "string",
            "created_at": "YYYY-MM or ''",
            "revisions": [{"date": "YYYY-MM or ''", "note": "string or ''"}],
            "owner_department_zh": "string or ''",
            "conditions": ["string", "..."],
            "key_actions": ["string", "..."],
            "red_flags": ["string", "..."],
            "followup_department_zh": "string or ''",
            "followup_note": "string or ''",
        },
        "retrieval": {
            "aliases": ["string", "..."],
            "keywords": ["string", "..."],
        },
    }

    user_prompt = (
        "請根據下列內容萃取 metadata，並依 schema 輸出 JSON。\n\n"
        f"### hints\n{json.dumps(hints, ensure_ascii=False)}\n\n"
        f"### schema\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"### content\n{content}\n"
    )
    return system_prompt, user_prompt


def _merge_llm_metadata_into_frontmatter(
    existing: Dict[str, Any],
    llm_obj: Dict[str, Any],
    *,
    force: bool,
    file_stem: str,
) -> Dict[str, Any]:
    """
    把 LLM 產出的欄位 merge 回 front matter。
    - 只處理範本相關欄位，並保留既有 type/id/source/updated_at/last_reviewed 等。
    - 預設只覆寫「空白/佔位值」，force=True 才覆寫現有值。
    """
    data = existing

    def should_set(current: Any) -> bool:
        if force:
            return True
        if current is None:
            return True
        if isinstance(current, str):
            if current.strip() == "":
                return True
            if _is_placeholder_title(current, file_stem):
                return True
        if isinstance(current, list) and len(current) == 0:
            return True
        if isinstance(current, dict) and len(current) == 0:
            return True
        return False

    # title/summary
    cur_title = _get_nested(data, ["title", "zh-Hant"], "")
    new_title = (llm_obj.get("title_zh_hant") or "").strip()
    if new_title and should_set(cur_title):
        _set_nested(data, ["title", "zh-Hant"], new_title)

    cur_summary = _get_nested(data, ["summary", "zh-Hant"], "")
    new_summary = (llm_obj.get("summary_zh_hant") or "").strip()
    if new_summary and should_set(cur_summary):
        _set_nested(data, ["summary", "zh-Hant"], new_summary)

    # tags
    new_tags = llm_obj.get("tags")
    if isinstance(new_tags, list):
        new_tags = _unique_keep_order([str(x) for x in new_tags])
        if "衛教" not in new_tags:
            new_tags = ["衛教"] + new_tags
        cur_tags = data.get("tags")
        if should_set(cur_tags):
            data["tags"] = new_tags

    # education block
    edu = llm_obj.get("education") if isinstance(llm_obj.get("education"), dict) else {}
    if "education" not in data or not isinstance(data["education"], dict):
        data["education"] = {}

    # category/code/created_at
    for k_llm, k_path in [
        ("category", ["education", "category"]),
        ("code", ["education", "code"]),
        ("created_at", ["education", "created_at"]),
    ]:
        v = (edu.get(k_llm) or "").strip() if isinstance(edu.get(k_llm), str) else edu.get(k_llm)
        cur = _get_nested(data, k_path, "")
        if v and should_set(cur):
            _set_nested(data, k_path, v)

    # revisions
    revs = edu.get("revisions")
    if isinstance(revs, list) and len(revs) > 0:
        # normalize shape
        norm_revs = []
        for r in revs[:3]:
            if isinstance(r, dict):
                norm_revs.append(
                    {
                        "date": str(r.get("date") or "").strip(),
                        "note": str(r.get("note") or "").strip(),
                    }
                )
        cur_revs = _get_nested(data, ["education", "revisions"], [])
        if should_set(cur_revs):
            _set_nested(data, ["education", "revisions"], norm_revs)

    # owner department
    owner_dep = (edu.get("owner_department_zh") or "").strip() if isinstance(edu.get("owner_department_zh"), str) else ""
    cur_owner_dep = _get_nested(data, ["education", "owner", "department_zh"], "")
    if owner_dep and should_set(cur_owner_dep):
        _set_nested(data, ["education", "owner", "department_zh"], owner_dep)

    # conditions/key_actions/red_flags
    for key in ["conditions", "key_actions", "red_flags"]:
        v = edu.get(key)
        if isinstance(v, list):
            v_norm = _unique_keep_order([str(x) for x in v])
            cur_v = _get_nested(data, ["education", key], [])
            if should_set(cur_v):
                _set_nested(data, ["education", key], v_norm)

    # followup
    f_dep = (edu.get("followup_department_zh") or "").strip() if isinstance(edu.get("followup_department_zh"), str) else ""
    f_note = (edu.get("followup_note") or "").strip() if isinstance(edu.get("followup_note"), str) else ""
    cur_f_dep = _get_nested(data, ["education", "followup", "department_zh"], "")
    cur_f_note = _get_nested(data, ["education", "followup", "note"], "")
    if f_dep and should_set(cur_f_dep):
        _set_nested(data, ["education", "followup", "department_zh"], f_dep)
    if f_note and should_set(cur_f_note):
        _set_nested(data, ["education", "followup", "note"], f_note)

    # retrieval
    ret = llm_obj.get("retrieval") if isinstance(llm_obj.get("retrieval"), dict) else {}
    if "retrieval" not in data or not isinstance(data["retrieval"], dict):
        data["retrieval"] = {}
    for key in ["aliases", "keywords"]:
        v = ret.get(key)
        if isinstance(v, list):
            v_norm = _unique_keep_order([str(x) for x in v])
            cur_v = _get_nested(data, ["retrieval", key], [])
            if should_set(cur_v):
                _set_nested(data, ["retrieval", key], v_norm)

    # safety defaults
    if not _get_nested(data, ["org", "name_zh"]):
        _set_nested(data, ["org", "name_zh"], "屏東基督教醫院")
    if not data.get("lang"):
        data["lang"] = "zh-Hant"
    if not data.get("audience"):
        data["audience"] = ["patient", "family"]
    if "source" not in data or not isinstance(data["source"], dict):
        data["source"] = {"url": "內部衛教 PDF"}
    if not _get_nested(data, ["source", "url"]):
        _set_nested(data, ["source", "url"], "內部衛教 PDF")

    return data


def _apply_metadata_updates(path: Path, dry_run: bool) -> FileRunResult:
    original = path.read_text(encoding="utf-8")
    yaml_text, rest = _split_front_matter(original)
    if yaml_text is None:
        return FileRunResult(
            changed=False,
            status="no_front_matter",
            elapsed_s=0.0,
            usage=UsageStats(),
        )

    data = _parse_front_matter_yaml(yaml_text)

    file_stem = path.stem
    body = rest  # must remain unchanged

    # ---- LLM extraction ----
    vinfo = _parse_version_info(body)
    title_hint = _extract_title_from_fulltext(body)

    cfg = _load_llm_config()
    # runtime flags (set by main via setattr)
    max_chars = int(getattr(_apply_metadata_updates, "_max_chars", 12000) or 12000)
    full_doc = bool(getattr(_apply_metadata_updates, "_full_doc", False))
    tail_chars = int(getattr(_apply_metadata_updates, "_tail_chars", 0) or 0)
    chunked = bool(getattr(_apply_metadata_updates, "_chunked", False))
    chunk_chars = int(getattr(_apply_metadata_updates, "_chunk_chars", 6000) or 6000)
    chunk_overlap = int(getattr(_apply_metadata_updates, "_chunk_overlap", 300) or 300)
    chunk_max = int(getattr(_apply_metadata_updates, "_chunk_max", 0) or 0)
    chunk_final_max_chars = int(getattr(_apply_metadata_updates, "_chunk_final_max_chars", 16000) or 16000)
    chunk_carry = bool(getattr(_apply_metadata_updates, "_chunk_carry", False))
    chunk_carry_max_chars = int(getattr(_apply_metadata_updates, "_chunk_carry_max_chars", 4000) or 4000)
    chunk_carry_summary_window = int(getattr(_apply_metadata_updates, "_chunk_carry_summary_window", 3) or 3)
    system_prompt, user_prompt = _build_llm_prompts(
        file_name=path.name,
        title_hint=title_hint,
        version_hint=vinfo,
        body=body,
        max_chars=max_chars,
        full_doc=full_doc,
        tail_chars=tail_chars,
    )

    usage_total = UsageStats()
    elapsed_total = 0.0

    if chunked:
        # 1) chunk-wise extraction to digest full document safely
        chunks = _split_text_into_chunks(body, chunk_chars=chunk_chars, overlap_chars=chunk_overlap)
        if chunk_max and chunk_max > 0:
            chunks = chunks[:chunk_max]

        chunk_summaries: List[str] = []
        merged_signals: Dict[str, Any] = {}

        chunk_system = (
            "你是「衛教文件分段萃取器」。\n"
            "任務：對單一片段抽取「metadata 線索」與「片段重點摘要」。\n"
            "輸出規則（非常重要）：\n"
            "- 只輸出嚴格 JSON 物件，不得包含 code fence，不得包含任何說明文字。\n"
            "- 使用繁體中文（zh-Hant）。\n"
            "- 只萃取你在片段中能支持的資訊；不確定就留空或省略。\n"
        )
        chunk_schema = {
            "chunk_summary_zh_hant": "string (<=5 sentences)",
            "signals": {
                "title_candidates": ["string", "..."],
                "tags": ["string", "..."],
                "departments_zh": ["string", "..."],
                "education_codes": ["string", "..."],
                "created_at_ym": ["YYYY-MM", "..."],
                "revision_ym": ["YYYY-MM", "..."],
                "revision_notes": ["string", "..."],
                "conditions": ["string", "..."],
                "key_actions": ["string", "..."],
                "red_flags": ["string", "..."],
                "followup_departments_zh": ["string", "..."],
                "aliases": ["string", "..."],
                "keywords": ["string", "..."],
            },
        }

        for idx, chunk in enumerate(chunks, start=1):
            carry_text = ""
            if chunk_carry:
                # Provide a bounded "running context" to reduce drift.
                win = max(0, int(chunk_carry_summary_window))
                recent = chunk_summaries[-win:] if win > 0 else []
                carry_lines = []
                if recent:
                    carry_lines.append("recent_summaries:")
                    for i, s in enumerate(recent, start=max(1, len(chunk_summaries) - len(recent) + 1)):
                        carry_lines.append(f"- ({i}) {s}")
                if merged_signals:
                    carry_lines.append("")
                    carry_lines.append("running_signals_json:")
                    carry_lines.append(json.dumps(merged_signals, ensure_ascii=False))
                carry_text = "\n".join(carry_lines).strip()
                if chunk_carry_max_chars and chunk_carry_max_chars > 0:
                    carry_text = carry_text[:chunk_carry_max_chars]

            chunk_user = (
                f"### file\n{path.name}\n"
                f"### chunk\n{idx}/{len(chunks)}\n"
                f"### schema\n{json.dumps(chunk_schema, ensure_ascii=False)}\n"
                + (f"### running_context\n{carry_text}\n" if carry_text else "")
                + f"### content\n{chunk}\n"
            )
            out_text, u, e = _call_chat_completions(cfg, system_prompt=chunk_system, user_prompt=chunk_user)
            usage_total.add(u)
            elapsed_total += float(e or 0.0)

            obj, err, elapsed_total = _parse_llm_json_with_one_repair(
                cfg, llm_text=out_text, usage_total=usage_total, elapsed_total=elapsed_total
            )
            if err or not isinstance(obj, dict):
                return FileRunResult(
                    changed=False,
                    status=f"chunked_failed[{idx}/{len(chunks)}]: {err or 'invalid_json'}",
                    elapsed_s=float(elapsed_total),
                    usage=usage_total,
                )

            summ = str(obj.get("chunk_summary_zh_hant") or "").strip()
            if summ:
                chunk_summaries.append(summ)
            sig = obj.get("signals") if isinstance(obj.get("signals"), dict) else {}
            merged_signals = _merge_signals(merged_signals, sig if isinstance(sig, dict) else {})

        # 2) final metadata call using digest
        digest_lines = []
        digest_lines.append("chunk_summaries:")
        for i, s in enumerate(chunk_summaries, start=1):
            digest_lines.append(f"- ({i}) {s}")
        digest_lines.append("")
        digest_lines.append("merged_signals_json:")
        digest_lines.append(json.dumps(merged_signals, ensure_ascii=False))
        digest_text = "\n".join(digest_lines)

        final_system, final_user = _build_final_metadata_prompts_from_text(
            file_name=path.name,
            title_hint=title_hint,
            version_hint=vinfo,
            text=digest_text,
            max_chars=chunk_final_max_chars,
        )
        llm_text, u2, e2 = _call_chat_completions(cfg, system_prompt=final_system, user_prompt=final_user)
        usage_total.add(u2)
        elapsed_total += float(e2 or 0.0)
    else:
        llm_text, usage_primary, elapsed_primary = _call_chat_completions(
            cfg, system_prompt=system_prompt, user_prompt=user_prompt
        )
        usage_total.add(usage_primary)
        elapsed_total += float(elapsed_primary or 0.0)

    llm_obj, err, elapsed_total = _parse_llm_json_with_one_repair(
        cfg, llm_text=llm_text, usage_total=usage_total, elapsed_total=elapsed_total
    )
    if err or not isinstance(llm_obj, dict):
        return FileRunResult(
            changed=False,
            status=err or "llm_json_parse_failed",
            elapsed_s=float(elapsed_total),
            usage=usage_total,
        )

    # merge
    force = bool(getattr(_apply_metadata_updates, "_force", False))
    merged = _merge_llm_metadata_into_frontmatter(
        data,
        llm_obj,
        force=force,
        file_stem=file_stem,
    )
    data = merged

    # Re-emit YAML in canonical-ish order (respecting existing data when present)
    canonical: Dict[str, Any] = {}
    for k in ["type", "id", "title", "summary", "org", "lang", "audience", "tags", "education", "retrieval", "source", "updated_at", "last_reviewed"]:
        if k in data:
            canonical[k] = data[k]

    new_yaml = _emit_yaml(canonical)
    new_text = "---\n" + new_yaml + "---\n" + rest

    changed = new_text != original
    if changed and not dry_run:
        path.write_text(new_text, encoding="utf-8")
    status = "updated" if changed else "no_change"
    status += (
        f" ({elapsed_total:.2f}s, "
        f"tokens_in={usage_total.prompt_tokens} tokens_out={usage_total.completion_tokens})"
    )
    return FileRunResult(
        changed=changed,
        status=status,
        elapsed_s=float(elapsed_total),
        usage=usage_total,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="目標資料夾（包含 *.md）")
    ap.add_argument(
        "--file",
        action="append",
        default=[],
        help="只處理指定檔案（可重複指定；可用相對/絕對路徑）。指定後會忽略 --dir 內的其他檔案。",
    )
    ap.add_argument(
        "--file-list",
        default="",
        help="只處理指定檔案清單（檔案內每行一個路徑；可含空行與 # 註解）。指定後會忽略 --dir 內的其他檔案。",
    )
    ap.add_argument("--exclude", action="append", default=[], help="排除檔名（可重複指定）")
    ap.add_argument("--dry-run", action="store_true", help="只顯示變更統計，不寫檔")
    ap.add_argument("--limit", type=int, default=0, help="只處理前 N 個檔案（0 = 不限制）")
    ap.add_argument("--force", action="store_true", help="強制以 LLM 結果覆寫現有 metadata（預設只補空白/佔位值）")
    ap.add_argument("--model", default="", help="覆寫使用的模型（預設用 settings.chat_model 或環境變數 CHAT_MODEL）")
    ap.add_argument("--no-progress", action="store_true", help="不輸出逐檔進度（預設會顯示進度/耗時/tokens/ETA）")
    ap.add_argument(
        "--full-doc",
        action="store_true",
        help="將整份 Markdown 主體內容提供給 LLM 參考（可能耗時與 tokens 大幅增加；仍會受 --max-chars 影響）",
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="提供給 LLM 的最大字元數（0 表示不截斷）。預設 12000。",
    )
    ap.add_argument(
        "--tail-chars",
        type=int,
        default=0,
        help="僅在 --full-doc 且 --max-chars>0 且內容超長時生效：保留結尾 N 字元，並用 head+(省略)+tail 組合（避免只看到開頭）。預設 0（不保留尾端）。",
    )
    ap.add_argument(
        "--chunked",
        action="store_true",
        help="分段萃取模式：先將全文分片逐段抽取重點與線索，再用統整內容產出 metadata（適合超長文件、降低單次爆掉風險）。",
    )
    ap.add_argument("--chunk-chars", type=int, default=6000, help="chunked 模式：每片最大字元數（預設 6000）。")
    ap.add_argument("--chunk-overlap", type=int, default=300, help="chunked 模式：相鄰片段重疊字元數（預設 300）。")
    ap.add_argument("--chunk-max", type=int, default=0, help="chunked 模式：最多處理前 N 片（0=不限制）。")
    ap.add_argument(
        "--chunk-final-max-chars",
        type=int,
        default=16000,
        help="chunked 模式：最終統整那次 LLM 的輸入最大字元數（預設 16000）。",
    )
    ap.add_argument(
        "--chunk-carry",
        action="store_true",
        help="chunked 模式：在處理每個片段時帶入「目前累積摘要/線索」以降低失焦（會增加 tokens）。",
    )
    ap.add_argument(
        "--chunk-carry-max-chars",
        type=int,
        default=4000,
        help="chunked 模式：帶入的累積上下文最大字元數（避免越滾越大）。預設 4000。",
    )
    ap.add_argument(
        "--chunk-carry-summary-window",
        type=int,
        default=3,
        help="chunked 模式：帶入最近幾段摘要作為上下文（預設 3）。",
    )
    args = ap.parse_args()

    target_dir = Path(args.dir)
    excludes = set(args.exclude or [])

    if not target_dir.exists():
        raise SystemExit(f"dir not found: {target_dir}")

    # pre-load LLM config early to fail-fast (unless dry-run wants to skip)
    if not args.dry_run:
        _ = _load_llm_config(model_override=args.model or None)

    # Build file list (either explicit list, or scan directory)
    explicit_files: List[Path] = []
    if args.file_list:
        file_list_path = Path(args.file_list)
        if not file_list_path.exists():
            raise SystemExit(f"file-list not found: {file_list_path}")
        for raw in file_list_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            explicit_files.append(Path(line))
    if args.file:
        explicit_files.extend([Path(x) for x in args.file])

    if explicit_files:
        md_files = []
        for p in explicit_files:
            # allow relative paths from project root
            if not p.is_absolute():
                p = (_PROJECT_ROOT / p).resolve()
            if p.name in excludes:
                continue
            if not p.exists():
                raise SystemExit(f"file not found: {p}")
            if p.suffix.lower() != ".md":
                raise SystemExit(f"not a markdown file: {p}")
            md_files.append(p)
        # de-dup while preserving order
        seen = set()
        uniq: List[Path] = []
        for p in md_files:
            rp = str(p)
            if rp in seen:
                continue
            seen.add(rp)
            uniq.append(p)
        md_files = uniq
    else:
        md_files = sorted([p for p in target_dir.glob("*.md") if p.is_file()])
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
        total += 1
        idx = total
        # stash force flag on function object (simple way to avoid passing through too many layers)
        setattr(_apply_metadata_updates, "_force", bool(args.force))
        setattr(_apply_metadata_updates, "_full_doc", bool(args.full_doc))
        setattr(_apply_metadata_updates, "_max_chars", int(args.max_chars))
        setattr(_apply_metadata_updates, "_tail_chars", int(args.tail_chars))
        setattr(_apply_metadata_updates, "_chunked", bool(args.chunked))
        setattr(_apply_metadata_updates, "_chunk_chars", int(args.chunk_chars))
        setattr(_apply_metadata_updates, "_chunk_overlap", int(args.chunk_overlap))
        setattr(_apply_metadata_updates, "_chunk_max", int(args.chunk_max))
        setattr(_apply_metadata_updates, "_chunk_final_max_chars", int(args.chunk_final_max_chars))
        setattr(_apply_metadata_updates, "_chunk_carry", bool(args.chunk_carry))
        setattr(_apply_metadata_updates, "_chunk_carry_max_chars", int(args.chunk_carry_max_chars))
        setattr(_apply_metadata_updates, "_chunk_carry_summary_window", int(args.chunk_carry_summary_window))
        if args.model:
            # allow per-run model override
            os.environ.setdefault("CHAT_MODEL", args.model)
        per_started = time.monotonic()
        res = _apply_metadata_updates(p, dry_run=args.dry_run)
        per_elapsed = time.monotonic() - per_started

        usage_sum.add(res.usage)
        elapsed_sum += float(res.elapsed_s or per_elapsed)

        if res.changed:
            changed_count += 1
        if (
            "llm_json_parse_failed" in res.status
            or "LLM API error" in res.status
            or "LLM API HTTPError" in res.status
            or "no_front_matter" in res.status
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


