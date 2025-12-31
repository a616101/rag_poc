#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 結構化抽取 → Markdown（自動辨識：文字型 / 投影片型 / 掃描影像型）

設計原則
- 文字型：以 PyMuPDF 抽取文字，做基本段落/清單重建，保留可用於 RAG 的結構。
- 投影片型：以「每頁一節」輸出，優先抓每頁最大字級作為標題，其餘轉 bullet。
- 掃描影像型：若系統有 tesseract，啟用 OCR；否則輸出頁面圖片並在 Markdown 放提示。

需求
- Python >= 3.10
- 套件：pymupdf, pillow, pytesseract（OCR 需要系統 tesseract 命令）

用法範例
  source .venv/bin/activate
  python scripts/pdf_structured_extract.py \
    --input rag_test_data/source/衛教單原始檔 \
    --output rag_test_data/docs/衛教單_md \
    --max-pages 50
"""

from __future__ import annotations

import argparse
import base64
import io
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Literal

import fitz  # PyMuPDF
from PIL import Image
import httpx
import pypdfium2 as pdfium

try:
    import pytesseract  # type: ignore

    _HAS_PYTESSERACT = True
except Exception:
    pytesseract = None
    _HAS_PYTESSERACT = False


PdfKind = Literal["text", "slides", "scanned", "mixed"]
PdfEngine = Literal["fitz", "pdfium"]
FulltextMode = Literal["section", "details"]


@dataclass(frozen=True)
class PdfFeatures:
    pages: int
    sampled_pages: int
    avg_text_chars: float
    avg_img_area_ratio: float
    landscape_ratio: float
    font_size_span: float
    text_blocks: int
    img_blocks: int


def _iter_pdfs(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            return []
        return [input_path]
    # 針對批次：避免把已處理搬移的 done/ 也納入（會造成重複處理）
    pdfs: list[Path] = []
    for p in input_path.rglob("*.pdf"):
        if "done" in p.parts:
            continue
        pdfs.append(p)
    return sorted(pdfs)


def compute_features(pdf_path: Path, sample_pages: int = 12) -> PdfFeatures:
    doc = fitz.open(pdf_path)
    pages = doc.page_count
    sampled = min(pages, max(1, sample_pages))

    text_chars = 0
    landscape = 0
    img_blocks = 0
    text_blocks = 0
    img_area_ratio_sum = 0.0
    font_sizes: list[float] = []

    for i in range(sampled):
        page = doc.load_page(i)
        rect = page.rect
        if rect.width > rect.height:
            landscape += 1

        t = (page.get_text("text") or "").strip()
        text_chars += len(t)

        d = page.get_text("dict")
        area = float(rect.width * rect.height) or 1.0
        page_img_area = 0.0
        for b in d.get("blocks", []):
            btype = b.get("type")
            if btype == 1:
                img_blocks += 1
                x0, y0, x1, y1 = b.get("bbox", (0, 0, 0, 0))
                page_img_area += max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
            elif btype == 0:
                text_blocks += 1
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        fs = span.get("size")
                        if isinstance(fs, (int, float)):
                            font_sizes.append(float(fs))

        img_area_ratio_sum += page_img_area / area

    doc.close()

    fs_span = (max(font_sizes) - min(font_sizes)) if font_sizes else 0.0
    return PdfFeatures(
        pages=pages,
        sampled_pages=sampled,
        avg_text_chars=text_chars / sampled,
        avg_img_area_ratio=img_area_ratio_sum / sampled,
        landscape_ratio=landscape / sampled,
        font_size_span=fs_span,
        text_blocks=text_blocks,
        img_blocks=img_blocks,
    )


def detect_kind(f: PdfFeatures) -> PdfKind:
    # 掃描：幾乎無可抽文字，且圖片面積高（或根本沒有 text block）
    if (f.avg_text_chars < 30 and f.avg_img_area_ratio > 0.25) or (
        f.text_blocks == 0 and f.avg_img_area_ratio > 0.5
    ):
        return "scanned"

    # 投影片：橫向頁面占比高，或字級跨度大（標題/內文差異大）
    if (f.landscape_ratio > 0.6 or f.font_size_span >= 14.0) and f.avg_text_chars > 120:
        return "slides"

    # 文字：平均文字量足夠高
    if f.avg_text_chars >= 220:
        return "text"

    return "mixed"


_RE_LIST_NUM = re.compile(r"^\s*(\d+)[\.、\)]\s*(.+)\s*$")
_RE_LIST_PAREN = re.compile(r"^\s*[\(（](\d+)[\)）]\s*(.+)\s*$")
_RE_LIST_ZH = re.compile(r"^\s*([一二三四五六七八九十]+)[、\.]\s*(.+)\s*$")
_RE_DASH = re.compile(r"^\s*[-•‧]\s*(.+)\s*$")


def _normalize_line(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def _line_to_md(line: str) -> str:
    line = _normalize_line(line)
    if not line:
        return ""

    m = _RE_DASH.match(line)
    if m:
        return f"- {m.group(1).strip()}"

    m = _RE_LIST_PAREN.match(line)
    if m:
        return f"  - {m.group(2).strip()}"

    m = _RE_LIST_NUM.match(line)
    if m:
        return f"{m.group(1)}. {m.group(2).strip()}"

    m = _RE_LIST_ZH.match(line)
    if m:
        return f"- {m.group(2).strip()}"

    return line


def _extract_page_lines(page: fitz.Page) -> list[tuple[float, float, float, str, float]]:
    """
    以 dict 模式抽取文字，回傳每行：
    (y0, x0, x1, text, max_font_size)
    """
    d = page.get_text("dict")
    lines: list[tuple[float, float, float, str, float]] = []
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            # 同一行內依 x 排序，盡量保留英文空格，中文不額外插空格
            spans_sorted = sorted(spans, key=lambda s: float(s.get("bbox", [0, 0, 0, 0])[0]))
            parts: list[str] = []
            max_fs = 0.0
            x0 = 1e18
            x1 = 0.0
            y0 = 1e18
            for sp in spans_sorted:
                txt = sp.get("text") or ""
                if not txt.strip():
                    continue
                bbox = sp.get("bbox", [0, 0, 0, 0])
                x0 = min(x0, float(bbox[0]))
                y0 = min(y0, float(bbox[1]))
                x1 = max(x1, float(bbox[2]))
                fs = sp.get("size")
                if isinstance(fs, (int, float)):
                    max_fs = max(max_fs, float(fs))
                parts.append(txt)
            text = _normalize_line("".join(parts))
            if text:
                lines.append((y0, x0, x1, text, max_fs))

    lines.sort(key=lambda t: (t[0], t[1]))
    return lines


def _pick_title_first_page(doc: fitz.Document) -> str | None:
    if doc.page_count <= 0:
        return None
    page = doc.load_page(0)
    rect = page.rect
    top_limit = rect.height * 0.35
    best: tuple[float, str] | None = None  # (font_size, text)
    for y0, _x0, _x1, text, fs in _extract_page_lines(page):
        if y0 > top_limit:
            break
        if len(text) < 2:
            continue
        score = fs
        if best is None or score > best[0]:
            best = (score, text)
    return best[1] if best else None


def extract_markdown_text(pdf_path: Path, max_pages: int) -> str:
    doc = fitz.open(pdf_path)
    title = _pick_title_first_page(doc) or pdf_path.stem
    out: list[str] = [f"# {title}", ""]

    pages = min(doc.page_count, max_pages)
    for pi in range(pages):
        page = doc.load_page(pi)
        lines = _extract_page_lines(page)
        if pi > 0:
            out.append("")  # page separator (soft)
        for _y0, _x0, _x1, text, _fs in lines:
            md = _line_to_md(text)
            if md:
                out.append(md)

    doc.close()
    return "\n".join(out).strip() + "\n"


def extract_markdown_slides(pdf_path: Path, max_pages: int) -> str:
    doc = fitz.open(pdf_path)
    title = _pick_title_first_page(doc) or pdf_path.stem
    out: list[str] = [f"# {title}", ""]

    pages = min(doc.page_count, max_pages)
    for pi in range(pages):
        page = doc.load_page(pi)
        lines = _extract_page_lines(page)
        if not lines:
            out.append("- （此頁無可抽取文字）")
            out.append("")
            continue

        # 以該頁最大字級行作為 slide title
        best = max(lines, key=lambda t: t[4])
        slide_title = best[3]
        out.append(f"## {slide_title}")
        out.append("")

        for _y0, _x0, _x1, text, _fs in lines:
            if text == slide_title:
                continue
            md = _line_to_md(text)
            if not md:
                continue
            # 投影片其餘行傾向 bullet（避免把每行都當段落）
            if re.match(r"^\d+\.\s+", md) or md.startswith("- "):
                out.append(md)
            else:
                out.append(f"- {md}")
        out.append("")

    doc.close()
    return "\n".join(out).strip() + "\n"


def _tesseract_available() -> bool:
    return shutil.which("tesseract") is not None and _HAS_PYTESSERACT


def _tesseract_list_langs() -> set[str]:
    """
    取得本機 tesseract 可用語言包（失敗則回傳空集合）。
    """
    if shutil.which("tesseract") is None:
        return set()
    try:
        p = subprocess.run(
            ["tesseract", "--list-langs"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return set()
    langs: set[str] = set()
    for line in (p.stdout or "").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("list of available"):
            continue
        langs.add(line)
    return langs


def _pick_ocr_lang(requested: str) -> str:
    """
    requested:
      - "auto"：自動挑選（優先 chi_tra+eng → chi_sim+eng → chi_tra → chi_sim → eng）
      - 其他：若其中包含的語言包不齊全，會自動剔除不可用者並回退到 eng
    """
    available = _tesseract_list_langs()
    if not available:
        return "eng"

    if requested == "auto":
        candidates = ["chi_tra+eng", "chi_sim+eng", "chi_tra", "chi_sim", "eng"]
        for c in candidates:
            parts = c.split("+")
            if all(p in available for p in parts):
                return c
        return "eng"

    # requested 指定：只保留實際存在的語言
    parts = [p for p in requested.split("+") if p in available]
    if not parts:
        return "eng"
    # 確保至少有 eng，避免空白結果
    if "eng" in available and "eng" not in parts:
        parts.append("eng")
    return "+".join(parts)


def _extract_frontmatter(text: str) -> dict[str, str]:
    """
    從（文字抽取/OCR）內容中擷取常見欄位，提供更結構化的 Markdown。
    這裡只做「穩健、低風險」的提取：標題/編號/製訂日期/修訂日期。
    """
    t = text.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)

    code = ""
    m = re.search(r"編\s*號\s*([A-Z]{1,4}[^ \n]{0,30})", t)
    if m:
        code = m.group(1).strip()

    created = ""
    m = re.search(r"製訂日期\s*(\d{4})[./\-]\s*(\d{1,2})", t)
    if m:
        created = f"{m.group(1)}-{m.group(2).zfill(2)}"

    revised = ""
    # 例如：七修日期 2022.10 / 九修日期 2023.06
    m = re.search(r"[一二三四五六七八九十]+修日期\s*(\d{4})[./\-]\s*(\d{1,2})", t)
    if m:
        revised = f"{m.group(1)}-{m.group(2).zfill(2)}"

    # 標題：嘗試找「氣喘病人返家注意事項」這種，通常在前 40% 內容，且不含日期/編號關鍵字
    title = ""
    head = t[: max(4000, int(len(t) * 0.4))]
    for line in head.splitlines():
        line = _normalize_line(line)
        if not line or len(line) < 4 or len(line) > 40:
            continue
        if any(k in line for k in ["編號", "編 號", "製訂日期", "修日期"]):
            continue
        if re.search(r"\d{4}[./\-]\d{1,2}", line):
            continue
        # 過於通用的醫院抬頭略過
        if "屏東基督教醫院" in line and len(line) < 16:
            continue
        title = line
        break

    fm: dict[str, str] = {}
    if title:
        fm["title"] = title
    if code:
        fm["code"] = code
    if created:
        fm["created_at"] = created
    if revised:
        fm["revised_at"] = revised
    return fm


def _format_frontmatter(fm: dict[str, str]) -> str:
    if not fm:
        return ""
    lines = ["---", "type: pdf.extracted"]
    for k in ["title", "code", "created_at", "revised_at"]:
        v = fm.get(k)
        if v:
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def _ocr_page_to_text(page: fitz.Page, dpi: int, lang: str) -> str:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return (pytesseract.image_to_string(img, lang=lang) if pytesseract else "").strip()  # type: ignore[misc]


def _page_img_area_ratio(page: fitz.Page) -> float:
    """
    估算頁面「圖片區塊」面積占比（0~1）。
    用於決定是否需要做 VLM 圖像解讀。
    """
    rect = page.rect
    area = float(rect.width * rect.height) or 1.0
    d = page.get_text("dict")
    img_area = 0.0
    for b in d.get("blocks", []):
        if b.get("type") != 1:
            continue
        x0, y0, x1, y1 = b.get("bbox", (0, 0, 0, 0))
        img_area += max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
    return max(0.0, min(1.0, img_area / area))


def _pixmap_to_data_url(pix: fitz.Pixmap) -> str:
    png_bytes = pix.tobytes("png")
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"

def _png_bytes_to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _emit_page_image_markdown(
    out: list[str],
    *,
    pix: fitz.Pixmap,
    page_no: int,
    images_dir: Path,
    embed_images: bool,
    save_images: bool,
) -> None:
    """
    將頁面圖片寫入 Markdown。
    - embed_images=True：使用 data URI (base64) 取代檔案連結
    - save_images=True：仍落地存 PNG（供除錯/追溯），不影響 markdown 的引用方式
    """
    img_path = images_dir / f"page-{page_no:03d}.png"
    if save_images:
        images_dir.mkdir(parents=True, exist_ok=True)
        if not img_path.exists():
            pix.save(str(img_path))

    if embed_images:
        out.append(f"![page]({_pixmap_to_data_url(pix)})")
    else:
        out.append(f"![page]({(images_dir.name + '/' + img_path.name)})")


def _emit_page_image_markdown_png(
    out: list[str],
    *,
    png_bytes: bytes,
    page_no: int,
    images_dir: Path,
    embed_images: bool,
    save_images: bool,
) -> None:
    """
    與 _emit_page_image_markdown 類似，但輸入是 png bytes（適用 pdfium render）。
    """
    img_path = images_dir / f"page-{page_no:03d}.png"
    if save_images:
        images_dir.mkdir(parents=True, exist_ok=True)
        if not img_path.exists():
            img_path.write_bytes(png_bytes)
    if embed_images:
        out.append(f"![page]({_png_bytes_to_data_url(png_bytes)})")
    else:
        out.append(f"![page]({(images_dir.name + '/' + img_path.name)})")


def _lmstudio_vision_explain(
    base_url: str,
    model: str,
    data_url: str,
    ocr_hint: str,
    timeout_s: float,
    include_uncertainties: bool,
) -> str:
    """
    呼叫 LM Studio（OpenAI 相容）多模態模型產生圖像解讀。
    - 不使用 API key（LM Studio 多數情況不需要）
    - 以繁中輸出
    """
    system = (
        "你是醫院衛教文件的內容編輯。請用繁體中文輸出。\n"
        "目標：把這一頁的內容「重新排版成結構化 Markdown」，而不是貼出 OCR 原文。\n"
        "規則：\n"
        "- 不要逐字貼回 OCR 文字；請用你自己的話重寫、整理。\n"
        "- 你需要主動修正 OCR 常見錯誤（例如：1009→100%、甘歲→25歲、% 遺失、字母亂碼），並把校正後的結果直接寫進正文。\n"
        "- 只有在真的無法從圖像與上下文判斷時，才用〔不確定〕標記該小段資訊；不要另外列一段『需要確認』清單（除非使用者要求）。\n"
        "- 若有流程/步驟，用序列清單。\n"
        "- 若有表格/比較，轉成 Markdown 表格。\n"
        "- 只輸出整理後的 Markdown（不要加前後解釋）。"
    )
    # 注意：此輸出會被拼接到同一份文件中，避免使用「本頁」等逐頁語氣，減少讀者誤解為逐頁摘要。
    sections = [
        "## 主旨",
        "## 重點整理",
        "## 流程/步驟（如有）",
        "## 注意事項/警示（如有）",
        "## 表格/比較（如有，請用 Markdown 表格）",
        "## 版本資訊（若圖片中有文件編號/制定日期/修訂資訊，請列出）",
    ]
    if include_uncertainties:
        sections.append("## 需要確認的資訊（僅列出真的無法判斷的地方）")

    user_text = (
        "請將此頁內容整理為結構化 Markdown，推薦使用以下段落（可依實際內容增減）：\n"
        + "\n".join(sections)
        + "\n\n你可以參考下方 OCR（可能有錯字）來輔助，但不要複製貼上。\n"
        "請直接在正文中完成必要的自動校正。\n\n"
        f"OCR 參考：\n{ocr_hint[:1800]}"
    )
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(base_url.rstrip("/") + "/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        content = ""
    return (content or "").strip()


def _lmstudio_vision_fulltext_layout(
    base_url: str,
    model: str,
    data_url: str,
    ocr_hint: str,
    timeout_s: float,
) -> str:
    """
    用視覺模型輸出「全文逐字轉寫 + Markdown 排版」。
    核心要求：不摘要、不改寫、不補充，只做排版；看不清楚要標記〔無法辨識〕。
    """
    system = (
        "你是文件數位化編輯。請用繁體中文輸出。\n"
        "任務：將頁面內容『逐字轉寫』成 Markdown，並做合理排版。\n"
        "嚴格規則：\n"
        "- 不要摘要、不要改寫、不要增添任何圖片上沒有的資訊。\n"
        "- 你只能做：換行、段落、清單、表格（Markdown table）、粗體/標題等排版。\n"
        "- 不要用 Markdown code fence 包住輸出（不要輸出 ``` 或 ```markdown）。\n"
        "- 數字、單位、日期、代碼要盡可能保留原樣。\n"
        "- 任何看不清楚/不確定的字句，用〔無法辨識〕標記，不要猜。\n"
        "- 只輸出轉寫後的 Markdown（不要加前後解釋）。"
    )
    user_text = (
        "請依圖片逐字轉寫並排版為 Markdown。\n"
        "請直接輸出 Markdown 內容本體，不要用 ``` 或 ```markdown 包起來。\n"
        "可參考下方 OCR（可能有錯字）以幫助定位，但不可照抄 OCR 錯字；"
        "一切以圖片為準，若圖片也看不清楚就標記〔無法辨識〕。\n\n"
        f"OCR 參考（可能有錯字）：\n{ocr_hint[:2500]}"
    )
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(base_url.rstrip("/") + "/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        content = ""
    return (content or "").strip()


def _lmstudio_text_rewrite(
    base_url: str,
    model: str,
    text: str,
    timeout_s: float,
    include_uncertainties: bool,
) -> str:
    """
    純文字重寫（用於 text 頁面，或掃描頁在沒有 vision 時仍要輸出結構化版面）。
    """
    system = (
        "你是醫院衛教文件的內容編輯。請用繁體中文輸出。\n"
        "目標：把輸入文字「重新排版成結構化 Markdown」，不要逐字貼回原文。\n"
        "你需要主動修正 OCR/抽取常見錯誤（數字、% 符號、年月、常見錯字），並把校正後結果直接寫入正文。\n"
        "只有在真的無法判斷時，才用〔不確定〕標記該小段資訊；不要另外列一段『需要確認』清單（除非使用者要求）。\n"
        "只輸出整理後的 Markdown（不要加前後解釋）。"
    )
    sections = [
        "## 主旨",
        "## 重點整理",
        "## 流程/步驟（如有）",
        "## 注意事項/警示（如有）",
        "## 常見問答（如有）",
        "## 版本資訊（若文字中有文件編號/制定日期/修訂資訊，請列出）",
    ]
    if include_uncertainties:
        sections.append("## 需要確認的資訊（僅列出真的無法判斷的地方）")

    user = (
        "請將以下內容整理為結構化 Markdown，推薦使用以下段落（可依實際內容增減）：\n"
        + "\n".join(sections)
        + "\n\n內容（可能有錯字，請自行校正後重寫，不要貼原文）：\n"
        f"{text[:4000]}"
    )
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ],
    }
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(base_url.rstrip("/") + "/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        content = ""
    return (content or "").strip()


def _lmstudio_document_consolidate(
    *,
    base_url: str,
    model: str,
    title: str,
    per_page_structured: str,
    timeout_s: float,
) -> str:
    """
    第二階段：把「逐頁結構化內容」整合成「整份文件」的統整版。
    嚴格禁止虛構：只能使用輸入中明確存在的資訊；不確定就省略或標記〔無法辨識〕。
    """
    system = (
        "你是醫院衛教文件的總編輯。請用繁體中文輸出。\n"
        "任務：把多段『逐頁結構化內容』整合成一份『整份文件的統整版 Markdown』。\n"
        "嚴格規則（務必遵守）：\n"
        "- 只能使用輸入內容中明確出現的資訊；不得推論、不得補充、不得編造。\n"
        "- 若資料不足或不確定：直接省略該點，或用〔無法辨識〕標記（不要猜）。\n"
        "- 允許合併同義/重複內容、去重、調整排序與版面，但不得改變原意。\n"
        "- 數字、百分比、費用、日期、版本資訊要維持一致；若多處數值矛盾，請並列並註明〔原文不一致〕。\n"
        "- 不要以『第 n 頁』描述；本輸出應是一篇完整文章。\n"
        "- 不要用 Markdown code fence 包住輸出（不要輸出 ``` 或 ```markdown）。\n"
        "- 請務必只輸出一組固定結構的章節標題（每個標題最多出現一次），且不要輸出『## 全文』。\n"
        "  必須依序包含以下章節（即使某章節沒有內容，也要保留標題並以 '- ' 表示空）：\n"
        "  1) ## 主旨\n"
        "  2) ## 重點整理\n"
        "  3) ## 流程/步驟\n"
        "  4) ## 注意事項/警示\n"
        "  5) ## 表格/比較\n"
        "  6) ## 版本資訊\n"
        "- 只輸出統整後的 Markdown（不要加前後解釋）。"
    )
    user = (
        f"文件標題：{title}\n\n"
        "請依『嚴格規則』產出統整版 Markdown（只允許那 6 個章節標題各出現一次）。\n\n"
        "逐頁結構化內容（來源；只能依此統整）：\n"
        f"{per_page_structured[:18000]}"
    )
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ],
    }
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(base_url.rstrip("/") + "/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        content = ""
    return (content or "").strip()


def _extract_fulltext_from_extracted_md(md: str) -> str:
    """
    從第一階段輸出中取出 '## 全文' 之後的內容（不含標題行）。
    """
    if not md:
        return ""
    marker = "\n## 全文\n"
    idx = md.find(marker)
    if idx < 0:
        return ""
    return md[idx + len(marker) :].strip()


def _ocr_text_to_markdown(text: str) -> str:
    """
    將 OCR 的純文字做輕量 Markdown 重建（避免整段塞進 code fence）。
    """
    lines_in = text.splitlines()
    out: list[str] = []
    buf: list[str] = []

    def flush_paragraph() -> None:
        nonlocal buf
        if not buf:
            return
        para = " ".join(buf).strip()
        para = re.sub(r"\s+", " ", para)
        if para:
            out.append(para)
        buf = []

    for raw in lines_in:
        line = _normalize_line(raw)
        if not line:
            flush_paragraph()
            out.append("")
            continue
        md_line = _line_to_md(line)
        # 遇到清單就先 flush 段落，再輸出 item
        if md_line.startswith(("-", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
            flush_paragraph()
            out.append(md_line)
        else:
            buf.append(md_line)

    flush_paragraph()
    # 清掉多餘空行（最多保留 1 個）
    cleaned: list[str] = []
    prev_empty = False
    for l in out:
        empty = not l.strip()
        if empty and prev_empty:
            continue
        cleaned.append(l)
        prev_empty = empty
    return "\n".join(cleaned).strip()


def _strip_markdown_code_fences(md: str) -> str:
    """
    去除模型常見的輸出外框：
      ```markdown
      ...
      ```
    只保留 fence 內的內容。
    """
    s = (md or "").strip()
    if not s:
        return ""
    lines = s.splitlines()
    if len(lines) >= 2 and lines[0].lstrip().startswith("```") and lines[-1].strip() == "```":
        inner = "\n".join(lines[1:-1]).strip()
        return inner
    return s


def _merge_fulltext_pages(fulltexts: list[str]) -> str:
    """
    將各頁全文合併為同一篇連續 Markdown（同一份文章）。
    - 會自動移除每頁可能被模型包上的 code fences
    - 以空行分隔頁與頁的銜接（避免段落黏在一起）
    """
    def norm_line(s: str) -> str:
        # 盡量穩健：忽略多餘空白與行尾強制換行用的兩個空白
        s = s.replace("\u00a0", " ")
        s = s.strip()
        s = re.sub(r"\s+", " ", s)
        return s

    # 先把每頁清洗成 lines（保留原始行，但比較用 norm）
    pages_lines: list[list[str]] = []
    for ft in fulltexts:
        cleaned = _strip_markdown_code_fences(ft).strip()
        if not cleaned:
            continue
        lines = [ln.rstrip() for ln in cleaned.splitlines()]
        # 去掉首尾空行
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if lines:
            pages_lines.append(lines)

    if not pages_lines:
        return ""

    # 以第一頁的前幾行作為「可能的頁首」候選（後續頁若重複則移除）
    first_hdr_norm = [norm_line(x) for x in pages_lines[0][:6] if norm_line(x)]
    merged_lines: list[str] = []

    for idx, lines in enumerate(pages_lines):
        work = list(lines)

        # 後續頁：移除重複頁首（和第一頁頁首候選一致）
        if idx > 0 and first_hdr_norm:
            while work and norm_line(work[0]) in first_hdr_norm:
                work.pop(0)
            while work and not work[0].strip():
                work.pop(0)

        # 移除與目前已合併內容「尾端」重疊的區段（避免跨頁重複）
        if merged_lines and work:
            tail_norm = [norm_line(x) for x in merged_lines[-12:] if norm_line(x)]
            head_norm = [norm_line(x) for x in work[:12] if norm_line(x)]
            max_k = min(len(tail_norm), len(head_norm))
            overlap = 0
            for k in range(max_k, 0, -1):
                if tail_norm[-k:] == head_norm[:k]:
                    overlap = k
                    break
            if overlap > 0:
                # 從 work 前面移除 overlap（用原始 work 行數近似）
                removed = 0
                new_work: list[str] = []
                for ln in work:
                    if removed < overlap and norm_line(ln):
                        removed += 1
                        continue
                    new_work.append(ln)
                work = new_work

        if not work:
            continue

        if merged_lines:
            merged_lines.append("")  # 用一個空行銜接頁與頁
        merged_lines.extend(work)

    merged = "\n".join(merged_lines).strip()
    merged = re.sub(r"\n{3,}", "\n\n", merged)
    return merged


def _emit_fulltext_block(
    out: list[str],
    *,
    fulltext_md: str,
    mode: FulltextMode,
    title: str,
) -> None:
    """
    將「全文（原始提取）」輸出到 Markdown。
    - section：直接顯示（可用於 RAG chunking）
    - details：折疊顯示（避免文件太長）
    """
    content0 = _strip_markdown_code_fences(fulltext_md)
    content = content0.strip() if content0.strip() else "- （無全文內容）"
    if mode == "details":
        out.append("<details>")
        out.append(f"<summary>{title}</summary>")
        out.append("")
        out.append(content)
        out.append("")
        out.append("</details>")
    else:
        out.append(f"### {title}")
        out.append("")
        out.append(content)


def extract_markdown_ocr(
    pdf_path: Path,
    max_pages: int,
    images_dir: Path,
    ocr_lang: str,
    dpi: int,
    vision_base_url: str | None = None,
    vision_model: str | None = None,
    vision_timeout_s: float = 90.0,
    include_source_text: bool = False,
    include_uncertainties: bool = False,
    verbose: bool = False,
    embed_images: bool = False,
    save_images: bool = True,
    include_frontmatter: bool = False,
    include_fulltext: bool = True,
    fulltext_mode: FulltextMode = "section",
) -> str:
    """
    針對掃描 PDF：
    - 若可 OCR：輸出 OCR 文字（仍會同時輸出每頁圖片以便追溯）
    - 若不可 OCR：僅輸出圖片 + 佔位提示
    """
    doc = fitz.open(pdf_path)
    pages = min(doc.page_count, max_pages)

    title = pdf_path.stem
    out: list[str] = []
    chosen_lang = _pick_ocr_lang(ocr_lang)

    can_ocr = _tesseract_available()
    if not can_ocr:
        # 仍輸出可用的「每頁圖片」以便後續人工/外部流程 OCR
        out.append(f"# {title}")
        out.append("")
        out.append(
            "> 注意：此 PDF 判定為掃描影像型，但目前環境未偵測到 `tesseract`，"
            "因此無法 OCR。已改為輸出每頁圖片供後續處理。"
        )
        out.append("")

    fulltexts: list[str] = []

    for pi in range(pages):
        if verbose and doc.page_count > 0:
            print(f"  - page {pi+1}/{doc.page_count}")
        page = doc.load_page(pi)
        pix = page.get_pixmap(dpi=dpi, alpha=False)

        if not can_ocr:
            # 不輸出頁面圖片
            continue

        # 真的 OCR：先 OCR 再做 Markdown 重建
        txt = _ocr_page_to_text(page, dpi=dpi, lang=chosen_lang)
        if pi == 0:
            # 用 OCR 內容抽前言 frontmatter & 標題
            fm = _extract_frontmatter(txt)
            title2 = fm.get("title") or title
            if include_frontmatter:
                out.append(_format_frontmatter(fm).rstrip("\n"))
            out.append(f"# {title2}")
            out.append("")
            # 不輸出 OCR banner（避免輸出雜訊）
            out.append("")

        # 不輸出頁面圖片
        # 輸出「結構化整理」：優先 vision（看得懂圖），否則退回純文字重寫
        structured: str = ""
        if vision_base_url and vision_model:
            try:
                data_url = _pixmap_to_data_url(pix)
                structured = _lmstudio_vision_explain(
                    base_url=vision_base_url,
                    model=vision_model,
                    data_url=data_url,
                    ocr_hint=txt,
                    timeout_s=vision_timeout_s,
                    include_uncertainties=include_uncertainties,
                )
            except Exception as e:
                structured = f"> 圖像解讀失敗：{type(e).__name__}: {e}"
        elif vision_base_url and vision_model is None:
            structured = "> 未提供 --vision-model，無法做圖像解讀。"
        elif vision_base_url is None and vision_model:
            structured = "> 未提供 --vision-base-url，無法做圖像解讀。"
        else:
            # 沒有 vision：仍用文字重寫出結構
            if vision_base_url and vision_model:
                structured = ""
            else:
                # 若使用者沒提供 vision 參數，仍可用同一台 LM Studio 走純文字重寫
                # 但沒有 base_url/model 就只能輸出最小資訊
                structured = ""

        if not structured:
            if txt and vision_base_url and vision_model:
                # 理論上會走到上面 vision，不應到這裡
                structured = _ocr_text_to_markdown(txt)
            elif txt:
                structured = _ocr_text_to_markdown(txt)
            else:
                structured = "- （OCR 無結果）"

        if include_fulltext:
            fulltext_md = ""
            if vision_base_url and vision_model:
                try:
                    fulltext_md = _lmstudio_vision_fulltext_layout(
                        base_url=vision_base_url,
                        model=vision_model,
                        data_url=_pixmap_to_data_url(pix),
                        ocr_hint=txt,
                        timeout_s=vision_timeout_s,
                    )
                except Exception:
                    fulltext_md = ""
            if not fulltext_md:
                fulltext_md = _ocr_text_to_markdown(txt) if txt else ""
            fulltexts.append(fulltext_md)

        out.append(structured)
        out.append("")

        if include_source_text and txt:
            out.append("<details>")
            out.append("<summary>來源文字（OCR 原文，僅供除錯）</summary>")
            out.append("")
            out.append("```")
            out.append(txt)
            out.append("```")
            out.append("")
            out.append("</details>")
            out.append("")

    doc.close()
    if include_fulltext and fulltexts:
        out.append("")
        out.append("## 全文")
        out.append("")
        merged = _merge_fulltext_pages(fulltexts)
        if fulltext_mode == "details":
            out.append("<details>")
            out.append("<summary>全文（視覺模型逐字轉寫＋排版；無法辨識會標記）</summary>")
            out.append("")
            out.append(merged if merged else "- （無全文內容）")
            out.append("")
            out.append("</details>")
            out.append("")
        else:
            out.append(merged if merged else "- （無全文內容）")
            out.append("")
    return "\n".join(out).strip() + "\n"


def extract_markdown_ocr_pdfium(
    pdf_path: Path,
    max_pages: int,
    images_dir: Path,
    ocr_lang: str,
    dpi: int,
    vision_base_url: str | None = None,
    vision_model: str | None = None,
    vision_timeout_s: float = 90.0,
    include_source_text: bool = False,
    include_uncertainties: bool = False,
    verbose: bool = False,
    embed_images: bool = False,
    save_images: bool = True,
    include_frontmatter: bool = False,
    include_fulltext: bool = True,
    fulltext_mode: FulltextMode = "section",
) -> str:
    """
    掃描/疑難 PDF 的安全路徑：用 pdfium 渲染每頁，再做 OCR + VLM 結構化。
    適用於特定 PDF 會讓 PyMuPDF segfault 的情況。
    """
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        total_pages = len(doc)
        pages = min(total_pages, max_pages)
        title = pdf_path.stem
        out: list[str] = []
        chosen_lang = _pick_ocr_lang(ocr_lang)
        can_ocr = _tesseract_available()

        if not can_ocr:
            out.append(f"# {title}")
            out.append("")
            out.append(
                "> 注意：此 PDF 判定為掃描影像型，但目前環境未偵測到 `tesseract`，"
                "因此無法 OCR。已改為輸出每頁圖片供後續處理。"
            )
            out.append("")

        fulltexts: list[str] = []

        for pi in range(pages):
            if verbose and total_pages:
                print(f"  - page {pi+1}/{total_pages} (pdfium)")

            page = doc.get_page(pi)
            try:
                pil_img = page.render(scale=dpi / 72).to_pil()
            finally:
                try:
                    page.close()
                except Exception:
                    pass

            if pil_img.mode not in ("RGB", "L"):
                pil_img = pil_img.convert("RGB")

            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            # 不落地存圖、不輸出圖片（僅供 VLM 內部使用）

            if pi == 0:
                txt0 = (
                    (pytesseract.image_to_string(pil_img, lang=chosen_lang).strip())
                    if (can_ocr and pytesseract)
                    else ""
                )
                fm = _extract_frontmatter(txt0)
                title2 = fm.get("title") or title
                if include_frontmatter:
                    out.append(_format_frontmatter(fm).rstrip("\n"))
                out.append(f"# {title2}")
                out.append("")
                # 不輸出 OCR banner（避免輸出雜訊）
                out.append("")

            # 不輸出頁面圖片

            txt = (
                (pytesseract.image_to_string(pil_img, lang=chosen_lang).strip())
                if (can_ocr and pytesseract)
                else ""
            )

            structured = ""
            if vision_base_url and vision_model:
                try:
                    structured = _lmstudio_vision_explain(
                        base_url=vision_base_url,
                        model=vision_model,
                        data_url=_png_bytes_to_data_url(png_bytes),
                        ocr_hint=txt,
                        timeout_s=vision_timeout_s,
                        include_uncertainties=include_uncertainties,
                    )
                except Exception as e:
                    structured = f"> 圖像解讀失敗：{type(e).__name__}: {e}"

            if not structured:
                structured = _ocr_text_to_markdown(txt) if txt else "- （OCR 無結果）"

            out.append(structured)
            out.append("")

            if include_fulltext:
                fulltext_md = ""
                if vision_base_url and vision_model:
                    try:
                        fulltext_md = _lmstudio_vision_fulltext_layout(
                            base_url=vision_base_url,
                            model=vision_model,
                            data_url=_png_bytes_to_data_url(png_bytes),
                            ocr_hint=txt,
                            timeout_s=vision_timeout_s,
                        )
                    except Exception:
                        fulltext_md = ""
                if not fulltext_md:
                    fulltext_md = _ocr_text_to_markdown(txt) if txt else ""
                fulltexts.append(fulltext_md)

            if include_source_text and txt:
                out.append("<details>")
                out.append("<summary>來源文字（OCR 原文，僅供除錯）</summary>")
                out.append("")
                out.append("```")
                out.append(txt)
                out.append("```")
                out.append("")
                out.append("</details>")
                out.append("")

        if include_fulltext and fulltexts:
            out.append("")
            out.append("## 全文")
            out.append("")
            merged = _merge_fulltext_pages(fulltexts)
            if fulltext_mode == "details":
                out.append("<details>")
                out.append("<summary>全文（視覺模型逐字轉寫＋排版；無法辨識會標記）</summary>")
                out.append("")
                out.append(merged if merged else "- （無全文內容）")
                out.append("")
                out.append("</details>")
                out.append("")
            else:
                out.append(merged if merged else "- （無全文內容）")
                out.append("")
        return "\n".join(out).strip() + "\n"
    finally:
        try:
            doc.close()
        except Exception:
            pass


def extract_markdown_hybrid(
    pdf_path: Path,
    max_pages: int,
    images_dir: Path,
    ocr_lang: str,
    ocr_dpi: int,
    min_text_chars_per_page: int = 80,
    vision_base_url: str | None = None,
    vision_model: str | None = None,
    vision_img_ratio: float = 0.22,
    vision_timeout_s: float = 90.0,
    include_source_text: bool = False,
    include_uncertainties: bool = False,
    vision_always: bool = False,
    verbose: bool = False,
    embed_images: bool = False,
    save_images: bool = True,
    include_frontmatter: bool = False,
    include_fulltext: bool = True,
    fulltext_mode: FulltextMode = "section",
) -> str:
    """
    更聰明的提取策略（推薦用於 mixed 或甚至 text）：
    - 每頁先嘗試文字抽取
    - 若該頁文字太少 → OCR（有 tesseract）或輸出圖片 + 提示
    """
    doc = fitz.open(pdf_path)
    title_guess = _pick_title_first_page(doc) or pdf_path.stem
    pages = min(doc.page_count, max_pages)

    can_ocr = _tesseract_available()
    chosen_lang = _pick_ocr_lang(ocr_lang) if can_ocr else "eng"

    # 用第一頁「可得到的文字」建立 frontmatter（若第一頁也沒文字且可 OCR，會改用 OCR）
    first_page = doc.load_page(0) if pages > 0 else None
    seed_text = ""
    if first_page is not None:
        seed_text = (first_page.get_text("text") or "").strip()
        if len(seed_text) < min_text_chars_per_page and can_ocr:
            seed_text = _ocr_page_to_text(first_page, dpi=ocr_dpi, lang=chosen_lang)

    fm = _extract_frontmatter(seed_text)
    title = fm.get("title") or title_guess

    out: list[str] = []
    if include_frontmatter:
        fm_block = _format_frontmatter(fm).strip()
        if fm_block:
            out.append(fm_block)
            out.append("")
    out.append(f"# {title}")
    out.append("")
    # 不輸出 OCR banner（避免輸出雜訊）
    out.append("")

    fulltexts: list[str] = []
    for pi in range(pages):
        if verbose and doc.page_count > 0:
            print(f"  - page {pi+1}/{doc.page_count}")
        page = doc.load_page(pi)
        page_text = (page.get_text("text") or "").strip()
        use_ocr = len(page_text) < min_text_chars_per_page

        # 頁面圖片占比，用於決定是否做 VLM 解讀
        img_ratio = _page_img_area_ratio(page)

        # vision 判斷：達門檻或強制 every page
        vision_enabled = bool(vision_base_url and vision_model)
        vision_for_this_page = vision_enabled and (vision_always or img_ratio >= vision_img_ratio)

        if use_ocr:
            pix = page.get_pixmap(dpi=ocr_dpi, alpha=False)
            # 不輸出頁面圖片（僅供 VLM 內部使用）

            txt = _ocr_page_to_text(page, dpi=ocr_dpi, lang=chosen_lang) if can_ocr else ""

            structured = ""
            if vision_for_this_page:
                try:
                    data_url = _pixmap_to_data_url(pix)
                    structured = _lmstudio_vision_explain(
                        base_url=vision_base_url,
                        model=vision_model,
                        data_url=data_url,
                        ocr_hint=txt,
                        timeout_s=vision_timeout_s,
                        include_uncertainties=include_uncertainties,
                    )
                except Exception as e:
                    structured = f"> 圖像解讀失敗：{type(e).__name__}: {e}"

            if not structured:
                if txt and vision_base_url and vision_model:
                    # 若圖片占比未達門檻，但仍希望有結構化內容：改走文字重寫（同模型）
                    structured = _lmstudio_text_rewrite(
                        base_url=vision_base_url,
                        model=vision_model,
                        text=txt,
                        timeout_s=vision_timeout_s,
                        include_uncertainties=include_uncertainties,
                    )
                elif txt:
                    structured = _ocr_text_to_markdown(txt)
                else:
                    structured = "- （此頁文字不足，且 OCR 無結果/不可用）"

            out.append(structured)
            out.append("")

            if include_fulltext:
                fulltext_md = ""
                if vision_base_url and vision_model:
                    try:
                        fulltext_md = _lmstudio_vision_fulltext_layout(
                            base_url=vision_base_url,
                            model=vision_model,
                            data_url=_pixmap_to_data_url(pix),
                            ocr_hint=txt,
                            timeout_s=vision_timeout_s,
                        )
                    except Exception:
                        fulltext_md = ""
                if not fulltext_md:
                    fulltext_md = _ocr_text_to_markdown(txt) if txt else ""
                fulltexts.append(fulltext_md)

            if include_source_text and txt:
                out.append("<details>")
                out.append("<summary>來源文字（OCR 原文，僅供除錯）</summary>")
                out.append("")
                out.append("```")
                out.append(txt)
                out.append("```")
                out.append("")
                out.append("</details>")
                out.append("")
            continue

        # 文字抽取頁：走原本字塊抽取
        lines = _extract_page_lines(page)
        extracted = "\n".join(
            [_line_to_md(t[3]) for t in lines if _line_to_md(t[3])]
        ).strip()

        # 預設不輸出原文（避免混淆），改用 LLM 結構化重寫
        structured = ""
        if vision_base_url and vision_model:
            try:
                # 若圖片占比高，用 vision；否則用 text rewrite
                if vision_for_this_page:
                    pix = page.get_pixmap(dpi=ocr_dpi, alpha=False)
                    # 不落地存圖、不輸出圖片（僅供 VLM 內部使用）
                    data_url = _pixmap_to_data_url(pix)
                    structured = _lmstudio_vision_explain(
                        base_url=vision_base_url,
                        model=vision_model,
                        data_url=data_url,
                        ocr_hint=page_text or extracted,
                        timeout_s=vision_timeout_s,
                        include_uncertainties=include_uncertainties,
                    )
                else:
                    structured = _lmstudio_text_rewrite(
                        base_url=vision_base_url,
                        model=vision_model,
                        text=page_text or extracted,
                        timeout_s=vision_timeout_s,
                        include_uncertainties=include_uncertainties,
                    )
            except Exception as e:
                structured = f"> 結構化整理失敗：{type(e).__name__}: {e}"
        else:
            # 沒有 LLM 端點時，只能退回抽取文字（仍盡量整理成 md）
            structured = extracted or page_text or "- （此頁無可抽取文字）"

        out.append(structured)
        out.append("")

        if include_fulltext:
            src = page_text or extracted
            fulltext_md = ""
            # 優先用視覺模型逐字轉寫（對版面/欄位/表格會更接近原稿）
            if vision_base_url and vision_model:
                try:
                    # 文字頁不一定有 pix，但可直接渲染頁面再交給 VLM
                    pix2 = page.get_pixmap(dpi=ocr_dpi, alpha=False)
                    fulltext_md = _lmstudio_vision_fulltext_layout(
                        base_url=vision_base_url,
                        model=vision_model,
                        data_url=_pixmap_to_data_url(pix2),
                        ocr_hint=src,
                        timeout_s=vision_timeout_s,
                    )
                except Exception:
                    fulltext_md = ""
            if not fulltext_md:
                fulltext_md = _ocr_text_to_markdown(src) if src else ""
            fulltexts.append(fulltext_md)

        if include_source_text and (page_text or extracted):
            out.append("<details>")
            out.append("<summary>來源文字（抽取原文，僅供除錯）</summary>")
            out.append("")
            out.append("```")
            out.append((page_text or extracted).strip())
            out.append("```")
            out.append("")
            out.append("</details>")
            out.append("")

    doc.close()
    if include_fulltext and fulltexts:
        out.append("")
        out.append("## 全文")
        out.append("")
        merged = _merge_fulltext_pages(fulltexts)
        if fulltext_mode == "details":
            out.append("<details>")
            out.append("<summary>全文（視覺模型逐字轉寫＋排版；無法辨識會標記）</summary>")
            out.append("")
            out.append(merged if merged else "- （無全文內容）")
            out.append("")
            out.append("</details>")
            out.append("")
        else:
            out.append(merged if merged else "- （無全文內容）")
            out.append("")
    return "\n".join(out).strip() + "\n"


def extract_markdown_vision_only(
    pdf_path: Path,
    max_pages: int,
    images_dir: Path,
    ocr_lang: str,
    ocr_dpi: int,
    *,
    vision_base_url: str,
    vision_model: str,
    vision_timeout_s: float = 90.0,
    include_source_text: bool = False,
    include_uncertainties: bool = False,
    verbose: bool = False,
    embed_images: bool = False,
    save_images: bool = True,
    include_frontmatter: bool = False,
    include_fulltext: bool = True,
    fulltext_mode: FulltextMode = "section",
) -> str:
    """
    統一採用 vision model 的抽取方式（不再做 text/slides/scanned 分流判斷）：
    - 每頁渲染成圖片
    - 用 VLM 產出結構化 Markdown
    - （可選）輸出全文：用 VLM 逐字轉寫+排版後，再跨頁合併為同一篇文章
    - tesseract OCR 僅作為 vision 的提示（如果可用）
    """
    doc = fitz.open(pdf_path)
    pages = min(doc.page_count, max_pages)
    can_ocr = _tesseract_available()
    chosen_lang = _pick_ocr_lang(ocr_lang) if can_ocr else "eng"

    out: list[str] = [f"# {pdf_path.stem}", ""]
    # 不輸出 OCR banner（避免輸出雜訊）
    out.append("")

    fulltexts: list[str] = []
    for pi in range(pages):
        if verbose and doc.page_count > 0:
            print(f"  - page {pi+1}/{doc.page_count} (vision-only)")
        page = doc.load_page(pi)
        pix = page.get_pixmap(dpi=ocr_dpi, alpha=False)
        # 不輸出頁面圖片（僅供 VLM 內部使用）

        ocr_hint = ""
        if can_ocr:
            try:
                ocr_hint = _ocr_page_to_text(page, dpi=ocr_dpi, lang=chosen_lang)
            except Exception:
                ocr_hint = ""

        structured = ""
        try:
            structured = _lmstudio_vision_explain(
                base_url=vision_base_url,
                model=vision_model,
                data_url=_pixmap_to_data_url(pix),
                ocr_hint=ocr_hint,
                timeout_s=vision_timeout_s,
                include_uncertainties=include_uncertainties,
            )
        except Exception as e:
            structured = f"> 結構化整理失敗：{type(e).__name__}: {e}"

        if structured:
            out.append(structured)
            out.append("")

        if include_fulltext:
            ft = ""
            try:
                ft = _lmstudio_vision_fulltext_layout(
                    base_url=vision_base_url,
                    model=vision_model,
                    data_url=_pixmap_to_data_url(pix),
                    ocr_hint=ocr_hint,
                    timeout_s=vision_timeout_s,
                )
            except Exception:
                ft = ""
            if not ft:
                ft = _ocr_text_to_markdown(ocr_hint) if ocr_hint else ""
            fulltexts.append(ft)

        if include_source_text and ocr_hint:
            out.append("<details>")
            out.append("<summary>來源文字（OCR 原文，僅供除錯）</summary>")
            out.append("")
            out.append("```")
            out.append(ocr_hint)
            out.append("```")
            out.append("")
            out.append("</details>")
            out.append("")

    doc.close()

    if include_fulltext:
        out.append("## 全文")
        out.append("")
        merged = _merge_fulltext_pages(fulltexts)
        if fulltext_mode == "details":
            out.append("<details>")
            out.append("<summary>全文（視覺模型逐字轉寫＋排版；無法辨識會標記）</summary>")
            out.append("")
            out.append(merged if merged else "- （無全文內容）")
            out.append("")
            out.append("</details>")
            out.append("")
        else:
            out.append(merged if merged else "- （無全文內容）")
            out.append("")

    return "\n".join(out).strip() + "\n"


def extract_markdown_vision_only_pdfium(
    pdf_path: Path,
    max_pages: int,
    images_dir: Path,
    ocr_lang: str,
    ocr_dpi: int,
    *,
    vision_base_url: str,
    vision_model: str,
    vision_timeout_s: float = 90.0,
    include_source_text: bool = False,
    include_uncertainties: bool = False,
    verbose: bool = False,
    embed_images: bool = False,
    save_images: bool = True,
    include_frontmatter: bool = False,
    include_fulltext: bool = True,
    fulltext_mode: FulltextMode = "section",
) -> str:
    """
    vision-only 的 pdfium 安全路徑：避免少數 PDF 讓 fitz segfault。
    """
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        total_pages = len(doc)
        pages = min(total_pages, max_pages)
        can_ocr = _tesseract_available()
        chosen_lang = _pick_ocr_lang(ocr_lang) if can_ocr else "eng"

        out: list[str] = [f"# {pdf_path.stem}", ""]
        # 不輸出 OCR banner（避免輸出雜訊）
        out.append("")

        fulltexts: list[str] = []
        for pi in range(pages):
            if verbose and total_pages:
                print(f"  - page {pi+1}/{total_pages} (vision-only pdfium)")

            page = doc.get_page(pi)
            try:
                pil_img = page.render(scale=ocr_dpi / 72).to_pil()
            finally:
                try:
                    page.close()
                except Exception:
                    pass

            if pil_img.mode not in ("RGB", "L"):
                pil_img = pil_img.convert("RGB")

            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            # 不輸出頁面圖片（僅供 VLM 內部使用）

            ocr_hint = ""
            if can_ocr and pytesseract:
                try:
                    ocr_hint = (pytesseract.image_to_string(pil_img, lang=chosen_lang).strip())
                except Exception:
                    ocr_hint = ""

            structured = ""
            try:
                structured = _lmstudio_vision_explain(
                    base_url=vision_base_url,
                    model=vision_model,
                    data_url=_png_bytes_to_data_url(png_bytes),
                    ocr_hint=ocr_hint,
                    timeout_s=vision_timeout_s,
                    include_uncertainties=include_uncertainties,
                )
            except Exception as e:
                structured = f"> 結構化整理失敗：{type(e).__name__}: {e}"

            if structured:
                out.append(structured)
                out.append("")

            if include_fulltext:
                ft = ""
                try:
                    ft = _lmstudio_vision_fulltext_layout(
                        base_url=vision_base_url,
                        model=vision_model,
                        data_url=_png_bytes_to_data_url(png_bytes),
                        ocr_hint=ocr_hint,
                        timeout_s=vision_timeout_s,
                    )
                except Exception:
                    ft = ""
                if not ft:
                    ft = _ocr_text_to_markdown(ocr_hint) if ocr_hint else ""
                fulltexts.append(ft)

            if include_source_text and ocr_hint:
                out.append("<details>")
                out.append("<summary>來源文字（OCR 原文，僅供除錯）</summary>")
                out.append("")
                out.append("```")
                out.append(ocr_hint)
                out.append("```")
                out.append("")
                out.append("</details>")
                out.append("")

        if include_fulltext:
            out.append("## 全文")
            out.append("")
            merged = _merge_fulltext_pages(fulltexts)
            if fulltext_mode == "details":
                out.append("<details>")
                out.append("<summary>全文（視覺模型逐字轉寫＋排版；無法辨識會標記）</summary>")
                out.append("")
                out.append(merged if merged else "- （無全文內容）")
                out.append("")
                out.append("</details>")
                out.append("")
            else:
                out.append(merged if merged else "- （無全文內容）")
                out.append("")

        return "\n".join(out).strip() + "\n"
    finally:
        try:
            doc.close()
        except Exception:
            pass


def _extract_first_heading_title(md: str) -> str:
    for line in md.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _extract_section_text(md: str, heading: str) -> str:
    """
    取出 '## {heading}' 到下一個 '## ' 之間的內容（不含標題行）。
    """
    lines = md.splitlines()
    out: list[str] = []
    in_sec = False
    for line in lines:
        if line.startswith("## "):
            cur = line[3:].strip()
            if in_sec:
                break
            in_sec = cur == heading
            continue
        if in_sec:
            out.append(line)
    return "\n".join(out).strip()


def _split_per_page_structured(md: str) -> list[str]:
    """
    從第一階段輸出中切出「逐頁結構化」片段：
    - 去掉全文區塊（## 全文 之後）
    - 優先以圖片標記（若存在）或 '## 主旨' 標題作為分隔點
    """
    if not md.strip():
        return []
    # 去掉全文區塊（全文用另一條軌道，不拿來做統整的依據）
    md2 = md
    idx = md2.find("\n## 全文\n")
    if idx >= 0:
        md2 = md2[:idx].strip()

    # 1) 若仍有圖片標記（舊輸出），用它來切段
    if "![page](" in md2:
        parts = re.split(r"!\[page\]\([^)]+\)\s*", md2)
        chunks: list[str] = []
        for p in parts:
            s = p.strip()
            if not s:
                continue
            chunks.append(s)
        return chunks

    # 2) 新輸出已移除圖片：改以 '## 主旨' 作為每段起點（不提頁碼）
    starts = [m.start() for m in re.finditer(r"(?m)^## 主旨\s*$", md2)]
    if len(starts) <= 1:
        return [md2.strip()] if md2.strip() else []

    chunks2: list[str] = []
    for i, st in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(md2)
        seg = md2[st:end].strip()
        if seg:
            chunks2.append(seg)
    return chunks2


def _join_per_page_structured_for_llm(chunks: list[str]) -> str:
    """
    將逐頁片段串成 LLM 輸入，避免提及頁碼但保留邊界。
    """
    out: list[str] = []
    for i, c in enumerate(chunks, start=1):
        out.append(f"--- 片段 {i} ---")
        out.append(c.strip())
        out.append("")
    return "\n".join(out).strip()


def _extract_bullets(section_text: str, limit: int = 8) -> list[str]:
    items: list[str] = []
    for line in section_text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            items.append(line[2:].strip())
        if len(items) >= limit:
            break
    return items


def _find_version_info(md: str) -> dict[str, str]:
    """
    從（VLM 結構化）內容中盡量抽出版本資訊。
    只做保守抽取：找到才填，找不到就空。
    """
    text = md
    # code
    code = ""
    m = re.search(r"(?:文件編號|編\s*號)\s*[:：]?\s*([A-Z]{1,4}\s*[^\n]{0,30})", text)
    if m:
        code = m.group(1).strip()
    # created_at
    created = ""
    m = re.search(r"(?:制定日期|製訂日期)\s*[:：]?\s*(\d{4})[./\-]\s*(\d{1,2})", text)
    if m:
        created = f"{m.group(1)}-{m.group(2).zfill(2)}"
    # English: Version Number & Date: Jul. 31, 2017
    if not created:
        m = re.search(
            r"Version\s*Number\s*&\s*Date\s*[:：]?\s*([A-Za-z]{3,9})\.?\s*(\d{1,2})\s*,\s*(\d{4})",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            mon = m.group(1).lower()[:3]
            mon_map = {
                "jan": "01",
                "feb": "02",
                "mar": "03",
                "apr": "04",
                "may": "05",
                "jun": "06",
                "jul": "07",
                "aug": "08",
                "sep": "09",
                "oct": "10",
                "nov": "11",
                "dec": "12",
            }
            mm = mon_map.get(mon, "")
            if mm:
                created = f"{m.group(3)}-{mm}"
    # revised_at
    revised = ""
    m = re.search(r"(?:修訂日期|[一二三四五六七八九十]+修日期)\s*[:：]?\s*(\d{4})[./\-]\s*(\d{1,2})", text)
    if m:
        revised = f"{m.group(1)}-{m.group(2).zfill(2)}"
    # revision note (e.g. 七修)
    rev_note = ""
    m = re.search(r"([一二三四五六七八九十]+)修", text)
    if m:
        rev_note = f"{m.group(1)}修"
    return {"code": code, "created_at": created, "revised_at": revised, "revision_note": rev_note}


def _guess_category_from_title(title: str) -> str:
    # 僅在標題/內容出現明確關鍵字才填，避免亂猜
    if "檢查前" in title or ("檢查" in title and "前" in title):
        return "pre_exam"
    if "檢查後" in title or ("檢查" in title and "後" in title):
        return "post_exam"
    if "術前" in title:
        return "pre_op"
    if "術後" in title:
        return "post_op"
    if "用藥" in title or "藥物" in title:
        return "medication"
    if "飲食" in title or "營養" in title:
        return "self_care"
    if "返家" in title or "出院" in title:
        return "discharge"
    # 不確定就留空（不要虛構）
    return ""


def _render_handout_frontmatter(
    *,
    stable_id: str,
    title: str,
    summary: str,
    org_name_zh: str,
    tags: list[str],
    category: str,
    code: str,
    created_at: str,
    revisions: list[dict[str, str]],
    department_zh: str,
    conditions: list[str],
    key_actions: list[str],
    red_flags: list[str],
    followup_dept: str,
    followup_note: str,
    aliases: list[str],
    keywords: list[str],
) -> str:
    today = date.today().strftime("%Y-%m-%d")
    lines: list[str] = ["---"]
    lines.append("type: education.handout")
    lines.append(f"id: edu-{stable_id}")
    lines.append("title:")
    lines.append(f"  zh-Hant: {title}" if title else "  zh-Hant: ")
    lines.append("summary:")
    lines.append(f"  zh-Hant: {summary}" if summary else "  zh-Hant: ")
    lines.append("org:")
    lines.append(f"  name_zh: {org_name_zh}" if org_name_zh else "  name_zh: ")
    lines.append("lang: zh-Hant")
    lines.append("audience: [patient, family]")
    safe_tags = ["衛教"] + [t for t in tags if t and t != "衛教"]
    lines.append(f"tags: [{', '.join(safe_tags)}]")
    lines.append("")
    lines.append("education:")
    lines.append(f"  category: {category}" if category else "  category: ")
    lines.append(f"  code: {code}" if code else "  code: ")
    lines.append(f"  created_at: {created_at}" if created_at else "  created_at: ")
    if revisions:
        lines.append("  revisions:")
        for r in revisions:
            lines.append(f"    - date: {r.get('date','')}".rstrip())
            note = r.get("note", "")
            lines.append(f"      note: {note}" if note else "      note: ")
    lines.append("  owner:")
    lines.append(f"    department_zh: {department_zh}" if department_zh else "    department_zh: ")
    if conditions:
        lines.append("  conditions:")
        for c in conditions:
            lines.append(f"    - {c}")
    if key_actions:
        lines.append("  key_actions:")
        for a in key_actions:
            lines.append(f"    - {a}")
    if red_flags:
        lines.append("  red_flags:")
        for rf in red_flags:
            lines.append(f"    - {rf}")
    lines.append("  followup:")
    lines.append(f"    department_zh: {followup_dept}" if followup_dept else "    department_zh: ")
    lines.append(f"    note: {followup_note}" if followup_note else "    note: ")
    lines.append("")
    lines.append("retrieval:")
    if aliases:
        lines.append("  aliases:")
        for a in aliases:
            lines.append(f"    - {a}")
    else:
        lines.append("  aliases: []")
    if keywords:
        lines.append("  keywords:")
        for k in keywords:
            lines.append(f"    - {k}")
    else:
        lines.append("  keywords: []")
    lines.append("")
    lines.append("source:")
    lines.append("  url: 內部衛教 PDF")
    lines.append(f"  captured_at: {today}")
    lines.append(f"updated_at: {today}")
    lines.append(f"last_reviewed: {today}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def convert_to_education_handout_markdown(
    *,
    pdf_stem: str,
    extracted_md: str,
    org_name_zh_default: str = "",
    consolidate_base_url: str = "",
    consolidate_model: str = "",
    consolidate_timeout_s: float = 90.0,
) -> str:
    """
    第二階段：在 VLM 已產出的結構化內容之後，套用 education.handout 範本 frontmatter + 章節。
    原則：只用 extracted_md 內可找到的資訊，不做虛構；缺資料留空。
    """
    title = _extract_first_heading_title(extracted_md) or pdf_stem

    # summary：優先取第 1 段的「主旨」（避免逐頁語氣）
    summary = ""
    chunks0 = _split_per_page_structured(extracted_md)
    first_page = chunks0[0] if chunks0 else extracted_md
    summary = _extract_section_text(first_page, "主旨").strip()
    if not summary:
        # 相容舊輸出：以前用「本頁主旨」
        summary = _extract_section_text(first_page, "本頁主旨").strip()

    # 重點/紅旗：從第 1 頁重點整理抓
    key_actions = _extract_bullets(_extract_section_text(first_page, "重點整理"), limit=6)
    red_flags: list[str] = []
    for line in extracted_md.splitlines():
        s = line.strip()
        if not s:
            continue
        if "就醫" in s or "立即" in s or "急診" in s:
            if s.startswith("- "):
                red_flags.append(s[2:].strip())
            elif s.startswith("1.") or s.startswith("2.") or s.startswith("3."):
                red_flags.append(s)
    red_flags = red_flags[:8]

    # 版本資訊：從正文（尤其是『版本資訊』段）抽
    ver = _find_version_info(extracted_md)
    revisions: list[dict[str, str]] = []
    if ver.get("revised_at") or ver.get("revision_note"):
        revisions.append({"date": ver.get("revised_at", ""), "note": ver.get("revision_note", "")})

    # org name：只從內容內找得到才填，找不到就留空
    org_name = org_name_zh_default
    if not org_name:
        if "屏東基督教醫院" in extracted_md:
            org_name = "屏東基督教醫院"

    # tags：只放不會虛構的
    tags: list[str] = []

    category = _guess_category_from_title(title)
    code = ver.get("code", "")
    created_at = ver.get("created_at", "")

    # 適用對象：保守從正文找「適用對象/適用狀況」字樣（找到才填）
    audience_lines: list[str] = []
    m = re.search(r"(?:適用對象|適用狀況)[^\n]*[：:─\-]\s*([^\n]{4,80})", extracted_md)
    if m:
        audience_lines.append(m.group(1).strip())

    # conditions：保守起見，直接用標題當主題（不新增不存在的病名）
    conditions = [title] if title else []

    # followup：沒有明確內容就留空
    followup_dept = ""
    followup_note = ""

    aliases = [title] if title else []
    # keywords：從標題抓幾個中文字 token（不發明新詞）
    kws = [w for w in re.findall(r"[\u4e00-\u9fff]{2,}", title)][:6]

    front = _render_handout_frontmatter(
        stable_id=pdf_stem,
        title=title,
        summary=summary,
        org_name_zh=org_name,
        tags=tags,
        category=category,
        code=code,
        created_at=created_at,
        revisions=revisions,
        department_zh="",
        conditions=conditions,
        key_actions=key_actions,
        red_flags=red_flags,
        followup_dept=followup_dept,
        followup_note=followup_note,
        aliases=aliases,
        keywords=kws,
    )

    # body：最終只輸出「單一統整結構」+ 文末全文（避免逐頁重複段落）
    body_lines: list[str] = []
    body_lines.append(f"# {title}")
    body_lines.append("")

    consolidated = ""
    if consolidate_base_url and consolidate_model:
        try:
            chunks = _split_per_page_structured(extracted_md)
            per_page_text = _join_per_page_structured_for_llm(chunks)
            consolidated = _lmstudio_document_consolidate(
                base_url=consolidate_base_url,
                model=consolidate_model,
                title=title,
                per_page_structured=per_page_text,
                timeout_s=consolidate_timeout_s,
            )
        except Exception as e:
            consolidated = f"> 統整失敗：{type(e).__name__}: {e}"

    # 統整內容（只應有一組：主旨/重點整理/流程/注意事項/表格/版本資訊）
    body_lines.append(consolidated.strip() if consolidated.strip() else "- （無統整內容）")
    body_lines.append("")

    # 文末全文
    body_lines.append("## 全文")
    body_lines.append("")
    fulltext = _extract_fulltext_from_extracted_md(extracted_md)
    body_lines.append(fulltext.strip() if fulltext.strip() else "- （無全文內容）")
    body_lines.append("")

    return front + "\n".join(body_lines).strip() + "\n"

def process_one(
    pdf_path: Path,
    output_dir: Path,
    max_pages: int,
    mode: Literal["auto", "text", "slides", "ocr"],
    sample_pages: int,
    ocr_lang: str,
    ocr_dpi: int,
    vision_base_url: str | None,
    vision_model: str | None,
    vision_img_ratio: float,
    vision_timeout_s: float,
    include_source_text: bool,
    include_uncertainties: bool,
    vision_always: bool,
    overwrite: bool,
    verbose: bool,
    embed_images: bool,
    save_images: bool,
    include_frontmatter: bool,
    output_format: str,
    pdf_engine: PdfEngine,
    include_fulltext: bool,
    fulltext_mode: FulltextMode,
) -> tuple[PdfKind, Path, bool]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_md = output_dir / f"{pdf_path.stem}.md"
    if out_md.exists() and not overwrite:
        # 已存在且不覆蓋：視為未處理（不要搬檔）
        return "text", out_md, False

    # 若指定 pdfium：完全避開 PyMuPDF 的解析/分類，統一當作掃描圖像走 OCR+VLM（較穩）
    if pdf_engine == "pdfium":
        kind: PdfKind = "scanned"
    elif mode == "auto":
        feats = compute_features(pdf_path, sample_pages=sample_pages)
        kind = detect_kind(feats)
    elif mode == "text":
        kind = "text"
    elif mode == "slides":
        kind = "slides"
    else:
        kind = "scanned"

    if verbose:
        print(f"[{kind}] {pdf_path.name}")

    # vision-only：使用者提供了 vision 端點後，統一用視覺模型解讀提取，不再做 text/slides/scanned 分流。
    # 仍保留 pdf_engine=pdfium 的穩定路徑（避免少數 PDF 讓 fitz segfault）。
    if vision_base_url and vision_model:
        images_dir = output_dir / f"{pdf_path.stem}_images"
        if pdf_engine == "pdfium":
            md = extract_markdown_vision_only_pdfium(
                pdf_path,
                max_pages=max_pages,
                images_dir=images_dir,
                ocr_lang=ocr_lang,
                ocr_dpi=ocr_dpi,
                vision_base_url=vision_base_url,
                vision_model=vision_model,
                vision_timeout_s=vision_timeout_s,
                include_source_text=include_source_text,
                include_uncertainties=include_uncertainties,
                verbose=verbose,
                embed_images=embed_images,
                save_images=save_images,
                include_frontmatter=include_frontmatter,
                include_fulltext=include_fulltext,
                fulltext_mode=fulltext_mode,
            )
        else:
            md = extract_markdown_vision_only(
                pdf_path,
                max_pages=max_pages,
                images_dir=images_dir,
                ocr_lang=ocr_lang,
                ocr_dpi=ocr_dpi,
                vision_base_url=vision_base_url,
                vision_model=vision_model,
                vision_timeout_s=vision_timeout_s,
                include_source_text=include_source_text,
                include_uncertainties=include_uncertainties,
                verbose=verbose,
                embed_images=embed_images,
                save_images=save_images,
                include_frontmatter=include_frontmatter,
                include_fulltext=include_fulltext,
                fulltext_mode=fulltext_mode,
            )
        final_md = md
        if output_format == "handout":
            final_md = convert_to_education_handout_markdown(
                pdf_stem=pdf_path.stem,
                extracted_md=md,
                consolidate_base_url=vision_base_url or "",
                consolidate_model=vision_model or "",
                consolidate_timeout_s=vision_timeout_s,
            )
        out_md.write_text(final_md, encoding="utf-8")
        return kind, out_md, True

    if kind == "slides":
        # slides 類文件（投影片/圖卡/決策輔助）若提供了 vision model，
        # 用 hybrid 讓 VLM 做更好的語意結構化，避免 handout 欄位抓不到而全空。
        if vision_base_url and vision_model:
            images_dir = output_dir / f"{pdf_path.stem}_images"
            md = extract_markdown_hybrid(
                pdf_path,
                max_pages=max_pages,
                images_dir=images_dir,
                ocr_lang=ocr_lang,
                ocr_dpi=ocr_dpi,
                vision_base_url=vision_base_url,
                vision_model=vision_model,
                vision_img_ratio=vision_img_ratio,
                vision_timeout_s=vision_timeout_s,
                include_source_text=include_source_text,
                include_uncertainties=include_uncertainties,
                # slides 預設強制每頁啟用 vision（這類文件圖表多，單靠抽字很碎）
                vision_always=True,
                verbose=verbose,
                embed_images=embed_images,
                save_images=save_images,
                include_frontmatter=include_frontmatter,
                include_fulltext=include_fulltext,
                fulltext_mode=fulltext_mode,
            )
        else:
            md = extract_markdown_slides(pdf_path, max_pages=max_pages)
    elif kind == "scanned":
        # 提示是否被 max_pages 截斷（避免誤以為漏頁）
        try:
            total_pages = fitz.open(pdf_path).page_count
        except Exception:
            total_pages = 0
        if verbose and total_pages and total_pages > max_pages:
            print(f"  ! 注意：此檔共有 {total_pages} 頁，但 --max-pages={max_pages}，只會處理前 {max_pages} 頁")
        images_dir = output_dir / f"{pdf_path.stem}_images"
        if pdf_engine == "pdfium":
            md = extract_markdown_ocr_pdfium(
                pdf_path,
                max_pages=max_pages,
                images_dir=images_dir,
                ocr_lang=ocr_lang,
                dpi=ocr_dpi,
                vision_base_url=vision_base_url,
                vision_model=vision_model,
                vision_timeout_s=vision_timeout_s,
                include_source_text=include_source_text,
                include_uncertainties=include_uncertainties,
                verbose=verbose,
                embed_images=embed_images,
                save_images=save_images,
                include_frontmatter=include_frontmatter,
                include_fulltext=include_fulltext,
                fulltext_mode=fulltext_mode,
            )
        else:
            md = extract_markdown_ocr(
                pdf_path,
                max_pages=max_pages,
                images_dir=images_dir,
                ocr_lang=ocr_lang,
                dpi=ocr_dpi,
                vision_base_url=vision_base_url,
                vision_model=vision_model,
                vision_timeout_s=vision_timeout_s,
                include_source_text=include_source_text,
                include_uncertainties=include_uncertainties,
                verbose=verbose,
                embed_images=embed_images,
                save_images=save_images,
                include_frontmatter=include_frontmatter,
                include_fulltext=include_fulltext,
                fulltext_mode=fulltext_mode,
            )
    else:
        try:
            total_pages = fitz.open(pdf_path).page_count
        except Exception:
            total_pages = 0
        if verbose and total_pages and total_pages > max_pages:
            print(f"  ! 注意：此檔共有 {total_pages} 頁，但 --max-pages={max_pages}，只會處理前 {max_pages} 頁")
        # 對 text/mixed 預設採用 hybrid：逐頁判斷文字量，不足就 OCR/輸出圖片
        images_dir = output_dir / f"{pdf_path.stem}_images"
        if pdf_engine == "pdfium":
            md = extract_markdown_ocr_pdfium(
                pdf_path,
                max_pages=max_pages,
                images_dir=images_dir,
                ocr_lang=ocr_lang,
                dpi=ocr_dpi,
                vision_base_url=vision_base_url,
                vision_model=vision_model,
                vision_timeout_s=vision_timeout_s,
                include_source_text=include_source_text,
                include_uncertainties=include_uncertainties,
                verbose=verbose,
                embed_images=embed_images,
                save_images=save_images,
                include_frontmatter=include_frontmatter,
                include_fulltext=include_fulltext,
                fulltext_mode=fulltext_mode,
            )
        else:
            md = extract_markdown_hybrid(
                pdf_path,
                max_pages=max_pages,
                images_dir=images_dir,
                ocr_lang=ocr_lang,
                ocr_dpi=ocr_dpi,
                vision_base_url=vision_base_url,
                vision_model=vision_model,
                vision_img_ratio=vision_img_ratio,
                vision_timeout_s=vision_timeout_s,
                include_source_text=include_source_text,
                include_uncertainties=include_uncertainties,
                vision_always=vision_always,
                verbose=verbose,
                embed_images=embed_images,
                save_images=save_images,
                include_frontmatter=include_frontmatter,
                include_fulltext=include_fulltext,
                fulltext_mode=fulltext_mode,
            )

    final_md = md
    if output_format == "handout":
        final_md = convert_to_education_handout_markdown(
            pdf_stem=pdf_path.stem,
            extracted_md=md,
            consolidate_base_url=vision_base_url or "",
            consolidate_model=vision_model or "",
            consolidate_timeout_s=vision_timeout_s,
        )

    out_md.write_text(final_md, encoding="utf-8")
    return kind, out_md, True


def _move_to_done(src_pdf: Path, done_dir: Path) -> Path:
    done_dir.mkdir(parents=True, exist_ok=True)
    dest = done_dir / src_pdf.name
    if not dest.exists():
        shutil.move(str(src_pdf), str(dest))
        return dest
    # 檔名衝突：自動加尾碼
    stem = src_pdf.stem
    suffix = src_pdf.suffix
    for i in range(1, 10_000):
        cand = done_dir / f"{stem}_{i}{suffix}"
        if not cand.exists():
            shutil.move(str(src_pdf), str(cand))
            return cand
    raise RuntimeError(f"done 目錄檔名衝突過多：{src_pdf.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF 結構化抽取 → Markdown（自動辨識）")
    parser.add_argument("--input", required=True, help="PDF 檔案或資料夾路徑")
    parser.add_argument(
        "--output",
        default="",
        help="輸出資料夾（若不填，資料夾模式預設輸出到 rag_test_data/docs/衛教單_md）",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "text", "slides", "ocr"],
        help="抽取模式：auto=自動辨識；ocr=強制走掃描/OCR 路線",
    )
    parser.add_argument("--max-pages", type=int, default=80, help="每個 PDF 最多處理頁數")
    parser.add_argument("--sample-pages", type=int, default=12, help="自動辨識用的抽樣頁數")
    parser.add_argument(
        "--ocr-lang",
        default="auto",
        help='tesseract OCR language；建議用 auto（會自動挑選 chi_tra/chi_sim/eng）',
    )
    parser.add_argument("--ocr-dpi", type=int, default=260, help="OCR 渲染 DPI（越高越慢）")
    parser.add_argument(
        "--vision-base-url",
        default="",
        help="LM Studio (OpenAI 相容) base URL，例如 http://192.168.50.152:1234",
    )
    parser.add_argument(
        "--vision-model",
        default="",
        help="多模態模型 id，例如 qwen/qwen3-vl-8b",
    )
    parser.add_argument(
        "--vision-img-ratio",
        type=float,
        default=0.22,
        help="頁面圖片面積占比達到此值才啟用圖像解讀（0~1）",
    )
    parser.add_argument(
        "--vision-timeout",
        type=float,
        default=90.0,
        help="圖像解讀 API timeout (秒)",
    )
    parser.add_argument(
        "--vision-always",
        action="store_true",
        help="強制每頁都做圖像結構化（即使圖片占比不高；較慢/較貴）",
    )
    parser.add_argument(
        "--include-source-text",
        action="store_true",
        help="在輸出中附上來源文字（OCR/抽取原文）的 details 區塊，僅供除錯（預設關閉）",
    )
    parser.add_argument(
        "--include-uncertainties",
        action="store_true",
        help="在輸出中加入『需要確認的資訊』段落（預設關閉；建議讓模型直接校正寫回正文）",
    )
    parser.add_argument(
        "--move-done",
        action="store_true",
        help="處理成功後，將原始 PDF 移動到 done 目錄（只搬『本次真的有處理』的檔案）",
    )
    parser.add_argument(
        "--done-dir",
        default="",
        help="done 目錄（預設為 input 資料夾下的 done/；若 input 是單檔，預設為該檔所在資料夾的 done/）",
    )
    parser.add_argument(
        "--embed-images",
        action="store_true",
        help="將 Markdown 圖片連結改為 base64 data URI（檔案會變很大）",
    )
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="不落地存 page-xxx.png（搭配 --embed-images 使用可避免產生大量圖片檔）",
    )
    parser.add_argument(
        "--include-frontmatter",
        action="store_true",
        help="輸出 YAML frontmatter（預設不輸出，以避免資訊混淆）",
    )
    parser.add_argument(
        "--no-fulltext",
        action="store_true",
        help="不輸出『全文（原始提取）』（預設會輸出全文以避免 LLM 曲解）",
    )
    parser.add_argument(
        "--fulltext-mode",
        default="section",
        choices=["section", "details"],
        help="全文呈現方式：section=直接顯示；details=折疊顯示（避免文件太長）",
    )
    parser.add_argument(
        "--output-format",
        default="raw",
        choices=["raw", "handout"],
        help="輸出格式：raw=只輸出提取後的 Markdown；handout=套用 education.handout 範本（第二階段轉換，避免虛構）",
    )
    parser.add_argument(
        "--isolate-per-pdf",
        action="store_true",
        help="批次時每個 PDF 以子行程獨立執行（避免 PyMuPDF 長時間處理導致 segfault；推薦開啟）",
    )
    parser.add_argument(
        "--pdf-engine",
        default="fitz",
        choices=["fitz", "pdfium"],
        help="PDF 處理引擎：fitz=PyMuPDF（快但少數檔可能 segfault）；pdfium=較穩（適合疑難檔）",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆蓋已存在輸出")
    parser.add_argument("--verbose", action="store_true", help="印出處理細節")

    args = parser.parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
    else:
        # 預設輸出位置：符合你這批資料的常見需求
        output_dir = (
            Path(__file__).resolve().parent.parent / "rag_test_data" / "docs" / "衛教單_md"
        )
        output_dir = output_dir.resolve()

    pdfs = list(_iter_pdfs(input_path))
    if not pdfs:
        raise SystemExit(f"找不到 PDF：{input_path}")

    # done dir 預設
    if args.done_dir:
        done_dir = Path(args.done_dir).expanduser().resolve()
    else:
        done_dir = (input_path if input_path.is_dir() else input_path.parent) / "done"
        done_dir = done_dir.resolve()

    counts: dict[str, int] = {"text": 0, "slides": 0, "scanned": 0, "mixed": 0}
    moved = 0
    skipped = 0
    failed = 0

    # 若是資料夾模式且啟用隔離：逐檔以子行程處理，避免 native 記憶體累積造成 segfault
    if args.isolate_per_pdf and input_path.is_dir():
        # 重新組裝 child args（移除 isolate flag，並把 input 換成單檔）
        argv = sys.argv[1:]
        base_args: list[str] = []
        i = 0
        while i < len(argv):
            a = argv[i]
            if a == "--isolate-per-pdf":
                i += 1
                continue
            if a == "--input":
                # skip flag + its value
                i += 2
                continue
            # 也支援 --input=...（若使用者這樣傳）
            if a.startswith("--input="):
                i += 1
                continue
            base_args.append(a)
            i += 1

        # 確保 child 不會再次 isolate
        if "--isolate-per-pdf" in base_args:
            base_args = [x for x in base_args if x != "--isolate-per-pdf"]

        for pdf in pdfs:
            # 若輸出已存在且未覆蓋，直接跳過（加速續跑）
            out_md = output_dir / f"{pdf.stem}.md"
            if out_md.exists() and not args.overwrite:
                skipped += 1
                continue

            cmd = [sys.executable, str(Path(__file__).resolve()), "--input", str(pdf)] + base_args
            try:
                p = subprocess.run(cmd, text=True)
                if p.returncode != 0:
                    failed += 1
                    print(f"! 子行程失敗：{pdf.name} (exit={p.returncode})")
                    # SIGSEGV: 自動用 pdfium 重試一次（避免 fitz 特定檔案崩潰）
                    if p.returncode in (-11, 139):
                        retry_cmd = cmd + ["--pdf-engine", "pdfium", "--mode", "ocr"]
                        print(f"  -> retry with pdfium: {pdf.name}")
                        p2 = subprocess.run(retry_cmd, text=True)
                        if p2.returncode == 0:
                            failed -= 1
                        else:
                            print(f"  -> retry failed: exit={p2.returncode}")
                else:
                    # 這裡不精準統計 doc kind（避免再解析），只統計成功數與搬檔由 child 端負責
                    pass
            except Exception as e:
                failed += 1
                print(f"! 子行程例外：{pdf.name} ({type(e).__name__}: {e})")

        print("完成（isolate-per-pdf）：")
        if skipped:
            print(f"- skipped（輸出已存在且未覆蓋）: {skipped}")
        if failed:
            print(f"- failed: {failed}")
        return

    for pdf in pdfs:
        try:
            kind, out_md, processed = process_one(
                pdf_path=pdf,
                output_dir=output_dir,
                max_pages=args.max_pages,
                mode=args.mode,
                sample_pages=args.sample_pages,
                ocr_lang=args.ocr_lang,
                ocr_dpi=args.ocr_dpi,
                vision_base_url=args.vision_base_url.strip() or None,
                vision_model=args.vision_model.strip() or None,
                vision_img_ratio=args.vision_img_ratio,
                vision_timeout_s=args.vision_timeout,
                include_source_text=args.include_source_text,
                include_uncertainties=args.include_uncertainties,
                vision_always=args.vision_always,
                overwrite=args.overwrite,
                verbose=args.verbose,
                embed_images=args.embed_images,
                save_images=(not args.no_save_images) or (not args.embed_images),
                include_frontmatter=args.include_frontmatter,
                output_format=args.output_format,
                pdf_engine=args.pdf_engine,
                include_fulltext=(not args.no_fulltext),
                fulltext_mode=args.fulltext_mode,
            )
            counts[kind] = counts.get(kind, 0) + 1
            if args.verbose:
                print(f"  -> {out_md}")

            if not processed:
                skipped += 1
                continue

            if args.move_done:
                dest = _move_to_done(pdf, done_dir=done_dir)
                moved += 1
                if args.verbose:
                    print(f"  -> moved to {dest}")

            # 釋放 PyMuPDF 內部快取/物件 store（降低長批次記憶體壓力）
            try:
                fitz.TOOLS.store_shrink(100)  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception as e:
            failed += 1
            print(f"! 失敗：{pdf.name} ({type(e).__name__}: {e})")

    print("完成：")
    for k in ["text", "slides", "mixed", "scanned"]:
        if counts.get(k, 0):
            print(f"- {k}: {counts[k]}")
    if skipped:
        print(f"- skipped（輸出已存在且未覆蓋）: {skipped}")
    if args.move_done:
        print(f"- moved_to_done: {moved} （done: {done_dir}）")
    if failed:
        print(f"- failed: {failed}")
    if counts.get("scanned", 0) and not _tesseract_available():
        print(
            "注意：偵測到 scanned PDF，但未安裝系統 `tesseract`，已改輸出圖片。"
            "若要 OCR，請安裝 tesseract（並包含 chi_tra 語言包）。"
        )


if __name__ == "__main__":
    main()


