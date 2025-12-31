# 文件處理腳本使用手冊（PDF → Markdown / education.handout）

本文件說明三支腳本的用途與建議流程：

- `scripts/pdf_structured_extract.py`：PDF →（結構化）Markdown（可選：VLM 圖像解讀、全文逐字轉寫、handout 格式輸出）
- `scripts/update_education_handout_summary_bulk.py`：批次重建 education.handout 的「統整內容」區塊（保留 `## 全文` 原文）
- `scripts/update_education_handout_metadata_bulk.py`：批次更新 education.handout 的 YAML front-matter（保留正文位元級不變）

> 建議：先把 PDF 轉成 `output-format=handout` 的 Markdown，再用後兩支腳本補強 metadata / 統整內容（特別是曾經因 timeout 只留下「統整失敗」的檔案）。

---

## 前置需求

### Python 與依賴

這些腳本依賴專案本身的 Python 環境。建議用 `uv`：

```bash
cd /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2
uv sync --dev
```

之後執行腳本可用：

- `uv run python scripts/...`
- 或直接 `python3 scripts/...`（前提：你已在正確的 venv/uv 環境）

### （可選）Tesseract OCR

若要處理「掃描影像型 PDF」並做 OCR，需要系統層的 `tesseract`，並建議安裝繁中語言包：

- **必需**：`tesseract` 命令可被找到（`which tesseract`）
- **建議**：語言包包含 `chi_tra`（繁中）與 `eng`

> 若未安裝 tesseract，掃描型 PDF 會走「輸出提示/或僅輸出內容不足」的保守路徑（不會硬猜）。

### （可選）OpenAI 相容 LLM / LM Studio

後兩支 bulk 腳本需要可呼叫的 OpenAI 相容 Chat Completions API（例如 LM Studio / vLLM / OpenAI）。

必要環境變數（擇一來源：環境變數或專案 settings）：

- `OPENAI_API_BASE`：例如 `http://127.0.0.1:1234/v1`
- `OPENAI_API_KEY`：LM Studio 通常可填任意字串
- `CHAT_MODEL`：例如 `gpt-4o-mini` 或你在 LM Studio 的模型 id

> `pdf_structured_extract.py` 若使用 `--vision-base-url/--vision-model` 也會呼叫 OpenAI 相容 API（Chat Completions）。

---

## 1) `scripts/pdf_structured_extract.py`（PDF → Markdown）

### 目標與輸出

此腳本會將 PDF 轉成 Markdown，並自動辨識 PDF 類型：

- **文字型（text）**：以 PyMuPDF 抽取文字，做基本段落/清單重建
- **投影片型（slides）**：每頁以較大字級行當標題，其他行轉 bullet
- **掃描影像型（scanned）**：若有 tesseract 則 OCR；也可配合 vision model 做「結構化重寫」與「全文逐字轉寫＋排版」
- **混合（mixed）**：預設採用 hybrid：逐頁判斷文字量，不足就 OCR/視覺解讀

並支援兩種輸出格式：

- `--output-format raw`：只輸出抽取後的 Markdown
- `--output-format handout`：第二階段把輸出包成 `education.handout` 的 front-matter +（統整內容）+ `## 全文`

### 常用範例

#### A. 單檔 PDF → raw Markdown

```bash
uv run python scripts/pdf_structured_extract.py \
  --input /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2/rag_test_data/source/衛教單原始檔/done/1701571983_300.pdf \
  --output /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2/rag_test_data/docs/衛教單_md \
  --max-pages 80
```

#### B. 批次資料夾 PDF → raw Markdown（建議隔離子行程避免 segfault）

```bash
uv run python scripts/pdf_structured_extract.py \
  --input /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2/rag_test_data/source/衛教單原始檔 \
  --output /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2/rag_test_data/docs/衛教單_md \
  --isolate-per-pdf \
  --max-pages 80 \
  --verbose
```

#### C. 批次處理完成後，把 PDF 搬到 done/（只搬「本次真的有處理」的檔案）

```bash
uv run python scripts/pdf_structured_extract.py \
  --input /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2/rag_test_data/source/衛教單原始檔 \
  --output /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2/rag_test_data/docs/衛教單_md \
  --isolate-per-pdf \
  --move-done \
  --verbose
```

#### D. 使用 vision model（推薦：版面/圖表/掃描檔品質更好）

> 一旦同時提供 `--vision-base-url` 與 `--vision-model`，腳本會改為 **vision-only**：每頁渲染成圖片交給 VLM 產生結構化 Markdown，並可輸出跨頁合併的 `## 全文`（逐字轉寫＋排版）。

```bash
uv run python scripts/pdf_structured_extract.py \
  --input /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2/rag_test_data/source/衛教單原始檔/done/1701571949_705.pdf \
  --output /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2/rag_test_data/docs/衛教單_md \
  --vision-base-url http://127.0.0.1:1234 \
  --vision-model qwen/qwen3-vl-8b \
  --ocr-lang auto \
  --ocr-dpi 260 \
  --max-pages 80
```

#### E. 直接輸出 `education.handout`（推薦用於後續 RAG 與欄位化）

```bash
uv run python scripts/pdf_structured_extract.py \
  --input /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2/rag_test_data/source/衛教單原始檔/done/1701571949_705.pdf \
  --output /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2/rag_test_data/cus/衛教 \
  --vision-base-url http://127.0.0.1:1234 \
  --vision-model qwen/qwen3-vl-8b \
  --output-format handout \
  --fulltext-mode details \
  --include-frontmatter
```

### 重要參數速查（挑常用）

- `--input`：PDF 單檔或資料夾
- `--output`：輸出資料夾（不填則預設到 `rag_test_data/docs/衛教單_md`）
- `--mode`：`auto|text|slides|ocr`（多數情況用 `auto`）
- `--max-pages`：每個 PDF 最多處理頁數（避免超長檔吃太久）
- `--isolate-per-pdf`：資料夾批次時建議開啟（每檔子行程，降低 PyMuPDF segfault 風險）
- `--pdf-engine`：`fitz|pdfium`（遇到 fitz 會崩潰的疑難檔，改 `pdfium`）
- `--vision-base-url/--vision-model`：啟用 VLM 視覺解讀與逐字全文排版（推薦）
- `--vision-timeout`：vision API timeout（秒）
- `--vision-always`：每頁都走 vision（較慢/較貴，但對投影片/圖表類更穩）
- `--output-format`：`raw|handout`
- `--no-fulltext`：不輸出 `## 全文`（預設會輸出，避免後續 LLM 曲解）
- `--fulltext-mode`：`section|details`（details 可折疊全文，避免檔案過長）
- `--overwrite`：覆蓋已存在輸出

### 常見問題

- **跑到一半 segfault / exit=-11**：批次請用 `--isolate-per-pdf`；仍不穩可加 `--pdf-engine pdfium --mode ocr`。
- **掃描 PDF 沒有 OCR 結果**：確認系統有 `tesseract` 且語言包含 `chi_tra`；或改用 `--vision-*` 走 VLM。
- **輸出太長**：用 `--fulltext-mode details` 或 `--no-fulltext`（但不建議關掉全文，除非你確定後續流程不需要）。

---

## 2) `scripts/update_education_handout_summary_bulk.py`（重建統整內容）

### 目標與保證

此腳本用 LLM 重新產生 `education.handout` 文件中的統整章節：

- `## 主旨`
- `## 重點整理`
- `## 流程/步驟`
- `## 注意事項/警示`
- `## 表格/比較`
- `## 版本資訊`

**重要保證**：

- 只會改動 YAML front-matter 之後、`## 全文` 之前的區塊（統整區）
- `## 全文` 與其後所有原文內容保持原樣（位元級不變）

### 常用範例

#### A. 只修復含「統整失敗」的檔案（推薦）

```bash
export OPENAI_API_BASE="http://127.0.0.1:1234/v1"
export OPENAI_API_KEY="dummy"
export CHAT_MODEL="gpt-4o-mini"

uv run python scripts/update_education_handout_summary_bulk.py \
  --dir /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2/rag_test_data/cus/衛教 \
  --only-failed \
  --chunked \
  --chunk-adaptive-split
```

#### B. 先 dry-run 抽查（不寫檔）

```bash
uv run python scripts/update_education_handout_summary_bulk.py \
  --dir /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2/rag_test_data/cus/衛教 \
  --only-failed \
  --dry-run \
  --limit 5
```

#### C. 只處理指定檔案（可重複 --file）

```bash
uv run python scripts/update_education_handout_summary_bulk.py \
  --dir /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2/rag_test_data/cus/衛教 \
  --file /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2/rag_test_data/cus/衛教/1701571949_705.md \
  --chunked
```

### 參數選擇建議

- **超長全文**：用 `--chunked`（預設建議）＋ `--chunk-adaptive-split`（更抗 timeout/壞 JSON）
- **模型較弱/容易 JSON 壞掉**：保留 `--chunked`，並可把 `--chunk-chars` 調小（例如 4000）
- **表格很多**：預設會壓縮大型表格（避免塞爆 prompt），通常不要關掉

---

## 3) `scripts/update_education_handout_metadata_bulk.py`（更新 YAML front-matter）

### 目標與保證

此腳本使用 LLM 從文件內容萃取 metadata，並更新檔案開頭的 YAML front-matter。

**重要保證**：

- 只會改動檔案開頭的 YAML front-matter（第一個 `---` 到第二個 `---`）
- 第二個 `---` 之後正文內容保持原樣（位元級不變）

### 常用範例

#### A. 先 dry-run + limit 抽查

```bash
export OPENAI_API_BASE="http://127.0.0.1:1234/v1"
export OPENAI_API_KEY="dummy"
export CHAT_MODEL="gpt-4o-mini"

uv run python scripts/update_education_handout_metadata_bulk.py \
  --dir /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2/rag_test_data/cus/衛教 \
  --limit 3 \
  --dry-run
```

#### B. 只補齊空白/佔位值（預設行為）

```bash
uv run python scripts/update_education_handout_metadata_bulk.py \
  --dir /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2/rag_test_data/cus/衛教
```

#### C. 強制覆寫既有 metadata（謹慎使用）

```bash
uv run python scripts/update_education_handout_metadata_bulk.py \
  --dir /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2/rag_test_data/cus/衛教 \
  --force
```

#### D. 超長文件：改用 chunked 分段萃取（更穩）

```bash
uv run python scripts/update_education_handout_metadata_bulk.py \
  --dir /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2/rag_test_data/cus/衛教 \
  --chunked \
  --chunk-carry
```

---

## 建議工作流（最常見）

### 工作流 A：PDF 新增入庫（推薦）

1) PDF → handout Markdown（含統整 + 全文）

```bash
uv run python scripts/pdf_structured_extract.py \
  --input /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2/rag_test_data/source/衛教單原始檔 \
  --output /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2/rag_test_data/cus/衛教 \
  --isolate-per-pdf \
  --vision-base-url http://127.0.0.1:1234 \
  --vision-model qwen/qwen3-vl-8b \
  --output-format handout \
  --fulltext-mode details
```

2) 若統整曾失敗（或你想重建統整品質）→ 跑 summary 重建

```bash
uv run python scripts/update_education_handout_summary_bulk.py \
  --dir /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2/rag_test_data/cus/衛教 \
  --only-failed \
  --chunked \
  --chunk-adaptive-split
```

3) 更新 metadata（keywords、aliases、owner、版本資訊等）

```bash
uv run python scripts/update_education_handout_metadata_bulk.py \
  --dir /Users/bruce/Projects/Labs/AI/chatbot_rag_ptch_v2/rag_test_data/cus/衛教 \
  --chunked \
  --chunk-carry
```

---

## 版本與風險提示

- 這些腳本以「保守、不虛構」為主：資訊不足會留空或標記（例如 `〔無〕` / `〔無法辨識〕`）。
- 若你把 `--force` 開下去，metadata 可能被模型覆寫；建議先 `--dry-run --limit` 抽查品質再套用。


