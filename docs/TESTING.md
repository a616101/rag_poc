# 測試指南

本文件說明 Chatbot RAG 系統的測試策略和工具。

## 測試類型

```
┌─────────────────────────────────────────────────────────────────┐
│                        測試金字塔                                 │
│                                                                   │
│                        ┌───────┐                                 │
│                        │  E2E  │  ← 端對端測試                    │
│                       ─┴───────┴─                                │
│                      │ Integration │  ← 整合測試                  │
│                     ─┴─────────────┴─                            │
│                    │    Unit Tests    │  ← 單元測試               │
│                   ─┴───────────────────┴─                        │
└─────────────────────────────────────────────────────────────────┘
```

## 測試腳本

所有測試腳本位於 `scripts/` 目錄：

| 腳本 | 用途 |
|------|------|
| `test_contextual_chunking.py` | 測試脈絡化分塊功能 |
| `test_context_expansion.py` | 測試上下文擴充功能 |
| `test_progressive_retrieval.py` | 測試漸進式檢索 |
| `analyze_retrieval_scores.py` | 分析檢索分數分佈 |
| `analyze_contextual_impact.py` | 分析脈絡化對檢索的影響 |

### 執行測試腳本

```bash
# 測試脈絡化分塊
python scripts/test_contextual_chunking.py

# 測試上下文擴充
python scripts/test_context_expansion.py

# 分析檢索分數
python scripts/analyze_retrieval_scores.py
```

## 單元測試

### Markdown 解析器測試

```python
# 測試章節解析
from chatbot_rag.services.markdown_parser import markdown_parser

content = """
# 主標題
## 第一章
### 1.1 小節
內容...
"""

sections = markdown_parser.parse_sections(content)
assert len(sections) == 3
assert sections[0].title == "主標題"
```

### 脈絡生成測試

```python
from chatbot_rag.services.contextual_chunking_service import (
    contextual_chunking_service
)

context = contextual_chunking_service.generate_context_level1(
    doc_metadata={"doc_title": "測試文件", "entry_type": "Policy"},
    section_path="第一章 > 1.1 小節"
)

assert "文件：測試文件" in context
assert "類型：Policy" in context
```

## 整合測試

### 檢索測試

```bash
curl -X POST http://localhost:8000/api/v1/rag/retrieve/test \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何登入平台？",
    "top_k": 5
  }'
```

### 健康檢查測試

```bash
# 基礎健康檢查
curl http://localhost:8000/api/v1/health

# RAG 系統健康檢查
curl http://localhost:8000/api/v1/rag/health
```

## 端對端測試

### 使用 Admin API

透過 `/api/v1/admin/experiments/run-e2e` 執行 E2E 測試：

```bash
curl -X POST http://localhost:8000/api/v1/admin/experiments/run-e2e \
  -H "Content-Type: application/json" \
  -d '{
    "category": "qa_retrieval",
    "limit": 10,
    "run_description": "快速驗證測試"
  }'
```

### 測試資料集

系統內建測試案例位於 `services/dataset_cases.py`：

```python
QA_RETRIEVAL_CASES = [
    {
        "id": "qa_001",
        "question": "平臺管理機關是誰？",
        "expected_answer": "行政院人事行政總處公務人力發展學院"
    },
    # ... 更多測試案例
]
```

## 評估指標

### 檢索評估

| 指標 | 說明 |
|------|------|
| 檢索成功率 | 成功檢索到相關文件的比例 |
| 平均分數 | 檢索結果的平均相似度分數 |
| Top-K 精準度 | 前 K 個結果中相關文件的比例 |

### 回答評估

| 指標 | 說明 |
|------|------|
| `answer_similarity` | 回答與預期答案的語義相似度 |
| `response_completeness` | 回答是否涵蓋關鍵資訊 |

### 評估器實作

```python
# services/evaluators.py

def answer_similarity(
    answer: str,
    expected_answer: str,
    embedding_service
) -> float:
    """計算答案相似度（0-1）"""
    answer_emb = embedding_service.embed_text(answer)
    expected_emb = embedding_service.embed_text(expected_answer)
    return cosine_similarity(answer_emb, expected_emb)

def response_completeness(
    answer: str,
    expected_answer: str,
    key_phrases: List[str]
) -> float:
    """計算回答完整度（0-1）"""
    matched = sum(1 for phrase in key_phrases if phrase in answer)
    return matched / len(key_phrases)
```

## 測試流程

### 1. 向量化測試文件

```bash
curl -X POST http://localhost:8000/api/v1/rag/vectorize \
  -H "Content-Type: application/json" \
  -d '{"source": "default", "mode": "override"}'
```

### 2. 驗證集合

```bash
curl http://localhost:8000/api/v1/rag/collection/info
```

### 3. 執行檢索測試

```bash
python scripts/test_progressive_retrieval.py
```

### 4. 執行 E2E 測試

```bash
curl -X POST http://localhost:8000/api/v1/admin/experiments/run-e2e \
  -H "Content-Type: application/json" \
  -d '{"category": "qa_retrieval", "limit": 100}'
```

### 5. 分析結果

在 Langfuse Dashboard 中查看實驗結果。

## 效能測試

### 檢索效能

```python
import time
from chatbot_rag.services.retriever_service import retriever_service

queries = ["如何登入？", "退款流程", "課程搜尋"]
times = []

for query in queries:
    start = time.time()
    retriever_service.retrieve(query, top_k=3)
    times.append(time.time() - start)

print(f"平均檢索時間: {sum(times)/len(times):.3f}s")
```

### SSE 串流效能

```bash
# 測試 SSE 連線
time curl -X POST http://localhost:8000/api/v1/rag/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "如何登入？"}' \
  --output /dev/null
```

## 持續整合

### GitHub Actions 範例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      qdrant:
        image: qdrant/qdrant
        ports:
          - 6333:6333
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v3
      
      - name: Install dependencies
        run: uv sync
      
      - name: Run tests
        run: |
          uv run python scripts/test_contextual_chunking.py
          uv run python scripts/test_progressive_retrieval.py
```

## 除錯技巧

### 啟用詳細日誌

```bash
DEBUG=true
LOG_LEVEL=DEBUG
LLM_STREAM_DEBUG=true
```

### 檢查 Qdrant 資料

```bash
# 查看集合資訊
curl http://localhost:6333/collections/documents

# 查看向量數量
curl http://localhost:6333/collections/documents/points/count
```

### 檢查 LLM 連線

```bash
curl $OPENAI_API_BASE/models
```

## 相關文件

- [Langfuse 整合](./LANGFUSE_INTEGRATION.md)
- [配置參考](./CONFIGURATION.md)
- [API 參考手冊](./API_REFERENCE.md)
