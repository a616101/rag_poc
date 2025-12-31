# Langfuse 整合指南

本文件說明 Chatbot RAG 系統與 Langfuse 的整合方式。

## 概述

系統使用 Langfuse 實現完整的可觀測性：

- **Trace**：追蹤每個請求的完整執行軌跡
- **Prompt Management**：集中管理和版本控制 Prompt
- **Scores**：收集用戶反饋評分
- **Datasets**：管理評估資料集
- **Experiments**：執行 Prompt 實驗

## 配置

### 環境變數

```bash
# Langfuse SDK 配置
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com

# 應用程式配置
LANGFUSE_PROMPT_ENABLED=true
LANGFUSE_PROMPT_LABEL=production
LANGFUSE_PROMPT_CACHE_TTL=300
```

### 使用自架 Langfuse

```bash
LANGFUSE_HOST=https://your-langfuse-instance.com
```

## Trace 追蹤

### 自動追蹤

系統自動為每個問答請求建立 Trace：

```python
# services/ask_stream/tracing.py

from langfuse import get_client

def create_trace_context(request_id: str, question: str):
    client = get_client()
    trace = client.trace(
        id=request_id,
        name="ask_stream",
        input={"question": question},
        tags=["unified-agent", "production"]
    )
    return trace
```

### Trace 結構

```
Trace: ask_stream
├── Span: guard
├── Span: language_normalizer
├── Span: planner
│   └── Generation: planner-llm-call
├── Span: query_builder
│   └── Generation: query-rewrite-llm-call
├── Span: tool_executor
│   └── Span: retrieve_documents
├── Span: retrieval_checker
├── Span: response_synth
│   └── Generation: response-llm-call
└── Span: telemetry
```

### 遙測事件

每個 Trace 結束時發送遙測摘要：

```json
{
  "trace_id": "d2d1e2ddd5ab558f8388c6d9cf510ac8",
  "intent": "simple_faq",
  "used_tools": ["retrieve_documents"],
  "user_language": "zh-hant",
  "is_out_of_scope": false,
  "prompt_versions": {
    "unified-agent-system": {"version": 3, "label": "production"}
  }
}
```

## Prompt Management

### 支援的 Prompt

| Prompt 名稱 | 用途 |
|-------------|------|
| `unified-agent-system` | 主要系統提示 |
| `planner-prompt` | 任務規劃提示 |
| `query-builder-prompt` | 查詢重寫提示 |
| `response-generator-prompt` | 回答生成提示 |

### Prompt 服務

```python
# services/prompt_service.py

class PromptService:
    def __init__(
        self,
        default_label: str = "production",
        cache_ttl_seconds: int = 300
    ):
        self.cache = {}
        self.default_label = default_label
        self.cache_ttl = cache_ttl_seconds
    
    def get_text_prompt(
        self,
        name: str,
        **variables
    ) -> Tuple[str, PromptMetadata]:
        """獲取並編譯文字 Prompt"""
        prompt = self._get_cached_or_fetch(name)
        compiled = prompt.compile(**variables)
        
        return compiled, PromptMetadata(
            name=name,
            version=prompt.version,
            label=prompt.label,
            langfuse_prompt=prompt
        )
```

### 使用 Prompt

```python
from chatbot_rag.services.prompt_service import PromptService

prompt_service = PromptService(default_label="production")

# 獲取 Prompt
content, metadata = prompt_service.get_text_prompt(
    "unified-agent-system",
    language_instruction="請用繁體中文回答",
    support_scope="這是一個學習平臺客服系統..."
)

# 在 Generation 中關聯 Prompt
generation = trace.generation(
    name="response",
    prompt=metadata.langfuse_prompt  # 關聯 Prompt 版本
)
```

### 初始化 Prompt

透過 Admin API 初始化：

```bash
curl -X POST http://localhost:8000/api/v1/admin/prompts/init
```

或使用腳本：

```bash
python scripts/init_prompts.py
```

## Scores（評分）

### 用戶反饋

收集用戶對回答的評分：

```bash
curl -X POST http://localhost:8000/api/v1/rag/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "trace_id": "d2d1e2ddd5ab558f8388c6d9cf510ac8",
    "score": "up",
    "comment": "回答很有幫助"
  }'
```

### 評分類型

| Score 名稱 | 說明 |
|------------|------|
| `user_feedback` | 用戶讚/倒讚 |
| `answer_similarity` | 答案相似度（自動評估） |
| `response_completeness` | 回答完整度（自動評估） |

## Datasets

### 建立資料集

```bash
curl -X POST http://localhost:8000/api/v1/admin/datasets/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "qa_retrieval",
    "description": "問答檢索評估資料集",
    "metadata": {"category": "qa_retrieval"}
  }'
```

### 上傳測試案例

```bash
curl -X POST http://localhost:8000/api/v1/admin/datasets/qa_retrieval/items \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "input": {"question": "如何登入平台？"},
        "expected_output": {"answer": "您可以透過..."}
      }
    ]
  }'
```

## Experiments

### 執行實驗

```bash
curl -X POST http://localhost:8000/api/v1/admin/experiments/run-e2e \
  -H "Content-Type: application/json" \
  -d '{
    "category": "qa_retrieval",
    "limit": 100,
    "run_description": "測試新的 Prompt 版本"
  }'
```

### 評估器

系統內建多種評估器：

| 評估器 | 說明 |
|--------|------|
| `answer_similarity` | 計算回答與預期答案的語義相似度 |
| `response_completeness` | 評估回答是否涵蓋關鍵資訊 |

## Dashboard 使用

### 查看 Traces

1. 登入 Langfuse Dashboard
2. 選擇專案
3. 進入 Traces 頁面
4. 篩選 `name:ask_stream`

### 分析 Prompt 效果

1. 進入 Prompts 頁面
2. 查看各版本的使用統計
3. 比較不同 Label 的效能

### 檢視評分

1. 進入 Scores 頁面
2. 篩選 `name:user_feedback`
3. 查看評分分佈

## 最佳實踐

### 1. Prompt 版本管理

- 使用 `staging` label 測試新版本
- 確認效果後再標記為 `production`
- 保留舊版本以便回滾

### 2. 快取策略

```bash
# 開發環境：短 TTL
LANGFUSE_PROMPT_CACHE_TTL=60

# 生產環境：較長 TTL
LANGFUSE_PROMPT_CACHE_TTL=300
```

### 3. 標籤使用

為 Trace 添加有意義的標籤：

```python
trace = client.trace(
    tags=["unified-agent", "production", "v2.0"]
)
```

### 4. 定期評估

- 設定定期執行實驗的排程
- 監控評分趨勢
- 收集足夠的用戶反饋

## 相關文件

- [配置參考](./CONFIGURATION.md)
- [API 參考手冊](./API_REFERENCE.md)
- [測試指南](./TESTING.md)
