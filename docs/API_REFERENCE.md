# API 參考手冊

本文件詳細說明 Chatbot RAG API 的所有端點。

## 基礎資訊

- **Base URL**: `http://localhost:8000`
- **API 版本**: `v1`
- **內容類型**: `application/json`
- **串流回應**: `text/event-stream`

---

## 基礎路由 `/api/v1/`

### GET /health

健康檢查端點。

**回應**
```json
{
  "status": "healthy"
}
```

### GET /hello/{name}

問候端點（測試用）。

**參數**
| 名稱 | 類型 | 說明 |
|------|------|------|
| name | string | 問候對象名稱 |

**回應**
```json
{
  "message": "Hello, {name}!"
}
```

---

## RAG 路由 `/api/v1/rag/`

### POST /vectorize

向量化文件，建立或更新向量索引。

**請求**
```json
{
  "source": "default",
  "mode": "override",
  "directory": "/path/to/docs"
}
```

| 欄位 | 類型 | 必填 | 說明 |
|------|------|------|------|
| source | string | 否 | `"default"` (預設目錄) 或 `"uploaded"` |
| mode | string | 否 | `"override"` (重建) 或 `"update"` (增量) |
| directory | string | 否 | 自訂文件目錄路徑 |

**回應**
```json
{
  "status": "success",
  "mode": "override",
  "source": "default",
  "documents_processed": 42,
  "chunks_created": 156,
  "vectors_stored": 156,
  "collection_name": "documents"
}
```

### GET /collection/info

查詢向量集合資訊。

**回應**
```json
{
  "collection_name": "documents",
  "vectors_count": 156,
  "points_count": 156,
  "status": "green"
}
```

### GET /health-check

RAG 系統健康檢查。

**回應**
```json
{
  "qdrant_connected": true,
  "embedding_service_connected": true,
  "collection_exists": true,
  "embedding_dimension": 768,
  "expected_dimension": 768,
  "dimension_match": true
}
```

### POST /test/retrieval

測試文件檢索。

**請求**
```json
{
  "query": "如何登入平台？",
  "top_k": 5,
  "score_threshold": 0.5
}
```

| 欄位 | 類型 | 必填 | 說明 |
|------|------|------|------|
| query | string | 是 | 檢索查詢 |
| top_k | integer | 否 | 返回文件數量，預設 3 |
| score_threshold | float | 否 | 相似度閾值 0-1，預設 0.5 |

**回應**
```json
{
  "status": "success",
  "query": "如何登入平台？",
  "documents_found": 3,
  "documents": [
    {
      "content": "用戶可以透過...",
      "source": "faq-01-first-login.md",
      "score": 0.89,
      "metadata": {
        "title": "首次登入說明",
        "section": "帳號管理"
      }
    }
  ]
}
```

---

## 智能問答端點

### POST /ask/stream

單輪智能問答（SSE 串流）。

**請求**
```json
{
  "question": "如何申請退款？",
  "top_k": 3,
  "llm_config": {
    "model": "gpt-4",
    "reasoning_effort": "medium",
    "reasoning_summary": "auto"
  }
}
```

| 欄位 | 類型 | 必填 | 說明 |
|------|------|------|------|
| question | string | 是 | 用戶問題 |
| top_k | integer | 否 | 檢索文件數量 |
| llm_config | object | 否 | LLM 配置覆蓋 |

**回應（SSE 事件流）**

```
event: node_event
data: {"source":"ask_stream","node":"guard","phase":"status","payload":{"stage":"GUARD_START"}}

event: node_event
data: {"source":"ask_stream","node":"planner","phase":"status","payload":{"task_type":"simple_faq","should_retrieve":true}}

event: llm_chunk
data: {"content":"根據"}

event: llm_chunk
data: {"content":"知識庫"}

event: llm_meta
data: {"model":"gpt-4","usage":{"prompt_tokens":500,"completion_tokens":150}}

event: final_response
data: {"answer":"根據知識庫...","documents_used":true,"steps":5}
```

**事件類型**

| 事件 | 說明 |
|------|------|
| `node_event` | 節點執行狀態 |
| `llm_chunk` | LLM 回應片段 |
| `llm_meta` | LLM 元資料（模型、Token 用量） |
| `final_response` | 最終回答 |
| `error` | 錯誤訊息 |

### POST /ask/stream_chat

多輪對話問答（SSE 串流）。

**請求**
```json
{
  "question": "那退款需要多久？",
  "conversation_history": [
    {"role": "user", "content": "如何申請退款？"},
    {"role": "assistant", "content": "您可以透過線上系統申請退款..."}
  ],
  "enable_conversation_summary": true,
  "conversation_summary": "用戶詢問退款相關問題",
  "top_k": 3
}
```

| 欄位 | 類型 | 必填 | 說明 |
|------|------|------|------|
| question | string | 是 | 當前問題 |
| conversation_history | array | 否 | 歷史對話記錄 |
| enable_conversation_summary | boolean | 否 | 是否使用對話摘要 |
| conversation_summary | string | 否 | 對話摘要 |
| top_k | integer | 否 | 檢索文件數量 |

**回應**

與 `/ask/stream` 相同的 SSE 事件流格式。

### POST /ask/stream_chat/conversation_summary

生成對話摘要。

**請求**
```json
{
  "conversation_history": [
    {"role": "user", "content": "如何申請退款？"},
    {"role": "assistant", "content": "您可以透過..."},
    {"role": "user", "content": "需要準備什麼文件？"},
    {"role": "assistant", "content": "您需要準備..."}
  ]
}
```

**回應**
```json
{
  "summary": "用戶詢問退款流程和所需文件，助手說明了線上申請步驟和必要文件清單。"
}
```

---

## 管理路由 `/api/v1/admin/`

### POST /prompts/init

初始化 Langfuse Prompts。

**回應**
```json
{
  "status": "success",
  "prompts_initialized": [
    "unified-agent-system",
    "planner-prompt",
    "query-builder-prompt"
  ]
}
```

### GET /prompts/status

查詢 Prompt 狀態。

**回應**
```json
{
  "enabled": true,
  "label": "production",
  "cache_ttl": 300,
  "prompts": {
    "unified-agent-system": {
      "version": 3,
      "last_updated": "2024-01-15T10:30:00Z"
    }
  }
}
```

### POST /datasets/create

建立 Langfuse Dataset。

**請求**
```json
{
  "name": "faq-evaluation",
  "description": "FAQ 問答評估資料集",
  "items": [
    {
      "input": {"question": "如何登入？"},
      "expected_output": {"answer": "..."}
    }
  ]
}
```

**回應**
```json
{
  "status": "success",
  "dataset_id": "ds_abc123",
  "items_created": 10
}
```

### POST /tools/export-schema

導出工具 Schema（JSON Schema 格式）。

**回應**
```json
{
  "tools": [
    {
      "name": "retrieve_documents",
      "description": "檢索知識庫文件",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {"type": "string"}
        }
      }
    }
  ]
}
```

### POST /experiments/run

執行 Prompt 實驗。

**請求**
```json
{
  "dataset_name": "faq-evaluation",
  "prompt_name": "unified-agent-system",
  "prompt_version": 3
}
```

**回應**
```json
{
  "status": "running",
  "experiment_id": "exp_xyz789",
  "total_items": 50
}
```

---

## 報表路由 `/api/v1/reports/`

### POST /generate

生成 Trace 報表（Excel 格式）。

**請求**
```json
{
  "start_date": "2024-01-01",
  "end_date": "2024-01-31",
  "filters": {
    "status": "success"
  }
}
```

**回應**

- **≤200 traces**：同步回傳 Excel 檔案
- **>200 traces**：回傳任務 ID，非同步處理

```json
{
  "status": "processing",
  "task_id": "task_abc123",
  "estimated_time": "5 minutes"
}
```

---

## 檔案路由 `/api/v1/files/`

### GET /download/{filename}

下載檔案（表單等）。

**參數**
| 名稱 | 類型 | 說明 |
|------|------|------|
| filename | string | 檔案名稱 |

**回應**

二進位檔案內容，含適當的 Content-Type 標頭。

---

## 錯誤回應

所有 API 使用統一的錯誤格式：

```json
{
  "detail": "錯誤描述",
  "error_code": "ERROR_CODE",
  "request_id": "req_abc123"
}
```

**常見錯誤碼**

| HTTP 狀態碼 | 錯誤碼 | 說明 |
|------------|--------|------|
| 400 | INVALID_REQUEST | 請求格式錯誤 |
| 400 | QUESTION_TOO_LONG | 問題超過長度限制 |
| 400 | INJECTION_DETECTED | 偵測到注入攻擊 |
| 404 | COLLECTION_NOT_FOUND | 向量集合不存在 |
| 503 | QDRANT_UNAVAILABLE | Qdrant 服務不可用 |
| 503 | LLM_UNAVAILABLE | LLM 服務不可用 |

---

## 請求標頭

**標準標頭**
```
Content-Type: application/json
Accept: application/json
```

**串流請求**
```
Accept: text/event-stream
```

**可選標頭**
```
X-Request-ID: custom-request-id
```

---

## 速率限制

目前版本未實作速率限制。建議在生產環境透過反向代理（如 Nginx）設定。

---

## 相關文件

- [SSE 串流處理](./SSE_STREAMING.md) - 串流回應詳細說明
- [配置參考](./CONFIGURATION.md) - API 相關配置
- [安全防護](./SECURITY.md) - 輸入驗證和安全機制
