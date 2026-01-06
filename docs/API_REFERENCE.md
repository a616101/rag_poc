# API 參考手冊

本文件詳細說明 ChatBot GraphRAG API 的所有端點。

## 基礎資訊

- **Base URL**: `http://localhost:18000`
- **API 版本**: `v1`
- **內容類型**: `application/json`
- **串流回應**: `text/event-stream`

---

## 健康檢查端點

### GET /health

基礎健康檢查。

**回應**
```json
{
  "status": "healthy"
}
```

### GET /health/ready

就緒檢查（驗證所有服務連接）。

**回應**
```json
{
  "status": "ready",
  "services": {
    "qdrant": "connected",
    "nebula": "connected",
    "opensearch": "connected",
    "postgres": "connected",
    "redis": "connected",
    "minio": "connected"
  }
}
```

### GET /health/live

存活檢查。

**回應**
```json
{
  "status": "alive"
}
```

### GET /health/concurrency

LLM 並發狀態。

**回應**
```json
{
  "llm_semaphores": {
    "chat": {"available": 3, "max": 5},
    "embedding": {"available": 8, "max": 10}
  },
  "queue_length": 2
}
```

---

## 向量化 API `/api/v1/rag/vectorize`

### POST /vectorize

異步文件攝取與向量化。

**請求**
```json
{
  "source": "default",
  "mode": "override",
  "directory": "/path/to/docs",
  "options": {
    "build_graph": true,
    "detect_communities": true
  }
}
```

| 欄位 | 類型 | 必填 | 說明 |
|------|------|------|------|
| source | string | 否 | `"default"` 或 `"uploaded"` |
| mode | string | 否 | `"override"` (重建) 或 `"update"` (增量) |
| directory | string | 否 | 自訂文件目錄路徑 |
| options.build_graph | boolean | 否 | 是否建立知識圖譜 |
| options.detect_communities | boolean | 否 | 是否執行社群偵測 |

**回應**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "向量化作業已提交"
}
```

### GET /vectorize/status/{job_id}

查詢向量化作業進度。

**回應**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress": {
    "documents_processed": 25,
    "total_documents": 42,
    "chunks_created": 156,
    "entities_extracted": 89,
    "relations_extracted": 45,
    "current_phase": "entity_extraction"
  }
}
```

### GET /vectorize/directory

列出可用的文件目錄。

**回應**
```json
{
  "directories": [
    {"path": "default", "files": 42},
    {"path": "uploaded", "files": 15}
  ]
}
```

---

## 問答 API `/api/v1/rag/ask`

### POST /ask/stream

Responses API 格式的串流問答。

**請求**
```json
{
  "question": "如何申請退款？",
  "top_k": 3,
  "query_mode": "auto",
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
| top_k | integer | 否 | 檢索文件數量，預設 3 |
| query_mode | string | 否 | `"auto"`, `"local"`, `"global"`, `"drift"` |
| llm_config | object | 否 | LLM 配置覆蓋 |

**回應（SSE 事件流）**

```
event: status
data: {"node":"guard","phase":"start","message":"安全檢查中..."}

event: status
data: {"node":"intent","phase":"complete","intent":"retrieval","query_mode":"local"}

event: retrieval
data: {"sources":[{"title":"退款說明","score":0.92}],"entity_count":5}

event: reasoning
data: {"content":"正在分析退款流程..."}

event: answer
data: {"delta":"根據"}

event: answer
data: {"delta":"知識庫"}

event: meta
data: {"model":"gpt-4","usage":{"prompt_tokens":500,"completion_tokens":150}}

event: meta_summary
data: {"trace_id":"abc123","total_tokens":650,"latency_ms":1250}
```

**事件類型**

| 事件 | 說明 |
|------|------|
| `status` | 節點執行狀態 |
| `rewrite_llm` | Query 重寫過程 |
| `retrieval` | 檢索結果資訊 |
| `reasoning` | LLM 推理內容 |
| `answer` | 回答片段 (delta) |
| `meta` | Token 使用統計 |
| `meta_summary` | 完整統計摘要 |
| `sources` | 參考來源 |
| `error` | 錯誤訊息 |

### POST /ask/stream_chat

OpenAI Chat API 相容格式的串流問答。

**請求**
```json
{
  "messages": [
    {"role": "system", "content": "你是一個有幫助的助手"},
    {"role": "user", "content": "如何申請退款？"},
    {"role": "assistant", "content": "您可以透過線上系統申請..."},
    {"role": "user", "content": "需要多久時間？"}
  ],
  "stream": true,
  "include_sources": true,
  "top_k": 3
}
```

| 欄位 | 類型 | 必填 | 說明 |
|------|------|------|------|
| messages | array | 是 | OpenAI 格式的訊息陣列 |
| stream | boolean | 否 | 是否串流回應，預設 true |
| include_sources | boolean | 否 | 是否包含參考來源 |
| top_k | integer | 否 | 檢索文件數量 |

**回應（OpenAI 相容 SSE）**

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"delta":{"content":"根據"},"index":0}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"delta":{"content":"知識庫"},"index":0}]}

data: [DONE]
```

### POST /feedback

提交用戶反饋。

**請求**
```json
{
  "trace_id": "d2d1e2ddd5ab558f8388c6d9cf510ac8",
  "score": "up",
  "comment": "回答很有幫助"
}
```

| 欄位 | 類型 | 必填 | 說明 |
|------|------|------|------|
| trace_id | string | 是 | 追蹤 ID |
| score | string | 是 | `"up"` 或 `"down"` |
| comment | string | 否 | 用戶評論 |

**回應**
```json
{
  "status": "success",
  "feedback_id": "fb_abc123"
}
```

---

## 快取管理 API `/api/v1/admin/cache`

### GET /cache/stats

查詢快取統計。

**回應**
```json
{
  "semantic_cache": {
    "entries": 1250,
    "hit_rate": 0.35,
    "memory_mb": 128
  },
  "index_version": "2024-01-15T10:30:00Z",
  "prompt_version": 3
}
```

### POST /cache/invalidate

使快取失效。

**請求**
```json
{
  "pattern": "退款*",
  "scope": "semantic"
}
```

**回應**
```json
{
  "status": "success",
  "entries_invalidated": 15
}
```

### DELETE /cache/clear

清除所有快取。

**回應**
```json
{
  "status": "success",
  "entries_cleared": 1250
}
```

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
| 404 | JOB_NOT_FOUND | 向量化作業不存在 |
| 429 | RATE_LIMITED | 請求過於頻繁 |
| 503 | SERVICE_UNAVAILABLE | 服務暫時不可用 |
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

**多租戶標頭**
```
X-Tenant-ID: tenant_abc
X-ACL-Groups: group1,group2
```

**可選標頭**
```
X-Request-ID: custom-request-id
```

---

## 查詢模式說明

| 模式 | 說明 | 適用場景 |
|------|------|----------|
| `auto` | 自動選擇最佳模式 | 一般查詢 |
| `local` | 從實體開始的 2-hop 圖遍歷 | 特定實體相關問題 |
| `global` | 基於社群摘要的全局檢索 | 概覽性問題 |
| `drift` | 動態多輪探索 | 複雜推理問題 |

---

## 相關文件

- [SSE 串流處理](./SSE_STREAMING.md) - 串流回應詳細說明
- [配置參考](./CONFIGURATION.md) - API 相關配置
- [安全防護](./SECURITY.md) - 輸入驗證和安全機制
- [系統架構](./chatbot_graphrag_architecture.md) - GraphRAG 架構說明
