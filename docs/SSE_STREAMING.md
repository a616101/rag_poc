# SSE 串流處理

本文件說明 Chatbot RAG 系統的 Server-Sent Events (SSE) 串流實作。

## 概述

系統使用 SSE 實現即時回應串流，讓用戶在 LLM 生成回答時能即時看到進度和內容。

### 為什麼選擇 SSE？

| 特性 | SSE | WebSocket |
|------|-----|-----------|
| 複雜度 | 低 | 高 |
| 瀏覽器支援 | 原生支援 | 原生支援 |
| 自動重連 | 是 | 需自行實作 |
| 雙向通訊 | 否 | 是 |
| 適用場景 | 伺服器推送 | 雙向即時通訊 |

對於問答系統，SSE 是理想選擇：單向（伺服器 → 客戶端）、簡單、可靠。

## 端點

### `/api/v1/rag/ask/stream`

主要問答端點，使用 Responses API backend。

```bash
curl -X POST http://localhost:8000/api/v1/rag/ask/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"question": "如何登入平台？"}'
```

### `/api/v1/rag/ask/stream_chat`

多輪對話端點，使用 Chat API backend。

```bash
curl -X POST http://localhost:8000/api/v1/rag/ask/stream_chat \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "question": "那退款流程呢？",
    "conversation_history": [...]
  }'
```

## 事件格式

所有事件使用 JSON 格式：

```
data: {"source":"ask_stream","node":"planner","phase":"status","payload":{...}}

data: {"source":"ask_stream","channel":"llm","content":"根據"}

data: {"source":"ask_stream","stage":"telemetry_summary","payload":{...}}
```

## 事件類型

### 階段事件 (Stage Events)

標識工作流程進度：

| Stage | 說明 |
|-------|------|
| `guard_start` | 輸入檢查開始 |
| `guard_end` | 輸入檢查完成 |
| `language_normalizer_start` | 語言標準化開始 |
| `language_normalizer_done` | 語言標準化完成 |
| `planner_start` | 任務規劃開始 |
| `planner_done` | 任務規劃完成 |
| `query_builder_start` | 查詢建構開始 |
| `query_builder_done` | 查詢建構完成 |
| `tool_executor_start` | 工具執行開始 |
| `tool_executor_call` | 工具呼叫 |
| `tool_executor_result` | 工具執行結果 |
| `tool_executor_done` | 工具執行完成 |
| `retrieval_checker_start` | 檢索結果檢查開始 |
| `retrieval_checker_done` | 檢索結果檢查完成 |
| `response_generating` | 開始生成回答 |
| `response_reasoning` | 推理過程（如有） |
| `response_done` | 回答生成完成 |
| `telemetry_summary` | 遙測摘要 |

### LLM 串流事件 (LLM Streaming)

回答內容的即時串流：

```json
{
  "source": "ask_stream",
  "channel": "llm",
  "content": "根據"
}
```

```json
{
  "source": "ask_stream",
  "channel": "llm",
  "content": "知識庫"
}
```

### 工具執行事件

```json
{
  "source": "ask_stream",
  "stage": "tool_executor_call",
  "payload": {
    "tool_name": "retrieve_documents",
    "args": {"query": "登入 操作步驟"}
  }
}
```

```json
{
  "source": "ask_stream",
  "stage": "tool_executor_result",
  "payload": {
    "tool_name": "retrieve_documents",
    "documents_count": 3
  }
}
```

### 遙測摘要事件

完成後返回追蹤資訊：

```json
{
  "source": "ask_stream",
  "stage": "telemetry_summary",
  "payload": {
    "trace_id": "d2d1e2ddd5ab558f8388c6d9cf510ac8",
    "intent": "simple_faq",
    "used_tools": ["retrieve_documents"],
    "user_language": "zh-hant",
    "is_out_of_scope": false,
    "prompt_versions": {
      "unified-agent-system": {"version": 3, "label": "production"}
    }
  }
}
```

## 前端整合

### JavaScript EventSource

```javascript
const eventSource = new EventSource('/api/v1/rag/ask/stream', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ question: '如何登入？' })
});

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.channel === 'llm') {
    // 即時顯示回答內容
    appendToAnswer(data.content);
  } else if (data.stage === 'tool_executor_call') {
    // 顯示工具執行狀態
    showToolStatus(data.payload.tool_name);
  } else if (data.stage === 'telemetry_summary') {
    // 保存 trace_id 供 feedback 使用
    saveTraceId(data.payload.trace_id);
  }
};

eventSource.onerror = (error) => {
  console.error('SSE Error:', error);
  eventSource.close();
};
```

### Angular HttpClient

```typescript
import { HttpClient } from '@angular/common/http';

streamQuestion(question: string): Observable<SSEEvent> {
  return new Observable(observer => {
    const eventSource = new EventSource(
      `/api/v1/rag/ask/stream?question=${encodeURIComponent(question)}`
    );
    
    eventSource.onmessage = (event) => {
      observer.next(JSON.parse(event.data));
    };
    
    eventSource.onerror = (error) => {
      observer.error(error);
      eventSource.close();
    };
    
    return () => eventSource.close();
  });
}
```

### Fetch API（支援 POST）

```javascript
async function streamAsk(question) {
  const response = await fetch('/api/v1/rag/ask/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream'
    },
    body: JSON.stringify({ question })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const text = decoder.decode(value);
    const lines = text.split('\n');
    
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        handleEvent(data);
      }
    }
  }
}
```

## 錯誤處理

### 錯誤事件

```json
{
  "source": "ask_stream",
  "stage": "error",
  "payload": {
    "error": "LLM service unavailable",
    "code": "LLM_UNAVAILABLE"
  }
}
```

### 客戶端斷線處理

後端會檢測客戶端斷線並提早終止處理：

```python
async for event in ask_stream_service.stream_events(
    request=request,
    is_disconnected=http_request.is_disconnected,
):
    yield "data: " + json.dumps(event) + "\n\n"
```

## 效能考量

### 背壓處理

系統透過 `asyncio` 自然處理背壓。若客戶端消費速度慢，伺服器會暫停生成。

### 逾時設定

建議設定合理的逾時：
- 客戶端：60 秒
- 反向代理（Nginx）：120 秒

```nginx
location /api/v1/rag/ask/stream {
    proxy_read_timeout 120s;
    proxy_buffering off;
    proxy_cache off;
}
```

### 連線數限制

SSE 連線會佔用伺服器資源，建議：
- 設定合理的 `MAX_CONNECTIONS`
- 使用連線池
- 考慮使用負載均衡

## 相關文件

- [API 參考手冊](./API_REFERENCE.md)
- [Agentic RAG 工作流程](./AGENTIC_RAG_FLOW.md)
- [部署指南](./DEPLOYMENT.md)
