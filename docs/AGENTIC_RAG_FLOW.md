# Agentic RAG 工作流程

本文件詳細說明 Chatbot RAG 系統的 Agentic RAG 架構和工作流程。

## 什麼是 Agentic RAG？

傳統 RAG 採用固定的「查詢 → 檢索 → 生成」流程，而 Agentic RAG 引入智能代理概念：

| 傳統 RAG | Agentic RAG |
|----------|-------------|
| 固定流程 | 條件分支、動態決策 |
| 單一檢索 | 多工具協作 |
| 無狀態 | 具備上下文感知 |
| 查詢即檢索 | 規劃後決策是否檢索 |

## LangGraph 計算圖

系統使用 LangGraph 實現 9 個節點的計算圖：

```
┌─────────────────────────────────────────────────────────────────┐
│                        LangGraph Workflow                        │
│                                                                   │
│  START                                                            │
│    │                                                              │
│    ▼                                                              │
│  ┌─────────┐                                                     │
│  │  guard  │ ← 輸入安全檢查                                       │
│  └────┬────┘                                                     │
│       │                                                           │
│       ▼                                                           │
│  ┌─────────────────────┐                                         │
│  │ language_normalizer │ ← 語言檢測、問題標準化                    │
│  └──────────┬──────────┘                                         │
│             │                                                     │
│             ▼                                                     │
│       ┌───────────┐                                              │
│       │  planner  │ ← 意圖分析、任務規劃                           │
│       └─────┬─────┘                                              │
│             │                                                     │
│      ┌──────┴──────┬───────────────┐                             │
│      ▼             ▼               ▼                              │
│  followup     query_builder   response_synth                      │
│  _transform        │               ▲                              │
│      │             ▼               │                              │
│      │      ┌─────────────┐        │                              │
│      │      │tool_executor│        │                              │
│      │      └──────┬──────┘        │                              │
│      │             ▼               │                              │
│      │      ┌─────────────┐        │                              │
│      │      │ doc_grader  │────────┤                              │
│      │      └─────────────┘        │                              │
│      │                             │                              │
│      └─────────────────────────────┤                              │
│                                    ▼                              │
│                             ┌─────────────┐                       │
│                             │response_synth│                      │
│                             └──────┬──────┘                       │
│                                    │                              │
│                                    ▼                              │
│                             ┌─────────────┐                       │
│                             │  telemetry  │                       │
│                             └──────┬──────┘                       │
│                                    │                              │
│                                    ▼                              │
│                                   END                             │
└─────────────────────────────────────────────────────────────────┘
```

## 節點詳細說明

### 1. Guard（輸入防護）

**檔案**：`services/ask_stream/graph/nodes/guard.py`

**功能**：
- 檢查輸入是否符合安全規範
- 預留擴展點（目前直接通過）

**輸出**：
- `pass` → 繼續到 language_normalizer
- `blocked` → 直接結束，返回錯誤訊息

### 2. Language Normalizer（語言標準化）

**檔案**：`services/ask_stream/graph/nodes/language_normalizer.py`

**功能**：
- 偵測用戶輸入語言（zh-hant, zh-hans, en 等）
- 標準化問題格式
- 處理對話歷史中的語言混用

**輸出**：
```python
{
    "user_language": "zh-hant",
    "normalized_question": "如何登入平台？",
    "latest_question": "如何登入平台？"
}
```

### 3. Planner（任務規劃）

**檔案**：`services/ask_stream/graph/nodes/planner.py`

**功能**：
- 分析用戶意圖（intent）
- 決定任務類型（task_type）
- 決定是否需要檢索（should_retrieve）
- 規劃工具呼叫（tool_calls）

**任務類型**：
| task_type | 說明 | 是否檢索 |
|-----------|------|----------|
| `simple_faq` | 簡單問答 | 是 |
| `form_download` | 表單下載 | 是 |
| `conversation_followup` | 追問 | 視情況 |
| `out_of_scope` | 超出範圍 | 否 |
| `greeting` | 問候 | 否 |

**輸出**：
```python
{
    "plan": {
        "task_type": "simple_faq",
        "should_retrieve": True,
        "tool_calls": [
            {"name": "retrieve_documents", "args": {"query": "登入平台"}}
        ]
    },
    "intent": "simple_faq"
}
```

### 4. Followup Transform（追問處理）

**檔案**：`services/ask_stream/graph/nodes/followup_transform.py`

**功能**：
- 處理多輪對話中的追問
- 結合對話歷史理解追問意圖
- 生成追問處理指令

**觸發條件**：`task_type == "conversation_followup"`

### 5. Query Builder（查詢建構）

**檔案**：`services/ask_stream/graph/nodes/query_builder.py`

**功能**：
- 使用 LLM 重寫用戶問題為更精確的檢索查詢
- 考慮對話上下文
- 支援多輪重試（loop）

**輸出**：
```python
{
    "retrieval": {
        "query": "e等公務園 登入 操作步驟",
        "loop": 1
    },
    "active_tool_calls": [...]
}
```

### 6. Tool Executor（工具執行）

**檔案**：`services/ask_stream/graph/nodes/tool_executor.py`

**功能**：
- 執行規劃的工具呼叫
- 支援多工具並行執行

**可用工具**：
| 工具名稱 | 功能 |
|----------|------|
| `retrieve_documents` | 從向量資料庫檢索文件 |
| `get_form_download_links` | 取得表單下載連結 |
| `export_form_file` | 導出表單檔案 |

**漸進式檢索策略**：
```python
RETRIEVAL_THRESHOLDS = [0.65, 0.50, 0.35]
# 先嘗試高精度 (0.65)
# 失敗則降低閾值 (0.50)
# 最後嘗試高召回 (0.35)
```

### 7. Retrieval Checker（檢索結果檢查）

**檔案**：`services/ask_stream/graph/nodes/retrieval_checker.py`

**功能**：
- 檢查檢索結果是否存在
- 決定是否需要重新檢索

**輸出**：
- `raw_chunks > 0` → 繼續到 response_synth
- `raw_chunks == 0` 且 `loop < max_loop` → 返回 query_builder 重試
- `raw_chunks == 0` 且 `loop >= max_loop` → 標記為 fallback

### 8. Response Synth（回應合成）

**檔案**：`services/ask_stream/graph/nodes/response.py`

**功能**：
- 根據檢索結果生成最終回答
- 支援串流輸出（SSE）
- 處理不同任務類型的回答策略
- 支援推理摘要（reasoning_summary）

**回答策略**：
| 情境 | 策略 |
|------|------|
| 有相關文件 | 基於文件內容回答 |
| 超出範圍 | 禮貌說明服務範圍 |
| 追問 | 結合對話歷史回答 |
| 表單下載 | 提供下載連結 |

### 9. Telemetry（遙測）

**檔案**：`services/ask_stream/graph/nodes/telemetry.py`

**功能**：
- 記錄執行軌跡到 Langfuse
- 收集評估所需數據
- 生成摘要資訊

**記錄內容**：
```python
{
    "trace_id": "abc123",
    "intent": "simple_faq",
    "used_tools": ["retrieve_documents"],
    "user_language": "zh-hant",
    "is_out_of_scope": False,
    "prompt_versions": {...}
}
```

## 路由邏輯

### route_after_guard

```python
def route_after_guard(state):
    if state.get("guard_blocked"):
        return "end"
    return "language_normalizer"
```

### route_after_planner

```python
def route_after_planner(state):
    plan = state.get("plan", {})
    task_type = plan.get("task_type", "")
    
    if task_type == "conversation_followup":
        return "followup_transform"
    elif plan.get("should_retrieve"):
        return "query_builder"
    else:
        return "response_synth"
```

### route_after_retrieval_checker

```python
def route_after_retrieval_checker(state):
    retrieval = state.get("retrieval", {})
    status = retrieval.get("status")

    if status == "retry":
        return "query_builder"  # 重試
    return "response_synth"
```

## 狀態管理

系統使用 LangGraph 的 `State` 類型管理狀態：

```python
class State(TypedDict):
    messages: List[BaseMessage]
    retry_count: int
    user_language: str
    top_k: int
    used_tools: List[str]
    retrieval: Dict[str, Any]
    intent: str
    plan: TaskPlan
    context: str
    final_answer: str
    # ... 更多欄位
```

## 執行流程範例

### 範例 1：簡單問答

```
用戶: "如何登入平台？"

1. guard: 通過
2. language_normalizer: zh-hant
3. planner: task_type=simple_faq, should_retrieve=True
4. query_builder: query="e等公務園 登入 操作步驟"
5. tool_executor: retrieve_documents → 3 篇文件
6. retrieval_checker: 3 篇相關
7. response_synth: 生成回答
8. telemetry: 記錄軌跡
```

### 範例 2：超出範圍

```
用戶: "今天天氣如何？"

1. guard: 通過
2. language_normalizer: zh-hant
3. planner: task_type=out_of_scope, should_retrieve=False
4. response_synth: "抱歉，我是 e 等公務園智能客服..."
5. telemetry: 記錄軌跡
```

### 範例 3：重試檢索

```
用戶: "XXX（模糊問題）"

1. guard: 通過
2. language_normalizer: zh-hant
3. planner: task_type=simple_faq, should_retrieve=True
4. query_builder: query="XXX 相關" (loop=1)
5. tool_executor: retrieve_documents (threshold=0.65) → 0 篇
6. retrieval_checker: 0 篇 → 重試
7. query_builder: query="XXX 說明" (loop=2)
8. tool_executor: retrieve_documents (threshold=0.50) → 2 篇
9. retrieval_checker: 2 篇相關
10. response_synth: 生成回答
11. telemetry: 記錄軌跡
```

## 相關文件

- [系統架構](./ARCHITECTURE.md)
- [SSE 串流處理](./SSE_STREAMING.md)
- [API 參考手冊](./API_REFERENCE.md)
