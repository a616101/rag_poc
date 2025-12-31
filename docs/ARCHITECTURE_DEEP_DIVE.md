# RAG Chatbot 架構深度解析

## 目錄

1. [系統架構總覽](#1-系統架構總覽)
2. [高併發請求處理機制](#2-高併發請求處理機制)
3. [請求生命週期完整流程](#3-請求生命週期完整流程)
4. [LangGraph 執行引擎](#4-langgraph-執行引擎)
5. [節點詳細解析](#5-節點詳細解析)
6. [SSE 事件串流機制](#6-sse-事件串流機制)
7. [併發控制與背壓機制](#7-併發控制與背壓機制)
8. [遙測與追蹤系統](#8-遙測與追蹤系統)

---

## 1. 系統架構總覽

### 1.1 技術堆疊

```
┌─────────────────────────────────────────────────────────────────┐
│                        前端應用程式                              │
│                    (SSE EventSource 客戶端)                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI 應用層                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ CORS 中間件 │→│ 日誌中間件   │→│ 效能監控中間件           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    API 路由層                                ││
│  │  /api/v1/rag/ask/stream      (Responses backend)            ││
│  │  /api/v1/rag/ask/stream_chat (Chat backend)                 ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Service 層 (ask_stream)                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  AskStreamService                            ││
│  │  • stream_events() / stream_events_chat()                   ││
│  │  • 建立初始狀態、管理事件流                                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                │                                 │
│                                ▼                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              LangGraph 執行引擎                              ││
│  │  • graph.astream() 非同步串流執行                           ││
│  │  • 條件路由、狀態傳遞                                       ││
│  │  • 9 個處理節點                                             ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
┌───────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│   LLM Service     │ │ Retriever       │ │ Semantic Cache      │
│  (OpenAI API)     │ │ Service         │ │ Service             │
│  • Chat           │ │ • Qdrant        │ │ • 語意相似度查詢    │
│  • Responses      │ │ • Embedding     │ │ • 快取儲存/失效     │
│  • Embedding      │ └─────────────────┘ └─────────────────────┘
└───────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Langfuse 遙測平台                           │
│  • Trace 追蹤                                                    │
│  • Prompt 管理                                                   │
│  • 評分回饋                                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 核心元件說明

| 元件 | 職責 | 關鍵特性 |
|------|------|---------|
| **FastAPI** | HTTP 伺服器 | 原生 async、SSE 支援 |
| **LangGraph** | 工作流引擎 | 狀態機、條件路由、async 節點 |
| **LangChain** | LLM 抽象層 | 統一介面、callback 支援 |
| **Qdrant** | 向量資料庫 | 語意搜尋、相似度過濾 |
| **Langfuse** | 可觀測性平台 | Trace、Prompt 管理 |

---

## 2. 高併發請求處理機制

### 2.1 Event Loop 運作原理

```
                    Python asyncio Event Loop
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│   │Request 1│  │Request 2│  │Request 3│  │Request N│  ...      │
│   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘           │
│        │            │            │            │                  │
│        ▼            ▼            ▼            ▼                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    Task Queue                            │   │
│   │  [coroutine1, coroutine2, coroutine3, ... coroutineN]   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   Event Loop                             │   │
│   │                                                          │   │
│   │   while True:                                            │   │
│   │       ready_tasks = get_ready_tasks()                   │   │
│   │       for task in ready_tasks:                          │   │
│   │           task.step()  # 執行到下一個 await              │   │
│   │           if task.waiting_for_io:                       │   │
│   │               register_io_callback(task)                │   │
│   │           elif task.done:                               │   │
│   │               handle_result(task)                       │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 非同步執行流程（優化後）

```
時間軸 ──────────────────────────────────────────────────────────▶

Request 1: ─────┬─await LLM─┬─────┬─await LLM─┬─────┬─完成
                │           │     │           │     │
Request 2: ─────┼───────────┼─────┼───────────┼─────┼─await LLM─┬─完成
                │           │     │           │     │           │
Request 3: ─────┼───────────┼─────┼───────────┼─────┼───────────┼─await─┬─完成
                │           │     │           │     │           │       │
                ▼           ▼     ▼           ▼     ▼           ▼       ▼
Event Loop:    [1]        [2,3] [1,2]       [3]  [1,2,3]      [2,3]   [3]
               執行         執行  執行        執行  執行        執行    執行

說明：
• 當 Request 1 等待 LLM 回應時，Event Loop 可以處理 Request 2, 3
• 每個 await 點都是 Event Loop 切換任務的機會
• 單執行緒可同時處理多個請求（非阻塞）
```

### 2.3 同步 vs 非同步對比

```
【同步阻塞模式（優化前）】

Request 1: ████████████████████████████████████████████ 完成
                                                       ↓
Request 2:                                             ████████████████████ 完成
                                                                           ↓
Request 3:                                                                 ████████ 完成

總時間: Request1 + Request2 + Request3 (串行)


【非同步非阻塞模式（優化後）】

Request 1: ██░░██░░██░░██████ 完成
Request 2: ░░██░░██░░██░░████████ 完成
Request 3: ░░░░██░░██░░██░░██████████ 完成

█ = CPU 執行
░ = 等待 I/O（其他請求可執行）

總時間: max(Request1, Request2, Request3) (並行)
```

---

## 3. 請求生命週期完整流程

### 3.1 完整時序圖

```
┌────────┐     ┌─────────┐     ┌───────────┐     ┌──────────┐     ┌─────────┐
│ Client │     │ FastAPI │     │ Service   │     │LangGraph │     │   LLM   │
└───┬────┘     └────┬────┘     └─────┬─────┘     └────┬─────┘     └────┬────┘
    │               │                │                │                │
    │ POST /ask/stream               │                │                │
    │──────────────>│                │                │                │
    │               │                │                │                │
    │               │ StreamingResponse               │                │
    │               │ (SSE)          │                │                │
    │               │                │                │                │
    │               │ stream_events()│                │                │
    │               │───────────────>│                │                │
    │               │                │                │                │
    │               │                │ build_ask_graph()               │
    │               │                │───────────────>│                │
    │               │                │                │                │
    │               │                │ graph.astream()│                │
    │               │                │───────────────>│                │
    │               │                │                │                │
    │               │                │                │ [guard 節點]   │
    │               │                │<───────────────│                │
    │ SSE: guard_end│<───────────────│                │                │
    │<──────────────│                │                │                │
    │               │                │                │                │
    │               │                │                │ [planner 節點] │
    │               │                │                │ await ainvoke()│
    │               │                │                │───────────────>│
    │               │                │                │<───────────────│
    │               │                │<───────────────│                │
    │ SSE: planner_done              │                │                │
    │<──────────────│<───────────────│                │                │
    │               │                │                │                │
    │               │                │                │ [query_builder]│
    │               │                │                │ await ainvoke()│
    │               │                │                │───────────────>│
    │               │                │                │<───────────────│
    │               │                │<───────────────│                │
    │ SSE: query_done│<──────────────│                │                │
    │<──────────────│                │                │                │
    │               │                │                │                │
    │               │                │                │ [tool_executor]│
    │               │                │                │ await retrieve │
    │ SSE: tool_result               │<───────────────│                │
    │<──────────────│<───────────────│                │                │
    │               │                │                │                │
    │               │                │                │ [response_synth]
    │               │                │                │ async for chunk│
    │               │                │                │───────────────>│
    │ SSE: chunk 1  │<───────────────│<───────────────│<───────────────│
    │<──────────────│                │                │                │
    │ SSE: chunk 2  │<───────────────│<───────────────│<───────────────│
    │<──────────────│                │                │                │
    │ SSE: chunk N  │<───────────────│<───────────────│<───────────────│
    │<──────────────│                │                │                │
    │               │                │                │                │
    │ SSE: meta_summary              │                │                │
    │<──────────────│<───────────────│<───────────────│                │
    │               │                │                │                │
    │ Connection Close               │                │                │
    │<──────────────│                │                │                │
    │               │                │                │                │
```

### 3.2 程式碼層級流程

```python
# 1. API 路由入口
# 檔案: src/chatbot_rag/api/rag_routes.py

@router.post("/ask/stream")
async def ask_question_stream(http_request: Request, request: QuestionRequest):
    async def event_generator():
        async for event in ask_stream_service.stream_events(
            request=request,
            is_disconnected=http_request.is_disconnected,
        ):
            yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# 2. Service 層
# 檔案: src/chatbot_rag/services/ask_stream/service.py

async def stream_events(request, is_disconnected) -> AsyncIterator[dict]:
    async for event in run_stream_graph(
        request,
        is_disconnected,
        agent_backend="responses",
    ):
        yield event


async def run_stream_graph(request, is_disconnected, *, agent_backend):
    # 2.1 建立初始狀態
    init_state = build_initial_stream_state(request)

    # 2.2 取得或編譯 Graph（有快取）
    graph = build_ask_graph(
        base_llm_params=base_llm_params,
        agent_backend=agent_backend,
        prompt_service=prompt_service,
    )

    # 2.3 建立 Langfuse Trace 上下文
    with create_trace_context(...) as trace_ctx:
        # 2.4 執行 Graph 並串流事件
        async for event in graph.astream(
            init_state,
            stream_mode="custom",
            config={"callbacks": [trace_ctx.handler]},
        ):
            # 2.5 檢查客戶端是否斷線
            if await is_disconnected():
                break

            # 2.6 處理並轉發事件
            yield event

        # 2.7 發送最終摘要
        yield summary_event


# 3. Graph 工廠
# 檔案: src/chatbot_rag/services/ask_stream/graph/factory.py

_graph_cache: Dict[str, CompiledStateGraph] = {}

def build_ask_graph(base_llm_params, *, agent_backend, prompt_service):
    # 3.1 計算快取鍵
    cache_key = _make_cache_key(base_llm_params, agent_backend)

    # 3.2 檢查快取
    if cache_key in _graph_cache:
        return _graph_cache[cache_key]

    # 3.3 編譯新 Graph
    factory = UnifiedAgentGraphFactory(...)
    compiled = factory.compile()

    # 3.4 存入快取
    _graph_cache[cache_key] = compiled
    return compiled
```

---

## 4. LangGraph 執行引擎

### 4.1 Graph 結構定義

```
                              ┌─────────────────────────────────────┐
                              │           StateGraph                │
                              │                                     │
START ──▶ guard ──▶ language_normalizer ──▶ cache_lookup           │
                                              │                     │
                    ┌─────────────────────────┴────────┐            │
                    │                                  │            │
                    ▼                                  ▼            │
            [cache_hit=True]                  [cache_hit=False]     │
                    │                                  │            │
                    ▼                                  ▼            │
            cache_response                        planner           │
                    │                                  │            │
                    │       ┌──────────────┬──────────┴────────┐   │
                    │       │              │                   │   │
                    │       ▼              ▼                   ▼   │
                    │  [followup]   [no_retrieve]        [retrieve]│
                    │       │              │                   │   │
                    │       ▼              │                   ▼   │
                    │  followup_transform  │            query_builder
                    │       │              │                   │   │
                    │       │              │                   ▼   │
                    │       │              │            tool_executor
                    │       │              │                   │   │
                    │       │              │                   ▼   │
                    │       │              │         retrieval_checker
                    │       │              │              │    │   │
                    │       │              │    ┌─────────┘    │   │
                    │       │              │    │ [retry]      │   │
                    │       │              │    ▼              │   │
                    │       │              │  query_builder    │   │
                    │       │              │    │    (loop)    │   │
                    │       │              │    └──────────────┘   │
                    │       │              │              │        │
                    │       └──────────────┴──────────────┘        │
                    │                      │                       │
                    │                      ▼                       │
                    │               response_synth                 │
                    │                      │                       │
                    └──────────────────────┤                       │
                                           ▼                       │
                                     cache_store                   │
                                           │                       │
                                           ▼                       │
                                       telemetry ──▶ END           │
                              │                                     │
                              └─────────────────────────────────────┘
```

### 4.2 條件路由邏輯

```python
# 檔案: src/chatbot_rag/services/ask_stream/graph/routing.py

def route_after_guard(state: State) -> str:
    """Guard 節點後的路由"""
    if state.get("guard_blocked"):
        return END
    return "language_normalizer"


def route_after_cache_lookup(state: State) -> str:
    """快取查詢後的路由"""
    if state.get("cache_hit"):
        return "cache_response"
    return "planner"


def route_after_planner(state: State) -> str:
    """Planner 節點後的路由"""
    task_type = state.get("intent", "")
    should_retrieve = state.get("should_retrieve", True)
    prev_answer = state.get("prev_answer_normalized", "")

    # 追問模式：有前文回答且判定為追問
    if task_type == "conversation_followup" and prev_answer:
        return "followup_transform"

    # 不需要檢索：直接生成回答
    if not should_retrieve:
        return "response_synth"

    # 需要檢索：進入查詢建構
    return "query_builder"


def route_after_retrieval_checker(state: State) -> str:
    """檢索檢查後的路由"""
    retrieval = state.get("retrieval", {})
    status = retrieval.get("status", "")

    if status == "retry":
        # 重試：回到 query_builder
        return "query_builder"

    # 成功或放棄：進入回答生成
    return "response_synth"
```

### 4.3 State 結構定義

```python
# 檔案: src/chatbot_rag/llm/__init__.py

class State(TypedDict):
    # 訊息歷史
    messages: List[BaseMessage]

    # 使用者資訊
    user_language: str                    # 偵測到的語言
    latest_question: str                  # 原始問題
    normalized_question: str              # 正規化後的問題

    # Guard 狀態
    guard_blocked: bool

    # 快取狀態
    cache_hit: bool
    cached_answer: Optional[str]

    # Planner 輸出
    plan: Dict[str, Any]                  # 任務計畫
    intent: str                           # 任務類型
    should_retrieve: bool                 # 是否需要檢索
    followup_instruction: str             # 追問指示

    # 檢索狀態
    retrieval: Dict[str, Any]
    #   - loop: int                       # 當前迴圈次數
    #   - raw_chunks: List[str]           # 原始檢索結果
    #   - status: str                     # pending/retry/relevant/fallback
    #   - query: str                      # 最終查詢
    #   - next_strategy: str              # Smart Retry 策略

    # 工具執行
    active_tool_calls: List[Dict]
    used_tools: List[str]

    # 回答生成
    context: str                          # 合併後的上下文
    final_answer: str                     # 最終回答
    conversation_summary: str             # 對話摘要

    # 評估欄位（供 Langfuse 使用）
    eval_question: str
    eval_context: str
    eval_answer: str
    eval_query_rewrite: str
```

---

## 5. 節點詳細解析

### 5.1 Guard 節點

```python
# 檔案: src/chatbot_rag/services/ask_stream/graph/nodes/guard.py

async def guard_node(state: State) -> State:
    """
    安全檢查節點

    目前為預留節點，可擴充：
    - 敏感詞過濾
    - 請求頻率限制
    - 用戶權限驗證
    """
    writer = get_stream_writer()

    # 發送開始事件
    emit_node_event(writer, node="guard", stage=GUARD_START)

    # 執行檢查（目前無實作）
    blocked = False

    # 發送結束事件
    emit_node_event(writer, node="guard", stage=GUARD_END, blocked=blocked)

    # 更新狀態
    state["guard_blocked"] = blocked
    return state
```

### 5.2 Language Normalizer 節點

```python
# 檔案: src/chatbot_rag/services/ask_stream/graph/nodes/language_normalizer.py

async def language_normalizer(state: State) -> State:
    """
    語言正規化節點

    功能：
    1. 偵測使用者語言
    2. 將問題和前文統一轉換為目標語言
    3. 降低多語言造成的語意飄移
    """
    # 偵測語言
    user_question = extract_latest_human_message(state["messages"])
    detected_language = detect_preferred_language(user_question)

    # 正規化問題（必要時翻譯）
    normalized_question, usage, _ = await _ensure_text_in_language(
        user_question,
        detected_language
    )

    # 正規化前文回答
    prev_answer = extract_last_ai_message(state["messages"])
    normalized_prev, _, _ = await _ensure_text_in_language(
        prev_answer,
        detected_language
    )

    # 更新狀態
    state["user_language"] = detected_language
    state["normalized_question"] = normalized_question
    state["prev_answer_normalized"] = normalized_prev

    return state
```

### 5.3 Planner 節點

```python
# 檔案: src/chatbot_rag/services/ask_stream/graph/nodes/planner.py

async def planner_node(state: State) -> State:
    """
    任務規劃節點

    使用 LLM 判斷：
    - task_type: 任務類型
    - should_retrieve: 是否需要檢索
    - tool_calls: 預計使用的工具
    - transform_instruction: 追問轉換指示
    """
    # 呼叫 LLM 進行任務規劃
    task_plan, usage, _ = await _llm_plan_unified_task(
        agent_llm=planner_llm,
        user_question=state["normalized_question"],
        last_ai_content=state.get("prev_answer_normalized", ""),
        user_language=state["user_language"],
    )

    # 解析結果
    task_type = task_plan["task_type"]
    # 可能值: simple_faq, form_download, form_export,
    #         conversation_followup, out_of_scope

    # 更新狀態
    state["plan"] = task_plan
    state["intent"] = task_type
    state["should_retrieve"] = task_plan["should_retrieve"]
    state["followup_instruction"] = task_plan.get("transform_instruction", "")

    return state
```

### 5.4 Query Builder 節點

```python
# 檔案: src/chatbot_rag/services/ask_stream/graph/nodes/query_builder.py

async def query_builder(state: State) -> State:
    """
    查詢建構節點

    功能：
    1. 使用 LLM 重寫檢索查詢（加入上下文）
    2. Smart Retry: 根據迴圈次數選擇變異策略
    3. 翻譯查詢為繁體中文（若非中文）
    """
    loop = state["retrieval"]["loop"] + 1

    # Smart Retry 策略
    if loop > 1:
        strategy = get_strategy_for_loop(loop)
        varied_result = await generate_variation_async(
            base_query,
            strategy,
            query_rewriter_llm,
        )

    # LLM 重寫查詢
    rewritten_query, usage, _, _ = await _rewrite_query_with_context(
        llm=query_rewriter_llm,
        seed_query=base_query,
        plan=state["plan"],
        latest_question=state["latest_question"],
        prev_answer=state.get("prev_answer_normalized", ""),
        messages=state["messages"],
    )

    # 翻譯為繁中（若需要）
    if query_lang not in ("zh-hant", "zh-hans"):
        effective_query, _, _ = await translate_query_to_zh_hant_async(
            rewritten_query
        )

    # 更新狀態
    state["retrieval"]["query"] = effective_query
    state["retrieval"]["loop"] = loop

    return state
```

### 5.5 Tool Executor 節點

```python
# 檔案: src/chatbot_rag/services/ask_stream/graph/nodes/tool_executor.py

async def tool_executor(state: State) -> State:
    """
    工具執行節點

    功能：
    1. 執行檢索工具（retrieve_documents_tool）
    2. 支援 Multi-Query Retrieval（平行執行多個查詢）
    3. RRF 合併多查詢結果
    4. 去重處理
    """
    decomposed_queries = state["retrieval"].get("decomposed_queries", [])

    # Multi-Query Retrieval
    if len(decomposed_queries) > 1:
        tasks = [
            _execute_retrieve(tool, q)
            for q in decomposed_queries[:3]
        ]
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=10.0,
        )

        # RRF 合併
        merged = _rrf_merge(all_results, k=60)
    else:
        # 單一查詢
        result = await tool.ainvoke({"query": query})

    # 去重
    raw_chunks = _deduplicate_chunks(merged)

    # 更新狀態
    state["retrieval"]["raw_chunks"] = raw_chunks
    state["used_tools"].append(tool_name)

    return state
```

### 5.6 Response Synth 節點

```python
# 檔案: src/chatbot_rag/services/ask_stream/graph/nodes/response.py

async def response_node(state: State) -> State:
    """
    回答生成節點

    功能：
    1. 根據 task_type 選擇 LLM 參數
    2. 構建回答 prompt（包含上下文和歷史）
    3. 流式生成回答（SSE）
    4. 生成對話摘要（可選）
    """
    # 動態建立 LLM
    streaming_llm = _create_streaming_llm(task_type)

    # 構建 prompt
    final_messages = [
        SystemMessage(content=system_prompt),
        SystemMessage(content=context_bundle),  # 上下文
    ]
    if conversation_summary:
        final_messages.append(SystemMessage(content=summary_context))
    if history_messages:
        final_messages.extend(history_messages)
    final_messages.append(HumanMessage(content=normalized_question))

    # 流式生成
    async for chunk in streaming_llm.astream(final_messages):
        # 解析 chunk
        delta = chunk.additional_kwargs.get("delta", "")
        channel = chunk.additional_kwargs.get("channel", "")

        # 發送 SSE 事件
        if channel == "reasoning":
            writer({"channel": "reasoning", "delta": delta})
        elif delta:
            writer({"channel": "answer", "delta": delta})
            answer_tokens.append(delta)

    # 生成對話摘要
    if summary_enabled:
        updated_summary, _, _ = await _summarize_conversation_history(
            prev_summary=state.get("conversation_summary", ""),
            latest_user=latest_question,
            latest_answer=final_answer,
        )

    # 更新狀態
    state["final_answer"] = "".join(answer_tokens)
    state["conversation_summary"] = updated_summary

    return state
```

### 5.7 Telemetry 節點

```python
# 檔案: src/chatbot_rag/services/ask_stream/graph/nodes/telemetry.py

async def telemetry_node(state: State) -> State:
    """
    遙測彙整節點

    功能：
    1. 採樣判斷（減少高流量開銷）
    2. 非同步更新 Langfuse trace
    3. 發送最終狀態事件
    """
    # 採樣判斷
    if telemetry_sampler.should_sample():
        # 非同步更新 Langfuse
        await _update_langfuse_trace_async(
            tags=["unified_agent", f"intent:{intent}"],
            metadata={
                "intent": intent,
                "is_out_of_scope": is_out_of_scope,
                "used_tools": used_tools,
            },
            output={
                "eval_question": eval_question,
                "eval_context": eval_context,
                "eval_answer": final_answer,
            },
        )
    else:
        logger.debug("[TELEMETRY] Request not sampled")

    # 發送狀態事件
    emit_node_event(writer, node="telemetry", stage=TELEMETRY_SUMMARY)

    return state


async def _update_langfuse_trace_async(tags, metadata, output):
    """使用 anyio.to_thread 包裝同步 Langfuse 呼叫"""
    def _update_trace():
        langfuse = get_langfuse_client()
        langfuse.update_current_trace(
            tags=tags,
            metadata=metadata,
            output=output,
        )

    await anyio.to_thread.run_sync(_update_trace)
```

---

## 6. SSE 事件串流機制

### 6.1 事件格式

```javascript
// SSE 事件格式
data: {"source":"ask_stream","node":"guard","channel":"status","stage":"guard_end","blocked":false}\n\n
data: {"source":"ask_stream","node":"planner","channel":"status","stage":"planner_done","intent":"simple_faq"}\n\n
data: {"source":"ask_stream","node":"response_synth","channel":"answer","delta":"根據"}\n\n
data: {"source":"ask_stream","node":"response_synth","channel":"answer","delta":"官方文檔"}\n\n
data: {"request_id":"abc123","trace_id":"xyz789","channel":"meta_summary","summary":{...}}\n\n
```

### 6.2 事件通道分類

| 通道 | 用途 | 發送時機 |
|------|------|---------|
| `status` | 狀態更新 | 每個節點開始/結束 |
| `answer` | 回答內容 | response_synth 串流 |
| `reasoning` | 推理過程 | response_synth (Responses API) |
| `meta` | Token 統計 | 各 LLM 呼叫後 |
| `meta_summary` | 最終摘要 | 流程結束時 |

### 6.3 完整事件序列範例

```javascript
// 1. Guard 節點
{"node":"guard","channel":"status","stage":"guard_start"}
{"node":"guard","channel":"status","stage":"guard_end","blocked":false}

// 2. Language Normalizer 節點
{"node":"language_normalizer","channel":"status","stage":"language_normalizer_start"}
{"node":"language_normalizer","channel":"status","stage":"language_normalizer_done","user_language":"zh-hant"}

// 3. Cache Lookup 節點
{"node":"cache_lookup","channel":"status","stage":"cache_lookup_start"}
{"node":"cache_lookup","channel":"status","stage":"cache_miss"}

// 4. Planner 節點
{"node":"planner","channel":"status","stage":"planner_start"}
{"node":"planner","channel":"status","stage":"planner_done","intent":"simple_faq","should_retrieve":true}
{"node":"planner","channel":"meta","usage":{"total_tokens":156,"input_tokens":120,"output_tokens":36}}

// 5. Query Builder 節點
{"node":"query_builder","channel":"status","stage":"query_builder_start"}
{"node":"query_builder","channel":"status","stage":"query_builder_done","query":"如何登入 e 等公務園學習平臺"}
{"node":"query_builder","channel":"meta","usage":{"total_tokens":89}}

// 6. Tool Executor 節點
{"node":"tool_executor","channel":"status","stage":"tool_executor_start"}
{"node":"tool_executor","channel":"status","stage":"tool_executor_call","tool_name":"retrieve_documents_tool"}
{"node":"tool_executor","channel":"status","stage":"tool_executor_result","tool_output":"[truncated]"}
{"node":"tool_executor","channel":"status","stage":"tool_executor_done","used_tools":["retrieve_documents_tool"],"documents_count":3}

// 7. Retrieval Checker 節點
{"node":"retrieval_checker","channel":"status","stage":"retrieval_checker_start"}
{"node":"retrieval_checker","channel":"status","stage":"retrieval_checker_done","documents_count":3}

// 8. Response Synth 節點（串流）
{"node":"response_synth","channel":"status","stage":"response_generating"}
{"node":"response_synth","channel":"answer","delta":"根據"}
{"node":"response_synth","channel":"answer","delta":"官方"}
{"node":"response_synth","channel":"answer","delta":"文檔，"}
{"node":"response_synth","channel":"answer","delta":"登入"}
{"node":"response_synth","channel":"answer","delta":"步驟"}
{"node":"response_synth","channel":"answer","delta":"如下："}
{"node":"response_synth","channel":"answer","delta":"\n\n1. "}
// ... 更多 chunks ...
{"node":"response_synth","channel":"status","stage":"response_done","loops":1}
{"node":"response_synth","channel":"meta","usage":{"total_tokens":523}}

// 9. Cache Store 節點
{"node":"cache_store","channel":"status","stage":"cache_store_done"}

// 10. Telemetry 節點
{"node":"telemetry","channel":"status","stage":"telemetry_summary"}

// 11. 最終摘要
{
  "request_id": "abc12345",
  "trace_id": "d2d1e2ddd5ab558f8388c6d9cf510ac8",
  "channel": "meta_summary",
  "summary": {
    "question": "如何登入平台",
    "intent": "simple_faq",
    "search_query": "如何登入 e 等公務園學習平臺",
    "guard_blocked": false,
    "is_out_of_scope": false,
    "agent_loops": 1,
    "agent_used_tools": ["retrieve_documents_tool"],
    "total_usage": {
      "total_tokens": 768,
      "input_tokens": 456,
      "output_tokens": 312
    },
    "trace_id": "d2d1e2ddd5ab558f8388c6d9cf510ac8"
  }
}
```

---

## 7. 併發控制與背壓機制

> ✅ **實施狀態：已完成整合**
>
> `LLMConcurrencyManager` 模組已建立並在 FastAPI lifespan 中初始化，所有 LLM 呼叫點皆已整合 Semaphore 控制。
>
> 目前狀態：
> - ✅ 模組已建立 (`src/chatbot_rag/core/concurrency.py`)
> - ✅ 已在 `main.py` lifespan 中初始化
> - ✅ **已整合**：各節點的 LLM 呼叫已包裝 `llm_concurrency.acquire()` 或 `with_llm_semaphore()`
> - ✅ **優先級排序**：可選功能，讓已完成較多 LLM 呼叫的請求優先執行

### 7.1 LLM Concurrency Manager

```python
# 檔案: src/chatbot_rag/core/concurrency.py

# 請求上下文追蹤
@dataclass
class RequestContext:
    """追蹤單一請求的 LLM 呼叫進度。"""
    request_id: str
    completed_calls: int = 0
    start_time: float = field(default_factory=time.monotonic)

    def increment(self) -> None:
        """完成一個 LLM 呼叫後增加計數。"""
        self.completed_calls += 1

# ContextVar 用於跨 async 呼叫傳遞請求上下文
request_context_var: ContextVar[Optional[RequestContext]] = ContextVar(
    "request_context", default=None
)

# 優先級佇列項目
@dataclass(order=True)
class PriorityItem:
    priority: int                           # 越小越優先
    timestamp: float = field(compare=False) # 用於飢餓防護
    request_id: str = field(compare=False)
    event: asyncio.Event = field(compare=False)


class LLMConcurrencyManager:
    """
    LLM 並發管理器

    支援兩種排程模式：
    1. FIFO 模式（預設）：先到先服務
    2. 優先級模式：已完成較多 LLM 呼叫的請求優先執行

    使用 asyncio.Semaphore 實現背壓控制：
    - 限制同時進行的 LLM 請求數
    - 防止後端 LLM 服務過載
    - 提供逾時機制
    """

    def __init__(self):
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._stats: Dict[str, Dict[str, int]] = {}
        self._priority_enabled: bool = False
        self._starvation_threshold: float = 5.0
        self._priority_queues: Dict[str, List[PriorityItem]] = {}

    def initialize(self):
        """在 FastAPI lifespan 中初始化"""
        self._semaphores = {
            "default": asyncio.Semaphore(settings.llm_max_concurrent_default),
            "chat": asyncio.Semaphore(settings.llm_max_concurrent_chat),
            "responses": asyncio.Semaphore(settings.llm_max_concurrent_responses),
            "embedding": asyncio.Semaphore(settings.llm_max_concurrent_embedding),
        }

        # 優先級排序初始化
        self._priority_enabled = settings.llm_priority_enabled
        self._starvation_threshold = settings.llm_priority_starvation_threshold

        if self._priority_enabled:
            self._priority_queues = {backend: [] for backend in self._semaphores}

    def _calculate_priority(self, completed_calls: int, wait_start: float) -> int:
        """
        計算優先級（越小越優先）

        基礎優先級 = -(completed_calls + 1)
        飢餓防護：等待超過門檻後額外提升
        """
        base_priority = -(completed_calls + 1)

        wait_time = time.monotonic() - wait_start
        if wait_time > self._starvation_threshold:
            starvation_boost = -int((wait_time - self._starvation_threshold) * 10)
            return base_priority + starvation_boost

        return base_priority

    @asynccontextmanager
    async def acquire(self, backend: str = "default", timeout: Optional[float] = None):
        """
        取得並發鎖（支援優先級排序）

        使用方式：
            async with llm_concurrency.acquire("chat"):
                result = await llm.ainvoke(messages)
        """
        ctx = request_context_var.get()

        # 如果未啟用優先級或無上下文，使用 FIFO
        if not self._priority_enabled or ctx is None:
            async with self._acquire_fifo(backend, timeout):
                yield
            return

        # 優先級模式
        async with self._acquire_priority(backend, ctx, timeout):
            yield

        # 完成後增加計數
        ctx.increment()
```

### 7.2 已整合的 LLM 呼叫點

所有節點的 LLM 呼叫已整合 Semaphore 背壓控制：

| 節點 | 檔案 | LLM 呼叫 | Backend |
|------|------|---------|---------|
| **language_normalizer** | `nodes/language_normalizer.py:53` | `converter.ainvoke()` | `responses` |
| **planner** | `nodes/planner.py:203` | `agent_llm.ainvoke()` | 動態 |
| **query_variation** | `services/query_variation.py:123` | `llm.ainvoke()` | `default` |
| **language_utils** | `services/language_utils.py:289` | `translator.ainvoke()` | 動態 |
| **query_builder** | `nodes/query_builder.py:122` | `llm.ainvoke()` | 動態 |
| **response_synth** | `nodes/response.py:130` | `summary_llm.ainvoke()` | `responses` |
| **response_synth** | `nodes/response.py:387` | `streaming_llm.astream()` | 動態 |

**整合策略：每層獨立控制**

一個 API 請求可能觸發多次 LLM 呼叫（例如 query_builder 會呼叫 query_variation 和 language_utils），
每次呼叫都獨立獲取 Semaphore。這是正確的行為：限制的是「同時進行的 LLM 請求數」而非「API 請求數」。

### 7.3 優先級排序機制

#### 7.3.1 問題背景：FIFO 的尾延遲惡化

```
Without Priority (FIFO):
Semaphore 隊列: A-1, B-1, C-1, A-2, B-2, C-2, A-3, B-3, C-3...
                 ↑ 完全不考慮屬於哪個主請求

Request A: [LLM-1: 100ms] → wait[300ms] → [LLM-2: 100ms] → wait[400ms] → [LLM-3]
Total: ~1000ms（大部分時間在等待）
```

#### 7.3.2 解決方案：Progress-Based Priority

```
優先級公式: priority = -(completed_calls + 1)

Request A 開始:  priority = -1
A 完成 LLM-1:    priority = -2（提升）
A 完成 LLM-2:    priority = -3（再提升）

With Priority:
Request A: [LLM-1: 100ms] → [LLM-2: 100ms] → [LLM-3: 100ms]
Total: ~300ms（優先級提升後減少等待時間）
```

#### 7.3.3 飢餓防護機制

```python
def _calculate_priority(self, completed_calls: int, wait_start: float) -> int:
    base_priority = -(completed_calls + 1)

    # 等待超過 starvation_threshold 秒後，每秒額外提升 10 點優先級
    wait_time = time.monotonic() - wait_start
    if wait_time > self._starvation_threshold:
        starvation_boost = -int((wait_time - self._starvation_threshold) * 10)
        return base_priority + starvation_boost

    return base_priority
```

這確保新請求不會永遠被後來的請求搶佔。

#### 7.3.4 請求上下文設定

```python
# 檔案: src/chatbot_rag/services/ask_stream/service.py

async def run_stream_graph(request, is_disconnected, *, agent_backend, ...):
    request_id = uuid.uuid4().hex[:8]

    # 建立並設定請求上下文（用於優先級排序）
    request_ctx = RequestContext(request_id=request_id)
    context_token = request_context_var.set(request_ctx)
    llm_concurrency.register_request(request_ctx)

    try:
        async for event in graph.astream(...):
            yield event
    finally:
        # 清理請求上下文
        llm_concurrency.unregister_request(request_id)
        request_context_var.reset(context_token)
```

#### 7.3.5 配置參數

```bash
# .env

# 優先級排序（可選）
LLM_PRIORITY_ENABLED=false                   # 預設關閉
LLM_PRIORITY_STARVATION_THRESHOLD=5.0        # 飢餓門檻（秒）
```

| 場景 | 建議配置 |
|------|---------|
| 低併發（< 50 QPS） | `LLM_PRIORITY_ENABLED=false`（FIFO 足夠） |
| 中併發（50-200 QPS） | `LLM_PRIORITY_ENABLED=true`, `THRESHOLD=5.0` |
| 高併發（> 200 QPS） | `LLM_PRIORITY_ENABLED=true`, `THRESHOLD=3.0` |

### 7.4 整合後的預期運作圖

```
                        LLM Concurrency Manager
                    ┌─────────────────────────────┐
                    │                             │
Request 1 ─────────▶│  ┌─────────────────────┐   │
Request 2 ─────────▶│  │   Semaphore (15)    │   │───────▶ LLM Backend
Request 3 ─────────▶│  │                     │   │         (Responses API)
    ...             │  │  [■■■■■■■■■░░░░░░]   │   │
Request N ─────────▶│  │   9/15 slots used   │   │
                    │  └─────────────────────┘   │
                    │                             │
Request N+1 ───────▶│  ⏳ 等待 slot 釋放         │
Request N+2 ───────▶│  ⏳ 等待 slot 釋放         │
                    │                             │
                    └─────────────────────────────┘

■ = 已使用的 slot
░ = 可用的 slot
⏳ = 等待中的請求
```

### 7.4 統計監控 API

系統提供 REST API 來即時監控 LLM 並發狀態：

#### 7.4.1 詳細狀態 API

```bash
GET /api/v1/admin/concurrency/status
```

返回各後端的完整狀態：

```json
{
  "chat": {
    "limit": 20,
    "available": 15,
    "in_progress": 5,
    "waiting": 3,
    "total_acquired": 1000,
    "total_released": 995,
    "total_timeout": 2
  },
  "responses": {
    "limit": 15,
    "available": 10,
    "in_progress": 5,
    "waiting": 0,
    "total_acquired": 500,
    "total_released": 495,
    "total_timeout": 0
  },
  "embedding": { ... },
  "default": { ... }
}
```

| 欄位 | 說明 |
|------|------|
| `limit` | 該後端的最大並發數（由 settings 設定） |
| `available` | 目前可用的 semaphore 槽位 |
| `in_progress` | 正在執行 LLM 呼叫的請求數 |
| `waiting` | 正在排隊等待 semaphore 的請求數 |
| `total_acquired` | 累計成功獲取 semaphore 的次數 |
| `total_released` | 累計釋放 semaphore 的次數 |
| `total_timeout` | 累計等待逾時的次數 |

#### 7.4.2 摘要 API

```bash
GET /api/v1/admin/concurrency/summary
```

返回彙總資訊：

```json
{
  "total_in_progress": 12,
  "total_waiting": 5,
  "by_backend": {
    "chat": {"in_progress": 5, "waiting": 2},
    "responses": {"in_progress": 7, "waiting": 3},
    "embedding": {"in_progress": 0, "waiting": 0},
    "default": {"in_progress": 0, "waiting": 0}
  }
}
```

#### 7.4.3 優先級狀態 API

```bash
GET /api/v1/admin/concurrency/priority
```

返回優先級排序狀態（若已啟用）：

```json
{
  "priority_enabled": true,
  "starvation_threshold": 5.0,
  "active_requests": 3,
  "queues": {
    "responses": {
      "length": 5,
      "top_priorities": [-4, -3, -2, -2, -1]
    },
    "chat": {
      "length": 2,
      "top_priorities": [-2, -1]
    }
  }
}
```

#### 7.4.4 使用範例

```bash
# 監控即時並發狀態
curl http://localhost:8000/api/v1/admin/concurrency/status

# 取得簡要摘要
curl http://localhost:8000/api/v1/admin/concurrency/summary

# 取得優先級佇列狀態
curl http://localhost:8000/api/v1/admin/concurrency/priority

# 搭配 watch 持續監控
watch -n 1 'curl -s http://localhost:8000/api/v1/admin/concurrency/summary | jq'
```

#### 7.4.5 程式碼內監控

```python
from chatbot_rag.core.concurrency import llm_concurrency

# 詳細狀態（與 API 相同）
status = llm_concurrency.get_status()

# 摘要狀態
summary = llm_concurrency.get_summary()

# 基本統計
stats = llm_concurrency.get_stats()
# {"chat": {"acquired": 1523, "released": 1520, "timeout": 3}, ...}

# 可用 slot
slots = llm_concurrency.get_available_slots()
# {"chat": 18, "responses": 13, "embedding": 30}

# 優先級佇列狀態（若啟用）
priority_stats = llm_concurrency.get_priority_stats()
# {"priority_enabled": true, "queues": {...}, ...}
```

---

## 8. 遙測與追蹤系統

### 8.1 Langfuse 整合架構

```
┌─────────────────────────────────────────────────────────────────┐
│                         Langfuse                                 │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                        Traces                             │   │
│  │                                                           │   │
│  │  Trace: ask-stream-workflow (request_id)                 │   │
│  │  ├─ input: {question, conversation_summary}              │   │
│  │  ├─ tags: [unified_agent__responses, request:abc123]     │   │
│  │  ├─ metadata: {top_k, agent_backend}                     │   │
│  │  │                                                        │   │
│  │  ├─ Span: planner                                        │   │
│  │  │   ├─ input: [SystemMessage, HumanMessage]             │   │
│  │  │   ├─ output: TaskPlanOutput                           │   │
│  │  │   └─ usage: {total: 156, input: 120, output: 36}      │   │
│  │  │                                                        │   │
│  │  ├─ Span: query_builder                                  │   │
│  │  │   └─ ...                                              │   │
│  │  │                                                        │   │
│  │  ├─ Span: tool_executor                                  │   │
│  │  │   └─ tool_calls: [retrieve_documents_tool]            │   │
│  │  │                                                        │   │
│  │  ├─ Span: response_synth                                 │   │
│  │  │   ├─ streaming: true                                  │   │
│  │  │   └─ usage: {total: 523, input: 234, output: 289}     │   │
│  │  │                                                        │   │
│  │  └─ output: {eval_question, eval_context, eval_answer}   │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                       Prompts                             │   │
│  │                                                           │   │
│  │  • planner-system (production)                           │   │
│  │  • query-rewriter-system (production)                    │   │
│  │  • unified-agent-system (production)                     │   │
│  │  • conversation-summarizer (production)                  │   │
│  │  • ...                                                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                       Scores                              │   │
│  │                                                           │   │
│  │  trace_id: d2d1e2ddd5ab558f8388c6d9cf510ac8              │   │
│  │  ├─ user_feedback: 1 (thumbs up)                         │   │
│  │  └─ comment: null                                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 採樣機制

```python
# 檔案: src/chatbot_rag/core/telemetry_sampler.py

class TelemetrySampler:
    """
    遙測採樣器

    用途：
    - 減少高流量時的 Langfuse API 呼叫
    - 降低網路開銷和延遲
    - 保持統計上的代表性
    """

    def __init__(self, sample_rate: float = 1.0):
        self._sample_rate = sample_rate  # 0.0 ~ 1.0
        self._total_checks = 0
        self._sampled_count = 0

    def should_sample(self) -> bool:
        self._total_checks += 1

        if self._sample_rate >= 1.0:
            self._sampled_count += 1
            return True

        if self._sample_rate <= 0.0:
            return False

        result = random.random() < self._sample_rate
        if result:
            self._sampled_count += 1
        return result

    def get_stats(self) -> dict:
        actual_rate = self._sampled_count / self._total_checks if self._total_checks > 0 else 0
        return {
            "sample_rate": self._sample_rate,
            "total_checks": self._total_checks,
            "sampled_count": self._sampled_count,
            "actual_rate": actual_rate,
        }
```

### 8.3 採樣配置建議

| 環境 | 採樣率 | 說明 |
|------|--------|------|
| 開發 | `1.0` | 全量記錄，便於除錯 |
| 測試 | `0.5` | 50% 採樣 |
| 生產 | `0.1` | 10% 採樣，降低開銷 |
| 高流量生產 | `0.05` | 5% 採樣 |

---

## 附錄：關鍵檔案索引

| 類別 | 檔案路徑 |
|------|---------|
| API 路由 | `src/chatbot_rag/api/rag_routes.py` |
| Service 層 | `src/chatbot_rag/services/ask_stream/service.py` |
| Graph 工廠 | `src/chatbot_rag/services/ask_stream/graph/factory.py` |
| 路由邏輯 | `src/chatbot_rag/services/ask_stream/graph/routing.py` |
| 節點定義 | `src/chatbot_rag/services/ask_stream/graph/nodes/*.py` |
| 併發控制 | `src/chatbot_rag/core/concurrency.py` |
| 遙測採樣 | `src/chatbot_rag/core/telemetry_sampler.py` |
| 配置 | `src/chatbot_rag/core/config.py` |
| LLM 模型 | `src/chatbot_rag/llm/responses_chat_model.py` |

---

*文件版本: 2.0*
*最後更新: 2025-12-11*
