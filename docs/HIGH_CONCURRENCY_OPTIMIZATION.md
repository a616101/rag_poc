# RAG Chatbot 高併發優化技術文件

## 目錄

1. [原始問題分析](#1-原始問題分析)
2. [優化策略總覽](#2-優化策略總覽)
3. [已完成的改造](#3-已完成的改造)
4. [尚待優化的項目](#4-尚待優化的項目)
5. [配置參數說明](#5-配置參數說明)
6. [效能監控建議](#6-效能監控建議)
7. [優化前後對比](#7-優化前後對比)

---

## 1. 原始問題分析

### 1.1 問題背景

在高併發場景下（約 100 個同時使用者），系統出現嚴重的延遲惡化。主要 API 端點：

- `/api/v1/rag/ask/stream` - Responses backend
- `/api/v1/rag/ask/stream_chat` - Chat backend

### 1.2 根本原因

**Event Loop 阻塞**：Python asyncio 使用單執行緒事件迴圈，當程式碼中存在同步阻塞呼叫時，整個 event loop 會停止處理其他請求。

原本的問題點：

| 類型 | 問題 | 影響 |
|------|------|------|
| LLM 呼叫 | 使用 `.invoke()` 同步方法 | 每次 LLM 呼叫阻塞約 1-3 秒 |
| Graph 串流 | 使用 `graph.stream()` 同步迭代 | 整個流程無法並發 |
| Langfuse 遙測 | 直接呼叫同步 SDK | 每次 trace 更新阻塞約 100-300ms |
| Graph 編譯 | 每次請求重新編譯 | 額外開銷約 50-100ms |

### 1.3 預期效果

- 延遲曲線：從「線性/超線性惡化」→「可預期的排隊模型」
- 單 worker 處理更多並發連線
- 降低記憶體使用（可減少 worker 數量）

---

## 2. 優化策略總覽

### 2.1 五階段優化計畫

```
Phase 1: 核心 Async 改造（最高優先級）
├── 1.1 LangGraph Streaming → graph.astream
├── 1.2 節點層級 LLM 呼叫 → ainvoke/astream
└── 1.3 ResponsesChatModel 新增 async 方法

Phase 2: Graph 編譯快取
└── 模組層級快取，避免重複編譯開銷

Phase 3: LLM Semaphore 背壓控制
└── 限制並發 LLM 請求數，防止後端過載

Phase 4: Langfuse 完整 Async 改造 + Sampling
├── 使用 anyio.to_thread 包裝同步呼叫
└── 採樣機制減少遙測開銷

Phase 5: Service 初始化改造
└── FastAPI lifespan 統一初始化
```

### 2.2 架構變更示意

```
之前（同步阻塞）：
Request → [sync graph.stream] → [sync LLM.invoke] → [sync Langfuse] → Response
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                          全部阻塞 event loop

之後（全非同步）：
Request → [async graph.astream] → [async LLM.ainvoke] → [async Langfuse] → Response
         ↓                        ↓                      ↓
         event loop 可處理其他請求 ← ← ← ← ← ← ← ← ← ← ← ←
```

---

## 3. 已完成的改造

### 3.1 Phase 1: 核心 Async 改造

#### 3.1.1 LangGraph Streaming

**檔案**: `src/chatbot_rag/services/ask_stream/service.py`

```python
# 之前
for event in graph.stream(init_state, stream_mode="custom", config=...):
    ...

# 之後
async for event in graph.astream(init_state, stream_mode="custom", config=...):
    ...
```

#### 3.1.2 節點層級 async 改造

所有 9 個 Graph 節點都已改為 async：

| 節點 | 檔案 | LLM 呼叫方式 |
|------|------|-------------|
| `guard` | `graph/nodes/guard.py` | 無 LLM |
| `language_normalizer` | `graph/nodes/language_normalizer.py` | `await converter.ainvoke()` |
| `planner` | `graph/nodes/planner.py` | `await agent_llm.ainvoke()` |
| `query_builder` | `graph/nodes/query_builder.py` | `await llm.ainvoke()` |
| `tool_executor` | `graph/nodes/tool_executor.py` | `await tool.ainvoke()` + `asyncio.gather()` |
| `retrieval_checker` | `graph/nodes/retrieval_checker.py` | 無 LLM |
| `followup_transform` | `graph/nodes/followup_transform.py` | 無 LLM |
| `response` | `graph/nodes/response.py` | `async for chunk in llm.astream()` |
| `telemetry` | `graph/nodes/telemetry.py` | `await anyio.to_thread.run_sync()` |

#### 3.1.3 ResponsesChatModel async 方法

**檔案**: `src/chatbot_rag/llm/responses_chat_model.py`

新增方法：
- `_agenerate()` - 非同步生成完整回應
- `_astream()` - 非同步串流生成

```python
@property
def async_client(self) -> AsyncOpenAI:
    if self._async_client is None:
        self._async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
    return self._async_client

async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
    response = await self.async_client.responses.create(...)
    ...

async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
    async for event in await self.async_client.responses.create(stream=True, ...):
        ...
```

### 3.2 Phase 2: Graph 編譯快取

**檔案**: `src/chatbot_rag/services/ask_stream/graph/factory.py`

```python
_graph_cache: Dict[str, CompiledStateGraph] = {}

def _make_cache_key(base_llm_params: dict, agent_backend: str) -> str:
    params_str = json.dumps(base_llm_params, sort_keys=True, default=str)
    return hashlib.md5(f"{params_str}:{agent_backend}".encode()).hexdigest()

def build_ask_graph(base_llm_params, *, agent_backend="chat", prompt_service=None):
    cache_key = _make_cache_key(base_llm_params, agent_backend)
    if cache_key in _graph_cache:
        logger.debug("[GRAPH] Cache hit for key=%s", cache_key[:8])
        return _graph_cache[cache_key]

    # ... 編譯 graph ...
    _graph_cache[cache_key] = compiled
    return compiled

def clear_graph_cache():
    """供測試或設定重載使用"""
    _graph_cache.clear()

def get_graph_cache_stats() -> dict:
    """取得快取統計"""
    return {"cache_size": len(_graph_cache), "cache_keys": list(_graph_cache.keys())}
```

### 3.3 Phase 3: LLM Semaphore 背壓控制 + 優先級排序 ✅

> ✅ **實施狀態：已完成整合**
>
> `LLMConcurrencyManager` 已建立並在 lifespan 中初始化，支援兩種排程模式：
> 1. **FIFO 模式**（預設）：先到先服務
> 2. **優先級模式**：已完成較多 LLM 呼叫的請求優先執行

**檔案**: `src/chatbot_rag/core/concurrency.py`

#### 3.3.1 核心架構

```python
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
```

#### 3.3.2 LLMConcurrencyManager

```python
class LLMConcurrencyManager:
    def __init__(self):
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._priority_enabled: bool = False
        self._starvation_threshold: float = 5.0
        self._priority_queues: Dict[str, List[PriorityItem]] = {}

    def initialize(self):
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
        """計算優先級（越小越優先）"""
        base_priority = -(completed_calls + 1)

        # 飢餓防護：等待超過門檻後額外提升
        wait_time = time.monotonic() - wait_start
        if wait_time > self._starvation_threshold:
            starvation_boost = -int((wait_time - self._starvation_threshold) * 10)
            return base_priority + starvation_boost

        return base_priority

    @asynccontextmanager
    async def acquire(self, backend: str = "default", timeout: Optional[float] = None):
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

llm_concurrency = LLMConcurrencyManager()

# Helper 函數，簡化整合（API 不變）
async def with_llm_semaphore(coro_func, backend="default"):
    async with llm_concurrency.acquire(backend):
        return await coro_func()
```

#### 3.3.3 優先級排序機制詳解

**問題：FIFO 的尾延遲惡化**

```
Without Priority (FIFO):
Semaphore 隊列: A-1, B-1, C-1, A-2, B-2, C-2, A-3, B-3, C-3...
                 ↑ 完全不考慮屬於哪個主請求

Request A: [LLM-1: 100ms] → wait[300ms] → [LLM-2: 100ms] → wait[400ms] → [LLM-3]
Total: ~1000ms（大部分時間在等待）
```

**解決方案：Progress-Based Priority**

```
With Priority:
優先級 = -(completed_calls + 1)

Request A 開始：priority = -1
A-1 完成後：   priority = -2（提升）
A-2 完成後：   priority = -3（再提升）

Request A: [LLM-1: 100ms] → [LLM-2: 100ms] → [LLM-3: 100ms]
Total: ~300ms（優先級提升後減少等待時間）
```

**飢餓防護機制**

```python
# 等待超過 starvation_threshold 秒後，每秒額外提升 10 點優先級
if wait_time > self._starvation_threshold:
    starvation_boost = -int((wait_time - self._starvation_threshold) * 10)
    return base_priority + starvation_boost
```

這確保新請求不會永遠被後來的請求搶佔。

#### 3.3.4 請求上下文設定

**檔案**: `src/chatbot_rag/services/ask_stream/service.py`

```python
async def run_stream_graph(request, is_disconnected, *, agent_backend, ...):
    request_id = uuid.uuid4().hex[:8]

    # 建立並設定請求上下文（用於優先級排序）
    request_ctx = RequestContext(request_id=request_id)
    context_token = request_context_var.set(request_ctx)
    llm_concurrency.register_request(request_ctx)

    try:
        # ... graph 執行邏輯 ...
        async for event in graph.astream(...):
            yield event
    finally:
        # 清理請求上下文
        llm_concurrency.unregister_request(request_id)
        request_context_var.reset(context_token)
```

#### 3.3.5 已整合的 LLM 呼叫點

| 節點 | 檔案 | LLM 呼叫 | Backend |
|------|------|---------|---------|
| language_normalizer | `nodes/language_normalizer.py:53` | `converter.ainvoke()` | `responses` |
| planner | `nodes/planner.py:203` | `agent_llm.ainvoke()` | 動態 |
| query_variation | `services/query_variation.py:123` | `llm.ainvoke()` | `default` |
| language_utils | `services/language_utils.py:289` | `translator.ainvoke()` | 動態 |
| query_builder | `nodes/query_builder.py:122` | `llm.ainvoke()` | 動態 |
| response_synth | `nodes/response.py:130` | `summary_llm.ainvoke()` | `responses` |
| response_synth | `nodes/response.py:387` | `streaming_llm.astream()` | 動態 |

**整合策略：每層獨立控制**

一個 API 請求可能觸發多次 LLM 呼叫（例如 query_builder 會呼叫 query_variation 和 language_utils），
每次呼叫都獨立獲取 Semaphore。優先級排序會讓已完成較多 LLM 呼叫的請求優先取得下一個 slot。

#### 3.3.6 配置參數

**檔案**: `src/chatbot_rag/core/config.py`

```python
# LLM 並發控制設定
llm_max_concurrent_default: int = 10   # 預設最大並發
llm_max_concurrent_chat: int = 20      # Chat 後端
llm_max_concurrent_responses: int = 100 # Responses 後端
llm_max_concurrent_embedding: int = 30 # Embedding
llm_request_timeout: float = 60.0      # 請求逾時

# LLM 優先級排序設定
llm_priority_enabled: bool = False      # 預設關閉
llm_priority_starvation_threshold: float = 5.0  # 飢餓門檻（秒）
```

### 3.4 Phase 4: Langfuse Async + Sampling

#### 3.4.1 遙測採樣器

**新檔案**: `src/chatbot_rag/core/telemetry_sampler.py`

```python
class TelemetrySampler:
    def __init__(self, sample_rate: float = 1.0):
        self._sample_rate = max(0.0, min(1.0, sample_rate))

    def should_sample(self) -> bool:
        if self._sample_rate >= 1.0:
            return True
        if self._sample_rate <= 0.0:
            return False
        return random.random() < self._sample_rate

telemetry_sampler = TelemetrySampler(sample_rate=1.0)

def initialize_telemetry_sampler() -> None:
    """從 settings 初始化採樣率"""
    telemetry_sampler.sample_rate = settings.langfuse_sample_rate
```

**配置**: `src/chatbot_rag/core/config.py`

```python
langfuse_sample_rate: float = 1.0  # 0.0~1.0，1.0=100%
```

#### 3.4.2 Telemetry 節點 async 包裝

**檔案**: `src/chatbot_rag/services/ask_stream/graph/nodes/telemetry.py`

```python
async def _update_langfuse_trace_async(tags, metadata, output) -> None:
    def _update_trace():
        langfuse = get_langfuse_client()
        langfuse.update_current_trace(tags=tags, metadata=metadata, output=output)

    await anyio.to_thread.run_sync(_update_trace)

async def telemetry_node(state: State) -> State:
    if telemetry_sampler.should_sample():
        await _update_langfuse_trace_async(tags, metadata, output)
    else:
        logger.debug("[TELEMETRY] Request not sampled, skipping Langfuse update")
    ...
```

### 3.5 Phase 5: Service 初始化改造

**檔案**: `src/chatbot_rag/main.py`

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # LLM 並發控制初始化（必須在 event loop 中）
    llm_concurrency.initialize()
    logger.info("LLM concurrency manager initialized")

    # 遙測採樣器初始化
    initialize_telemetry_sampler()

    # Langfuse Prompt 初始化
    if settings.langfuse_prompt_enabled:
        # ...

    yield

    # 關閉清理
    logger.info(f"Shutting down {settings.app_name}")
```

---

## 4. 尚待優化的項目

以下是經過全面檢查後發現的潛在阻塞點，建議未來進行優化：

### 4.1 高優先級

| 位置 | 問題 | 建議 |
|------|------|------|
| `graph/nodes/cache_response.py:73` | `time.sleep()` 阻塞 | 改用 `await asyncio.sleep()` |
| `api/admin_routes.py:982` | LLM Judge 使用 `.invoke()` | 改用 `.ainvoke()` |

### 4.2 中優先級

| 位置 | 問題 | 建議 |
|------|------|------|
| `services/contextual_chunking_service.py:127` | 批量處理中使用同步 LLM | 改用 `asyncio.gather()` + `.ainvoke()` |
| `api/admin_routes.py` (多行) | Langfuse 同步 API | 使用 `anyio.to_thread` 包裝 |
| `services/document_service.py:82,224` | 同步檔案 I/O | 使用 `aiofiles` |
| `services/form_export_service.py:162` | 同步 CSV 寫入 | 使用 `aiofiles` |

### 4.3 低優先級

| 位置 | 問題 | 建議 |
|------|------|------|
| `services/query_variation.py:176` | 保留同步版本 | 確保 async 路由不使用 |
| `services/language_utils.py:236` | 保留同步版本 | 確保 async 路由不使用 |
| `services/prompt_service.py:193` | Langfuse 同步呼叫 | 考慮 async 包裝 |
| `services/feedback_service.py:57` | Langfuse 同步呼叫 | 使用 `anyio.to_thread` |

---

## 5. 配置參數說明

### 5.1 vLLM 後端配置對照

本系統使用 vLLM 作為 LLM 推論後端。vLLM 的 `--max-num-seqs` 參數決定了同時處理的最大序列數，
本系統的 Semaphore 配置應與此參數匹配，以達到最佳背壓控制效果。

#### 5.1.1 GPU 硬體與 max-num-seqs 對照

| GPU | VRAM | 建議 max-num-seqs | 適用場景 |
|-----|------|------------------|---------|
| **RTX 6000 Pro** | 48GB | 16-20 | 開發/小規模部署 |
| **A100 40GB** | 40GB | 20-32 | 中規模生產環境 |
| **A100 80GB** | 80GB | 32-50 | 大規模生產環境 |
| **H100 80GB** | 80GB | 50-100 | 高併發生產環境 |

#### 5.1.2 Semaphore 配置與 vLLM 對應

**原則：Semaphore 限制應略低於或等於 vLLM max-num-seqs**

這樣可以：
1. 防止請求堆積在 vLLM 佇列中（增加延遲）
2. 讓背壓在應用層可控（有明確的 timeout 和監控）
3. 避免 vLLM OOM 或過載

### 5.2 .env 配置範例

#### 5.2.1 RTX 6000 Pro 配置（max-num-seqs=16）

```bash
# ==================== LLM 並發控制 ====================
LLM_MAX_CONCURRENT_DEFAULT=8     # 預設最大並發數
LLM_MAX_CONCURRENT_CHAT=16       # Chat API（對應 vLLM max-num-seqs）
LLM_MAX_CONCURRENT_RESPONSES=16  # Responses API
LLM_MAX_CONCURRENT_EMBEDDING=20  # Embedding（通常獨立服務）
LLM_REQUEST_TIMEOUT=60.0         # LLM 請求逾時（秒）

# ==================== LLM 優先級排序 ====================
LLM_PRIORITY_ENABLED=false       # 低併發場景不需要優先級排序
LLM_PRIORITY_STARVATION_THRESHOLD=5.0

# ==================== Langfuse 遙測 ====================
LANGFUSE_SAMPLE_RATE=1.0         # 開發環境全量記錄
```

#### 5.2.2 A100 40GB 配置（max-num-seqs=32）

```bash
# ==================== LLM 並發控制 ====================
LLM_MAX_CONCURRENT_DEFAULT=16    # 預設最大並發數
LLM_MAX_CONCURRENT_CHAT=32       # Chat API（對應 vLLM max-num-seqs）
LLM_MAX_CONCURRENT_RESPONSES=32  # Responses API
LLM_MAX_CONCURRENT_EMBEDDING=30  # Embedding
LLM_REQUEST_TIMEOUT=60.0

# ==================== LLM 優先級排序 ====================
LLM_PRIORITY_ENABLED=true        # 中併發建議啟用
LLM_PRIORITY_STARVATION_THRESHOLD=5.0

# ==================== Langfuse 遙測 ====================
LANGFUSE_SAMPLE_RATE=0.3         # 生產環境 30% 採樣
```

#### 5.2.3 A100 80GB 配置（max-num-seqs=50）

```bash
# ==================== LLM 並發控制 ====================
LLM_MAX_CONCURRENT_DEFAULT=25    # 預設最大並發數
LLM_MAX_CONCURRENT_CHAT=50       # Chat API（對應 vLLM max-num-seqs）
LLM_MAX_CONCURRENT_RESPONSES=50  # Responses API
LLM_MAX_CONCURRENT_EMBEDDING=40  # Embedding
LLM_REQUEST_TIMEOUT=60.0

# ==================== LLM 優先級排序 ====================
LLM_PRIORITY_ENABLED=true
LLM_PRIORITY_STARVATION_THRESHOLD=4.0  # 高併發可降低門檻

# ==================== Langfuse 遙測 ====================
LANGFUSE_SAMPLE_RATE=0.1         # 高流量 10% 採樣
```

#### 5.2.4 H100 配置（max-num-seqs=100）

```bash
# ==================== LLM 並發控制 ====================
LLM_MAX_CONCURRENT_DEFAULT=50    # 預設最大並發數
LLM_MAX_CONCURRENT_CHAT=100      # Chat API（對應 vLLM max-num-seqs）
LLM_MAX_CONCURRENT_RESPONSES=100 # Responses API
LLM_MAX_CONCURRENT_EMBEDDING=60  # Embedding
LLM_REQUEST_TIMEOUT=45.0         # H100 速度快，可縮短逾時

# ==================== LLM 優先級排序 ====================
LLM_PRIORITY_ENABLED=true
LLM_PRIORITY_STARVATION_THRESHOLD=3.0  # 高併發降低門檻

# ==================== Langfuse 遙測 ====================
LANGFUSE_SAMPLE_RATE=0.05        # 超高流量 5% 採樣
```

### 5.3 調參建議

#### 5.3.1 Semaphore 與 vLLM 配置關係

```
┌─────────────────────────────────────────────────────────────────┐
│                    請求流量控制示意                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  API 請求 ──▶ Semaphore (N) ──▶ vLLM (max-num-seqs=M)          │
│                   │                      │                       │
│                   ▼                      ▼                       │
│            應用層背壓控制           GPU 推論層控制               │
│            (可監控/timeout)         (固定佇列)                   │
│                                                                  │
│  建議關係：N ≤ M                                                 │
│  • N = M：充分利用 vLLM 容量                                     │
│  • N < M：保留餘裕，防止突發流量                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 5.3.2 遙測採樣率

| 場景 | 建議配置 | 說明 |
|------|---------|------|
| 開發環境 | `LANGFUSE_SAMPLE_RATE=1.0` | 全量記錄，便於除錯 |
| 測試環境 | `LANGFUSE_SAMPLE_RATE=0.5` | 50% 採樣 |
| 生產環境（< 100 QPS） | `LANGFUSE_SAMPLE_RATE=0.3` | 30% 採樣 |
| 生產環境（100-500 QPS） | `LANGFUSE_SAMPLE_RATE=0.1` | 10% 採樣 |
| 高流量生產（> 500 QPS） | `LANGFUSE_SAMPLE_RATE=0.05` | 5% 採樣 |

#### 5.3.3 優先級排序

| vLLM max-num-seqs | 建議配置 | 說明 |
|-------------------|---------|------|
| 16 | `PRIORITY_ENABLED=false` | 低併發，FIFO 足夠 |
| 20-32 | `PRIORITY_ENABLED=true`, `THRESHOLD=5.0` | 中併發，標準門檻 |
| 50 | `PRIORITY_ENABLED=true`, `THRESHOLD=4.0` | 高併發，略降門檻 |
| 100 | `PRIORITY_ENABLED=true`, `THRESHOLD=3.0` | 超高併發，快速響應 |

**注意事項：**
- 優先級排序會增加少量記憶體開銷（追蹤請求上下文）
- heapq 操作為 O(log n)，對效能影響極小
- 飢餓門檻設太低可能導致優先級排序失效（所有請求都被提升）
- 建議先在測試環境驗證配置，觀察 `/api/v1/admin/concurrency/status` 監控數據

---

## 6. 效能監控建議

### 6.1 並發監控 API

系統提供 REST API 來即時監控 LLM 並發狀態：

#### 6.1.1 詳細狀態 API

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

#### 6.1.2 摘要 API

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

#### 6.1.3 使用範例

```bash
# 監控即時並發狀態
curl http://localhost:8000/api/v1/admin/concurrency/status

# 取得簡要摘要
curl http://localhost:8000/api/v1/admin/concurrency/summary

# 搭配 watch 持續監控
watch -n 1 'curl -s http://localhost:8000/api/v1/admin/concurrency/summary | jq'
```

### 6.2 程式碼內監控

除了 API 外，也可以在程式碼中直接呼叫監控方法：

```python
from chatbot_rag.core.concurrency import llm_concurrency

# 詳細狀態（與 API 相同）
status = llm_concurrency.get_status()

# 摘要狀態
summary = llm_concurrency.get_summary()

# 基本統計
stats = llm_concurrency.get_stats()
# 返回: {"chat": {"acquired": N, "released": N, "timeout": N}, ...}

# 可用槽位
slots = llm_concurrency.get_available_slots()
# 返回: {"chat": 15, "responses": 10, ...}

# Graph 快取統計
from chatbot_rag.services.ask_stream.graph.factory import get_graph_cache_stats
cache_stats = get_graph_cache_stats()
# 返回: {"cache_size": 2, "cache_keys": ["abc123", "def456"]}

# 遙測採樣統計
from chatbot_rag.core.telemetry_sampler import telemetry_sampler
sampling_stats = telemetry_sampler.get_stats()
# 返回: {"sample_rate": 0.1, "total_checks": 1000, "sampled_count": 98, "actual_rate": 0.098}
```

### 6.3 負載測試建議

```bash
# 使用 locust 或 k6 進行負載測試
# 關注指標：
# - P50/P95/P99 延遲
# - 錯誤率
# - 並發連線數 vs 延遲曲線
```

### 6.4 日誌關鍵字

```bash
# 監控這些日誌訊息：
grep -E "\[CONCURRENCY\]|\[TELEMETRY\]|\[GRAPH\]" logs/app.log

# 重要訊息：
# [CONCURRENCY] Timeout acquiring semaphore  # 並發瓶頸
# [TELEMETRY] Request not sampled           # 採樣跳過
# [GRAPH] Cache hit                         # 快取命中
```

---

## 附錄：改動檔案清單

### 新增檔案

| 檔案 | 用途 |
|------|------|
| `src/chatbot_rag/core/concurrency.py` | LLM 並發管理器 |
| `src/chatbot_rag/core/telemetry_sampler.py` | 遙測採樣器 |
| `docs/HIGH_CONCURRENCY_OPTIMIZATION.md` | 本文件 |

### 修改檔案

| 檔案 | 變更類型 |
|------|---------|
| `src/chatbot_rag/core/config.py` | 新增配置項 |
| `src/chatbot_rag/main.py` | lifespan 初始化 |
| `src/chatbot_rag/services/ask_stream/service.py` | graph.astream |
| `src/chatbot_rag/services/ask_stream/graph/factory.py` | Graph 快取 |
| `src/chatbot_rag/services/ask_stream/graph/nodes/*.py` | 全部改為 async |
| `src/chatbot_rag/llm/responses_chat_model.py` | async 方法 |
| `src/chatbot_rag/services/language_utils.py` | async 版本函數 |
| `src/chatbot_rag/services/query_variation.py` | async 版本函數 |

---

## 7. 優化前後對比

本節詳細對比優化前後的架構差異，幫助理解各項改造的效果。

### 7.1 整體架構對比

#### 優化前：同步阻塞架構

```
┌─────────────────────────────────────────────────────────────────┐
│                      原始架構（優化前）                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Request ──▶ graph.stream() ──▶ LLM.invoke() ──▶ Langfuse       │
│                   │                  │               │           │
│                   ▼                  ▼               ▼           │
│              [阻塞 event loop]  [阻塞 1-3秒]   [阻塞 100-300ms]  │
│                                                                  │
│  問題：                                                          │
│  • 單一請求佔用整個 event loop                                   │
│  • 無法並發處理多個請求                                          │
│  • 延遲隨併發數線性增長                                          │
│  • 每次請求重新編譯 Graph（50-100ms 開銷）                       │
│  • 無 LLM 並發控制（可能壓垮後端）                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 優化後：全非同步架構

```
┌─────────────────────────────────────────────────────────────────┐
│                      優化後架構                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Request ──▶ graph.astream() ──▶ LLM.ainvoke() ──▶ async Langfuse
│                   │                  │               │           │
│                   ▼                  ▼               ▼           │
│              [await 釋放控制權] [await + Semaphore] [anyio.to_thread]
│                   │                  │               │           │
│                   └──────────────────┴───────────────┘           │
│                              │                                   │
│                              ▼                                   │
│                    Event Loop 可處理其他請求                     │
│                                                                  │
│  優點：                                                          │
│  • 單執行緒可處理數百個並發請求                                  │
│  • Graph 編譯快取（避免重複編譯）                                │
│  • LLM Semaphore 背壓控制（防止後端過載）                        │
│  • 優先級排序（減少尾延遲）                                      │
│  • 遙測採樣（降低 Langfuse 開銷）                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 LLM 呼叫方式對比

#### 優化前：同步呼叫

```python
# ❌ 同步方式（阻塞 event loop）

# planner 節點
def planner_node(state: State) -> State:
    result = agent_llm.invoke(messages)  # 阻塞 1-3 秒
    return state

# response 節點
def response_node(state: State) -> State:
    for chunk in llm.stream(messages):   # 同步迭代器，阻塞
        yield chunk
    return state
```

#### 優化後：非同步呼叫 + Semaphore 控制

```python
# ✅ 非同步方式 + 並發控制

# planner 節點
async def planner_node(state: State) -> State:
    async with llm_concurrency.acquire("responses"):  # 背壓控制
        result = await agent_llm.ainvoke(messages)    # 非阻塞
    return state

# response 節點
async def response_node(state: State) -> State:
    async with llm_concurrency.acquire(backend):
        async for chunk in llm.astream(messages):     # 非同步串流
            yield chunk
    return state
```

### 7.3 並發排程對比

#### 優化前：無控制的並發

```
100 個同時請求 ──────────────────▶ LLM 後端
                                      │
                                      ▼
                                  [後端過載]
                                      │
                                      ▼
                              所有請求延遲惡化
                              可能 timeout 或 crash
```

#### 優化後：Semaphore 背壓 + 優先級排序

```
                                    LLM Concurrency Manager
                                ┌─────────────────────────────┐
100 個同時請求                  │                             │
    │                           │  ┌─────────────────────┐   │
    ├─▶ Request 1 ─────────────▶│  │  Semaphore (15)     │   │───▶ LLM 後端
    ├─▶ Request 2 ─────────────▶│  │                     │   │    (受保護)
    ├─▶ Request 3 ─────────────▶│  │  [■■■■■■■■■░░░░░░]  │   │
    │   ...                     │  │   9/15 slots used   │   │
    ├─▶ Request 15 ────────────▶│  └─────────────────────┘   │
    │                           │                             │
    ├─▶ Request 16 ────────────▶│  ⏳ Priority Queue:        │
    ├─▶ Request 17 ────────────▶│     [-3] Request 5-LLM3   │◀─ 優先
    │   ...                     │     [-2] Request 8-LLM2   │
    └─▶ Request 100 ───────────▶│     [-1] Request 16-LLM1  │
                                │                             │
                                └─────────────────────────────┘
```

### 7.4 FIFO vs 優先級排序

#### FIFO 模式（預設）

```
Request A: [LLM-1] ─────wait─────[LLM-2] ─────wait─────[LLM-3] ─▶ 完成
Request B: ──[LLM-1] ────wait─────[LLM-2] ────wait─────[LLM-3] ─▶ 完成
Request C: ────[LLM-1] ───wait─────[LLM-2] ───wait─────[LLM-3] ─▶ 完成

Semaphore 佇列: A1, B1, C1, A2, B2, C2, A3, B3, C3
                 ↑ 完全不考慮請求進度

問題：
• Request A 完成 LLM-1 後，LLM-2 排在 B1, C1 後面
• 已投入資源的請求沒有優先權
• 尾延遲（P99）惡化
```

#### 優先級模式（可選）

```
優先級公式: priority = -(completed_calls + 1)

Request A 開始:  priority = -1
A 完成 LLM-1:    priority = -2（提升）
A 完成 LLM-2:    priority = -3（再提升）

執行順序:
Request A: [LLM-1] ──[LLM-2] ──[LLM-3] ─▶ 完成（最快）
Request B: ─────────[LLM-1] ──[LLM-2] ──[LLM-3] ─▶ 完成
Request C: ──────────────────[LLM-1] ──[LLM-2] ──[LLM-3] ─▶ 完成

效果：
• Request A 的後續 LLM 呼叫優先執行
• 單一請求的總延遲降低（但新請求等待時間略增）
• 尾延遲（P99）大幅改善

飢餓防護：
• 等待超過 5 秒的請求自動提升優先級
• 每秒額外加 10 點優先級
• 確保新請求不會永遠被搶佔
```

### 7.5 Graph 編譯快取對比

#### 優化前：每次請求重新編譯

```python
# ❌ 每次請求都編譯 Graph

async def run_stream_graph(request):
    # 每次都重新建立工廠並編譯
    factory = UnifiedAgentGraphFactory(...)
    graph = factory.compile()  # 50-100ms 開銷

    async for event in graph.astream(state):
        yield event
```

#### 優化後：模組層級快取

```python
# ✅ Graph 編譯結果快取

_graph_cache: Dict[str, CompiledStateGraph] = {}

def build_ask_graph(base_llm_params, *, agent_backend):
    cache_key = _make_cache_key(base_llm_params, agent_backend)

    if cache_key in _graph_cache:
        return _graph_cache[cache_key]  # 命中快取，0ms

    # 首次編譯
    factory = UnifiedAgentGraphFactory(...)
    compiled = factory.compile()
    _graph_cache[cache_key] = compiled
    return compiled
```

### 7.6 遙測開銷對比

#### 優化前：全量同步遙測

```python
# ❌ 每次請求都同步更新 Langfuse

def telemetry_node(state: State) -> State:
    langfuse.update_current_trace(...)  # 同步呼叫，阻塞 100-300ms
    return state
```

#### 優化後：採樣 + 非同步

```python
# ✅ 採樣 + anyio.to_thread 非同步

async def telemetry_node(state: State) -> State:
    if telemetry_sampler.should_sample():  # 10% 採樣
        await _update_langfuse_trace_async(...)  # 非阻塞
    return state

async def _update_langfuse_trace_async(tags, metadata, output):
    await anyio.to_thread.run_sync(
        lambda: langfuse.update_current_trace(...)
    )
```

### 7.7 效能預期對比

| 指標 | 優化前 | 優化後 | 改善 |
|------|--------|--------|------|
| 單 worker 並發能力 | ~10 請求 | ~100+ 請求 | **10x+** |
| Graph 編譯開銷 | 50-100ms/請求 | 0ms（快取命中） | **100%** |
| 遙測開銷（高流量） | 100-300ms/請求 | ~30ms（採樣後） | **70-90%** |
| P50 延遲（100 QPS） | 線性增長 | 穩定 | **顯著** |
| P99 延遲（優先級模式） | 嚴重惡化 | 可控 | **顯著** |
| LLM 後端負載 | 無控制（可能過載） | Semaphore 保護 | **穩定** |

### 7.8 配置對比

#### 優化前：無可調參數

```bash
# 無任何併發或效能相關設定
```

#### 優化後：精細調控

```bash
# .env

# LLM 並發控制
LLM_MAX_CONCURRENT_DEFAULT=10
LLM_MAX_CONCURRENT_CHAT=20
LLM_MAX_CONCURRENT_RESPONSES=100
LLM_MAX_CONCURRENT_EMBEDDING=30
LLM_REQUEST_TIMEOUT=60.0

# 優先級排序（可選）
LLM_PRIORITY_ENABLED=false
LLM_PRIORITY_STARVATION_THRESHOLD=5.0

# 遙測採樣
LANGFUSE_SAMPLE_RATE=0.1
```

### 7.9 監控能力對比

#### 優化前：無監控

```
# 無法得知：
# - 當前有多少 LLM 請求在執行
# - 有多少請求在等待
# - 是否有 timeout
# - Graph 快取是否命中
```

#### 優化後：完整監控 API

```bash
# 即時監控
curl http://localhost:8000/api/v1/admin/concurrency/status
curl http://localhost:8000/api/v1/admin/concurrency/summary
curl http://localhost:8000/api/v1/admin/concurrency/priority

# 返回：
# - 各後端的 slot 使用狀況
# - 等待佇列長度
# - 累計統計（acquired/released/timeout）
# - 優先級佇列狀態
```

---

*文件版本: 2.0*
*最後更新: 2025-12-11*
