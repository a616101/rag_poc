# ChatBot GraphRAG 系統架構文檔

> **版本**: 4.1.0
> **更新日期**: 2025-12-31

## 目錄

1. [系統概述](#系統概述)
2. [技術架構](#技術架構)
3. [目錄結構詳解](#目錄結構詳解)
4. [LLM 並發控制](#llm-並發控制)
5. [向量處理與 Embedding](#向量處理與-embedding)
6. [知識圖譜建立](#知識圖譜建立)
7. [Hybrid Search 實作](#hybrid-search-實作)
8. [OpenSearch 操作指南](#opensearch-操作指南)
9. [LangGraph 工作流程](#langgraph-工作流程)
10. [GraphRAG 檢索模式](#graphrag-檢索模式)
11. [資料攝取與分塊](#資料攝取與分塊)
12. [狀態管理](#狀態管理)
13. [API 端點](#api-端點)
14. [文件向量化 API](#文件向量化-api)
15. [環境變數配置](#環境變數配置)
16. [開發環境與部署](#開發環境與部署)
17. [附錄：與 chatbot_rag 的比較](#附錄與-chatbot_rag-的比較)

---

## 系統概述

ChatBot GraphRAG 是一個**生產級別的 GraphRAG（圖增強檢索生成）API**，使用 FastAPI、LangGraph 和多種向量/圖數據庫構建。相較於傳統 RAG 系統，GraphRAG 結合了：

- **向量檢索** (Dense + Sparse + Full-text)
- **知識圖譜** (Entity-Relation Graph via NebulaGraph)
- **社群偵測** (Leiden Algorithm)
- **多模式檢索** (LOCAL / GLOBAL / DRIFT)

### 核心特色

| 特色 | 說明 |
|------|------|
| **三路混合檢索** | Dense (768維) + Sparse (SPLADE) + Full-text (BM25) + RRF 融合 |
| **知識圖譜增強** | NebulaGraph 2-hop 圖遍歷，14 種實體類型，11 種關係類型 |
| **四種查詢模式** | Direct / Local / Global / Drift 智能路由 |
| **Leiden 社群偵測** | 多層級社群結構 (Level 0-3)，社群報告生成 |
| **LLM 並發控制** | 多後端 Semaphore + 優先級排程 + 飢餓保護 |
| **接地性評估** | 雙層評估 (啟發式 + 10% Ragas 抽樣) |
| **人機協作 (HITL)** | PostgreSQL 持久化檢查點，支援中斷/恢復 |
| **版本感知快取** | 基於 index_version + prompt_version 的語義快取 |
| **多租戶 ACL** | 完整的 tenant_id + acl_groups 存取控制 |
| **OWASP 安全防護** | LLM01 (Prompt Injection) + LLM02 (Output Handling) |

---

## 技術架構

### 技術堆疊

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer                                │
│                  FastAPI + ORJSON + SSE                          │
├─────────────────────────────────────────────────────────────────┤
│                   LLM Concurrency Control                        │
│      Multi-Backend Semaphores + Priority Queue + Backpressure    │
├─────────────────────────────────────────────────────────────────┤
│                      Workflow Engine                             │
│            LangGraph StateGraph (22 Nodes)                       │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│   Vector DB  │  Graph DB    │  Full-text   │  Object Storage    │
│   Qdrant     │  NebulaGraph │  OpenSearch  │  MinIO             │
│   (3 集合)   │  (圖譜空間)  │  (BM25 索引) │  (S3 相容)         │
├──────────────┴──────────────┴──────────────┴────────────────────┤
│                      Relational DB                               │
│                      PostgreSQL                                  │
├─────────────────────────────────────────────────────────────────┤
│                      LLM Provider                                │
│              OpenAI-Compatible API (LMStudio, TWCC)              │
├─────────────────────────────────────────────────────────────────┤
│                      Observability                               │
│                   Langfuse (自建) + Ragas                        │
└─────────────────────────────────────────────────────────────────┘
```

### 資料流架構

```
使用者查詢
    ↓
[API Layer] FastAPI + SSE
    ↓
[Concurrency Control] LLM Semaphore 取得許可
    ↓
[Guard] OWASP LLM01 安全檢查
    ↓
[ACL] 多租戶存取控制
    ↓
[Cache] 版本感知語義快取查詢
    │
    ├─ [HIT] → 返回快取回答
    │
    └─ [MISS] → [Intent Router] 意圖分類
                    │
    ┌───────────────┼───────────────────────┐
    │               │                       │
 DIRECT          LOCAL                  GLOBAL/DRIFT
    │               │                       │
    ↓               ↓                       ↓
[直接回答]   [Hybrid Search]        [Community Reports]
                    │                       │
                    ↓                       ↓
              [RRF Merge]           [Followups 生成]
                    │                       │
                    ↓                       │
               [Rerank]  ←──────────────────┘
                    │
                    ↓
           [Graph Traversal]
                    │
                    ↓
           [Context Building]
                    │
                    ↓
        [Groundedness Evaluation]
           │        │         │
         pass    retry   needs_review
           │        │         │
           ↓        ↓         ↓
      [Final]  [Targeted] [HITL]
      [Answer]  [Retry]   [Interrupt]
           │        │         │
           └────────┴─────────┘
                    ↓
             [Cache Store]
                    ↓
              [Telemetry]
                    ↓
                  END
```

---

## 目錄結構詳解

```
src/chatbot_graphrag/
├── main.py                           # FastAPI 應用程式入口 + 並發初始化
├── cli.py                            # CLI 命令列介面
│
├── core/                             # 核心配置
│   ├── config.py                     # Pydantic Settings (270+ 參數)
│   ├── constants.py                  # 枚舉與常數定義 (8 個枚舉)
│   └── concurrency.py                # LLM 並發控制管理器 (NEW)
│
├── api/                              # API 路由層
│   └── routes/
│       ├── ask_stream.py             # 原生格式串流 API
│       ├── ask_stream_chat.py        # OpenAI 相容串流 API
│       ├── cache_admin.py            # 快取管理 API
│       └── vectorize.py              # 向量化 API
│
├── graph_workflow/                   # LangGraph 計算圖 (核心)
│   ├── types.py                      # GraphRAGState 狀態定義 (40+ 欄位)
│   ├── factory.py                    # 圖構建工廠 (HITL 支援)
│   ├── routing.py                    # 6 個條件路由函數
│   ├── budget.py                     # 循環預算管理
│   ├── tracing.py                    # Langfuse 追蹤整合
│   └── nodes/                        # 22 個工作流節點
│       ├── guard.py                  # 輸入安全檢查 + 並發控制
│       ├── acl.py                    # 存取控制驗證
│       ├── normalize.py              # 語言偵測與正規化
│       ├── cache.py                  # 快取操作 (3 節點)
│       ├── intent.py                 # 意圖路由 (2 節點)
│       ├── retrieval.py              # 混合搜尋 (5 節點)
│       ├── rerank.py                 # 交叉編碼器重排
│       ├── graph.py                  # 圖萃取與遍歷 (3 節點)
│       ├── context.py                # 上下文構建 (3 節點)
│       ├── quality.py                # 品質評估 (3 節點)
│       ├── output.py                 # 最終輸出 + 並發控制 (2 節點)
│       └── status.py                 # 狀態事件發送
│
├── models/                           # 資料模型
│   ├── pydantic/                     # API 請求/回應模型
│   │   ├── graph.py                  # Entity, Relation, Community
│   │   ├── ingestion.py              # Document, Chunk
│   │   ├── requests.py               # API 請求模型
│   │   └── responses.py              # API 回應模型
│   └── sqlalchemy/                   # ORM 模型
│       ├── base.py                   # 基礎模型 + Mixins
│       ├── acl.py                    # ACL 模型
│       ├── doc.py                    # 文檔模型
│       └── chunk.py                  # 文檔塊模型
│
└── services/                         # 服務層 (單例模式)
    ├── ask_service.py                # 問答服務主類
    │
    ├── llm/                          # LLM 工廠 + 並發包裝
    │   ├── factory.py                # 模型工廠 + 並發模型方法
    │   ├── concurrent_llm.py         # 並發控制包裝器 (NEW)
    │   ├── responses_chat_model.py   # Responses API 包裝
    │   └── responses_accumulator.py  # 串流累積器
    │
    ├── vector/                       # 向量服務
    │   ├── embedding_service.py      # Dense + Sparse 嵌入 + 並發控制
    │   └── qdrant_service.py         # Qdrant 操作 (3 集合)
    │
    ├── search/                       # 搜尋服務
    │   ├── hybrid_search.py          # RRF 三路融合
    │   ├── opensearch_service.py     # BM25 全文搜尋
    │   ├── query_decomposer.py       # LLM 查詢分解 + 並發控制
    │   └── query_classifier.py       # 查詢類型分類
    │
    ├── graph/                        # 知識圖譜服務
    │   ├── nebula_client.py          # NebulaGraph 客戶端
    │   ├── entity_extractor.py       # LLM 實體萃取 + 並發控制
    │   ├── relation_extractor.py     # LLM 關係萃取 + 並發控制
    │   ├── batch_loader.py           # 批量載入圖譜
    │   ├── community_detector.py     # Leiden 社群偵測
    │   └── community_summarizer.py   # 社群報告生成 + 並發控制
    │
    ├── retrieval/                    # 多模式檢索
    │   ├── local_mode.py             # 實體級檢索
    │   ├── global_mode.py            # 社群級檢索
    │   └── drift_mode.py             # 探索級檢索
    │
    ├── ingestion/                    # 資料攝取
    │   ├── coordinator.py            # 雙管線協調器
    │   ├── curated_pipeline.py       # YAML+Markdown 管線
    │   ├── raw_pipeline.py           # PDF/DOCX 管線
    │   ├── schema_validator.py       # 模式驗證
    │   └── chunkers/                 # 6 種專用分塊器
    │       ├── base.py               # 基礎分塊器
    │       ├── generic.py            # 通用分塊
    │       ├── physician.py          # 醫師資料
    │       ├── procedure.py          # 流程步驟
    │       ├── guide.py              # 指南/交通
    │       └── hospital_team.py      # 醫院團隊
    │
    ├── storage/                      # 物件儲存
    │   └── minio_service.py          # MinIO S3 操作
    │
    ├── cache/                        # 快取服務
    │   └── __init__.py               # Redis 快取
    │
    ├── evaluation/                   # 評估服務
    │   └── ragas_evaluator.py        # Ragas 評估框架
    │
    └── tracing/                      # 追蹤服務
        └── __init__.py               # Langfuse 整合
```

---

## LLM 並發控制

### 架構概述

**檔案**: `core/concurrency.py`

GraphRAG 實現了完整的 LLM 並發控制機制，防止高負載下系統崩潰：

```
                     LLM 請求
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  LLMConcurrencyManager                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   default   │  │    chat     │  │  responses  │          │
│  │  Semaphore  │  │  Semaphore  │  │  Semaphore  │          │
│  │   (10)      │  │   (20)      │  │   (50)      │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                              │
│  ┌─────────────┐  ┌──────────────────────────────────────┐  │
│  │  embedding  │  │         Priority Queue               │  │
│  │  Semaphore  │  │  (FIFO / Priority with Starvation    │  │
│  │   (30)      │  │         Protection)                  │  │
│  └─────────────┘  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                         ↓
                    LLM API Call
```

### 多後端 Semaphore 配置

```python
# 預設並發限制
llm_max_concurrent_default: int = 10      # 預設後端
llm_max_concurrent_chat: int = 20         # Chat 完成 API
llm_max_concurrent_responses: int = 50    # Responses API (較快)
llm_max_concurrent_embedding: int = 30    # Embedding API

# 優先級排程 (可選)
llm_priority_enabled: bool = False
llm_priority_starvation_threshold: float = 5.0  # 秒
```

### LLMConcurrencyManager 類別

```python
class LLMConcurrencyManager:
    """
    LLM 並發控制管理器

    特性:
    - 多後端 Semaphore 隔離
    - 可選優先級排程
    - 飢餓保護機制
    - 請求上下文追蹤
    """

    def initialize(self) -> None:
        """初始化各後端的 Semaphore"""

    @asynccontextmanager
    async def acquire(self, backend: str = "default"):
        """取得指定後端的並發許可"""

    def get_status(self) -> dict[str, dict]:
        """獲取各後端的並發狀態"""

    def get_summary(self) -> dict:
        """獲取總體並發摘要"""
```

### 使用方式

#### 方式一：Context Manager

```python
from chatbot_graphrag.core.concurrency import llm_concurrency

async def my_llm_call():
    async with llm_concurrency.acquire("chat"):
        response = await client.chat.completions.create(...)
    return response
```

#### 方式二：Helper Functions

```python
from chatbot_graphrag.core.concurrency import with_chat_semaphore

async def my_llm_call():
    response = await with_chat_semaphore(
        lambda: llm.ainvoke(messages)
    )
    return response
```

### 已整合並發控制的服務

| 服務檔案 | 後端類型 | 說明 |
|---------|----------|------|
| `graph_workflow/nodes/guard.py` | chat | 輸入安全檢查 |
| `graph_workflow/nodes/output.py` | chat/responses | 串流回應生成 |
| `services/search/query_decomposer.py` | chat | 查詢分解 |
| `services/vector/embedding_service.py` | embedding | 向量嵌入 |
| `services/graph/entity_extractor.py` | chat | 實體提取 |
| `services/graph/relation_extractor.py` | chat | 關係提取 |
| `services/graph/community_summarizer.py` | chat | 社群摘要 |

### 監控端點

```bash
# 獲取並發狀態
curl http://localhost:18000/health/concurrency

# 回應範例
{
  "status": "healthy",
  "backends": {
    "default": {"limit": 10, "in_use": 2, "waiting": 0},
    "chat": {"limit": 20, "in_use": 5, "waiting": 1},
    "responses": {"limit": 50, "in_use": 10, "waiting": 0},
    "embedding": {"limit": 30, "in_use": 8, "waiting": 0}
  },
  "summary": {
    "total_in_use": 25,
    "total_waiting": 1,
    "total_limit": 110
  }
}
```

---

## 向量處理與 Embedding

### Embedding Service 架構

**檔案**: `services/vector/embedding_service.py`

GraphRAG 使用 **雙向量策略**，同時生成密集向量和稀疏向量：

#### 密集向量 (Dense Embeddings)

```python
async def embed_texts(texts: list[str], batch_size: int = 20) -> list[list[float]]:
    """
    使用 OpenAI 相容 API 生成密集向量

    - 模型: text-embedding-embeddinggemma-300m-qat
    - 維度: 768
    - 批次大小: 20
    - 並發控制: llm_concurrency.acquire("embedding")
    """
    from chatbot_graphrag.core.concurrency import llm_concurrency

    async with llm_concurrency.acquire("embedding"):
        response = await self._client.embeddings.create(
            model=self._model,
            input=batch,
        )
```

#### 稀疏向量 (Sparse Embeddings - SPLADE)

```python
async def sparse_embed_texts(texts: list[str], batch_size: int = 8) -> list[dict[int, float]]:
    """
    使用 SPLADE 模型生成稀疏向量

    - 模型: naver/splade-cocondenser-ensembledistil
    - 輸出格式: {token_id: weight}
    - 激活函數: log(1 + ReLU(x))
    - GPU 加速: 自動檢測 CUDA
    """
```

#### 並行向量生成

```python
async def embed_with_sparse(texts: list[str]) -> list[dict]:
    """
    並行生成雙向量

    返回: [{"dense": [...], "sparse": {...}}, ...]
    """
    dense_task = self.embed_texts(texts)
    sparse_task = self.sparse_embed_texts(texts)
    dense, sparse = await asyncio.gather(dense_task, sparse_task)
```

### Qdrant 向量資料庫

**檔案**: `services/vector/qdrant_service.py`

GraphRAG 在 Qdrant 中維護 **3 個集合**：

#### 1. graphrag_chunks (文檔分塊)

```python
vectors_config = {
    "dense": VectorParams(
        size=768,
        distance=Distance.COSINE
    )
}
sparse_vectors_config = {
    "sparse": SparseVectorParams(
        index=SparseIndexParams(on_disk=False)
    )
}

# Payload 索引 (用於過濾)
payload_indexes = [
    "doc_id",       # 文檔 ID
    "doc_type",     # 文檔類型
    "chunk_type",   # 分塊類型
    "acl_groups",   # 存取控制群組
    "tenant_id",    # 多租戶隔離
    "entity_ids"    # 關聯實體 (圖遍歷用)
]
```

#### 2. graphrag_cache (語義快取)

```python
# 只使用密集向量
# 相似度閾值: 0.90
# TTL: 3600 秒
```

#### 3. graphrag_communities (社群報告)

```python
# 存儲社群級別摘要
# 支援 level 過濾 (0-3)
# 用於 GLOBAL/DRIFT 模式
```

### UUID 生成策略

```python
CHUNK_UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

def string_to_uuid(s: str) -> str:
    """確定性 UUID5 生成，相同輸入產生相同 UUID"""
    return str(uuid.uuid5(CHUNK_UUID_NAMESPACE, s))
```

---

## 知識圖譜建立

### NebulaGraph 配置

**檔案**: `services/graph/nebula_client.py`

```python
# 連線配置
nebula_host = "nebula-graphd"
nebula_port = 9669
nebula_space = "graphrag"
nebula_pool_size = 10
nebula_timeout = 30000  # ms
```

### 圖譜 Schema 定義

**檔案**: `core/constants.py`

#### 實體類型 (14 種)

```python
class EntityType(str, Enum):
    PERSON = "person"           # 醫生、護士、員工
    DEPARTMENT = "department"   # 科室/部門
    PROCEDURE = "procedure"     # 醫療程序/服務
    LOCATION = "location"       # 地點
    BUILDING = "building"       # 建築物
    FLOOR = "floor"             # 樓層
    ROOM = "room"               # 房間/診間
    FORM = "form"               # 表單/文件
    MEDICATION = "medication"   # 藥物
    EQUIPMENT = "equipment"     # 設備
    SERVICE = "service"         # 服務項目
    CONDITION = "condition"     # 疾病/症狀
    TRANSPORT = "transport"     # 運輸方式
    CONTACT = "contact"         # 聯繫方式
```

#### 關係類型 (11 種)

```python
class RelationType(str, Enum):
    BELONGS_TO = "belongs_to"      # 歸屬
    WORKS_IN = "works_in"          # 工作地點
    PERFORMS = "performs"          # 執行
    LOCATED_AT = "located_at"      # 位置
    REQUIRES = "requires"          # 需求
    MENTIONS = "mentions"          # 提及
    RELATED_TO = "related_to"      # 一般關聯
    TREATS = "treats"              # 治療
    CONNECTS_TO = "connects_to"    # 連接
    PART_OF = "part_of"            # 部分-整體
    MEMBER_OF = "member_of"        # 成員
```

### 知識圖譜建立流程

```
Document Input
    ↓
[1] Entity Extraction (LLM + 並發控制)
    ├── 溫度: 0.1
    ├── Max tokens: 4000
    ├── 並發控制: llm_concurrency.acquire("chat")
    └── 重試: 指數退避 (2s, 4s, 8s)
    ↓
[2] Entity JSON Parsing
    ├── Markdown code block 提取
    ├── 平衡括號修復
    └── 重複實體合併
    ↓
[3] Relation Extraction (LLM + 並發控制)
    ├── 至少 2 個實體才執行
    ├── 並發控制: llm_concurrency.acquire("chat")
    ├── 權重: 0.0-1.0
    └── 精確/部分實體匹配
    ↓
[4] Batch Load to NebulaGraph
    ├── INSERT VERTEX entity
    ├── INSERT EDGE {edge_type}
    └── 更新 Qdrant chunk 的 entity_ids
    ↓
[5] Community Detection (Leiden)
    ├── igraph + leidenalg
    ├── 分辨率: resolution * (2^level)
    └── 最多 3 層級
    ↓
[6] Community Summarization (LLM + 並發控制)
    ├── 溫度: 0.3
    ├── 並發控制: llm_concurrency.acquire("chat")
    ├── 關鍵實體: 最多 5 個
    └── 主題萃取
    ↓
Knowledge Graph Ready
```

### Entity ID 生成

```python
def _generate_entity_id(name: str, entity_type: EntityType) -> str:
    """
    一致性 ID 生成:
    hash_input = f"{entity_type.value}:{name.lower()}"
    hash_hex = SHA256(hash_input)[:16]
    return f"e_{entity_type.value}_{hash_hex}"
    """
```

### nGQL 查詢範例

#### 圖遍歷

```sql
GET SUBGRAPH WITH PROP 2 STEPS FROM "e_person_abc123"
OVER works_in, belongs_to, performs
YIELD VERTICES AS nodes, EDGES AS edges;
```

#### 實體搜尋

```sql
LOOKUP ON entity
WHERE entity.entity_type == "person"
  AND entity.name STARTS WITH "王"
YIELD id(vertex) as vid, properties(vertex) as props
LIMIT 20;
```

---

## Hybrid Search 實作

### 三路搜尋架構

**檔案**: `services/search/hybrid_search.py`

```
                    Query
                      ↓
    ┌─────────────────┼─────────────────┐
    ↓                 ↓                 ↓
Dense Search    Sparse Search    Full-text Search
 (Qdrant)        (SPLADE)        (OpenSearch BM25)
    ↓                 ↓                 ↓
    └─────────────────┼─────────────────┘
                      ↓
              RRF Fusion (k=60)
                      ↓
               Final Results
```

### 並行執行

```python
async def search(query: str, query_embedding: list[float], ...) -> list[SearchResult]:
    # 並行執行三路搜尋
    dense_task = self._dense_search(query_embedding, ...)
    sparse_task = self._sparse_search(sparse_embedding, ...)
    fts_task = self._fts_search(query, ...)

    results = await asyncio.gather(
        dense_task, sparse_task, fts_task,
        return_exceptions=True
    )
```

### RRF 融合公式

```
RRF_score(d) = Σ (weight_i / (k + rank_i(d)))

其中:
- weight_i: 搜尋模態權重 (dense=0.4, sparse=0.3, fts=0.3)
- k: RRF 常數 (預設 60)
- rank_i(d): 文檔 d 在第 i 個搜尋結果中的排名
```

### 查詢感知權重調整

```python
QUERY_TYPE_WEIGHTS = {
    QueryType.FACTUAL:      (0.50, 0.25, 0.25),  # 密集優先
    QueryType.CONCEPTUAL:   (0.45, 0.30, 0.25),  # 平衡
    QueryType.NAVIGATIONAL: (0.25, 0.35, 0.40),  # 關鍵字優先
    QueryType.PROCEDURAL:   (0.40, 0.30, 0.30),  # 密集 + 關鍵字
    QueryType.COMPARATIVE:  (0.50, 0.25, 0.25),  # 密集優先
    QueryType.AGGREGATE:    (0.35, 0.35, 0.30),  # 社群級優先
}
```

### 進度搜尋 (Progressive Search)

```python
async def search_progressive(
    query: str,
    thresholds: tuple = (0.65, 0.50, 0.35)  # 高/中/低
) -> tuple[list[SearchResult], str]:
    """
    嘗試高閾值 → 中閾值 → 低閾值
    至少需要 3 個結果才視為成功
    """
    for threshold in thresholds:
        results = await self.search(query, min_score=threshold)
        if len(results) >= 3:
            return results, threshold
```

### OpenSearch 全文搜尋

**檔案**: `services/search/opensearch_service.py`

```python
# 中文分詞配置
"analyzer": {
    "chinese_analyzer": {
        "type": "custom",
        "tokenizer": "standard",
        "filter": ["lowercase", "cjk_width", "cjk_bigram"]
    }
}

# 多字段搜尋 (BM25)
"multi_match": {
    "query": query,
    "fields": [
        "content^1.0",
        "contextual_content^0.8",
        "title^2.0",          # 標題權重最高
        "section_title^1.5",
        "entity_names^1.2"
    ],
    "type": "best_fields",
    "fuzziness": "AUTO"
}
```

### 查詢分解

**檔案**: `services/search/query_decomposer.py`

```python
@dataclass
class DecomposedQuery:
    query_type: str           # "entity_lookup", "reverse_lookup", "general"
    physician_name: str       # 醫師名稱 (若有)
    department: str           # 科室 (若有)
    property_type: str        # 屬性類型 (門診時間、專長等)
    resolved_query: str       # 上下文解析後的完整問題
    sub_queries: list[str]    # 多角度子查詢

# 範例:
# 輸入: "王醫師的門診時間"
# 輸出:
#   query_type: "entity_lookup"
#   physician_name: "王"
#   property_type: "schedule"
#   sub_queries: ["王醫師 門診", "王醫師 看診時間", "王 門診時刻表"]
```

---

## OpenSearch 操作指南

### Docker Compose 配置

**檔案**: `docker-compose.graphrag.yml`

```yaml
# OpenSearch 3.x (Full-text Search)
opensearch:
  image: opensearchproject/opensearch:3.4.0
  container_name: graphrag-opensearch
  environment:
    - cluster.name=graphrag-cluster
    - node.name=opensearch-node1
    - discovery.type=single-node
    - bootstrap.memory_lock=true
    - OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m
    - DISABLE_SECURITY_PLUGIN=true
  ports:
    - "19200:9200"   # REST API
    - "19600:9600"   # Performance Analyzer
  volumes:
    - opensearch-data:/usr/share/opensearch/data

# OpenSearch Dashboards (debug profile)
opensearch-dashboards:
  image: opensearchproject/opensearch-dashboards:3.4.0
  container_name: graphrag-opensearch-dashboards
  environment:
    - OPENSEARCH_HOSTS=["http://opensearch:9200"]
    - DISABLE_SECURITY_DASHBOARDS_PLUGIN=true
  ports:
    - "15601:5601"
  profiles:
    - debug
    - development
```

### 啟動 OpenSearch + Dashboards

```bash
# 啟動 OpenSearch + Dashboards (debug profile)
docker compose -f docker-compose.graphrag.yml --profile debug up -d opensearch opensearch-dashboards

# 或者啟動完整開發環境 (包含 Dashboards)
docker compose -f docker-compose.graphrag.yml --profile development up -d
```

### 連線資訊

| 服務 | URL | 說明 |
|------|-----|------|
| OpenSearch REST API | http://localhost:19200 | REST API 端點 |
| OpenSearch Dashboards | http://localhost:15601 | Web UI |

### 索引 Schema

**索引名稱**: `graphrag_chunks`

| 欄位 | 類型 | 分析器 | 用途 |
|------|------|--------|------|
| `chunk_id` | keyword | - | 唯一識別碼 |
| `doc_id` | keyword | - | 來源文件 ID |
| `doc_type` | keyword | - | 文件類型 |
| `chunk_type` | keyword | - | 分塊類型 |
| `content` | text | CJK + bigram | 主要內容（全文搜尋） |
| `contextual_content` | text | CJK | 擴展上下文 |
| `title` | text | CJK, boost 2.0 | 文件標題（最高優先） |
| `section_title` | text | CJK, boost 1.5 | 章節標題 |
| `entity_names` | text | CJK | 實體名稱 |
| `language` | keyword | - | 語言代碼 |
| `acl_groups` | keyword | - | 存取控制群組 |
| `tenant_id` | keyword | - | 多租戶隔離 |
| `department` | keyword | - | 部門過濾 |
| `created_at` | date | - | 建立時間 |
| `updated_at` | date | - | 更新時間 |

### OpenSearch Dashboards 操作

#### 1. 開啟 Dev Tools

1. 瀏覽器開啟 http://localhost:15601
2. 點擊左側選單 **☰ → Management → Dev Tools**
3. 在 Console 中輸入查詢

#### 2. 常用查詢範例

**檢查叢集健康狀態**:
```json
GET _cluster/health
```

**查看索引資訊**:
```json
GET graphrag_chunks/_mapping

GET graphrag_chunks/_count

GET _cat/indices?v
```

**全文搜尋（BM25）**:
```json
GET graphrag_chunks/_search
{
  "query": {
    "multi_match": {
      "query": "你要搜尋的關鍵字",
      "fields": ["content^1.0", "title^2.0", "section_title^1.5", "entity_names^1.2"],
      "type": "best_fields",
      "fuzziness": "AUTO"
    }
  },
  "highlight": {
    "fields": {
      "content": { "fragment_size": 200, "number_of_fragments": 3 }
    }
  },
  "size": 10
}
```

**精確短語搜尋**:
```json
GET graphrag_chunks/_search
{
  "query": {
    "match_phrase": {
      "content": {
        "query": "完整短語",
        "slop": 2
      }
    }
  }
}
```

**依文件類型過濾**:
```json
GET graphrag_chunks/_search
{
  "query": {
    "bool": {
      "must": [
        { "match": { "content": "關鍵字" } }
      ],
      "filter": [
        { "term": { "doc_type": "procedure" } }
      ]
    }
  }
}
```

**實體搜尋**:
```json
GET graphrag_chunks/_search
{
  "query": {
    "bool": {
      "should": [
        { "match": { "entity_names": "王醫師" } },
        { "match": { "entity_names": "心臟科" } }
      ],
      "minimum_should_match": 1
    }
  }
}
```

**聚合統計**:
```json
GET graphrag_chunks/_search
{
  "size": 0,
  "aggs": {
    "doc_types": {
      "terms": { "field": "doc_type", "size": 20 }
    },
    "departments": {
      "terms": { "field": "department", "size": 20 }
    }
  }
}
```

**相似文件搜尋 (More Like This)**:
```json
GET graphrag_chunks/_search
{
  "query": {
    "more_like_this": {
      "fields": ["content", "title"],
      "like": [
        { "_index": "graphrag_chunks", "_id": "chunk_id_here" }
      ],
      "min_term_freq": 1,
      "min_doc_freq": 1
    }
  }
}
```

### 命令列驗證

```bash
# 檢查 OpenSearch 是否正常
curl http://localhost:19200/_cluster/health?pretty

# 檢查索引是否存在
curl http://localhost:19200/_cat/indices?v

# 查看索引文件數量
curl http://localhost:19200/graphrag_chunks/_count?pretty

# 搜尋測試
curl -X GET "http://localhost:19200/graphrag_chunks/_search?pretty" \
  -H "Content-Type: application/json" \
  -d '{"query": {"match_all": {}}, "size": 1}'

# 搜尋特定關鍵字
curl -X GET "http://localhost:19200/graphrag_chunks/_search?pretty" \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "multi_match": {
        "query": "住院",
        "fields": ["content", "title"]
      }
    },
    "size": 5
  }'
```

### 與 GraphRAG 整合流程

```
使用者查詢
    ↓
HybridSearchService.search()
    ├→ Dense Search (Qdrant 向量搜尋)
    ├→ Sparse Search (Qdrant SPLADE)
    └→ FTS Search (OpenSearch BM25) ← 這裡
         ├ 多欄位 BM25 搜尋
         ├ ACL/tenant 過濾
         ├ Highlight 生成
         └ 返回評分結果
    ↓
RRF Fusion (權重: dense=0.4, sparse=0.3, fts=0.3)
    ↓
Reranking (cross-encoder)
    ↓
Response Generation
```

---

## LangGraph 工作流程

### 22 節點完整流程圖

```
START
  │
  ▼
┌─────────────┐
│    guard    │──────────────────────────────────────────┐
└──────┬──────┘                                          │
       │ continue                                   blocked
       ▼                                                 │
┌─────────────┐                                          │
│     acl     │──────────────────────────────────────────┤
└──────┬──────┘                                          │
       │ continue                                  denied │
       ▼                                                 │
┌─────────────┐                                          │
│  normalize  │                                          │
└──────┬──────┘                                          │
       │                                                 │
       ▼                                                 │
┌───────────────┐                                        │
│ cache_lookup  │                                        │
└───────┬───────┘                                        │
  ┌─────┴─────┐                                          │
  │           │                                          │
 hit         miss                                        │
  │           │                                          │
  ▼           ▼                                          │
┌─────────┐ ┌───────────────┐                            │
│ cache_  │ │ intent_router │                            │
│response │ └───────┬───────┘                            │
└────┬────┘   ┌─────┼─────────┬────────────┐             │
     │        │     │         │            │             │
     │     direct  local   global        drift           │
     │        │     │         │            │             │
     │        ▼     │         ▼            │             │
     │  ┌──────────┐│    ┌─────────────────┴───┐         │
     │  │ direct_  ││    │ community_reports   │         │
     │  │ answer   ││    └──────────┬──────────┘         │
     │  └────┬─────┘│               │                    │
     │       │      │               ▼                    │
     │       │      │    ┌─────────────────┐             │
     │       │      │    │    followups    │             │
     │       │      │    └────────┬────────┘             │
     │       │      │             │                      │
     │       │      ▼             ▼                      │
     │       │    ┌─────────────────┐                    │
     │       │    │   hybrid_seed   │◄───────────────────┤
     │       │    └────────┬────────┘                    │
     │       │             │                             │
     │       │             ▼                             │
     │       │    ┌─────────────────┐                    │
     │       │    │    rrf_merge    │                    │
     │       │    └────────┬────────┘                    │
     │       │             │                             │
     │       │             ▼                             │
     │       │    ┌─────────────────┐                    │
     │       │    │     rerank      │  (40 → 12)         │
     │       │    └────────┬────────┘                    │
     │       │             │                             │
     │       │             ▼                             │
     │       │    ┌─────────────────────┐                │
     │       │    │ graph_seed_extract  │                │
     │       │    └─────────┬───────────┘                │
     │       │              │                            │
     │       │              ▼                            │
     │       │    ┌─────────────────────┐                │
     │       │    │   graph_traverse    │ (2-hop)        │
     │       │    └─────────┬───────────┘                │
     │       │              │                            │
     │       │              ▼                            │
     │       │    ┌─────────────────────┐                │
     │       │    │subgraph_to_queries  │                │
     │       │    └─────────┬───────────┘                │
     │       │              │                            │
     │       │              ▼                            │
     │       │    ┌─────────────────────┐                │
     │       │    │     hop_hybrid      │◄───────────┐   │
     │       │    └─────────┬───────────┘            │   │
     │       │              │                        │   │
     │       │              ▼                        │   │
     │       │    ┌─────────────────────┐            │   │
     │       │    │   chunk_expander    │            │   │
     │       │    └─────────┬───────────┘            │   │
     │       │              │                        │   │
     │       │              ▼                        │   │
     │       │    ┌─────────────────────┐            │   │
     │       │    │   context_packer    │            │   │
     │       │    └─────────┬───────────┘            │   │
     │       │              │                        │   │
     │       │              ▼                        │   │
     │       │    ┌─────────────────────┐            │   │
     │       │    │   evidence_table    │            │   │
     │       │    └─────────┬───────────┘            │   │
     │       │              │                        │   │
     │       │              ▼                        │   │
     │       │    ┌─────────────────────┐            │   │
     │       │    │    groundedness     │            │   │
     │       │    └─────────┬───────────┘            │   │
     │       │        ┌─────┼─────┐                  │   │
     │       │        │     │     │                  │   │
     │       │      pass  retry  needs_review        │   │
     │       │        │     │     │                  │   │
     │       │        │     ▼     ▼                  │   │
     │       │        │  ┌────────────────┐          │   │
     │       │        │  │ targeted_retry │──────────┘   │
     │       │        │  └────────────────┘              │
     │       │        │           │ exhausted            │
     │       │        │           ▼                      │
     │       │        │  ┌────────────────┐              │
     │       │        │  │ interrupt_hitl │───► WAIT     │
     │       │        │  └────────┬───────┘              │
     │       │        │           │ continue/timeout     │
     │       │        ▼           ▼                      │
     │       └───────►┌─────────────────┐◄───────────────┘
     │                │  final_answer   │ (+ 並發控制)
     │                └────────┬────────┘
     │                         │
     ▼                         ▼
┌─────────────┐         ┌─────────────┐
│             │◄────────│ cache_store │
│             │         └─────────────┘
│  telemetry  │
│             │
└──────┬──────┘
       │
       ▼
      END
```

### 6 個條件路由函數

**檔案**: `graph_workflow/routing.py`

| 路由函數 | 返回值 | 決策邏輯 |
|----------|--------|----------|
| `route_after_guard` | blocked / continue | 檢查 guard_blocked 或 acl_denied |
| `route_after_cache` | hit / miss | 檢查快取命中 |
| `route_by_intent` | direct / local / global / drift | 基於模式匹配 |
| `route_after_groundedness` | pass / retry / needs_review | 基於分數和預算 |
| `route_by_budget` | continue / exhausted | 檢查預算限制 |
| `route_hitl` | wait / continue / timeout | HITL 狀態管理 |

### 節點功能總覽

| # | 節點名稱 | 檔案 | 功能說明 | 並發控制 |
|---|----------|------|----------|----------|
| 1 | guard_node | guard.py | OWASP LLM01 輸入安全檢查 | ✓ chat |
| 2 | acl_node | acl.py | 多租戶存取控制驗證 | - |
| 3 | normalize_node | normalize.py | 語言偵測與問題標準化 | - |
| 4 | cache_lookup_node | cache.py | 版本感知語義快取查詢 | - |
| 5 | cache_response_node | cache.py | 快取回應輸出 | - |
| 6 | cache_store_node | cache.py | 快取儲存 | - |
| 7 | intent_router_node | intent.py | 意圖路由分類 | - |
| 8 | direct_answer_node | intent.py | 直接回答生成 | - |
| 9 | hybrid_seed_node | retrieval.py | 三路混合搜尋 + 查詢分解 | - |
| 10 | community_reports_node | retrieval.py | 社群報告檢索 | - |
| 11 | followups_node | retrieval.py | 追問查詢生成 | - |
| 12 | rrf_merge_node | retrieval.py | RRF 結果合併 | - |
| 13 | hop_hybrid_node | retrieval.py | 跳躍混合搜尋 | - |
| 14 | rerank_node | rerank.py | 交叉編碼器重排 (40→12) | - |
| 15 | graph_seed_extract_node | graph.py | 從 chunks 萃取實體種子 | - |
| 16 | graph_traverse_node | graph.py | NebulaGraph 2-hop 遍歷 | - |
| 17 | subgraph_to_queries_node | graph.py | 子圖轉追問查詢 | - |
| 18 | chunk_expander_node | context.py | 區塊上下文擴展 | - |
| 19 | context_packer_node | context.py | OWASP LLM01 上下文轉義 | - |
| 20 | evidence_table_node | context.py | 結構化證據表構建 | - |
| 21 | groundedness_node | quality.py | 雙層接地性評估 (啟發式 + Ragas) | - |
| 22 | targeted_retry_node | quality.py | 智能失敗分析與重試策略 | - |
| 23 | interrupt_hitl_node | quality.py | HITL 中斷點設置 | - |
| 24 | final_answer_node | output.py | 最終回答生成 + OWASP LLM02 | ✓ chat/responses |
| 25 | telemetry_node | output.py | Langfuse 追蹤記錄 | - |

### 預算管理

**檔案**: `graph_workflow/types.py`

```python
@dataclass
class LoopBudget:
    max_loops: int = 3                    # 最大循環次數
    max_new_queries: int = 8              # 最大查詢次數
    max_context_tokens: int = 12000       # 最大上下文 tokens
    max_wall_time_seconds: float = 15.0   # 最大執行時間 (秒)

    # 當前使用量
    current_loops: int = 0
    current_queries: int = 0
    current_tokens: int = 0
    start_time: float = 0.0

    def is_exhausted(self) -> bool:
        """任何預算耗盡即返回 True"""
        elapsed = time.time() - self.start_time
        return (
            self.current_loops >= self.max_loops
            or self.current_queries >= self.max_new_queries
            or self.current_tokens >= self.max_context_tokens
            or elapsed >= self.max_wall_time_seconds
        )
```

---

## GraphRAG 檢索模式

### 三種模式對比

| 特性 | LOCAL | GLOBAL | DRIFT |
|------|-------|--------|-------|
| **焦點** | 實體級精準 | 社群級聚合 | 迭代式探索 |
| **初始步驟** | 混合搜尋 | 社群報告 | 全局搜尋 |
| **迭代次數** | 1 次 | 1 次 | 最多 3 次 |
| **結果塊數** | 12 | 12 | 15 |
| **執行時間** | <2s | 2-5s | 5-15s |
| **圖遍歷深度** | 2 跳 | 1 跳 | 2 跳 |
| **社群數** | N/A | 5 | 3→8 |
| **適用場景** | 特定實體查詢 | 概覽/比較問題 | 複雜研究問題 |

### LOCAL 模式 (實體導向)

**檔案**: `services/retrieval/local_mode.py`

```
Seed Results (20條)
        ↓
Extract Entity Seeds (10個實體)
        ↓
Graph Traversal (最多 2 跳)
        ↓
Entity Chunks + Seed Results
        ↓
Expand & Merge (最多 12條)
        ↓
Final Results
```

**配置**:
```python
LocalModeConfig:
    seed_limit: 20
    min_seed_score: 0.35
    max_hops: 2
    max_entities_per_hop: 10
    max_expanded_chunks: 12
```

### GLOBAL 模式 (社群導向)

**檔案**: `services/retrieval/global_mode.py`

```
Query Embedding
        ↓
Search Community Reports (5個社群)
        ↓
Generate Follow-up Queries (3個追問)
        ↓
Local Mode Drill-Down (並行執行 3 個)
        ↓
Merge Results
        ↓
Final Chunks (12條)
```

**配置**:
```python
GlobalModeConfig:
    max_communities: 5
    min_community_score: 0.3
    community_levels: [1, 2]  # 高層 + 細節
    max_followups: 3
    enable_drill_down: True
```

### DRIFT 模式 (探索導向)

**檔案**: `services/retrieval/drift_mode.py`

```
Initial Global Search
        ↓
While novelty_score > 0.3 AND iterations < 3 AND time < 15s:
    ├─ Get Neighboring Communities
    ├─ Generate Expansion Query
    ├─ Execute Local Search
    ├─ Calculate Novelty (new_chunks / all_chunks)
    └─ Boost Multi-Found Chunks
        ↓
Rank Final Chunks
        ↓
Return Results (15條)
```

**配置**:
```python
DriftModeConfig:
    max_iterations: 3
    max_queries: 8
    max_wall_time_seconds: 15.0
    initial_communities: 3
    expansion_factor: 1.5
    max_communities: 8
    explore_depth: 2
    min_novelty_score: 0.3
    max_final_chunks: 15
```

**終止條件**:
- `max_iterations` 達到
- `max_queries` 達到
- `max_wall_time_seconds` 超時
- `novelty_score < 0.3` (新穎性過低)

---

## 資料攝取與分塊

### 雙管線架構

**檔案**: `services/ingestion/coordinator.py`

```
DocumentInput
      ↓
IngestionCoordinator (路由判斷)
      │
      ├─→ [CURATED Pipeline] (YAML + Markdown)
      │    ├── 解析 YAML frontmatter
      │    ├── 驗證 Schema
      │    ├── 根據 doc_type 選擇 Chunker
      │    └── 生成結構化 chunks
      │
      └─→ [RAW Pipeline] (PDF/DOCX/HTML/TXT)
           ├── 文件內容提取
           ├── 推斷 doc_type
           └── 通用分塊
      ↓
Parallel Storage
├── MinIO: 規範化 JSON
├── Qdrant: 向量索引
├── OpenSearch: 全文索引
├── NebulaGraph: 知識圖譜 (可選)
└── PostgreSQL: 元資料
```

### 6 種專用分塊器

**檔案**: `services/ingestion/chunkers/`

| 分塊器 | 適用文檔類型 | 特殊處理 |
|--------|-------------|----------|
| `generic.py` | GENERIC, FAQ | 基礎章節分割 |
| `physician.py` | PHYSICIAN | 醫師檔案、專長、門診時間表 |
| `procedure.py` | PROCEDURE | 步驟、條件、費用、表單 |
| `guide.py` | GUIDE_LOCATION, GUIDE_TRANSPORT | 位置資訊、交通方式 |
| `hospital_team.py` | HOSPITAL_TEAM | 團隊成員、服務範圍、服務時間 |

### Chunk 結構

```python
Chunk(
    id="c_{doc_id}_{position}_{content_hash}",
    doc_id=doc_id,
    doc_version=version,
    content=text,
    metadata=ChunkMetadata(
        chunk_type=ChunkType,       # PARAGRAPH, TABLE, LIST, STEPS, etc.
        section_title=str,
        position_in_doc=int,
        parent_chunk_id=optional,
        entity_ids=list[str],       # 關聯實體
        custom_fields=dict
    ),
    char_count=len,
    contextual_content=optional     # 相鄰 chunk 上下文
)
```

### YAML Frontmatter 範例

```yaml
---
doc_type: physician
title: 王大明 醫師
description: 心臟科專家
department: 心臟內科
language: zh-TW
name:
  zh-Hant: 王大明
  en: Wang
specialties:
  - 心臟超音波
  - 心導管檢查
schedule:
  - day: 週一
    period: 上午
    location: 特診
contact:
  phone: "02-1234-5678"
  email: "wang@hospital.tw"
education:
  - degree: 醫學博士
    school: 國立台灣大學
acl_groups: [public, staff]
---

# 王大明醫師的專業背景

[正文內容...]
```

---

## 狀態管理

### GraphRAGState 完整結構

**檔案**: `graph_workflow/types.py`

```python
class GraphRAGState(TypedDict, total=False):
    # === 核心輸入 ===
    messages: list[BaseMessage]       # 對話歷史
    question: str                     # 原始問題
    normalized_question: str          # 標準化問題
    resolved_question: str            # 上下文解析後問題
    user_language: str                # 語言代碼 (zh-TW, en, ja)

    # === 安全與存取控制 ===
    guard_blocked: bool               # 被安全檢查阻擋
    guard_reason: str                 # 阻擋原因
    acl_denied: bool                  # 存取被拒絕
    acl_groups: list[str]             # 使用者 ACL 群組
    tenant_id: str                    # 租戶 ID
    filter_context: FilterContext     # 多租戶過濾上下文

    # === 意圖與路由 ===
    query_mode: str                   # "local", "global", "drift", "direct"
    intent_reasoning: str             # 意圖判斷理由

    # === 循環預算 ===
    budget: LoopBudget                # 預算追蹤物件

    # === 檢索狀態 ===
    seed_results: RetrievalResult     # 初始搜尋結果
    community_reports: list[dict]     # 社群摘要
    followup_queries: list[str]       # 追問查詢
    graph_subgraph: dict              # 圖遍歷結果
    merged_results: RetrievalResult   # 合併後結果
    reranked_chunks: list[SearchResult]  # 重排後 chunks

    # === 上下文構建 ===
    expanded_chunks: list[SearchResult]  # 擴展後 chunks
    context_text: str                 # 打包的上下文
    context_tokens: int               # 上下文 token 數

    # === 證據與接地性 ===
    evidence_table: list[EvidenceItem]  # 結構化證據
    groundedness_status: str          # "pass", "retry", "needs_review"
    groundedness_score: float         # 0.0-1.0

    # === 人機協作 (HITL) ===
    hitl_required: bool               # 需要人工審核
    hitl_resolved: bool               # 人工審核完成
    hitl_feedback: str                # 人工回饋
    hitl_approved: bool               # 人工批准狀態
    hitl_triggered_at: str            # ISO 時間戳
    hitl_timeout_at: float            # Unix 超時時間
    hitl_timed_out: bool              # 是否超時
    hitl_rejected: bool               # 是否被拒絕

    # === 版本控制 ===
    index_version: str                # 向量索引版本
    pipeline_version: str             # 管線版本
    prompt_version: str               # 提示版本
    config_hash: str                  # 配置雜湊

    # === 輸出 ===
    final_answer: str                 # 最終回答
    confidence: float                 # 信心分數

    # === 可觀測性 ===
    trace_id: str                     # Langfuse trace ID
    retrieval_path: list[str]         # 檢索路徑記錄
    timing: dict[str, float]          # 各階段耗時 (ms)

    # === LLM 後端 ===
    agent_backend: str                # "responses" 或 "chat"
```

---

## API 端點

### 端點總覽

| 端點 | 方法 | 功能 |
|------|------|------|
| `/api/v1/rag/ask/stream` | POST | 原生格式串流問答 |
| `/api/v1/rag/ask/stream_chat` | POST | OpenAI 相容串流問答 |
| `/api/v1/rag/vectorize` | POST | 單一文件向量化 |
| `/api/v1/rag/vectorize/file` | POST | 檔案上傳向量化 |
| `/api/v1/rag/vectorize/directory` | POST | 目錄批次向量化 |
| `/health` | GET | 基本健康檢查 |
| `/health/ready` | GET | 服務就緒檢查 |
| `/health/live` | GET | 存活探針 |
| `/health/concurrency` | GET | 並發狀態監控 |

### `/api/v1/rag/ask/stream` - 原生格式

**請求**:
```json
{
    "question": "什麼是糖尿病？",
    "acl_groups": ["public"],
    "tenant_id": "default",
    "include_sources": true,
    "enable_hitl": false
}
```

**SSE 事件**:
```
data: {"event": "response.start", "trace_id": "abc123"}
data: {"event": "response.chunk", "content": "糖尿病是"}
data: {"event": "response.sources", "sources": [...]}
data: {"event": "response.done", "confidence": 0.95}
```

### `/api/v1/rag/ask/stream_chat` - OpenAI 相容

**請求**:
```json
{
    "messages": [
        {"role": "system", "content": "你是專業的醫療助理"},
        {"role": "user", "content": "什麼是糖尿病？"}
    ],
    "stream": true,
    "backend": "responses"
}
```

**SSE 回應** (OpenAI 格式):
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"糖尿病是"}}]}
data: [DONE]
```

**Python OpenAI SDK 整合**:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:18000/api/v1/rag",
    api_key="not-needed"
)

stream = client.chat.completions.create(
    model="graphrag",
    messages=[{"role": "user", "content": "什麼是糖尿病？"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### `/health/concurrency` - 並發監控

**回應**:
```json
{
    "status": "healthy",
    "backends": {
        "default": {"limit": 10, "in_use": 2, "waiting": 0},
        "chat": {"limit": 20, "in_use": 5, "waiting": 1},
        "responses": {"limit": 50, "in_use": 10, "waiting": 0},
        "embedding": {"limit": 30, "in_use": 8, "waiting": 0}
    },
    "summary": {
        "total_in_use": 25,
        "total_waiting": 1,
        "total_limit": 110
    }
}
```

---

## 文件向量化 API

### 端點總覽

| 端點 | 方法 | 用途 |
|------|------|------|
| `/api/v1/rag/vectorize` | POST | 單一文檔向量化（JSON） |
| `/api/v1/rag/vectorize/file` | POST | 檔案上傳向量化 |
| `/api/v1/rag/vectorize/directory` | POST | 批次目錄掃描向量化 |
| `/api/v1/rag/vectorize/status/{job_id}` | GET | 查詢工作狀態 |

### 處理管道類型

| Pipeline | 適用檔案 | 說明 |
|----------|---------|------|
| `curated` | `.md` 檔案 | 解析 YAML frontmatter，根據 doc_type 選擇專用 Chunker |
| `raw` | `.pdf`, `.docx`, `.txt` 等 | 使用通用 Chunker 處理 |

### 處理模式

| 模式 | 說明 |
|------|------|
| `update` | 增量更新，只處理新增或修改的文檔（預設） |
| `override` | 覆蓋模式，清空現有資料後重建（⚠️ 不可逆）|

### `/api/v1/rag/vectorize` - 單一文檔向量化

**請求**:
```bash
curl -X POST "http://localhost:18000/api/v1/rag/vectorize" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "---\ntitle: 住院須知\ndoc_type: procedure.admission\n---\n# 住院流程\n內容...",
    "doc_type": "procedure.admission",
    "pipeline": "curated",
    "async_mode": false
  }'
```

**請求參數**:

| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| `content` | string | ✓ | 文檔內容（Markdown 或純文字） |
| `doc_type` | string | - | 文檔類型（如 procedure.admission） |
| `filename` | string | - | 原始檔名 |
| `metadata` | object | - | 額外 metadata |
| `acl_groups` | array | - | 存取控制群組（預設 ["public"]） |
| `tenant_id` | string | - | 租戶識別碼（預設 "default"） |
| `async_mode` | boolean | - | 是否非同步執行（預設 false） |
| `pipeline` | string | - | 處理管道（curated/raw，預設 curated） |

**回應範例**:
```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "status": "completed",
  "message": "已成功處理 1 個文檔",
  "chunk_count": 15,
  "entities_extracted": 8,
  "relations_extracted": 5
}
```

### `/api/v1/rag/vectorize/file` - 檔案上傳向量化

**請求**:
```bash
# 上傳 Markdown 檔案
curl -X POST "http://localhost:18000/api/v1/rag/vectorize/file" \
  -F "file=@住院須知.md" \
  -F "doc_type=procedure.admission" \
  -F "acl_groups=public"

# 上傳 PDF 檔案（自動使用 raw pipeline）
curl -X POST "http://localhost:18000/api/v1/rag/vectorize/file" \
  -F "file=@document.pdf" \
  -F "async_mode=true"
```

**表單參數**:

| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| `file` | file | ✓ | 上傳的檔案 |
| `doc_type` | string | - | 文檔類型 |
| `acl_groups` | string | - | 存取控制群組（逗號分隔） |
| `tenant_id` | string | - | 租戶識別碼 |
| `async_mode` | boolean | - | 是否非同步執行 |

### `/api/v1/rag/vectorize/directory` - 批次目錄向量化

**請求**:
```bash
# 使用預設目錄 (rag_test_data/docs)，增量更新
curl -X POST "http://localhost:18000/api/v1/rag/vectorize/directory" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "update",
    "enable_graph": true,
    "async_mode": true
  }'

# 指定目錄 + 覆蓋模式
curl -X POST "http://localhost:18000/api/v1/rag/vectorize/directory" \
  -H "Content-Type: application/json" \
  -d '{
    "directory": "/app/rag_test_data/docs",
    "mode": "override",
    "recursive": true,
    "file_extensions": [".md"],
    "enable_graph": true,
    "enable_community_detection": false,
    "async_mode": true
  }'
```

**請求參數**:

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `directory` | string | rag_test_data/docs | 目錄路徑 |
| `mode` | string | update | 處理模式（update/override） |
| `pipeline` | string | curated | 處理管道（curated/raw） |
| `recursive` | boolean | true | 是否遞迴掃描子目錄 |
| `file_extensions` | array | [".md"] | 要處理的副檔名列表 |
| `async_mode` | boolean | true | 是否非同步執行 |
| `acl_groups` | array | ["public"] | 存取控制群組 |
| `tenant_id` | string | "default" | 租戶識別碼 |
| `enable_graph` | boolean | true | 是否啟用圖譜抽取（實體、關係） |
| `enable_community_detection` | boolean | false | 是否啟用社群偵測（需先啟用 enable_graph） |

**回應範例**（非同步模式）:
```json
{
  "job_id": "job_batch123456",
  "status": "processing",
  "message": "正在處理 50 個文檔...",
  "documents_processed": 0,
  "progress": 0
}
```

### `/api/v1/rag/vectorize/status/{job_id}` - 查詢工作狀態

**請求**:
```bash
curl "http://localhost:18000/api/v1/rag/vectorize/status/job_a1b2c3d4e5f6"
```

**回應範例**（執行中）:
```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "status": "running",
  "message": "已處理 25/50 個文檔",
  "documents_processed": 25,
  "documents_failed": 0,
  "progress": 50.0
}
```

**回應範例**（已完成）:
```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "status": "completed",
  "message": "已處理 50/50 個文檔",
  "documents_processed": 50,
  "documents_failed": 0,
  "chunk_count": 320,
  "entities_extracted": 150,
  "relations_extracted": 85,
  "progress": 100.0
}
```

### 完整流程範例

```bash
# 1. 啟動服務
docker compose -f docker-compose.graphrag.yml --profile development up -d

# 2. 初始化 MinIO buckets（首次執行）
docker compose -f docker-compose.graphrag.yml --profile init run --rm minio-init

# 3. 執行批次向量化（非同步）
curl -X POST "http://localhost:18000/api/v1/rag/vectorize/directory" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "update",
    "enable_graph": true,
    "async_mode": true
  }'
# 回應: {"job_id": "job_abc123", ...}

# 4. 查詢進度
curl "http://localhost:18000/api/v1/rag/vectorize/status/job_abc123"

# 5. 在 OpenSearch Dashboards 驗證（http://localhost:15601）
# Dev Tools 中執行:
# GET graphrag_chunks/_count

# 6. 測試問答
curl -X POST "http://localhost:18000/api/v1/rag/ask/stream" \
  -H "Content-Type: application/json" \
  -d '{"question": "住院需要準備什麼文件？"}'
```

### 資料攝取流程圖

```
Document Upload (API)
    ↓
IngestionCoordinator (路由判斷)
    │
    ├─→ [CURATED Pipeline] (YAML + Markdown)
    │    ├── 解析 YAML frontmatter
    │    ├── 驗證 Schema
    │    ├── 根據 doc_type 選擇 Chunker
    │    └── 生成結構化 chunks
    │
    └─→ [RAW Pipeline] (PDF/DOCX/HTML/TXT)
         ├── 文件內容提取
         ├── 推斷 doc_type
         └── 通用分塊
    ↓
Parallel Storage
├── MinIO: 規範化 JSON
├── Qdrant: 向量索引 (Dense + Sparse)
├── OpenSearch: 全文索引 (BM25)
├── NebulaGraph: 知識圖譜 (若 enable_graph=true)
└── PostgreSQL: 元資料
```

---

## 環境變數配置

### 基礎設施連線

```bash
# NebulaGraph
NEBULA_HOST=nebula-graphd
NEBULA_PORT=9669
NEBULA_GRAPH_SPACE=graphrag

# Qdrant
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION_CHUNKS=graphrag_chunks
QDRANT_COLLECTION_COMMUNITIES=graphrag_communities

# OpenSearch
OPENSEARCH_URL=http://opensearch:9200
OPENSEARCH_INDEX_CHUNKS=graphrag_chunks

# MinIO
MINIO_ENDPOINT=minio:9000
MINIO_BUCKET_DOCUMENTS=documents

# PostgreSQL
POSTGRES_URL=postgresql://graphrag:graphrag_secret@postgres:5432/graphrag
```

### LLM 配置

```bash
OPENAI_API_BASE=http://192.168.50.152:1234/v1
OPENAI_API_KEY=lm-studio

EMBEDDING_MODEL=text-embedding-embeddinggemma-300m
EMBEDDING_DIMENSION=768
CHAT_MODEL=openai/gpt-oss-20b
CHAT_TEMPERATURE=0.1
```

### LLM 並發控制

```bash
# 並發限制
LLM_MAX_CONCURRENT_DEFAULT=10
LLM_MAX_CONCURRENT_CHAT=20
LLM_MAX_CONCURRENT_RESPONSES=50
LLM_MAX_CONCURRENT_EMBEDDING=30

# 優先級排程 (可選)
LLM_PRIORITY_ENABLED=false
LLM_PRIORITY_STARVATION_THRESHOLD=5.0
```

### GraphRAG 工作流程參數

```bash
# 循環預算
GRAPHRAG_MAX_LOOPS=3
GRAPHRAG_MAX_NEW_QUERIES=8
GRAPHRAG_MAX_CONTEXT_TOKENS=12000
GRAPHRAG_MAX_WALL_TIME_SECONDS=15.0

# 檢索參數
GRAPHRAG_SEED_TOP_K=40
GRAPHRAG_RERANK_TOP_K=12
GRAPHRAG_GRAPH_MAX_HOPS=2

# RRF 權重
GRAPHRAG_RRF_WEIGHT_DENSE=0.4
GRAPHRAG_RRF_WEIGHT_SPARSE=0.3
GRAPHRAG_RRF_WEIGHT_FTS=0.3

# 語義快取
SEMANTIC_CACHE_ENABLED=true
SEMANTIC_CACHE_SIMILARITY_THRESHOLD=0.90
```

### 安全與 HITL

```bash
ENABLE_INPUT_GUARD=true
MAX_QUESTION_LENGTH=1000
ENABLE_INJECTION_DETECTION=true

HITL_ENABLED=true
HITL_TIMEOUT_SECONDS=300.0
HITL_LOW_CONFIDENCE_THRESHOLD=0.4
```

### Langfuse 可觀測性

```bash
LANGFUSE_PUBLIC_KEY=lf_pk_graphrag_dev_key
LANGFUSE_SECRET_KEY=lf_sk_graphrag_dev_key
LANGFUSE_HOST=http://langfuse-web:3000
LANGFUSE_ENABLED=true
```

---

## 開發環境與部署

### Docker Compose 架構

```bash
# 開發模式 (App + 所有基礎設施 + Langfuse)
docker compose -f docker-compose.graphrag.yml --profile development up -d

# 生產模式
docker compose -f docker-compose.graphrag.yml --profile production up -d

# 僅基礎設施 (不含 App)
docker compose -f docker-compose.graphrag.yml up -d

# 初始化 MinIO buckets
docker compose -f docker-compose.graphrag.yml --profile init run --rm minio-init
```

### 服務埠號對照表

#### GraphRAG 核心服務

| 服務 | 內部埠 | 外部埠 | 說明 |
|------|--------|--------|------|
| **PostgreSQL** | 5432 | 15432 | 元資料庫 |
| **Redis** | 6379 | 16379 | 快取 |
| **Qdrant REST** | 6333 | 16333 | 向量搜尋 API |
| **Qdrant gRPC** | 6334 | 16334 | 向量搜尋 gRPC |
| **MinIO API** | 9000 | 19000 | 物件儲存 |
| **MinIO Console** | 9001 | 19001 | 管理介面 |
| **OpenSearch** | 9200 | 19200 | 全文搜尋 |
| **App** | 8000 | 18000 | GraphRAG API |
| **NebulaGraph** | 9669 | 29669 | 圖查詢服務 |

#### Langfuse 服務

| 服務 | 內部埠 | 外部埠 |
|------|--------|--------|
| **Langfuse Web** | 3000 | 23000 |
| **Langfuse PostgreSQL** | 5432 | 25432 |
| **ClickHouse** | 8123 | 28123 |

### 健康檢查

```bash
curl http://localhost:18000/health              # 基本健康
curl http://localhost:18000/health/ready        # 服務就緒
curl http://localhost:18000/health/concurrency  # 並發狀態
curl http://localhost:23000/api/health          # Langfuse
curl http://localhost:16333/health              # Qdrant
curl http://localhost:19200/_cluster/health     # OpenSearch
```

---

## 附錄：與 chatbot_rag 的比較

| 特性 | chatbot_rag | chatbot_graphrag |
|------|-------------|------------------|
| 檢索方式 | 向量搜尋 + 快取 | 向量 + 稀疏 + 圖 + 社群 |
| 知識圖譜 | 無 | NebulaGraph 完整支援 |
| 社群偵測 | 無 | Leiden Algorithm |
| 查詢模式 | 單一管線 | 4 種模式智能路由 |
| 儲存系統 | Qdrant, Redis | Qdrant + OpenSearch + MinIO + PostgreSQL + NebulaGraph |
| 文件處理 | 文件上傳 | 雙管線 + 6 種專用分塊器 |
| 多租戶 | 基本 | 完整 ACL + 租戶隔離 |
| HITL | 有限 | 完整 interrupt-before 機制 |
| 節點數 | 14 | 22 |
| 安全防護 | 基本 | OWASP LLM01 + LLM02 |
| 評估框架 | 無 | Ragas 10% 抽樣 |
| **LLM 並發控制** | ✓ 完整實現 | ✓ 完整移植 |
| 並發後端數 | 4 | 4 |
| 優先級排程 | ✓ | ✓ |
| 飢餓保護 | ✓ | ✓ |

---

*本文檔基於 `src/chatbot_graphrag` 原始碼分析生成，版本 4.1.0*
