# GraphRAG Docker Compose 啟動指南

本文件說明如何使用 Docker Compose 啟動 GraphRAG 系統，並詳細介紹各元件的架構與運作機制。

## 目錄

- [系統架構總覽](#系統架構總覽)
- [快速啟動](#快速啟動)
- [基礎設施元件](#基礎設施元件)
- [啟動模式說明](#啟動模式說明)
- [API 端點](#api-端點)
- [資料流程](#資料流程)
- [維運指令](#維運指令)
- [故障排除](#故障排除)

---

## 系統架構總覽

GraphRAG 是一個結合知識圖譜與向量檢索的增強型 RAG 系統，採用以下技術棧：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            GraphRAG 系統架構                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │   FastAPI   │────▶│  LangGraph  │────▶│   LLM API   │                   │
│  │  (18000)    │     │  Workflow   │     │ (OpenAI 相容)│                   │
│  └─────────────┘     └─────────────┘     └─────────────┘                   │
│         │                   │                                               │
│         ▼                   ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        服務層 (Services)                             │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐           │   │
│  │  │ Ingestion │ │ Retrieval │ │   Graph   │ │  Search   │           │   │
│  │  │ Pipeline  │ │   Modes   │ │ Traversal │ │  Hybrid   │           │   │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                   │                   │                           │
│         ▼                   ▼                   ▼                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      基礎設施層 (Infrastructure)                      │   │
│  │                                                                       │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │   │
│  │  │NebulaGraph│  │  Qdrant  │  │OpenSearch│  │ Postgres │  │  Redis │ │   │
│  │  │  (圖庫)   │  │ (向量庫) │  │ (全文檢索)│  │ (中繼資料)│  │ (快取) │ │   │
│  │  │  29669   │  │  16333   │  │  19200   │  │  15432   │  │ 16379  │ │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └────────┘ │   │
│  │                                                                       │   │
│  │  ┌──────────┐                                                        │   │
│  │  │  MinIO   │                                                        │   │
│  │  │(物件存儲)│                                                        │   │
│  │  │19000/19001│                                                       │   │
│  │  └──────────┘                                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 元件職責

| 元件 | 用途 | 外部埠號 |
|------|------|----------|
| **NebulaGraph** | 知識圖譜存儲 (Entity-Relation-Event) | 29669 |
| **Qdrant** | 向量資料庫 (Dense + Sparse vectors) | 16333, 16334 |
| **OpenSearch** | 全文檢索引擎 | 19200 |
| **PostgreSQL** | 中繼資料存儲 (Doc, Chunk, ACL) | 15432 |
| **Redis** | 任務佇列與快取 | 16379 |
| **MinIO** | 物件存儲 (原始文件、Chunk) | 19000, 19001 |
| **FastAPI** | GraphRAG API 應用程式 | 18000 |

---

## 快速啟動

### 前置需求

- Docker 20.10+
- Docker Compose 2.0+
- 至少 8GB RAM (建議 16GB)
- 至少 20GB 磁碟空間

### 步驟 1：建立環境檔案

```bash
# 複製環境範本
cp .env.example .env

# 編輯環境變數 (設定 LLM API 等)
vim .env
```

必要的環境變數：

```bash
# LLM API 設定 (OpenAI 相容)
OPENAI_API_BASE=http://192.168.50.152:1234/v1
OPENAI_API_KEY=lm-studio

# Embedding 模型
EMBEDDING_MODEL=text-embedding-embeddinggemma-300m-qat
EMBEDDING_DIMENSION=768

# Chat 模型
CHAT_MODEL=openai/gpt-oss-20b
```

### 步驟 2：啟動基礎設施

```bash
# 啟動所有基礎設施服務 (包含 NebulaGraph v3.8.0)
docker compose -f docker-compose.graphrag.yml up -d

# 等待所有服務健康 (約 60-90 秒)
# 注意：nebula-console 會自動執行 ADD HOSTS 註冊 storaged
docker compose -f docker-compose.graphrag.yml ps

# 確認 NebulaGraph 已正確初始化
docker logs graphrag-nebula-console 2>&1 | grep -i "registered\|ready"

# 初始化 MinIO buckets (只需執行一次)
docker compose -f docker-compose.graphrag.yml --profile init up minio-init
```

**NebulaGraph 初始化說明**:
- `nebula-console` 容器會在啟動後自動執行 `ADD HOSTS` 命令
- 這會將 `nebula-storaged:9779` 註冊到 metad 叢集中
- 首次啟動可能需要等待 30-60 秒完成註冊

### 步驟 3：啟動應用程式

```bash
# 開發模式 (含 hot-reload)
docker compose -f docker-compose.graphrag.yml --profile development up app-dev

# 正式環境模式
docker compose -f docker-compose.graphrag.yml --profile production up app-prod
```

### 步驟 4：驗證服務

```bash
# 健康檢查
curl http://localhost:18000/health

# 就緒檢查 (確認所有依賴服務)
curl http://localhost:18000/health/ready
```

---

## 基礎設施元件

### NebulaGraph (知識圖譜) - v3.8.0

NebulaGraph 是分散式圖資料庫，用於存儲：

- **Entity (實體)**: 人物、地點、組織、概念
- **Relation (關係)**: 實體之間的連結
- **Event (事件)**: 時間相關的發生事項
- **Community Report (社群報告)**: Leiden 演算法產生的社群摘要

```
┌─────────────────────────────────────────────────────────────────┐
│                    NebulaGraph 叢集 (v3.8.0)                     │
│                                                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐ │
│  │  metad     │  │ storaged   │  │  graphd    │  │ console  │ │
│  │ (中繼資料)  │──│  (存儲)    │──│  (查詢)    │──│(自動註冊) │ │
│  │   9559     │  │   9779     │  │  29669     │  │          │ │
│  └────────────┘  └────────────┘  └────────────┘  └──────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**啟動順序**: metad → storaged → graphd → console (自動執行 ADD HOSTS)

**重要**: `nebula-console` 會在啟動時自動執行 `ADD HOSTS` 命令，將 storaged 註冊到叢集中。這是 NebulaGraph 正常運作的必要步驟。

**連線測試**:
```bash
# 進入 console 容器
docker exec -it graphrag-nebula-console nebula-console -addr nebula-graphd -port 9669 -u root -p nebula

# 測試指令
nebula> SHOW HOSTS;        # 應該看到 nebula-storaged:9779
nebula> SHOW SPACES;       # 列出所有 space
```

### Qdrant (向量資料庫)

Qdrant 支援雙向量模式：

- **Dense Vector**: 語意嵌入 (768 維)
- **Sparse Vector**: SPLADE 稀疏向量 (關鍵詞權重)

```python
# Collections 結構
graphrag_chunks      # 文件 chunk 向量
graphrag_cache       # 語意快取
graphrag_communities # 社群報告向量
```

**Web UI**: http://localhost:16333/dashboard

### OpenSearch (全文檢索)

提供 BM25 全文檢索，與向量搜尋進行 RRF (Reciprocal Rank Fusion) 融合。

**Dashboards UI**: http://localhost:15601 (需啟用 debug profile)

### PostgreSQL (中繼資料)

存儲結構化中繼資料：

```sql
-- 主要資料表
docs          -- 文件記錄
doc_versions  -- 文件版本
chunks        -- Chunk 中繼資料
acl_entries   -- 存取控制列表
tenant_scopes -- 多租戶範圍
trace_links   -- 追蹤連結
```

### Redis (快取與佇列)

- **快取**: 語意查詢快取
- **佇列**: 背景任務 (ingestion, community 計算)

### MinIO (物件存儲)

```
buckets/
├── documents/   # 原始文件
├── chunks/      # Chunk 內容
├── canonical/   # 正規化版本
├── assets/      # 附件 (圖片等)
└── reports/     # 社群報告
```

**Console UI**: http://localhost:19001 (帳號: minioadmin / minioadmin123)

---

## 啟動模式說明

### Profile 說明

Docker Compose 使用 `--profile` 來區分不同啟動模式：

| Profile | 用途 | 啟動的服務 |
|---------|------|-----------|
| (預設) | 基礎設施 | nebula, qdrant, opensearch, postgres, redis, minio |
| `init` | 初始化 | minio-init (建立 buckets) |
| `development` | 開發模式 | app-dev (含 hot-reload) |
| `production` | 正式環境 | app-prod (8 workers) |
| `workers` | 背景工作 | ingest-worker, community-worker |
| `debug` | 除錯工具 | nebula-console, opensearch-dashboards |

### 常用啟動組合

```bash
# 1. 僅基礎設施 (用於本機開發)
docker compose -f docker-compose.graphrag.yml up -d

# 2. 完整開發環境
docker compose -f docker-compose.graphrag.yml \
  --profile development \
  --profile init \
  up -d

# 3. 正式環境 + 背景工作者
docker compose -f docker-compose.graphrag.yml \
  --profile production \
  --profile workers \
  up -d

# 4. 開發環境 + 除錯工具
docker compose -f docker-compose.graphrag.yml \
  --profile development \
  --profile debug \
  up -d
```

---

## API 端點

### 文件向量化 (Ingestion)

```bash
# POST /api/v1/rag/vectorize - JSON 內容
curl -X POST http://localhost:18000/api/v1/rag/vectorize \
  -H "Content-Type: application/json" \
  -d '{
    "content": "# 文件標題\n\n這是文件內容...",
    "doc_type": "procedure",
    "tenant_id": "default",
    "acl_groups": ["public"],
    "async_mode": false
  }'

# POST /api/v1/rag/vectorize/file - 檔案上傳
curl -X POST http://localhost:18000/api/v1/rag/vectorize/file \
  -F "file=@document.md" \
  -F "doc_type=guide" \
  -F "tenant_id=default"

# GET /api/v1/rag/vectorize/status/{job_id} - 查詢任務狀態
curl http://localhost:18000/api/v1/rag/vectorize/status/abc123
```

### 問答串流 (Ask Stream)

```bash
# POST /api/v1/rag/ask/stream - SSE 串流 (Responses API 格式)
curl -X POST http://localhost:18000/api/v1/rag/ask/stream \
  -H "Content-Type: application/json" \
  -d '{
    "question": "如何申請慢性病連續處方籤？",
    "tenant_id": "default",
    "stream": true,
    "include_sources": true
  }'

# 事件格式:
# data: {"type": "response.start", "trace_id": "..."}
# data: {"type": "response.chunk", "content": "..."}
# data: {"type": "response.sources", "sources": [...]}
# data: {"type": "response.done", "confidence": 0.85}
```

### Chat API (OpenAI 相容)

```bash
# POST /api/v1/rag/ask/stream_chat - OpenAI Chat 格式
curl -X POST http://localhost:18000/api/v1/rag/ask/stream_chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "你是醫院客服助理"},
      {"role": "user", "content": "掛號時間是什麼時候？"}
    ],
    "stream": true,
    "temperature": 0.1
  }'

# 回應格式 (OpenAI 相容):
# data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"..."}}]}
# data: [DONE]
```

### 健康檢查

```bash
# 基本健康檢查
curl http://localhost:18000/health
# {"status": "healthy", "version": "2.0.0"}

# 就緒檢查 (含依賴服務)
curl http://localhost:18000/health/ready
# {"status": "ready", "checks": {"nebula": true, "qdrant": true, "opensearch": true}}

# 存活檢查 (Kubernetes liveness probe)
curl http://localhost:18000/health/live
# {"status": "alive"}
```

---

## 資料流程

### 文件向量化流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           文件向量化流程                                     │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────┐
    │  文件輸入  │
    │ (MD/PDF)  │
    └────┬─────┘
         │
         ▼
    ┌──────────────────────────────────────────────────────┐
    │              Pipeline 分流                            │
    │                                                      │
    │   ┌────────────────┐      ┌────────────────┐        │
    │   │ Curated Pipeline│      │  Raw Pipeline  │        │
    │   │  (YAML+MD)      │      │ (PDF/DOCX/HTML)│        │
    │   │                │      │                │        │
    │   │ • Schema 驗證   │      │ • 格式轉換      │        │
    │   │ • 欄位抽取      │      │ • 智能分割      │        │
    │   │ • 型別路由      │      │ • 結構推斷      │        │
    │   └───────┬────────┘      └───────┬────────┘        │
    │           │                       │                 │
    │           └───────────┬───────────┘                 │
    │                       │                             │
    └───────────────────────┼─────────────────────────────┘
                            │
                            ▼
    ┌──────────────────────────────────────────────────────┐
    │              Chunking (智能分塊)                      │
    │                                                      │
    │   • Contextual Chunking (含上下文摘要)               │
    │   • 根據 doc_type 使用專用 chunker                   │
    │   • 保留結構邊界 (標題、段落、表格)                    │
    │                                                      │
    └───────────────────────┬──────────────────────────────┘
                            │
                            ▼
    ┌──────────────────────────────────────────────────────┐
    │              Entity Extraction (實體抽取)             │
    │                                                      │
    │   • LLM 抽取 Entity、Relation、Event                 │
    │   • 建立 Chunk → Entity 連結                         │
    │                                                      │
    └───────────────────────┬──────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
    ┌─────────┐       ┌─────────┐       ┌─────────┐
    │ Qdrant  │       │NebulaGraph│      │OpenSearch│
    │         │       │         │       │         │
    │ Dense + │       │ Entity  │       │ BM25    │
    │ Sparse  │       │ Relation│       │ 全文索引 │
    │ Vectors │       │ Event   │       │         │
    └─────────┘       └─────────┘       └─────────┘
```

### 查詢處理流程 (LangGraph Workflow)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LangGraph 查詢處理流程                                │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────┐
                              │  使用者  │
                              │  提問   │
                              └────┬────┘
                                   │
                                   ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  PHASE 1: 前處理                                                      │
    │                                                                      │
    │  guard → acl → normalize → cache_lookup                              │
    │                                                                      │
    │  • guard: 安全檢查 (注入偵測、有害內容)                                 │
    │  • acl: 存取控制驗證                                                  │
    │  • normalize: 語言偵測、問題正規化                                     │
    │  • cache_lookup: 語意快取查詢                                         │
    └─────────────────────────────────┬────────────────────────────────────┘
                                      │
                          ┌───────────┴───────────┐
                          │                       │
                    [cache hit]              [cache miss]
                          │                       │
                          ▼                       ▼
                   cache_response         intent_router
                          │                       │
                          │         ┌─────────────┼─────────────┐
                          │         │             │             │
                          │    [direct]      [local]       [global/drift]
                          │         │             │             │
                          │         ▼             ▼             ▼
                          │   direct_answer  hybrid_seed  community_reports
                          │         │             │             │
    ┌─────────────────────┼─────────┼─────────────┴─────────────┘
    │                     │         │             │
    │  PHASE 2: 檢索      │         │             ▼
    │                     │         │      ┌─────────────────┐
    │                     │         │      │ followups       │
    │                     │         │      │ (追蹤查詢)       │
    │                     │         │      └────────┬────────┘
    │                     │         │               │
    │                     │         └───────┬───────┘
    │                     │                 │
    │                     │                 ▼
    │                     │           rrf_merge
    │                     │           (RRF 融合)
    │                     │                 │
    │                     │                 ▼
    │                     │            rerank
    │                     │           (40→12)
    └─────────────────────┼─────────────────┼──────────────────────────────┘
                          │                 │
    ┌─────────────────────┼─────────────────┼──────────────────────────────┐
    │  PHASE 3: 圖譜探索  │                 │                              │
    │                     │                 ▼                              │
    │                     │      graph_seed_extract                        │
    │                     │            │                                   │
    │                     │            ▼                                   │
    │                     │       graph_traverse                           │
    │                     │       (2-hop 遍歷)                              │
    │                     │            │                                   │
    │                     │            ▼                                   │
    │                     │      hop_hybrid                                │
    │                     │      (額外檢索)                                 │
    └─────────────────────┼────────────┼───────────────────────────────────┘
                          │            │
    ┌─────────────────────┼────────────┼───────────────────────────────────┐
    │  PHASE 4: 上下文    │            │                                   │
    │                     │            ▼                                   │
    │                     │      chunk_expander                            │
    │                     │      (上下文擴展)                               │
    │                     │            │                                   │
    │                     │            ▼                                   │
    │                     │      context_packer                            │
    │                     │      (上下文打包)                               │
    │                     │            │                                   │
    │                     │            ▼                                   │
    │                     │      evidence_table                            │
    │                     │      (證據表建立)                               │
    └─────────────────────┼────────────┼───────────────────────────────────┘
                          │            │
    ┌─────────────────────┼────────────┼───────────────────────────────────┐
    │  PHASE 5: 品質檢查  │            │                                   │
    │                     │            ▼                                   │
    │                     │       groundedness                             │
    │                     │       (紮實度評估)                              │
    │                     │            │                                   │
    │                     │    ┌───────┼───────┐                          │
    │                     │    │       │       │                          │
    │                     │ [pass]  [retry] [needs_review]                │
    │                     │    │       │       │                          │
    │                     │    │       ▼       ▼                          │
    │                     │    │   targeted  interrupt_hitl               │
    │                     │    │    _retry    (人工審核)                   │
    │                     │    │       │       │                          │
    │                     │    │       └───────┤                          │
    │                     │    │               │                          │
    └─────────────────────┼────┼───────────────┼───────────────────────────┘
                          │    │               │
    ┌─────────────────────┼────┼───────────────┼───────────────────────────┐
    │  PHASE 6: 輸出      │    │               │                          │
    │                     │    └───────┬───────┘                          │
    │                     │            │                                   │
    │                     └────────────┼───────────────────────────────────┤
    │                                  │                                   │
    │                                  ▼                                   │
    │                           final_answer                               │
    │                           (最終回答)                                  │
    │                                  │                                   │
    │                                  ▼                                   │
    │                            cache_store                               │
    │                            (快取儲存)                                 │
    │                                  │                                   │
    │                                  ▼                                   │
    │                            telemetry                                 │
    │                            (追蹤記錄)                                 │
    │                                  │                                   │
    │                                  ▼                                   │
    │                               END                                    │
    └──────────────────────────────────────────────────────────────────────┘
```

### 三種查詢模式

| 模式 | 觸發條件 | 處理方式 |
|------|---------|---------|
| **Local** | 特定實體查詢 | Hybrid Search → Graph Traversal → Rerank |
| **Global** | 總覽/摘要查詢 | Community Reports → Follow-ups → Synthesis |
| **DRIFT** | 深入探索查詢 | 迭代探索 → 動態擴展 → 多輪檢索 |

---

## 維運指令

### 服務管理

```bash
# 查看所有服務狀態
docker compose -f docker-compose.graphrag.yml ps

# 查看特定服務日誌
docker compose -f docker-compose.graphrag.yml logs -f app-dev
docker compose -f docker-compose.graphrag.yml logs -f nebula-graphd

# 重啟特定服務
docker compose -f docker-compose.graphrag.yml restart app-dev

# 停止所有服務
docker compose -f docker-compose.graphrag.yml down

# 停止並清除資料 (⚠️ 危險)
docker compose -f docker-compose.graphrag.yml down -v
```

### 資料管理

```bash
# 備份 PostgreSQL
docker exec graphrag-postgres pg_dump -U graphrag graphrag > backup.sql

# 還原 PostgreSQL
cat backup.sql | docker exec -i graphrag-postgres psql -U graphrag graphrag

# 檢視 MinIO 儲存
docker exec graphrag-minio mc ls local/documents

# 清空 Qdrant collection
curl -X DELETE http://localhost:16333/collections/graphrag_chunks
```

### 效能監控

```bash
# 檢視容器資源使用
docker stats

# 檢視 NebulaGraph 狀態
curl http://localhost:39669/status

# 檢視 OpenSearch 叢集健康
curl http://localhost:19200/_cluster/health?pretty

# 檢視 Qdrant 集合資訊
curl http://localhost:16333/collections/graphrag_chunks
```

---

## 故障排除

### 常見問題

#### 1. NebulaGraph 無法啟動

```bash
# 檢查各服務日誌
docker compose -f docker-compose.graphrag.yml logs nebula-metad
docker compose -f docker-compose.graphrag.yml logs nebula-storaged
docker compose -f docker-compose.graphrag.yml logs nebula-graphd
docker compose -f docker-compose.graphrag.yml logs nebula-console

# 確認 storage 是否已註冊
docker exec -it graphrag-nebula-console nebula-console \
  -addr nebula-graphd -port 9669 -u root -p nebula \
  -e 'SHOW HOSTS;'

# 手動註冊 storage (如果自動註冊失敗)
docker exec -it graphrag-nebula-console nebula-console \
  -addr nebula-graphd -port 9669 -u root -p nebula \
  -e 'ADD HOSTS "nebula-storaged":9779;'
```

**版本升級後無法啟動**:
```bash
# 清理舊版本數據 (⚠️ 會刪除所有圖數據)
docker compose -f docker-compose.graphrag.yml down -v

# 刪除 NebulaGraph volumes
docker volume rm $(docker volume ls -q | grep nebula) 2>/dev/null || true

# 重新啟動
docker compose -f docker-compose.graphrag.yml up -d
```

#### 2. OpenSearch 記憶體不足

```bash
# 增加 JVM heap size
# 修改 docker-compose.graphrag.yml:
# OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g
```

#### 3. 應用程式無法連線基礎設施

```bash
# 確認網路
docker network ls | grep graphrag
docker network inspect graphrag-network

# 確認服務健康
docker compose -f docker-compose.graphrag.yml ps

# 測試連線
docker run --rm --network graphrag-network curlimages/curl curl -s http://qdrant:6333/health
```

#### 4. MinIO Bucket 不存在

```bash
# 重新初始化
docker compose -f docker-compose.graphrag.yml --profile init up minio-init
```

### 日誌位置

| 服務 | 日誌位置 |
|------|---------|
| app-dev/prod | `./logs/graphrag.log` |
| NebulaGraph | Docker volume: `nebula-*-logs` |
| PostgreSQL | `docker logs graphrag-postgres` |
| 其他服務 | `docker compose logs <service>` |

---

## 附錄：環境變數完整列表

```bash
# Application
APP_NAME=GraphRAG API
APP_VERSION=2.0.0
DEBUG=false

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=8

# NebulaGraph
NEBULA_HOST=nebula-graphd
NEBULA_PORT=9669
NEBULA_USER=root
NEBULA_PASSWORD=nebula
NEBULA_SPACE=graphrag

# Qdrant
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION_CHUNKS=graphrag_chunks

# PostgreSQL
POSTGRES_URL=postgresql://graphrag:graphrag_secret@postgres:5432/graphrag

# OpenSearch
OPENSEARCH_URL=http://opensearch:9200

# Redis
REDIS_URL=redis://redis:6379/0

# MinIO
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_SECURE=false

# LLM (OpenAI 相容)
OPENAI_API_BASE=http://192.168.50.152:1234/v1
OPENAI_API_KEY=lm-studio
CHAT_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=text-embedding-embeddinggemma-300m-qat
EMBEDDING_DIMENSION=768

# GraphRAG Workflow
GRAPHRAG_MAX_LOOPS=3
GRAPHRAG_MAX_CONTEXT_TOKENS=12000
GRAPHRAG_MAX_WALL_TIME_SECONDS=15.0

# Logging
LOG_LEVEL=INFO
LOG_TO_CONSOLE=true
LOG_TO_FILE=true
```
