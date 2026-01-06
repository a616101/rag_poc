# ChatBot GraphRAG

企業級智能問答系統，基於 **GraphRAG**（Graph-enhanced Retrieval-Augmented Generation）架構，結合知識圖譜與向量檢索實現高精準度問答。

## 系統特色

- **GraphRAG 架構** - 結合知識圖譜 (NebulaGraph) + 向量檢索 (Qdrant) + 全文搜索 (OpenSearch)
- **三路混合檢索** - Dense (768維) + Sparse (SPLADE) + Full-text (BM25) + RRF 融合
- **四種查詢模式** - Direct / Local / Global / Drift 智能路由
- **Leiden 社群偵測** - 多層級社群結構，自動生成社群報告
- **LLM 並發控制** - 多後端 Semaphore + 優先級排程 + 飢餓保護
- **SSE 即時串流** - Server-Sent Events 實時推送執行進度和回答
- **完整觀測性** - Langfuse 整合追蹤、Prompt 管理、評估器
- **OWASP 安全防護** - LLM01 (Prompt Injection) + LLM02 (Output Handling)
- **多租戶 ACL** - 完整的 tenant_id + acl_groups 存取控制

## 技術棧

| 類別 | 技術 |
|------|------|
| **Web 框架** | FastAPI + Uvicorn (ASGI) |
| **LLM 框架** | LangChain + LangGraph (22 節點工作流) |
| **向量資料庫** | Qdrant (3 集合: chunks, entities, communities) |
| **圖資料庫** | NebulaGraph (實體-關係圖譜) |
| **全文搜索** | OpenSearch (BM25 索引) |
| **關聯式資料庫** | PostgreSQL (元數據、ACL、HITL 檢查點) |
| **物件儲存** | MinIO (S3 相容) |
| **快取** | Redis (語義快取) |
| **可觀測性** | Langfuse (自建) + Ragas |
| **前端** | Angular 20+ / Svelte 4 (Widget) |
| **容器化** | Docker + Docker Compose |
| **套件管理** | uv (Astral) |

## 快速開始

```bash
# 1. 複製環境變數設定
cp .env.graphrag.example .env.graphrag

# 2. 啟動服務 (開發模式)
docker compose --profile development up -d

# 3. 向量化文件
curl -X POST http://localhost:18000/api/v1/rag/vectorize \
  -H "Content-Type: application/json" \
  -d '{"source": "default", "mode": "override"}'

# 4. 測試問答
curl -X POST http://localhost:18000/api/v1/rag/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "如何登入平台？"}'
```

## 文件目錄

### 入門指南
| 文件 | 說明 |
|------|------|
| [GETTING_STARTED.md](./GETTING_STARTED.md) | 快速入門指南 |
| [CONFIGURATION.md](./CONFIGURATION.md) | 環境配置參考 |

### 架構設計
| 文件 | 說明 |
|------|------|
| [chatbot_graphrag_architecture.md](./chatbot_graphrag_architecture.md) | GraphRAG 系統架構 (v4.1.0) |
| [SSE_STREAMING.md](./SSE_STREAMING.md) | SSE 串流處理 |
| [HIGH_CONCURRENCY_OPTIMIZATION.md](./HIGH_CONCURRENCY_OPTIMIZATION.md) | 高並發優化指南 |

### API 與整合
| 文件 | 說明 |
|------|------|
| [API_REFERENCE.md](./API_REFERENCE.md) | API 參考手冊 |
| [LANGFUSE_INTEGRATION.md](./LANGFUSE_INTEGRATION.md) | Langfuse 整合（Trace / Prompt / Scores） |

### 運維指南
| 文件 | 說明 |
|------|------|
| [GRAPHRAG_DOCKER_GUIDE.md](./GRAPHRAG_DOCKER_GUIDE.md) | Docker Compose 啟動指南 |
| [DEPLOYMENT.md](./DEPLOYMENT.md) | 部署指南（Docker / Nginx） |
| [SECURITY.md](./SECURITY.md) | 安全防護機制 |
| [TESTING.md](./TESTING.md) | 測試指南 |

### 資料處理腳本
| 文件 | 說明 |
|------|------|
| [SCRIPTS_DOCUMENT_HANDOUT_PIPELINE.md](./SCRIPTS_DOCUMENT_HANDOUT_PIPELINE.md) | 資料處理腳本使用手冊 |

## 專案結構

```
chatbot_graphrag/
├── src/chatbot_graphrag/         # 核心應用程式碼
│   ├── api/routes/               # API 路由層 (4 個核心路由)
│   ├── core/                     # 配置、並發控制、常數
│   ├── db/                       # 資料庫連接、遷移、Repository
│   ├── domains/                  # 領域配置 (hospital 等)
│   ├── graph_workflow/           # LangGraph 工作流 (22 節點)
│   │   └── nodes/                # 計算節點實現
│   ├── models/                   # Pydantic + SQLAlchemy 模型
│   ├── services/                 # 業務邏輯層 (12 個服務模組)
│   │   ├── cache/                # 語義快取
│   │   ├── graph/                # 知識圖譜服務
│   │   ├── ingestion/            # 資料攝取管道
│   │   ├── llm/                  # LLM 並發控制
│   │   ├── retrieval/            # 三種檢索模式
│   │   ├── search/               # 混合搜索
│   │   ├── storage/              # MinIO 物件儲存
│   │   ├── tracing/              # Langfuse 追蹤
│   │   └── vector/               # Qdrant 向量服務
│   └── workers/                  # 背景作業處理
├── frontend/
│   ├── chatbot-ui/               # Angular 前端
│   └── chat-widget/              # Svelte 可嵌入式 Widget
├── tests/                        # 測試套件
├── scripts/                      # 工具腳本
└── docker-compose.yml            # Docker 編排
```

## 授權

MIT License
