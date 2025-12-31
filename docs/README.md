# Chatbot RAG API

企業級智能問答系統，基於 **Agentic RAG**（Retrieval-Augmented Generation）架構，使用 LangGraph 實現複雜的多步推理工作流。

## 系統特色

- **Agentic RAG 架構** - LangGraph 計算圖實現 9 節點智能工作流
- **SSE 即時串流** - Server-Sent Events 實時推送執行進度和回答
- **脈絡化分塊** - Contextual Chunking 提升檢索精準度
- **漸進式檢索** - 多級閾值策略平衡精準度與召回率
- **完整觀測性** - Langfuse 整合 Trace、Prompt 管理、評估器
- **多語言支援** - 自動語言檢測與回應語言適配
- **安全防護** - 輸入驗證、注入攻擊檢測、相關性檢查

## 技術棧

| 類別 | 技術 |
|------|------|
| **Web 框架** | FastAPI + Uvicorn (ASGI) |
| **LLM 框架** | LangChain + LangGraph |
| **向量資料庫** | Qdrant |
| **可觀測性** | Langfuse |
| **前端** | Angular 20+ |
| **容器化** | Docker + Docker Compose |
| **套件管理** | uv (Astral) |

## 快速開始

```bash
# 1. 複製環境變數設定
cp .env.example .env

# 2. 啟動服務 (開發模式)
docker compose up app-dev

# 3. 向量化文件
curl -X POST http://localhost:8000/api/v1/rag/vectorize \
  -H "Content-Type: application/json" \
  -d '{"source": "default", "mode": "override"}'

# 4. 測試問答
curl -X POST http://localhost:8000/api/v1/rag/ask/stream \
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
| [ARCHITECTURE.md](./ARCHITECTURE.md) | 系統架構說明 |
| [AGENTIC_RAG_FLOW.md](./AGENTIC_RAG_FLOW.md) | Agentic RAG 工作流程（9 節點 LangGraph） |
| [SSE_STREAMING.md](./SSE_STREAMING.md) | SSE 串流處理 |

### API 與整合
| 文件 | 說明 |
|------|------|
| [API_REFERENCE.md](./API_REFERENCE.md) | API 參考手冊 |
| [LANGFUSE_INTEGRATION.md](./LANGFUSE_INTEGRATION.md) | Langfuse 整合（Trace / Prompt / Scores） |

### GraphRAG 系統 (v2.0)
| 文件 | 說明 |
|------|------|
| [GRAPHRAG_DOCKER_GUIDE.md](./GRAPHRAG_DOCKER_GUIDE.md) | GraphRAG Docker Compose 啟動指南 |

### 運維指南
| 文件 | 說明 |
|------|------|
| [DEPLOYMENT.md](./DEPLOYMENT.md) | 部署指南（Docker / Nginx） |
| [SECURITY.md](./SECURITY.md) | 安全防護機制 |
| [TESTING.md](./TESTING.md) | 測試指南 |

### 資料處理腳本
| 文件 | 說明 |
|------|------|
| [SCRIPTS_DOCUMENT_HANDOUT_PIPELINE.md](./SCRIPTS_DOCUMENT_HANDOUT_PIPELINE.md) | PDF → Markdown / education.handout 產生、統整內容重建、metadata 批次更新（scripts 使用手冊） |

## 專案結構

```
chatbot_rag/
├── src/chatbot_rag/          # 核心應用程式碼
│   ├── api/                  # API 路由層
│   ├── core/                 # 配置、日誌、中介軟體
│   ├── models/               # Pydantic 資料模型
│   ├── services/             # 業務邏輯層
│   │   ├── ask_stream/       # Agentic RAG 核心
│   │   │   └── graph/        # LangGraph 計算圖
│   │   └── ...               # 其他服務
│   ├── llm/                  # LLM 整合層
│   └── utils/                # 工具函式
├── frontend/chatbot-ui/      # Angular 前端
├── tests/                    # 測試套件
├── scripts/                  # 分析腳本
├── rag_test_data/docs/       # 測試文件
└── docker-compose.yml        # Docker 編排
```

## 授權

MIT License
