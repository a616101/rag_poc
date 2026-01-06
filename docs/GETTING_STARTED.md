# 快速入門指南

本指南協助您在本地環境快速啟動 ChatBot GraphRAG 系統。

## 系統需求

- **Docker** 24.0+ 和 **Docker Compose** v2.0+
- **Python** 3.11+ (本地開發)
- **uv** 套件管理器 (本地開發)
- **Node.js** 20+ (前端開發)
- **本地 LLM 服務**：LMStudio / Ollama / vLLM（或 OpenAI API）

## 安裝步驟

### 1. 複製專案

```bash
git clone <repository-url>
cd chatbot_graphrag
```

### 2. 環境變數設定

```bash
cp .env.graphrag.example .env.graphrag
```

編輯 `.env.graphrag` 設定關鍵配置：

```bash
# LLM API 設定 (OpenAI 相容 API)
OPENAI_API_BASE=http://192.168.50.152:1234/v1
OPENAI_API_KEY=lm-studio
CHAT_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=text-embedding-embeddinggemma-300m-qat
EMBEDDING_DIMENSION=768

# 向量資料庫 (Qdrant)
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION_CHUNKS=graphrag_chunks
QDRANT_COLLECTION_ENTITIES=graphrag_entities
QDRANT_COLLECTION_COMMUNITIES=graphrag_communities

# 圖資料庫 (NebulaGraph)
NEBULA_HOST=nebula-graphd
NEBULA_PORT=9669
NEBULA_USER=root
NEBULA_PASSWORD=nebula

# 全文搜索 (OpenSearch)
OPENSEARCH_URL=http://opensearch:9200
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=Admin@123456

# 關聯式資料庫 (PostgreSQL)
POSTGRES_URL=postgresql+asyncpg://graphrag:graphrag@postgres:5432/graphrag

# 物件儲存 (MinIO)
MINIO_URL=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# 快取 (Redis)
REDIS_URL=redis://redis:6379/0
```

### 3. 啟動服務

**Docker 方式（推薦）：**

```bash
# 開發模式 (含自動重載)
docker compose --profile development up -d

# 查看日誌
docker compose logs -f app-dev
```

**本地開發方式：**

```bash
# 安裝依賴
uv sync

# 啟動開發伺服器
uv run graphrag-dev

# 或直接使用 uvicorn
uv run uvicorn chatbot_graphrag.main:app --reload --host 0.0.0.0 --port 18000
```

### 4. 驗證服務

```bash
# 基礎健康檢查
curl http://localhost:18000/health

# 就緒檢查 (驗證所有服務連接)
curl http://localhost:18000/health/ready

# 並發狀態檢查
curl http://localhost:18000/health/concurrency
```

預期回應：
```json
{
  "status": "healthy",
  "services": {
    "qdrant": "connected",
    "nebula": "connected",
    "opensearch": "connected",
    "postgres": "connected",
    "redis": "connected"
  }
}
```

## 資料庫遷移

首次啟動需要執行資料庫遷移：

```bash
# 使用 CLI
uv run graphrag-db upgrade

# 或 Docker
docker compose exec app-dev uv run graphrag-db upgrade
```

## 向量化文件

### 使用 API 向量化

```bash
curl -X POST http://localhost:18000/api/v1/rag/vectorize \
  -H "Content-Type: application/json" \
  -d '{
    "source": "default",
    "mode": "override"
  }'
```

此操作會回傳 `job_id`，可用於追蹤進度：

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending"
}
```

### 查看向量化進度

```bash
curl http://localhost:18000/api/v1/rag/vectorize/status/{job_id}
```

## 測試問答

### Responses API 格式 (stream)

```bash
curl -X POST http://localhost:18000/api/v1/rag/ask/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"question": "如何登入平台？"}'
```

### OpenAI Chat API 格式 (stream_chat)

```bash
curl -X POST http://localhost:18000/api/v1/rag/ask/stream_chat \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "messages": [
      {"role": "user", "content": "如何登入平台？"}
    ],
    "stream": true
  }'
```

### 多輪對話

```bash
curl -X POST http://localhost:18000/api/v1/rag/ask/stream_chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "我想了解退費相關問題"},
      {"role": "assistant", "content": "好的，請問您想了解哪方面的退費問題？"},
      {"role": "user", "content": "退款流程是什麼？"}
    ],
    "stream": true
  }'
```

## 前端開發

### Angular 前端

```bash
cd frontend/chatbot-ui
npm install
ng serve
```

存取 http://localhost:4200

### Svelte Widget

```bash
cd frontend/chat-widget
npm install
npm run dev
```

存取 http://localhost:4202

## 常見問題

### 服務連線失敗

確認所有服務已啟動：
```bash
docker compose ps
```

檢查個別服務：
```bash
# Qdrant
curl http://localhost:6333/collections

# OpenSearch
curl -u admin:Admin@123456 http://localhost:9200

# NebulaGraph
docker compose exec nebula-graphd nebula-console -u root -p nebula
```

### LLM API 連線錯誤

檢查 `OPENAI_API_BASE` 是否可存取：
```bash
curl $OPENAI_API_BASE/models
```

### 資料庫遷移失敗

確認 PostgreSQL 連線正常：
```bash
docker compose exec postgres psql -U graphrag -d graphrag -c "\dt"
```

## 下一步

- 閱讀 [系統架構](./chatbot_graphrag_architecture.md) 了解 GraphRAG 設計原理
- 參考 [API 手冊](./API_REFERENCE.md) 了解完整 API
- 查看 [配置參考](./CONFIGURATION.md) 調整系統行為
- 閱讀 [Docker 指南](./GRAPHRAG_DOCKER_GUIDE.md) 了解部署選項
