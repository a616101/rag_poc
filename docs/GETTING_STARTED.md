# 快速入門指南

本指南協助您在本地環境快速啟動 Chatbot RAG 系統。

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
cd chatbot_rag
```

### 2. 環境變數設定

```bash
cp .env.example .env
```

編輯 `.env` 設定關鍵配置：

```bash
# LLM API 設定 (OpenAI 相容 API)
OPENAI_API_BASE=http://192.168.50.152:1234/v1
OPENAI_API_KEY=lm-studio
CHAT_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=text-embedding-embeddinggemma-300m-qat
EMBEDDING_DIMENSION=768

# Qdrant 向量資料庫
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION_NAME=documents

# 公開存取 URL (用於表單下載連結)
PUBLIC_BASE_URL=http://localhost:8000
```

### 3. 啟動服務

**Docker 方式（推薦）：**

```bash
# 開發模式 (含自動重載)
docker compose up app-dev

# 或背景執行
docker compose up -d app-dev
```

**本地開發方式：**

```bash
# 安裝依賴
uv sync

# 啟動開發伺服器
uv run chatbot-dev

# 或直接使用 uvicorn
uv run uvicorn chatbot_rag.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 驗證服務

```bash
# 健康檢查
curl http://localhost:8000/api/v1/health

# RAG 系統檢查
curl http://localhost:8000/api/v1/rag/health
```

預期回應：
```json
{
  "qdrant_connected": true,
  "embedding_service_connected": true,
  "collection_exists": false,
  "embedding_dimension": 768,
  "expected_dimension": 768,
  "dimension_match": true
}
```

> **注意**：`collection_exists: false` 表示尚未向量化文件，需執行下一步。

## 向量化文件

### 使用預設測試文件

```bash
curl -X POST http://localhost:8000/api/v1/rag/vectorize \
  -H "Content-Type: application/json" \
  -d '{
    "source": "default",
    "mode": "override"
  }'
```

參數說明：
- `source`: `"default"` 使用 `rag_test_data/docs/` 目錄
- `mode`: `"override"` 重建集合，`"update"` 增量更新

### 查看集合資訊

```bash
curl http://localhost:8000/api/v1/rag/collection/info
```

回應範例：
```json
{
  "collection_name": "documents",
  "vectors_count": 156,
  "points_count": 156,
  "status": "green"
}
```

## 測試問答

### 單輪問答

```bash
curl -X POST http://localhost:8000/api/v1/rag/ask/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"question": "如何登入 e 等公務園？"}'
```

### 多輪對話

```bash
curl -X POST http://localhost:8000/api/v1/rag/ask/stream_chat \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "question": "退款流程是什麼？",
    "conversation_history": [
      {"role": "user", "content": "我想了解退費相關問題"},
      {"role": "assistant", "content": "好的，請問您想了解哪方面的退費問題？"}
    ]
  }'
```

## 前端開發

```bash
cd frontend/chatbot-ui
npm install
ng serve
```

存取 http://localhost:4200

## 常見問題

### Qdrant 連線失敗

確認 Qdrant 服務已啟動：
```bash
docker compose ps
curl http://localhost:6333/collections
```

### Embedding 維度不符

確認 `.env` 中的 `EMBEDDING_DIMENSION` 與實際模型輸出一致。

### LLM API 連線錯誤

檢查 `OPENAI_API_BASE` 是否可存取：
```bash
curl $OPENAI_API_BASE/models
```

## 下一步

- 閱讀 [系統架構](./ARCHITECTURE.md) 了解設計原理
- 參考 [API 手冊](./API_REFERENCE.md) 了解完整 API
- 查看 [配置參考](./CONFIGURATION.md) 調整系統行為
