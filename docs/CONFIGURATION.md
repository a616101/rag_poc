# 環境配置參考

本文件說明 Chatbot RAG API 的所有配置參數。所有設定皆可透過環境變數或 `.env` 檔案設定。

## 配置來源優先順序

1. 環境變數（最高優先）
2. `.env` 檔案
3. 預設值

---

## 應用程式基本設定

| 變數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `APP_NAME` | string | `Chatbot RAG API` | 應用程式名稱 |
| `APP_VERSION` | string | `0.1.0` | 應用程式版本 |
| `DEBUG` | bool | `false` | 除錯模式，啟用時輸出更多日誌 |

---

## 伺服器設定

| 變數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `HOST` | string | `0.0.0.0` | 監聽位址 |
| `PORT` | int | `8000` | 監聽埠號 |
| `RELOAD` | bool | `false` | 是否啟用自動重載（開發模式） |
| `PUBLIC_BASE_URL` | string | `http://localhost:8000` | 對外公開 URL，用於產生檔案下載連結 |

---

## CORS 跨域設定

| 變數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `CORS_ORIGINS_STR` | string | `http://localhost:3000` | 允許的來源，逗號分隔 |
| `CORS_ALLOW_CREDENTIALS` | bool | `true` | 是否允許傳送憑證 |
| `CORS_ALLOW_METHODS_STR` | string | `*` | 允許的 HTTP 方法 |
| `CORS_ALLOW_HEADERS_STR` | string | `*` | 允許的 HTTP 標頭 |

**範例**：
```bash
CORS_ORIGINS_STR=http://localhost:3000,http://localhost:4200,https://myapp.com
```

---

## 日誌記錄設定

| 變數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `LOG_LEVEL` | string | `INFO` | 日誌級別：DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `LOG_TO_CONSOLE` | bool | `true` | 是否輸出到控制台 |
| `LOG_TO_FILE` | bool | `false` | 是否輸出到檔案 |
| `LOG_FILE_PATH` | string | `logs/app.log` | 日誌檔案路徑 |
| `LOG_ROTATION` | string | `100 MB` | 日誌輪替大小 |
| `LOG_RETENTION` | string | `30 days` | 日誌保留時間 |
| `LOG_COLORIZE_FILE` | bool | `true` | 檔案日誌是否使用顏色 |

---

## 效能調校設定

| 變數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `WORKERS` | int | `1` | Worker 程序數量 |
| `MAX_CONNECTIONS` | int | `1000` | 最大連線數 |
| `BACKLOG` | int | `2048` | 待處理連線佇列大小 |
| `KEEPALIVE_TIMEOUT` | int | `65` | Keep-Alive 逾時時間（秒） |

---

## Qdrant 向量資料庫設定

| 變數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `QDRANT_HOST` | string | `qdrant` | Qdrant 主機名稱 |
| `QDRANT_PORT` | int | `6333` | Qdrant HTTP API 埠號 |
| `QDRANT_GRPC_PORT` | int | `6334` | Qdrant gRPC API 埠號 |
| `QDRANT_API_KEY` | string | ` ` | Qdrant API 金鑰（可選） |
| `QDRANT_URL` | string | `http://qdrant:6333` | Qdrant 完整 URL |
| `QDRANT_COLLECTION_NAME` | string | `documents` | 向量集合名稱 |

**Docker 環境**：
```bash
QDRANT_URL=http://qdrant:6333
```

**本地開發**：
```bash
QDRANT_URL=http://localhost:6333
```

---

## LLM / Embedding 設定

系統使用 OpenAI 相容 API，可搭配 LMStudio、Ollama、vLLM 等本地服務。

| 變數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `OPENAI_API_BASE` | string | `http://192.168.50.152:1234/v1` | LLM API 基礎 URL |
| `OPENAI_API_KEY` | string | `lm-studio` | API 金鑰 |
| `CHAT_MODEL` | string | `openai/gpt-oss-20b` | 聊天模型名稱 |
| `CHAT_TEMPERATURE` | float | `0.1` | 生成溫度（0.0-1.0） |
| `CHAT_MAX_TOKENS` | int | `2000` | 最大生成 token 數量 |
| `EMBEDDING_MODEL` | string | `text-embedding-embeddinggemma-300m-qat` | 嵌入模型名稱 |
| `EMBEDDING_DIMENSION` | int | `768` | 嵌入向量維度 |
| `LLM_STREAM_DEBUG` | bool | `false` | 是否輸出 LLM 串流詳細日誌 |

**使用 OpenAI API**：
```bash
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=sk-your-api-key
CHAT_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
```

**使用 Ollama**：
```bash
OPENAI_API_BASE=http://localhost:11434/v1
OPENAI_API_KEY=ollama
CHAT_MODEL=llama3.2
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768
```

---

## 文件處理設定

| 變數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `CHUNK_SIZE` | int | `500` | 文件分塊大小（字元數） |
| `CHUNK_OVERLAP` | int | `50` | 分塊重疊大小 |
| `DEFAULT_DOCS_PATH` | string | `rag_test_data/docs` | 預設文件目錄路徑 |

---

## Contextual Chunking 設定

Contextual Chunking 為每個 chunk 添加文件脈絡，提升檢索精準度。

| 變數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `CONTEXTUAL_CHUNKING_ENABLED` | bool | `true` | 是否啟用脈絡化分塊 |
| `CONTEXTUAL_CHUNKING_USE_LLM` | bool | `true` | 是否使用 LLM 生成語義脈絡（Level 2） |
| `CONTEXTUAL_CHUNKING_MODEL` | string | ` ` | 脈絡生成模型，空字串使用 CHAT_MODEL |
| `CONTEXTUAL_CHUNKING_TEMPERATURE` | float | `0.1` | 脈絡生成溫度 |
| `CONTEXTUAL_CHUNKING_MAX_TOKENS` | int | `150` | 脈絡描述最大 token 數 |

**Level 1（結構化脈絡）**：使用 frontmatter + section headers，不需 LLM
**Level 2（語義脈絡）**：使用 LLM 生成語義描述

```bash
# 只使用 Level 1（節省 LLM 成本）
CONTEXTUAL_CHUNKING_USE_LLM=false
```

---

## 輸入驗證與安全設定

| 變數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `ENABLE_INPUT_GUARD` | bool | `true` | 是否啟用輸入防護 |
| `MAX_QUESTION_LENGTH` | int | `1000` | 問題最大長度限制 |
| `ENABLE_RELEVANCE_CHECK` | bool | `true` | 是否啟用相關性檢查 |
| `ENABLE_INJECTION_DETECTION` | bool | `true` | 是否啟用注入攻擊偵測 |

---

## Langfuse Prompt Management 設定

透過 Langfuse 管理 Prompt，支援版本控制和 A/B 測試。

| 變數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `LANGFUSE_PROMPT_ENABLED` | bool | `true` | 是否啟用 Langfuse Prompt 管理 |
| `LANGFUSE_PROMPT_LABEL` | string | `production` | 預設 prompt label |
| `LANGFUSE_PROMPT_CACHE_TTL` | int | `300` | 快取 TTL（秒） |

**Langfuse SDK 環境變數**（由 Langfuse SDK 讀取）：
```bash
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## 完整 .env 範例

```bash
# === 應用程式 ===
APP_NAME=Chatbot RAG API
DEBUG=false

# === 伺服器 ===
HOST=0.0.0.0
PORT=8000
PUBLIC_BASE_URL=http://localhost:8000

# === CORS ===
CORS_ORIGINS_STR=http://localhost:3000,http://localhost:4200

# === 日誌 ===
LOG_LEVEL=INFO
LOG_TO_FILE=false

# === Qdrant ===
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION_NAME=documents

# === LLM ===
OPENAI_API_BASE=http://192.168.50.152:1234/v1
OPENAI_API_KEY=lm-studio
CHAT_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=text-embedding-embeddinggemma-300m-qat
EMBEDDING_DIMENSION=768

# === 文件處理 ===
CHUNK_SIZE=500
CHUNK_OVERLAP=50
DEFAULT_DOCS_PATH=rag_test_data/docs

# === Contextual Chunking ===
CONTEXTUAL_CHUNKING_ENABLED=true
CONTEXTUAL_CHUNKING_USE_LLM=false

# === 安全 ===
ENABLE_INPUT_GUARD=true
MAX_QUESTION_LENGTH=1000

# === Langfuse ===
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PROMPT_ENABLED=true
LANGFUSE_PROMPT_LABEL=production
```

---

## 相關文件

- [快速入門指南](./GETTING_STARTED.md)
- [部署指南](./DEPLOYMENT.md)
- [安全防護機制](./SECURITY.md)
