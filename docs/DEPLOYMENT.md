# 部署指南

本文件說明 Chatbot RAG 系統的部署方式。

## 部署架構

```
┌─────────────────────────────────────────────────────────────────┐
│                        生產環境架構                               │
│                                                                   │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────────┐  │
│  │   Nginx     │ ───▶ │  FastAPI    │ ───▶ │    Qdrant       │  │
│  │  (反向代理)  │      │  (API)      │      │  (向量資料庫)     │  │
│  └─────────────┘      └──────┬──────┘      └─────────────────┘  │
│                              │                                    │
│                              ▼                                    │
│                       ┌─────────────┐      ┌─────────────────┐  │
│                       │   LLM API   │      │    Langfuse     │  │
│                       │ (LMStudio)  │      │  (可觀測性)       │  │
│                       └─────────────┘      └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Docker Compose 部署

### 生產配置

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DEBUG=false
      - LOG_LEVEL=INFO
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - qdrant
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: always

volumes:
  qdrant_data:
```

### 啟動生產環境

```bash
# 建置映像
docker compose -f docker-compose.prod.yml build

# 啟動服務
docker compose -f docker-compose.prod.yml up -d

# 查看日誌
docker compose -f docker-compose.prod.yml logs -f app
```

## 環境變數配置

### 生產環境 .env

```bash
# === 應用程式 ===
APP_NAME=Chatbot RAG API
DEBUG=false

# === 伺服器 ===
HOST=0.0.0.0
PORT=8000
PUBLIC_BASE_URL=https://your-domain.com

# === CORS ===
CORS_ORIGINS_STR=https://your-domain.com,https://app.your-domain.com

# === 日誌 ===
LOG_LEVEL=INFO
LOG_TO_FILE=true
LOG_FILE_PATH=/var/log/chatbot-rag/app.log

# === Qdrant ===
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=your-secure-api-key
QDRANT_COLLECTION_NAME=documents

# === LLM ===
OPENAI_API_BASE=http://llm-server:1234/v1
OPENAI_API_KEY=your-api-key
CHAT_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=text-embedding-embeddinggemma-300m-qat
EMBEDDING_DIMENSION=768

# === 安全 ===
ENABLE_INPUT_GUARD=true
ENABLE_INJECTION_DETECTION=true
ENABLE_RELEVANCE_CHECK=true

# === Langfuse ===
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PROMPT_ENABLED=true
```

## Nginx 反向代理

### 配置範例

```nginx
# /etc/nginx/sites-available/chatbot-rag

upstream chatbot_api {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL 憑證
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # SSL 安全設定
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;

    # 安全標頭
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000" always;

    # 請求大小限制
    client_max_body_size 10M;

    # 速率限制
    limit_req zone=api burst=20 nodelay;

    # API 路由
    location /api/ {
        proxy_pass http://chatbot_api;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # SSE 支援
        proxy_set_header Connection '';
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 120s;
    }

    # SSE 端點特殊處理
    location /api/v1/rag/ask/stream {
        proxy_pass http://chatbot_api;
        proxy_http_version 1.1;
        proxy_set_header Connection '';
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        chunked_transfer_encoding off;
    }

    # 健康檢查
    location /health {
        proxy_pass http://chatbot_api/api/v1/health;
    }
}
```

### 速率限制區塊

```nginx
# /etc/nginx/conf.d/rate_limit.conf

limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=stream:10m rate=2r/s;
```

## 監控與日誌

### 健康檢查端點

```bash
# 基礎健康檢查
curl https://your-domain.com/api/v1/health

# RAG 系統健康檢查
curl https://your-domain.com/api/v1/rag/health
```

### 日誌收集

```yaml
# docker-compose.prod.yml 添加日誌配置
services:
  app:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
```

### Prometheus 指標（可選）

系統可擴展支援 Prometheus 指標：

```python
from prometheus_client import Counter, Histogram

request_counter = Counter('requests_total', 'Total requests')
latency_histogram = Histogram('request_latency_seconds', 'Request latency')
```

## 資料持久化

### Qdrant 資料備份

```bash
# 備份 Qdrant 資料
docker compose exec qdrant qdrant-backup --output /backup/qdrant_backup.tar.gz

# 恢復資料
docker compose exec qdrant qdrant-restore --input /backup/qdrant_backup.tar.gz
```

### 定期備份腳本

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR=/var/backups/chatbot-rag
DATE=$(date +%Y%m%d_%H%M%S)

# 建立備份目錄
mkdir -p $BACKUP_DIR

# 備份 Qdrant
docker compose exec -T qdrant tar czf - /qdrant/storage > $BACKUP_DIR/qdrant_$DATE.tar.gz

# 保留最近 7 天的備份
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

## 擴展部署

### 水平擴展

```yaml
# docker-compose.scale.yml
services:
  app:
    deploy:
      replicas: 3
    
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - app
```

### 負載均衡

```nginx
upstream chatbot_api {
    least_conn;
    server app_1:8000;
    server app_2:8000;
    server app_3:8000;
    keepalive 32;
}
```

## 故障排除

### 常見問題

#### 1. 容器無法啟動

```bash
# 檢查日誌
docker compose logs app

# 檢查資源
docker stats
```

#### 2. Qdrant 連線失敗

```bash
# 檢查 Qdrant 狀態
curl http://localhost:6333/collections

# 檢查網路
docker network ls
docker network inspect chatbot_rag_default
```

#### 3. LLM API 連線問題

```bash
# 測試連線
curl $OPENAI_API_BASE/models
```

### 回滾程序

```bash
# 查看之前的映像
docker images chatbot-rag

# 回滾到之前版本
docker compose down
docker tag chatbot-rag:previous chatbot-rag:latest
docker compose up -d
```

## 安全清單

- [ ] 啟用 HTTPS
- [ ] 配置 CORS 只允許信任的來源
- [ ] 設定 Qdrant API 金鑰
- [ ] 限制 API 速率
- [ ] 啟用輸入驗證
- [ ] 配置防火牆規則
- [ ] 定期備份資料
- [ ] 監控系統日誌

## 相關文件

- [配置參考](./CONFIGURATION.md)
- [安全防護機制](./SECURITY.md)
- [測試指南](./TESTING.md)
