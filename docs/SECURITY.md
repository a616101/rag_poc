# 安全防護機制

本文件說明 Chatbot RAG 系統的安全防護措施。

## 概述

系統實作多層安全防護，保護服務免受惡意輸入和攻擊：

```
用戶輸入
    │
    ▼
┌─────────────────┐
│  長度限制檢查    │ ← 第一道防線
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  注入攻擊偵測    │ ← 第二道防線
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  相關性檢查      │ ← 第三道防線
└────────┬────────┘
         │
         ▼
     正常處理
```

## 輸入驗證

### 問題長度限制

防止超長輸入造成資源耗盡：

```python
# 配置
MAX_QUESTION_LENGTH = 1000  # 預設 1000 字元
```

超過限制時返回：
```json
{
  "detail": "問題長度超過限制（最大 1000 字元）",
  "error_code": "QUESTION_TOO_LONG"
}
```

### 注入攻擊偵測

偵測常見的 Prompt Injection 攻擊模式：

**偵測模式**：
- `ignore previous`、`忽略先前指令`
- `system prompt`、`系統提示`
- `you are now`、`現在你是`
- `jailbreak`、`越獄`
- `DAN mode`
- `roleplay as`

**配置**：
```bash
ENABLE_INJECTION_DETECTION=true
```

**偵測到注入時返回**：
```json
{
  "detail": "偵測到可疑輸入模式",
  "error_code": "INJECTION_DETECTED"
}
```

### 相關性檢查

確保問題在服務範圍內：

**服務範圍**：
- 平臺登入與帳號設定
- 課程搜尋與閱讀操作
- 學習時數與認證
- 加盟機關後台管理
- 課程建置與掛置規範
- 各類課程掛置申請表與測驗題範本

**不支援**：
- 天氣查詢
- 一般程式碼撰寫
- 旅遊行程安排
- 其他與平臺無關的任務

**配置**：
```bash
ENABLE_RELEVANCE_CHECK=true
```

## Guard 節點

LangGraph 工作流程的第一個節點，負責統一的輸入檢查：

```python
# services/ask_stream/graph/nodes/guard.py

def guard_node(state: State) -> State:
    """輸入防護節點"""
    question = state.get("latest_question", "")
    
    # 檢查輸入
    if not is_valid_input(question):
        return {
            **state,
            "guard_blocked": True,
            "guard_reason": "Invalid input detected"
        }
    
    return state
```

## API 安全

### CORS 配置

控制跨域請求：

```bash
# 允許的來源
CORS_ORIGINS_STR=http://localhost:3000,https://myapp.com

# 是否允許憑證
CORS_ALLOW_CREDENTIALS=true

# 允許的方法
CORS_ALLOW_METHODS_STR=GET,POST,PUT,DELETE

# 允許的標頭
CORS_ALLOW_HEADERS_STR=*
```

### 請求日誌

所有請求都會記錄，便於審計：

```python
# core/middleware.py

class RequestLoggingMiddleware:
    async def __call__(self, request, call_next):
        request_id = uuid.uuid4().hex[:8]
        logger.info(f"[{request_id}] {request.method} {request.url.path}")
        # ...
```

### 敏感資料保護

- API 金鑰不記錄在日誌中
- 環境變數用於敏感配置
- 不在錯誤訊息中暴露內部細節

## 資料安全

### 向量資料庫

Qdrant 配置建議：

```bash
# 啟用 API 金鑰認證
QDRANT_API_KEY=your-secure-api-key
```

### LLM API

- 使用 HTTPS 連線
- API 金鑰安全儲存
- 不記錄完整的 prompt/response

## 安全配置清單

### 開發環境

```bash
DEBUG=true
ENABLE_INPUT_GUARD=true
ENABLE_INJECTION_DETECTION=true
ENABLE_RELEVANCE_CHECK=false  # 開發時可關閉
LOG_LEVEL=DEBUG
```

### 生產環境

```bash
DEBUG=false
ENABLE_INPUT_GUARD=true
ENABLE_INJECTION_DETECTION=true
ENABLE_RELEVANCE_CHECK=true
LOG_LEVEL=INFO

# 限制 CORS
CORS_ORIGINS_STR=https://your-production-domain.com

# 啟用 HTTPS（透過反向代理）
PUBLIC_BASE_URL=https://your-production-domain.com
```

## 安全最佳實踐

### 1. 定期更新依賴

```bash
# 使用 uv 更新
uv sync --upgrade

# 檢查安全漏洞
pip-audit
```

### 2. 環境隔離

- 使用 Docker 容器隔離
- 不共用敏感配置
- 分離開發/測試/生產環境

### 3. 監控與告警

- 監控異常請求模式
- 設定速率限制
- 記錄安全事件

### 4. 反向代理配置

```nginx
# Nginx 安全配置
server {
    # HTTPS
    listen 443 ssl;
    
    # 安全標頭
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    
    # 速率限制
    limit_req zone=api burst=20 nodelay;
    
    # 請求大小限制
    client_max_body_size 10M;
}
```

## 已知限制

1. **目前未實作**：
   - API 金鑰認證
   - JWT Token 驗證
   - 速率限制（需透過反向代理）

2. **建議在生產環境新增**：
   - WAF（Web Application Firewall）
   - DDoS 防護
   - IP 白名單

## 相關文件

- [配置參考](./CONFIGURATION.md)
- [部署指南](./DEPLOYMENT.md)
