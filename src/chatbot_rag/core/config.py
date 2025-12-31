"""
應用程式配置設定模組

此模組使用 Pydantic Settings 管理應用程式的所有配置參數，
包括伺服器設定、資料庫連線、LLM 設定、日誌配置等。
配置值可從環境變數或 .env 檔案讀取。
"""

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    應用程式設定類別

    使用 Pydantic BaseSettings 來管理配置，支援從環境變數和 .env 檔案讀取設定。
    所有設定都有預設值，可透過環境變數覆寫。
    """

    model_config = SettingsConfigDict(
        env_file=".env",  # 環境變數檔案路徑
        env_file_encoding="utf-8",  # 檔案編碼
        case_sensitive=False,  # 環境變數名稱不區分大小寫
        extra="ignore",  # 忽略額外的環境變數
    )

    # ==================== 應用程式基本設定 ====================
    app_name: str = "Chatbot RAG API"  # 應用程式名稱
    app_version: str = "0.1.0"  # 應用程式版本
    debug: bool = False  # 除錯模式開關

    # ==================== 伺服器設定 ====================
    host: str = "0.0.0.0"  # 伺服器監聽位址，0.0.0.0 表示監聽所有網路介面
    port: int = 8000  # 伺服器監聽埠號
    reload: bool = False  # 是否啟用自動重載（開發模式使用）

    # 對外提供給前端或文件使用的公開 Base URL（用來組完整下載連結等）
    # 例如：
    # - 開發環境: http://localhost:8000
    # - 正式環境: https://elearn-api.hrd.gov.tw
    public_base_url: str = "http://localhost:8000"

    # ==================== CORS 跨域資源共享設定 ====================
    # CORS 設定以逗號分隔的字串格式儲存，方便從環境變數讀取
    cors_origins_str: str = "http://localhost:3000"  # 允許的來源網域列表
    cors_allow_credentials: bool = True  # 是否允許傳送憑證（如 Cookie）
    cors_allow_methods_str: str = "*"  # 允許的 HTTP 方法列表
    cors_allow_headers_str: str = "*"  # 允許的 HTTP 標頭列表

    # ==================== 日誌記錄設定 ====================
    log_level: str = "INFO"  # 日誌級別（DEBUG, INFO, WARNING, ERROR, CRITICAL）
    log_to_console: bool = True  # 是否輸出日誌到控制台
    log_to_file: bool = False  # 是否輸出日誌到檔案
    log_file_path: str = "logs/app.log"  # 日誌檔案路徑
    log_rotation: str = "100 MB"  # 日誌檔案輪替大小
    log_retention: str = "30 days"  # 日誌檔案保留時間
    log_colorize_file: bool = True  # 是否在檔案日誌中使用顏色

    # ==================== 效能調校設定 ====================
    workers: int = 1  # Worker 程序數量
    max_connections: int = 1000  # 最大連線數
    backlog: int = 2048  # 待處理連線佇列大小
    keepalive_timeout: int = 65  # Keep-Alive 逾時時間（秒）

    # ==================== Qdrant 向量資料庫設定 ====================
    qdrant_host: str = "qdrant"  # Qdrant 伺服器主機名稱或 IP
    qdrant_port: int = 6333  # Qdrant HTTP API 埠號
    qdrant_grpc_port: int = 6334  # Qdrant gRPC API 埠號
    qdrant_api_key: str = ""  # Qdrant API 金鑰（可選）
    qdrant_url: str = "http://qdrant:6333"  # Qdrant 完整 URL
    qdrant_collection_name: str = "documents"  # 向量集合名稱

    # ==================== OpenAI 相容 LLM 設定（LMStudio）====================
    openai_api_base: str = "http://192.168.50.152:1234/v1"  # LLM API 基礎 URL
    openai_api_key: str = "lm-studio"  # API 金鑰
    embedding_model: str = "text-embedding-embeddinggemma-300m-qat"  # 嵌入模型名稱
    embedding_dimension: int = 768  # 嵌入向量維度
    chat_model: str = "openai/gpt-oss-20b"  # 聊天模型名稱
    chat_temperature: float = 0.1  # 生成溫度，降低以提高回應穩定性和一致性（原為 0.7）
    chat_max_tokens: int = 2000  # 最大生成 token 數量

    # ==================== LLM / RAG debug 設定 ====================
    # 控制 Responses 串流過程的詳細 log（/ask 端點使用）
    llm_stream_debug: bool = False

    # ==================== 文件處理設定 ====================
    chunk_size: int = 500  # 文件分塊大小（字元數）
    chunk_overlap: int = 50  # 分塊重疊大小，確保上下文連貫性
    default_docs_path: str = "rag_test_data/docs"  # 預設文件目錄路徑

    # ==================== Contextual Chunking 設定 ====================
    contextual_chunking_enabled: bool = True  # 是否啟用脈絡化分塊
    contextual_chunking_use_llm: bool = True  # 是否使用 LLM 生成語義脈絡（Level 2）
    contextual_chunking_model: str = ""  # 脈絡生成用的模型（空字串則使用 chat_model）
    contextual_chunking_temperature: float = 0.1  # 脈絡生成溫度（低溫確保一致性）
    contextual_chunking_max_tokens: int = 150  # 脈絡描述最大 token 數

    # ==================== 輸入驗證與安全設定 ====================
    enable_input_guard: bool = True  # 是否啟用輸入防護
    max_question_length: int = 1000  # 問題最大長度限制
    enable_relevance_check: bool = True  # 是否啟用相關性檢查
    enable_injection_detection: bool = True  # 是否啟用注入攻擊偵測

    # ==================== Langfuse Prompt Management 設定 ====================
    langfuse_prompt_enabled: bool = True  # 是否啟用 Langfuse Prompt Management
    langfuse_prompt_label: str = "production"  # 預設 prompt label (production/staging)
    langfuse_prompt_cache_ttl: int = 300  # 快取 TTL（秒），預設 5 分鐘
    langfuse_sample_rate: float = 1.0  # Langfuse 遙測採樣率（0.0 ~ 1.0），1.0 = 100%

    # ==================== Semantic Cache 語意快取設定 ====================
    semantic_cache_enabled: bool = True  # 是否啟用語意快取
    semantic_cache_collection_name: str = "semantic_cache"  # 快取集合名稱
    semantic_cache_similarity_threshold: float = 0.90  # 相似度閾值（平衡精準度與召回率）
    semantic_cache_ttl_seconds: int = 0  # 快取 TTL（秒），0 = 不設過期，只在文件更新時清除

    # ==================== LLM 並發控制設定 ====================
    llm_max_concurrent_default: int = 10  # 預設最大並發 LLM 請求數
    llm_max_concurrent_chat: int = 20  # Chat 後端最大並發數
    llm_max_concurrent_responses: int = 50  # Responses 後端最大並發數
    llm_max_concurrent_embedding: int = 30  # Embedding 最大並發數
    llm_request_timeout: float = 60.0  # LLM 請求逾時（秒）

    # ==================== LLM 優先級排序設定 ====================
    llm_priority_enabled: bool = False  # 是否啟用請求級優先級排序（預設關閉）
    llm_priority_starvation_threshold: float = 5.0  # 飢餓門檻（秒），超過後自動提升優先級

    # ==================== Reranker 設定 ====================
    reranker_enabled: bool = True  # 是否啟用 Reranker
    reranker_provider: str = "local"  # Reranker 提供者: openai, jina, cohere, local
    # Local provider 推薦模型（支援中文）：
    # - BAAI/bge-reranker-v2-m3（多語言，推薦）
    # - BAAI/bge-reranker-base（中文）
    # - cross-encoder/ms-marco-multilingual-MiniLM-L12-v2（多語言）
    reranker_model: str = "BAAI/bge-reranker-v2-m3"  # Reranker 模型名稱
    reranker_api_base: str = ""  # Reranker API 端點（空字串則使用 openai_api_base）
    reranker_api_key: str = ""  # Reranker API 金鑰（空字串則使用 openai_api_key）
    reranker_top_k: int = 20  # Rerank 前先取的候選數量
    reranker_score_threshold: float = 0.6  # Rerank 後的分數閾值（BGE 模型建議 0.6+）
    reranker_timeout: float = 30.0  # Reranker 請求逾時（秒）
    reranker_batch_size: int = 5  # OpenAI provider 每批評估的文件數

    # ==================== Crawl4AI 網頁爬取設定 ====================
    # Crawl4AI 使用本地 SDK（AsyncWebCrawler），無需額外的 Docker 服務
    crawl4ai_headless: bool = True  # 是否使用無頭模式
    crawl4ai_verbose: bool = False  # 是否啟用詳細日誌
    crawl4ai_timeout: float = 60.0  # 爬取逾時（秒）
    crawl4ai_max_concurrent: int = 5  # 最大並發請求數
    crawl4ai_default_max_depth: int = 3  # 預設爬取深度
    crawl4ai_default_max_pages: int = 100  # 預設最大頁面數

    # Crawl4AI 內容提取設定
    # extraction_mode: raw（原始 markdown）、fit（啟發式提取）、llm（LLM 智慧提取）、balanced（平衡模式）
    crawl4ai_extraction_mode: str = "fit"  # 預設使用 fit 模式
    crawl4ai_fit_threshold: float = 0.48  # fit 模式的過濾門檻（0-1，越高過濾越多）
    crawl4ai_llm_threshold: float = 0.38  # llm 模式的過濾門檻（較低以保留更多內容給 LLM 判斷）
    crawl4ai_balanced_threshold: float = 0.25  # balanced 模式的過濾門檻（更低，保留更多內容）
    crawl4ai_llm_provider: str = ""  # LLM provider（如 openai/gpt-4o-mini），空字串則使用 chat_model
    crawl4ai_llm_instruction: str = """網頁內容提取器。提取主要內容，移除雜訊，輸出標準 Markdown。

任務：完整保留主要內容，移除導航、廣告、頁首頁尾等雜訊。

格式要求：
- 標題獨立一行，前後空行
- 列表項目各自一行，列表前後空行
- 段落間空行分隔
- 分隔線獨立一行，前後空行
- 粗體連結不跨行

只輸出整理後的 Markdown 內容。"""  # LLM 模式的提取指令

    # 清理策略設定（JSON 格式，可從環境變數覆蓋）
    # 格式: {"domain": {"noise_exact": [...], "noise_patterns": [...], ...}}
    scraper_cleaning_strategies: str = ""  # JSON 格式的網域策略覆蓋設定

    # ==================== 內容黑名單設定 ====================
    # 預設內容黑名單：精確匹配的行（爬取後會被移除的雜訊內容）
    # 以逗號分隔的字串格式，方便從環境變數讀取
    scraper_content_blacklist_exact_str: str = "知道了"  # 精確匹配的行

    # 預設內容黑名單：正則表達式模式（匹配的行會被移除）
    # 以 ||| 分隔的字串格式，方便從環境變數讀取
    scraper_content_blacklist_patterns_str: str = r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$"  # 純時間戳行

    @computed_field
    @property
    def llm_params_by_task(self) -> dict[str, dict[str, object]]:
        """
        根據任務類型返回對應的 LLM 參數配置。

        不同任務類型需要不同的 temperature 和 reasoning_effort：
        - simple_faq: 一般問答，需要較高的推理能力
        - form_download: 表單下載，需要精確匹配
        - form_export: 表單匯出，需要精確匹配
        - conversation_followup: 對話追問，需要更靈活的回應
        - out_of_scope: 超出範圍，快速回應即可

        Returns:
            dict: 任務類型對應的 LLM 參數
        """
        return {
            "simple_faq": {"temperature": 0.3, "reasoning_effort": "medium"},
            "form_download": {"temperature": 0.1, "reasoning_effort": "low"},
            "form_export": {"temperature": 0.1, "reasoning_effort": "low"},
            "conversation_followup": {"temperature": 0.5, "reasoning_effort": "medium"},
            "out_of_scope": {"temperature": 0.2, "reasoning_effort": "low"},
        }

    def get_llm_params_for_task(self, task_type: str) -> dict[str, object]:
        """
        根據任務類型獲取 LLM 參數。

        Args:
            task_type: 任務類型

        Returns:
            dict: 對應的 LLM 參數，若無則返回預設值
        """
        default_params = {"temperature": 0.3, "reasoning_effort": "medium"}
        return self.llm_params_by_task.get(task_type, default_params)

    @computed_field
    @property
    def cors_origins(self) -> list[str]:
        """
        解析 CORS 允許來源列表

        將以逗號分隔的字串轉換為列表，並移除空白字元。

        Returns:
            list[str]: 允許的來源網域列表
        """
        return [origin.strip() for origin in self.cors_origins_str.split(",")]

    @computed_field
    @property
    def cors_allow_methods(self) -> list[str]:
        """
        解析 CORS 允許方法列表

        將以逗號分隔的字串轉換為列表，並移除空白字元。

        Returns:
            list[str]: 允許的 HTTP 方法列表
        """
        return [method.strip() for method in self.cors_allow_methods_str.split(",")]

    @computed_field
    @property
    def cors_allow_headers(self) -> list[str]:
        """
        解析 CORS 允許標頭列表

        將以逗號分隔的字串轉換為列表，並移除空白字元。

        Returns:
            list[str]: 允許的 HTTP 標頭列表
        """
        return [header.strip() for header in self.cors_allow_headers_str.split(",")]

    @computed_field
    @property
    def scraper_content_blacklist_exact(self) -> list[str]:
        """
        解析內容黑名單精確匹配列表

        將以逗號分隔的字串轉換為列表。

        Returns:
            list[str]: 精確匹配的黑名單字串列表
        """
        if not self.scraper_content_blacklist_exact_str:
            return []
        return [s.strip() for s in self.scraper_content_blacklist_exact_str.split(",") if s.strip()]

    @computed_field
    @property
    def scraper_content_blacklist_patterns(self) -> list[str]:
        """
        解析內容黑名單正則表達式列表

        將以 ||| 分隔的字串轉換為列表（使用 ||| 是因為正則中可能包含逗號）。

        Returns:
            list[str]: 正則表達式模式列表
        """
        if not self.scraper_content_blacklist_patterns_str:
            return []
        return [s.strip() for s in self.scraper_content_blacklist_patterns_str.split("|||") if s.strip()]


# 建立全域設定實例，供整個應用程式使用
settings = Settings()
