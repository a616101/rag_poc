"""
GraphRAG 設定管理模組 (Configuration Settings Module)

===============================================================================
模組概述 (Module Overview)
===============================================================================
此模組管理 GraphRAG 應用程式的所有配置參數，包括：

1. 基礎設施連線 (Infrastructure Connections):
   - NebulaGraph 圖資料庫
   - Qdrant 向量資料庫
   - OpenSearch 全文搜尋
   - PostgreSQL 關聯式資料庫
   - Redis 快取
   - MinIO 物件儲存

2. LLM 設定 (LLM Settings):
   - OpenAI 相容 API 端點
   - 模型選擇和參數
   - 嵌入模型配置

3. GraphRAG 工作流程參數 (Workflow Parameters):
   - 迴圈預算限制
   - 檢索參數
   - 接地性檢查閾值

4. 日誌與可觀測性 (Logging & Observability):
   - 日誌等級和輸出
   - Langfuse 整合
   - Ragas 評估

配置優先順序：環境變數 > .env 檔案 > 預設值
===============================================================================
"""

from typing import Any
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    GraphRAG 應用程式設定類別
    
    使用 Pydantic BaseSettings 進行配置管理，支援：
    - 從環境變數讀取配置
    - 從 .env 檔案讀取配置
    - 所有設定都有預設值，可透過環境變數覆蓋
    
    使用範例:
        >>> from chatbot_graphrag.core.config import settings
        >>> print(settings.debug)  # 讀取除錯模式設定
        >>> print(settings.nebula_host)  # 讀取 NebulaGraph 主機位址
    """

    # Pydantic 設定配置
    model_config = SettingsConfigDict(
        env_file=".env",           # 從 .env 檔案讀取
        env_file_encoding="utf-8", # 檔案編碼
        case_sensitive=False,      # 環境變數不區分大小寫
        extra="ignore",            # 忽略未定義的額外欄位
    )

    # =========================================================================
    # 應用程式設定 (Application Settings)
    # =========================================================================
    app_name: str = "GraphRAG API"       # 應用程式名稱
    app_version: str = "2.0.0"           # 應用程式版本
    debug: bool = False                  # 除錯模式（啟用 API 文件介面）

    # =========================================================================
    # 伺服器設定 (Server Settings)
    # =========================================================================
    host: str = "0.0.0.0"                # 監聽主機位址
    port: int = 8000                     # 監聽連接埠
    reload: bool = False                 # 是否啟用熱重載
    public_base_url: str = "http://localhost:8000"  # 公開 URL（用於回呼）

    # =========================================================================
    # CORS 設定 (Cross-Origin Resource Sharing)
    # =========================================================================
    cors_origins_str: str = "http://localhost:3000,http://localhost:4200"  # 允許的來源（逗號分隔）
    cors_allow_credentials: bool = True   # 是否允許攜帶憑證
    cors_allow_methods_str: str = "*"     # 允許的 HTTP 方法
    cors_allow_headers_str: str = "*"     # 允許的 HTTP 標頭

    # =========================================================================
    # 日誌設定 (Logging Settings)
    # =========================================================================
    log_level: str = "INFO"              # 日誌等級 (DEBUG/INFO/WARNING/ERROR)
    log_to_console: bool = True          # 是否輸出到控制台
    log_to_file: bool = False            # 是否輸出到檔案
    log_file_path: str = "logs/graphrag.log"  # 日誌檔案路徑
    log_rotation: str = "100 MB"         # 日誌輪替大小
    log_retention: str = "30 days"       # 日誌保留期限

    # =========================================================================
    # 效能設定 (Performance Settings)
    # =========================================================================
    workers: int = 1                     # Worker 進程數量
    max_connections: int = 1000          # 最大連線數
    backlog: int = 2048                  # 連線等待佇列大小
    keepalive_timeout: int = 65          # Keep-Alive 逾時（秒）

    # =========================================================================
    # NebulaGraph 圖資料庫設定
    # =========================================================================
    # NebulaGraph 用於儲存實體-關係-事件圖和社群報告
    nebula_host: str = "nebula-graphd"   # 主機位址
    nebula_port: int = 9669              # 連接埠
    nebula_user: str = "root"            # 使用者名稱
    nebula_password: str = "nebula"      # 密碼
    nebula_space: str = "graphrag"       # 圖空間名稱
    nebula_pool_size: int = 10           # 連線池大小
    nebula_timeout: int = 30000          # 逾時時間（毫秒）

    # =========================================================================
    # Qdrant 向量資料庫設定
    # =========================================================================
    # Qdrant 用於儲存文件區塊的向量嵌入，支援相似性搜尋
    qdrant_host: str = "qdrant"          # 主機位址
    qdrant_port: int = 6333              # HTTP 連接埠
    qdrant_grpc_port: int = 6334         # gRPC 連接埠
    qdrant_api_key: str = ""             # API 金鑰（可選）
    qdrant_url: str = "http://qdrant:6333"  # 完整 URL
    qdrant_collection_chunks: str = "graphrag_chunks"       # 文件區塊集合
    qdrant_collection_cache: str = "graphrag_cache"         # 語意快取集合
    qdrant_collection_communities: str = "graphrag_communities"  # 社群集合

    # =========================================================================
    # PostgreSQL 關聯式資料庫設定
    # =========================================================================
    # PostgreSQL 用於儲存結構化元資料和 LangGraph 檢查點
    postgres_url: str = "postgresql://graphrag:graphrag_secret@postgres:5432/graphrag"
    postgres_pool_size: int = 10         # 連線池大小
    postgres_max_overflow: int = 20      # 最大溢出連線數
    postgres_pool_recycle: int = 3600    # 連線回收時間（秒）

    # =========================================================================
    # OpenSearch 全文搜尋設定
    # =========================================================================
    # OpenSearch 用於 BM25 全文搜尋，與向量搜尋結合使用
    opensearch_url: str = "http://opensearch:9200"
    opensearch_index_chunks: str = "graphrag_chunks"  # 索引名稱
    opensearch_timeout: int = 30         # 逾時時間（秒）

    # =========================================================================
    # Redis 快取設定
    # =========================================================================
    # Redis 用於分散式快取和會話管理
    redis_url: str = "redis://redis:6379/0"
    redis_max_connections: int = 50      # 最大連線數

    # =========================================================================
    # MinIO 物件儲存設定
    # =========================================================================
    # MinIO 用於儲存原始文件、處理後的區塊和社群報告
    minio_endpoint: str = "minio:9000"   # 端點位址
    minio_access_key: str = "minioadmin" # 存取金鑰
    minio_secret_key: str = "minioadmin123"  # 秘密金鑰
    minio_secure: bool = False           # 是否使用 HTTPS
    minio_bucket_documents: str = "documents"   # 原始文件儲存桶
    minio_bucket_chunks: str = "chunks"         # 區塊儲存桶
    minio_bucket_canonical: str = "canonical"   # 標準化文件儲存桶
    minio_bucket_assets: str = "assets"         # 資產儲存桶
    minio_bucket_reports: str = "reports"       # 報告儲存桶

    # =========================================================================
    # LLM 設定 (OpenAI 相容格式)
    # =========================================================================
    # 支援任何 OpenAI API 相容的後端（如 LM Studio、vLLM、Ollama）
    openai_api_base: str = "http://192.168.50.152:1234/v1"  # API 基礎 URL
    openai_api_key: str = "lm-studio"    # API 金鑰
    embedding_model: str = "text-embedding-embeddinggemma-300m-qat"  # 嵌入模型
    embedding_dimension: int = 768        # 嵌入向量維度
    chat_model: str = "openai/gpt-oss-20b"  # 聊天模型
    chat_temperature: float = 0.1         # 生成溫度（0=確定性，1=創造性）
    chat_max_tokens: int = 4000           # 最大生成 Token 數

    # =========================================================================
    # 稀疏編碼器設定 (Sparse Encoder)
    # =========================================================================
    # SPLADE 稀疏編碼器，用於混合搜尋中的關鍵字匹配
    sparse_encoder_model: str = "naver/splade-cocondenser-ensembledistil"
    sparse_encoder_enabled: bool = True   # 是否啟用稀疏編碼

    # =========================================================================
    # 文件處理設定 (Document Processing)
    # =========================================================================
    chunk_size: int = 500                 # 區塊大小（Token 數）
    chunk_overlap: int = 50               # 區塊重疊（Token 數）
    default_docs_path: str = "rag_test_data/docs"  # 預設文件路徑
    # 上下文分塊設定（為每個區塊添加文件上下文）
    contextual_chunking_enabled: bool = True      # 是否啟用上下文分塊
    contextual_chunking_use_llm: bool = True      # 是否使用 LLM 生成上下文
    contextual_chunking_model: str = ""           # 上下文生成模型（空=使用 chat_model）
    contextual_chunking_temperature: float = 0.1  # 上下文生成溫度
    contextual_chunking_max_tokens: int = 150     # 上下文最大 Token 數

    # =========================================================================
    # GraphRAG 工作流程設定 (Workflow Settings)
    # =========================================================================
    
    # ----- 迴圈預算 (Loop Budget) -----
    # 控制 Agentic RAG 的迭代次數和資源消耗
    graphrag_max_loops: int = 3           # 最大迴圈次數
    graphrag_max_new_queries: int = 8     # 最大新查詢數
    graphrag_max_context_tokens: int = 12000  # 最大上下文 Token 數
    graphrag_max_wall_time_seconds: float = 15.0  # 最大執行時間（秒）

    # ----- 檢索設定 (Retrieval) -----
    graphrag_seed_top_k: int = 40         # 初始種子檢索數量
    graphrag_rerank_top_k: int = 12       # 重新排序後保留數量
    graphrag_graph_max_hops: int = 2      # 圖遍歷最大跳數
    graphrag_community_level_min: int = 0  # 社群層級最小值
    graphrag_community_level_max: int = 3  # 社群層級最大值

    # ----- 接地性檢查 (Groundedness) -----
    # 確保生成的回答有證據支持
    graphrag_groundedness_threshold: float = 0.7  # 接地性閾值
    graphrag_groundedness_max_retries: int = 2    # 最大重試次數

    # ----- RRF 融合權重 (Reciprocal Rank Fusion) -----
    # 混合搜尋時各種方法的權重
    graphrag_rrf_weight_dense: float = 0.4   # 密集向量搜尋權重
    graphrag_rrf_weight_sparse: float = 0.3  # 稀疏向量搜尋權重
    graphrag_rrf_weight_fts: float = 0.3     # 全文搜尋權重
    graphrag_rrf_k: int = 60                 # RRF 常數

    # =========================================================================
    # 社群偵測設定 (Community Detection)
    # =========================================================================
    # 用於識別圖中的主題群組
    community_detection_algorithm: str = "leiden"  # 演算法：leiden 或 louvain
    community_detection_resolution: float = 1.0    # 解析度（越高社群越小）
    community_report_max_tokens: int = 500         # 社群報告最大 Token 數
    community_report_model: str = ""               # 報告生成模型（空=使用 chat_model）

    # =========================================================================
    # 重新排序器設定 (Reranker)
    # =========================================================================
    # 用於對檢索結果進行精細排序
    reranker_enabled: bool = True         # 是否啟用重新排序
    reranker_provider: str = "local"      # 提供者：openai/jina/cohere/local
    reranker_model: str = "BAAI/bge-reranker-v2-m3"  # 模型名稱
    reranker_api_base: str = ""           # API 基礎 URL（遠端模型用）
    reranker_api_key: str = ""            # API 金鑰
    reranker_score_threshold: float = 0.5  # 分數閾值
    reranker_timeout: float = 30.0        # 逾時時間（秒）
    reranker_batch_size: int = 10         # 批次大小

    # =========================================================================
    # 輸入守衛設定 (Input Guard)
    # =========================================================================
    # 保護系統免受惡意輸入
    enable_input_guard: bool = True       # 是否啟用輸入守衛
    max_question_length: int = 1000       # 最大問題長度
    enable_relevance_check: bool = True   # 是否檢查相關性
    enable_injection_detection: bool = True  # 是否偵測注入攻擊

    # =========================================================================
    # 語意快取設定 (Semantic Cache)
    # =========================================================================
    # 對語意相似的查詢使用快取結果，減少重複計算
    semantic_cache_enabled: bool = True   # 是否啟用語意快取
    semantic_cache_similarity_threshold: float = 0.90  # 相似度閾值
    semantic_cache_ttl_seconds: int = 0   # 快取存活時間（0=永不過期）

    # =========================================================================
    # Langfuse 可觀測性設定
    # =========================================================================
    # Langfuse 用於追蹤 LLM 呼叫和評估
    langfuse_enabled: bool = True         # 是否啟用 Langfuse
    langfuse_public_key: str = ""         # 公開金鑰
    langfuse_secret_key: str = ""         # 秘密金鑰
    langfuse_host: str = "https://cloud.langfuse.com"  # 主機 URL
    langfuse_sample_rate: float = 1.0     # 採樣率（1.0=100%）
    langfuse_prompt_enabled: bool = True  # 是否啟用 Prompt 管理
    langfuse_prompt_label: str = "production"  # Prompt 標籤
    langfuse_prompt_cache_ttl: int = 300  # Prompt 快取時間（秒）

    # =========================================================================
    # LLM 並行設定 (Concurrency)
    # =========================================================================
    # 控制各類 LLM 操作的並行數量，防止後端過載
    llm_max_concurrent_default: int = 10   # 預設並行數
    llm_max_concurrent_chat: int = 20      # 聊天操作並行數
    llm_max_concurrent_responses: int = 50 # Responses API 並行數
    llm_max_concurrent_embedding: int = 30 # 嵌入操作並行數
    llm_request_timeout: float = 60.0      # 請求逾時（秒）

    # 優先權排程設定
    llm_priority_enabled: bool = False     # 是否啟用優先權排程
    llm_priority_starvation_threshold: float = 5.0  # 飢餓提升閾值（秒）

    # =========================================================================
    # HITL 人機協作設定 (Human-in-the-Loop)
    # =========================================================================
    # 高風險查詢需要人工審核
    hitl_enabled: bool = True              # 是否啟用 HITL
    hitl_pii_detection: bool = True        # 是否偵測個人資訊
    hitl_medical_advice_detection: bool = True  # 是否偵測醫療建議
    hitl_max_groundedness_retries: int = 2  # 接地性檢查最大重試
    hitl_timeout_seconds: float = 300.0    # 等待人工審核逾時（秒）
    hitl_persistence_enabled: bool = True  # 是否使用 PostgreSQL 持久化
    hitl_persistence_table: str = "langgraph_checkpoints"  # 檢查點表名

    # =========================================================================
    # 索引版本控制設定 (Index Versioning)
    # =========================================================================
    index_version: str = "v1.0.0"          # 索引版本
    pipeline_version: str = "v1.0.0"       # 管道版本
    enable_version_rollback: bool = True   # 是否允許版本回滾

    # =========================================================================
    # Ragas 評估設定
    # =========================================================================
    # Ragas 用於自動評估 RAG 系統品質
    ragas_enabled: bool = True             # 是否啟用評估
    ragas_sample_rate: float = 0.1         # 採樣率（0.1=10%）
    ragas_metrics: str = "faithfulness,answer_relevancy,context_precision"  # 評估指標
    ragas_llm_model: str = ""              # 評估用 LLM（空=使用 chat_model）
    ragas_embedding_model: str = ""        # 評估用嵌入模型（空=使用 embedding_model）
    ragas_timeout_seconds: float = 30.0    # 評估逾時（秒）
    ragas_batch_size: int = 5              # 批次大小

    # =========================================================================
    # 領域設定 (Domain Settings)
    # =========================================================================
    domain: str = "hospital"               # 領域：hospital（醫院）或 generic（通用）

    # =========================================================================
    # 計算欄位 (Computed Fields)
    # =========================================================================
    # 這些欄位從其他設定值動態計算得出

    @computed_field
    @property
    def cors_origins(self) -> list[str]:
        """
        解析 CORS 來源列表
        
        從逗號分隔的字串轉換為列表格式。
        
        Returns:
            list[str]: 允許的 CORS 來源列表
        """
        return [origin.strip() for origin in self.cors_origins_str.split(",")]

    @computed_field
    @property
    def cors_allow_methods(self) -> list[str]:
        """
        解析 CORS 允許的 HTTP 方法
        
        Returns:
            list[str]: 允許的 HTTP 方法列表
        """
        return [method.strip() for method in self.cors_allow_methods_str.split(",")]

    @computed_field
    @property
    def cors_allow_headers(self) -> list[str]:
        """
        解析 CORS 允許的 HTTP 標頭
        
        Returns:
            list[str]: 允許的 HTTP 標頭列表
        """
        return [header.strip() for header in self.cors_allow_headers_str.split(",")]

    @computed_field
    @property
    def graphrag_loop_budget(self) -> dict[str, Any]:
        """
        取得預設迴圈預算配置
        
        迴圈預算用於控制 Agentic RAG 的資源消耗，
        防止無限迴圈和過度消耗。
        
        Returns:
            dict: 包含 max_loops、max_new_queries 等的字典
        """
        return {
            "max_loops": self.graphrag_max_loops,
            "max_new_queries": self.graphrag_max_new_queries,
            "max_context_tokens": self.graphrag_max_context_tokens,
            "max_wall_time_seconds": self.graphrag_max_wall_time_seconds,
        }

    @computed_field
    @property
    def graphrag_rrf_weights(self) -> tuple[float, float, float]:
        """
        取得 RRF 融合權重
        
        Reciprocal Rank Fusion 用於合併多種檢索方法的結果。
        
        Returns:
            tuple: (密集向量權重, 稀疏向量權重, 全文搜尋權重)
        """
        return (
            self.graphrag_rrf_weight_dense,
            self.graphrag_rrf_weight_sparse,
            self.graphrag_rrf_weight_fts,
        )

    @computed_field
    @property
    def llm_params_by_task(self) -> dict[str, dict[str, object]]:
        """
        依任務類型取得 LLM 參數
        
        不同任務類型需要不同的溫度和推理設定：
        - retrieval: 需要精確、聚焦的檢索（低溫度）
        - synthesis: 需要平衡的創造力來生成答案（中溫度）
        - extraction: 需要高精度的實體/關係抽取（零溫度）
        - summarization: 需要簡潔準確的摘要（低溫度）
        - intent_analysis: 意圖分析（中溫度）
        - groundedness: 接地性檢查（零溫度，高推理）
        - community_report: 社群報告生成（中溫度）
        
        Returns:
            dict: 任務類型到 LLM 參數的對應
        """
        return {
            "retrieval": {"temperature": 0.1, "reasoning_effort": "low"},
            "synthesis": {"temperature": 0.3, "reasoning_effort": "medium"},
            "extraction": {"temperature": 0.0, "reasoning_effort": "low"},
            "summarization": {"temperature": 0.2, "reasoning_effort": "medium"},
            "intent_analysis": {"temperature": 0.2, "reasoning_effort": "medium"},
            "groundedness": {"temperature": 0.0, "reasoning_effort": "high"},
            "community_report": {"temperature": 0.3, "reasoning_effort": "medium"},
        }

    def get_llm_params_for_task(self, task_type: str) -> dict[str, object]:
        """
        取得特定任務類型的 LLM 參數
        
        Args:
            task_type: 任務類型識別碼
        
        Returns:
            dict: 該任務的 LLM 參數
        
        使用範例:
            >>> params = settings.get_llm_params_for_task("extraction")
            >>> print(params["temperature"])  # 0.0
        """
        default_params = {"temperature": 0.3, "reasoning_effort": "medium"}
        return self.llm_params_by_task.get(task_type, default_params)


# =============================================================================
# 全域設定實例 (Global Settings Instance)
# =============================================================================
# 應用程式啟動時建立，供所有模組共用
settings = Settings()
