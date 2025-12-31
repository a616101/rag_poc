"""
GraphRAG 常數與列舉定義模組 (Constants and Enumerations)

===============================================================================
模組概述 (Module Overview)
===============================================================================
此模組定義了 GraphRAG 系統中使用的所有常數、列舉和類型定義。

主要分類：
1. 文件類型 (Document Types) - 用於類型特定的分塊策略
2. 管道類型 (Pipeline Types) - 定義攝取管道種類
3. 工作狀態 (Job Status) - 追蹤攝取任務狀態
4. 圖類型 (Graph Types) - 實體和關係類型
5. 查詢模式 (Query Modes) - GraphRAG 查詢模式
6. SSE 事件階段 (SSE Stages) - 串流事件識別
7. 預設值 (Default Values) - 系統預設常數
===============================================================================
"""

from enum import Enum
from typing import Final


# =============================================================================
# 文件類型 (Document Types)
# =============================================================================

class DocType(str, Enum):
    """
    文件類型分類
    
    用於決定使用哪種分塊策略處理文件。
    不同類型的文件有不同的結構，需要專門的分塊器。
    """
    # ----- 流程文件 -----
    PROCEDURE = "procedure"              # 醫療/行政流程
    PROCESS = "process"                  # 一般流程
    PROCESS_GENERIC = "process.generic"  # 通用流程
    
    # ----- 指南文件 -----
    GUIDE_LOCATION = "guide.location"    # 位置指南（樓層、科室位置）
    GUIDE_TRANSPORT = "guide.transport"  # 交通指南（公車、停車）
    GUIDE_GENERIC = "guide.generic"      # 通用指南
    
    # ----- 人員與團隊 -----
    PHYSICIAN = "physician"              # 醫師簡介
    HOSPITAL_TEAM = "hospital_team"      # 醫療團隊介紹
    
    # ----- 教育文件 -----
    EDUCATION_HANDOUT = "education.handout"  # 衛教單張
    EDUCATION_GENERAL = "education"          # 一般衛教內容
    
    # ----- 其他 -----
    FAQ = "faq"                          # 常見問題
    GENERIC = "generic"                  # 通用文件


class PipelineType(str, Enum):
    """
    攝取管道類型
    
    系統支援兩種文件攝取管道：
    - CURATED: 處理帶有 YAML frontmatter 的 Markdown 文件（結構化）
    - RAW: 處理原始的 PDF/DOCX/HTML 文件（非結構化）
    """
    CURATED = "curated"  # 結構化管道：YAML frontmatter + Markdown
    RAW = "raw"          # 原始管道：PDF/DOCX/HTML


class ProcessMode(str, Enum):
    """
    文件處理模式
    
    決定如何處理已存在的向量：
    - UPDATE: 增量更新，保留現有向量
    - OVERRIDE: 完全重建，刪除現有向量後重新建立
    """
    UPDATE = "update"      # 增量更新模式
    OVERRIDE = "override"  # 覆蓋重建模式


# =============================================================================
# 工作狀態 (Job Status)
# =============================================================================

class JobStatus(str, Enum):
    """
    攝取工作狀態
    
    追蹤文件攝取任務的執行狀態。
    """
    PENDING = "pending"      # 等待中
    RUNNING = "running"      # 執行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 失敗
    PARTIAL = "partial"      # 部分完成


class DocStatus(str, Enum):
    """
    文件版本狀態
    
    追蹤文件在系統中的生命週期狀態。
    """
    ACTIVE = "active"        # 活躍（當前版本）
    DEPRECATED = "deprecated"  # 已棄用（舊版本）
    DELETED = "deleted"      # 已刪除


# =============================================================================
# 圖類型 (Graph Types)
# =============================================================================

class EntityType(str, Enum):
    """
    圖實體類型
    
    定義 NebulaGraph 中的實體（節點）類型。
    這些類型用於實體抽取和圖遍歷。
    """
    # ----- 人員相關 -----
    PERSON = "person"           # 人員（醫師、職員）
    
    # ----- 組織結構 -----
    DEPARTMENT = "department"   # 科室/部門
    SERVICE = "service"         # 服務項目
    
    # ----- 位置相關 -----
    LOCATION = "location"       # 位置（通用）
    BUILDING = "building"       # 建築物
    FLOOR = "floor"             # 樓層
    ROOM = "room"               # 房間
    
    # ----- 醫療相關 -----
    PROCEDURE = "procedure"     # 醫療流程/處置
    MEDICATION = "medication"   # 藥物
    EQUIPMENT = "equipment"     # 醫療設備
    CONDITION = "condition"     # 疾病/症狀
    
    # ----- 其他 -----
    FORM = "form"               # 表單/文件
    TRANSPORT = "transport"     # 交通方式
    CONTACT = "contact"         # 聯絡資訊
    COMMUNITY = "community"     # 圖社群


class RelationType(str, Enum):
    """
    圖關係類型
    
    定義 NebulaGraph 中的邊（關係）類型。
    這些關係連接不同的實體，形成知識圖譜。
    """
    BELONGS_TO = "belongs_to"   # 隸屬於（如：醫師隸屬於科室）
    WORKS_IN = "works_in"       # 工作於（如：醫師工作於診間）
    PERFORMS = "performs"       # 執行（如：醫師執行手術）
    LOCATED_AT = "located_at"   # 位於（如：科室位於某樓層）
    REQUIRES = "requires"       # 需要（如：流程需要某表單）
    MENTIONS = "mentions"       # 提及（如：文件提及某實體）
    RELATED_TO = "related_to"   # 相關於（通用關係）
    TREATS = "treats"           # 治療（如：醫師治療某疾病）
    CONNECTS_TO = "connects_to" # 連接到（如：樓層連接到電梯）
    PART_OF = "part_of"         # 屬於（如：房間屬於樓層）
    MEMBER_OF = "member_of"     # 成員（如：實體屬於社群）


# =============================================================================
# 查詢模式 (Query Modes)
# =============================================================================

class QueryMode(str, Enum):
    """
    GraphRAG 查詢模式
    
    系統支援四種查詢模式，適用於不同類型的問題：
    - LOCAL: 局部模式，從實體種子開始遍歷子圖
    - GLOBAL: 全域模式，先查社群報告再深入局部
    - DRIFT: 漂移模式，動態擴展社群搜尋範圍
    - DIRECT: 直接模式，無需檢索即可回答
    """
    LOCAL = "local"     # 局部模式：實體種子 → 圖遍歷 → 子圖
    GLOBAL = "global"   # 全域模式：社群報告 → 追問 → 局部深入
    DRIFT = "drift"     # 漂移模式：社群擴展 → 多次追問 → 深度
    DIRECT = "direct"   # 直接模式：無需檢索


class IntentType(str, Enum):
    """
    使用者意圖分類
    
    分析使用者問題的意圖，決定處理策略。
    """
    RETRIEVAL = "retrieval"         # 需要檢索文件
    DIRECT = "direct"               # 可直接回答
    FOLLOWUP = "followup"           # 追問問題
    OUT_OF_SCOPE = "out_of_scope"   # 超出領域範圍
    PRIVACY = "privacy"             # 隱私相關查詢
    AMBIGUOUS = "ambiguous"         # 需要澄清


# =============================================================================
# 接地性狀態 (Groundedness Status)
# =============================================================================

class GroundednessStatus(str, Enum):
    """
    接地性檢查結果
    
    接地性（Groundedness）確保回答有證據支持，
    不會產生幻覺或無根據的陳述。
    """
    PASS = "pass"                   # 通過：所有聲明都有證據支持
    RETRY = "retry"                 # 重試：需要更多證據
    NEEDS_REVIEW = "needs_review"   # 需審核：需要人工審核


# =============================================================================
# 區塊類型 (Chunk Types)
# =============================================================================

class ChunkType(str, Enum):
    """
    區塊類型分類
    
    定義文件分塊後各區塊的語意類型。
    不同類型的區塊在檢索時有不同的權重和用途。
    """
    # ----- 通用類型 -----
    METADATA = "metadata"       # 元資料區塊
    SUMMARY = "summary"         # 摘要區塊
    PARAGRAPH = "paragraph"     # 段落區塊
    TABLE = "table"             # 表格區塊
    LIST = "list"               # 列表區塊
    FAQ = "faq"                 # 常見問題區塊

    # ----- 流程文件專用 -----
    STEPS = "steps"             # 步驟說明
    FEES = "fees"               # 費用資訊
    TIMELINE = "timeline"       # 時間線/時程
    FORMS = "forms"             # 表單資訊
    REQUIREMENTS = "requirements"  # 需求/條件

    # ----- 位置指南專用 -----
    FLOORS = "floors"           # 樓層資訊
    WAYFINDING = "wayfinding"   # 導引資訊
    FLOORPLAN = "floorplan"     # 樓層平面圖
    MAP = "map"                 # 地圖

    # ----- 交通指南專用 -----
    TRANSPORT_MODE = "transport_mode"  # 交通方式
    PARKING = "parking"         # 停車資訊
    DIRECTIONS = "directions"   # 方向指引

    # ----- 醫師簡介專用 -----
    EXPERTISE = "expertise"     # 專長領域
    SCHEDULE = "schedule"       # 門診時間
    CONTACT = "contact"         # 聯絡資訊

    # ----- 團隊介紹專用 -----
    SERVICE_SCOPE = "service_scope"  # 服務範圍
    LOCATIONS = "locations"     # 服務地點
    METRICS = "metrics"         # 績效指標
    HIGHLIGHTS = "highlights"   # 特色亮點


# =============================================================================
# SSE 事件階段 (Server-Sent Events Stages)
# =============================================================================

class SSEStage(str, Enum):
    """
    SSE 串流事件階段識別碼
    
    定義串流回應中各處理階段的事件類型，
    讓前端能夠追蹤處理進度並顯示適當的 UI。
    """
    # ----- 輸入守衛階段 (Guard) -----
    GUARD_START = "guard_start"       # 開始輸入檢查
    GUARD_END = "guard_end"           # 輸入檢查完成
    GUARD_BLOCKED = "guard_blocked"   # 輸入被阻擋

    # ----- 存取控制階段 (ACL) -----
    ACL_START = "acl_start"           # 開始權限檢查
    ACL_END = "acl_end"               # 權限檢查完成
    ACL_DENIED = "acl_denied"         # 存取被拒絕

    # ----- 正規化階段 (Normalize) -----
    NORMALIZE_START = "normalize_start"  # 開始查詢正規化
    NORMALIZE_END = "normalize_end"      # 查詢正規化完成

    # ----- 意圖路由階段 (Intent Router) -----
    INTENT_START = "intent_start"     # 開始意圖分析
    INTENT_END = "intent_end"         # 意圖分析完成

    # ----- 檢索階段 (Retrieval) -----
    RETRIEVAL_START = "retrieval_start"           # 開始檢索
    HYBRID_SEED_DONE = "hybrid_seed_done"         # 混合搜尋種子完成
    COMMUNITY_REPORTS_DONE = "community_reports_done"  # 社群報告檢索完成
    RRF_MERGE_DONE = "rrf_merge_done"             # RRF 融合完成
    RERANK_DONE = "rerank_done"                   # 重新排序完成
    GRAPH_TRAVERSE_DONE = "graph_traverse_done"   # 圖遍歷完成
    HOP_HYBRID_DONE = "hop_hybrid_done"           # 跳躍混合搜尋完成
    RETRIEVAL_END = "retrieval_end"               # 檢索完成

    # ----- 證據與接地性階段 (Evidence & Groundedness) -----
    EVIDENCE_TABLE_DONE = "evidence_table_done"   # 證據表建立完成
    GROUNDEDNESS_START = "groundedness_start"     # 開始接地性檢查
    GROUNDEDNESS_END = "groundedness_end"         # 接地性檢查完成
    GROUNDEDNESS_RETRY = "groundedness_retry"     # 接地性重試

    # ----- 人機協作階段 (HITL) -----
    HITL_REQUIRED = "hitl_required"   # 需要人工審核
    HITL_RESOLVED = "hitl_resolved"   # 人工審核完成

    # ----- 回應生成階段 (Response) -----
    RESPONSE_START = "response_start"         # 開始生成回應
    RESPONSE_GENERATING = "response_generating"  # 正在生成回應
    RESPONSE_END = "response_end"             # 回應生成完成

    # ----- 元資訊階段 (Meta) -----
    META_SUMMARY = "meta_summary"     # 處理摘要資訊


# =============================================================================
# 預設值 (Default Values)
# =============================================================================

# ----- 迴圈預算預設值 -----
# 控制 Agentic RAG 的資源消耗上限
DEFAULT_LOOP_BUDGET: Final[dict] = {
    "max_loops": 3,              # 最大迴圈次數
    "max_new_queries": 8,        # 最大新查詢數
    "max_context_tokens": 12000, # 最大上下文 Token 數
    "max_wall_time_seconds": 15.0,  # 最大執行時間（秒）
}

# ----- RRF 融合預設值 -----
# Reciprocal Rank Fusion 權重：(密集向量, 稀疏向量, 全文搜尋)
DEFAULT_RRF_WEIGHTS: Final[tuple[float, float, float]] = (0.4, 0.3, 0.3)
DEFAULT_RRF_K: Final[int] = 60  # RRF 常數

# ----- 檢索分數閾值 -----
# 用於過濾低品質的檢索結果
SCORE_THRESHOLD_HIGH: Final[float] = 0.65    # 高品質閾值
SCORE_THRESHOLD_MEDIUM: Final[float] = 0.50  # 中品質閾值
SCORE_THRESHOLD_LOW: Final[float] = 0.35     # 低品質閾值

# ----- 圖遍歷參數 -----
MAX_GRAPH_HOPS: Final[int] = 2       # 最大跳躍數
MAX_COMMUNITY_LEVEL: Final[int] = 3  # 最大社群層級

# ----- 嵌入向量維度 -----
EMBEDDING_DIMENSION_DEFAULT: Final[int] = 768   # 預設維度（如 BGE、GTE）
EMBEDDING_DIMENSION_OPENAI: Final[int] = 1536   # OpenAI 維度

# =============================================================================
# NebulaGraph Schema 常數
# =============================================================================
# 定義 NebulaGraph 圖資料庫的 Schema 元素

# ----- 標籤（Tag）名稱 -----
NEBULA_ENTITY_TAG: Final[str] = "entity"        # 實體標籤
NEBULA_COMMUNITY_TAG: Final[str] = "community"  # 社群標籤
NEBULA_CHUNK_TAG: Final[str] = "chunk"          # 區塊標籤

# ----- 邊類型列表 -----
NEBULA_EDGE_TYPES: Final[list[str]] = [
    "belongs_to",    # 隸屬於
    "works_in",      # 工作於
    "performs",      # 執行
    "located_at",    # 位於
    "requires",      # 需要
    "mentions",      # 提及
    "related_to",    # 相關於
    "treats",        # 治療
    "connects_to",   # 連接到
    "part_of",       # 屬於
    "member_of",     # 成員
]
