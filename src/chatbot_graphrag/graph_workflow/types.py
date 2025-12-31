"""
GraphRAG LangGraph 狀態類型

定義 GraphRAG 工作流程的狀態 schema。

包含：
- QueryMode: 查詢模式列舉（local/global/drift/direct）
- GroundednessStatus: 落地性狀態列舉
- LoopBudget: 循環預算追蹤
- EvidenceItem: 證據項目
- RetrievalResult: 檢索結果
- FilterContext: 多租戶過濾上下文
- GraphRAGState: 主要的 LangGraph 狀態 TypedDict
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage

from chatbot_graphrag.services.search.hybrid_search import SearchResult


class QueryMode(str, Enum):
    """
    查詢路由模式。

    決定使用哪種檢索策略回答問題。
    """

    LOCAL = "local"  # 基於實體的檢索 - 從種子實體遍歷子圖
    GLOBAL = "global"  # 基於社群的檢索 - 使用社群報告
    DRIFT = "drift"  # 動態探索 - 多輪社群擴展
    DIRECT = "direct"  # 直接回答 - 無需檢索


class GroundednessStatus(str, Enum):
    """
    落地性評估狀態。

    評估回答是否有足夠的來源證據支持。
    """

    PASS = "pass"  # 證據支持回答
    RETRY = "retry"  # 需要更多證據
    NEEDS_REVIEW = "needs_review"  # 需要人工審核


@dataclass
class LoopBudget:
    """
    檢索循環的預算追蹤。

    控制 Agentic RAG 的資源消耗，防止無限循環。
    """

    max_loops: int = 3  # 最大循環次數
    max_new_queries: int = 8  # 最大子查詢數
    max_context_tokens: int = 12000  # 最大上下文 token 數
    max_wall_time_seconds: float = 15.0  # 最大牆鐘時間

    # 目前使用量
    current_loops: int = 0
    current_queries: int = 0
    current_tokens: int = 0
    start_time: float = 0.0

    def is_exhausted(self) -> bool:
        """檢查是否有任何預算耗盡。"""
        import time

        elapsed = time.time() - self.start_time if self.start_time > 0 else 0
        return (
            self.current_loops >= self.max_loops
            or self.current_queries >= self.max_new_queries
            or self.current_tokens >= self.max_context_tokens
            or elapsed >= self.max_wall_time_seconds
        )

    def can_continue(self) -> bool:
        """檢查是否可以繼續下一次迭代。"""
        return not self.is_exhausted()

    def increment_loop(self) -> None:
        """增加循環計數器。"""
        self.current_loops += 1

    def increment_queries(self, count: int = 1) -> None:
        """增加查詢計數器。"""
        self.current_queries += count

    def add_tokens(self, count: int) -> None:
        """增加 token 計數器。"""
        self.current_tokens += count

    def remaining_budget(self) -> dict[str, Any]:
        """取得剩餘預算。"""
        import time

        elapsed = time.time() - self.start_time if self.start_time > 0 else 0
        return {
            "loops": self.max_loops - self.current_loops,
            "queries": self.max_new_queries - self.current_queries,
            "tokens": self.max_context_tokens - self.current_tokens,
            "time_seconds": max(0, self.max_wall_time_seconds - elapsed),
        }


@dataclass
class EvidenceItem:
    """
    回答落地性的單一證據項目。

    用於追蹤回答中的每個主張是否有來源支持。
    """

    chunk_id: str  # 來源 chunk ID
    content: str  # 證據內容
    relevance_score: float  # 相關性分數
    source_doc: str  # 來源文件
    citation_index: int  # 引用索引
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """
    檢索操作的結果。

    包含檢索到的 chunk 和相關 metadata。
    """

    chunks: list[SearchResult]  # 檢索到的 chunk 列表
    query_used: str  # 使用的查詢
    mode: QueryMode  # 使用的查詢模式
    retrieval_path: list[str] = field(default_factory=list)  # 檢索路徑
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterContext:
    """
    多租戶存取控制的過濾上下文。

    此上下文由 ACL 節點設定，供所有下游檢索節點使用，
    確保適當的資料隔離。
    """

    tenant_id: str = "default"
    acl_groups: list[str] = field(default_factory=lambda: ["public"])
    department: Optional[str] = None
    doc_types: Optional[list[str]] = None

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典，用於建構過濾條件。"""
        result = {
            "tenant_id": self.tenant_id,
            "acl_groups": self.acl_groups,
        }
        if self.department:
            result["department"] = self.department
        if self.doc_types:
            result["doc_types"] = self.doc_types
        return result


class GraphRAGState(TypedDict, total=False):
    """
    GraphRAG 工作流程的 LangGraph 狀態。

    所有欄位都是可選的（total=False），允許部分狀態更新。
    這是工作流程中所有節點共享的狀態結構。
    """

    # === 核心輸入 ===
    messages: list[BaseMessage]  # 對話歷史
    question: str  # 原始使用者問題
    normalized_question: str  # 正規化/翻譯後的問題
    resolved_question: str  # 解析對話上下文後的問題（用於追問）
    user_language: str  # 偵測的使用者語言（如 "zh-TW"、"en"）

    # === 守衛 & ACL ===
    guard_blocked: bool  # 輸入被安全守衛阻擋
    guard_reason: str  # 阻擋原因
    acl_denied: bool  # 存取控制拒絕
    acl_groups: list[str]  # 使用者的 ACL 群組
    tenant_id: str  # 租戶識別碼
    filter_context: FilterContext  # 檢索用的多租戶過濾上下文

    # === 意圖 & 路由 ===
    query_mode: str  # "local"、"global"、"drift"、"direct"
    intent_reasoning: str  # LLM 對意圖的推理

    # === 循環預算 ===
    budget: LoopBudget  # 預算追蹤

    # === 檢索狀態 ===
    seed_results: RetrievalResult  # 初始混合搜尋結果
    community_reports: list[dict[str, Any]]  # 社群摘要
    followup_queries: list[str]  # 生成的追問查詢
    graph_subgraph: dict[str, Any]  # 提取的圖譜子圖
    merged_results: RetrievalResult  # 合併的檢索結果
    reranked_chunks: list[SearchResult]  # 跨編碼器重排序後

    # === 上下文建構 ===
    expanded_chunks: list[SearchResult]  # chunk 擴展後
    context_text: str  # 打包給 LLM 的上下文
    context_tokens: int  # 上下文的 token 數

    # === 證據 & 落地性 ===
    evidence_table: list[EvidenceItem]  # 結構化證據
    groundedness_status: str  # "pass"、"retry"、"needs_review"
    groundedness_score: float  # 0.0-1.0 信心分數
    retry_reason: str  # 觸發重試的原因
    draft_answer: str  # 最終輸出前的草稿回答（用於 Ragas 評估）

    # === Ragas 評估（第 4 階段）===
    ragas_metrics: dict[str, Any]  # Ragas 評估指標
    ragas_sampled: bool  # 此請求是否被 Ragas 取樣

    # === HITL（人機協作）===
    hitl_required: bool  # 需要人工審核
    hitl_resolved: bool  # 人工審核完成
    hitl_feedback: str  # 人工回饋（如有）
    hitl_approved: bool  # 人工批准狀態（第 3 階段）
    hitl_triggered_at: str  # HITL 觸發的 ISO 時間戳記
    hitl_timeout_at: float  # 逾時截止的 Unix 時間戳記
    hitl_timeout_seconds: float  # 配置的逾時時間
    hitl_timed_out: bool  # HITL 是否逾時
    hitl_rejected: bool  # 人工審核是否拒絕回答
    needs_review_reason: str  # 需要人工審核的原因
    needs_human_review: bool  # hitl_required 的別名

    # === 輸出 ===
    final_answer: str  # 生成的回答
    citations: list[str]  # 引用參考
    confidence: float  # 回答信心度 0.0-1.0

    # === 可觀測性 ===
    trace_id: str  # Langfuse 追蹤 ID
    retrieval_path: list[str]  # 通過檢索階段的路徑
    timing: dict[str, float]  # 階段計時（毫秒）
    error: str  # 錯誤訊息（如有）

    # === 版本控制（第 0 階段）===
    index_version: str  # 向量索引版本
    pipeline_version: str  # GraphRAG 管道版本
    prompt_version: str  # 提示詞版本（來自 Langfuse 或配置）
    config_hash: str  # 相關配置的雜湊（用於可重現性）

    # === LLM 後端 ===
    agent_backend: str  # "responses" 或 "chat" - 使用哪個 LLM API


def compute_config_hash() -> str:
    """
    計算相關配置設定的雜湊，用於可重現性。

    包含影響檢索和生成行為的設定。
    """
    import hashlib
    import json

    from chatbot_graphrag.core.config import settings

    config_dict = {
        "index_version": settings.index_version,
        "pipeline_version": settings.pipeline_version,
        "embedding_model": settings.embedding_model,
        "chat_model": settings.chat_model,
        "reranker_model": settings.reranker_model,
        "graphrag_rrf_weights": settings.graphrag_rrf_weights,
        "graphrag_seed_top_k": settings.graphrag_seed_top_k,
        "graphrag_rerank_top_k": settings.graphrag_rerank_top_k,
        "graphrag_groundedness_threshold": settings.graphrag_groundedness_threshold,
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


# 預設初始狀態
def create_initial_state(
    question: str,
    messages: Optional[list[BaseMessage]] = None,
    acl_groups: Optional[list[str]] = None,
    tenant_id: str = "default",
    agent_backend: str = "responses",
) -> GraphRAGState:
    """為新查詢建立初始狀態。"""
    import time

    from chatbot_graphrag.core.config import settings

    budget = LoopBudget()
    budget.start_time = time.time()

    return GraphRAGState(
        messages=messages or [],
        question=question,
        normalized_question="",
        resolved_question="",
        user_language="",
        guard_blocked=False,
        guard_reason="",
        acl_denied=False,
        acl_groups=acl_groups or ["public"],
        tenant_id=tenant_id,
        filter_context=FilterContext(
            tenant_id=tenant_id,
            acl_groups=acl_groups or ["public"],
        ),
        query_mode="",
        intent_reasoning="",
        budget=budget,
        seed_results=RetrievalResult(chunks=[], query_used="", mode=QueryMode.LOCAL),
        community_reports=[],
        followup_queries=[],
        graph_subgraph={},
        merged_results=RetrievalResult(chunks=[], query_used="", mode=QueryMode.LOCAL),
        reranked_chunks=[],
        expanded_chunks=[],
        context_text="",
        context_tokens=0,
        evidence_table=[],
        groundedness_status="",
        groundedness_score=0.0,
        retry_reason="",
        hitl_required=False,
        hitl_resolved=False,
        hitl_feedback="",
        final_answer="",
        citations=[],
        confidence=0.0,
        trace_id="",
        retrieval_path=[],
        timing={},
        error="",
        # Versioning fields (Phase 0)
        index_version=settings.index_version,
        pipeline_version=settings.pipeline_version,
        prompt_version=settings.langfuse_prompt_label or "default",
        config_hash=compute_config_hash(),
        # LLM Backend
        agent_backend=agent_backend,
    )
