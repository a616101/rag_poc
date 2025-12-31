"""
================================================================================
Ask Stream API Route - Responses API 格式的串流問答端點
================================================================================

本模組實現 GraphRAG 系統的串流問答 API，使用自定義的 SSE 事件格式。

端點列表：
---------
- POST /api/v1/rag/ask/stream   - 串流問答（SSE 格式）
- POST /api/v1/rag/ask          - 非串流問答（JSON 格式）

SSE 事件類型：
-------------
1. response.start   - 串流開始，包含 trace_id
2. response.chunk   - 回應文字片段
3. response.sources - 來源引用資訊
4. response.done    - 串流完成，包含信心分數
5. response.error   - 發生錯誤

主要特性：
---------
1. GraphRAG 工作流程整合
   - 支援 Local、Global、DRIFT 三種查詢模式
   - 自動意圖分析與路由
   - 知識圖譜增強檢索

2. ACL 存取控制
   - 支援多租戶隔離
   - 群組權限過濾

3. Langfuse 可觀測性
   - 完整的追蹤鏈路
   - 每次請求都有唯一 trace_id

使用範例：
---------
# 串流請求
curl -X POST "http://localhost:8000/api/v1/rag/ask/stream" \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "什麼是糖尿病？",
    "stream": true,
    "include_sources": true
  }'

# 非串流請求
curl -X POST "http://localhost:8000/api/v1/rag/ask" \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "什麼是糖尿病？"
  }'

依賴關係：
---------
- chatbot_graphrag.services.ask_service: 統一問答服務
- fastapi: Web 框架
- pydantic: 資料驗證

作者：GraphRAG Team
版本：1.1.0 - 整合 Langfuse tracing 改進
================================================================================
"""

import json
import logging
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from chatbot_graphrag.graph_workflow import build_graphrag_workflow_with_memory
from chatbot_graphrag.services.ask_service import ask_service

# ============================================================================
# 模組設定
# ============================================================================

logger = logging.getLogger(__name__)

# 建立路由器
# prefix: API 路徑前綴
# tags: OpenAPI 文件分類標籤
router = APIRouter(prefix="/api/v1/rag", tags=["ask"])


# ============================================================================
# 請求/回應模型
# ============================================================================


class AskRequest(BaseModel):
    """
    問答請求模型。

    用於接收使用者的問題和相關設定參數。

    屬性說明：
    ---------
    question : str
        使用者問題，長度限制 1-4000 字元。
        問題應該清晰明確，避免過於模糊的詢問。

    conversation_id : Optional[str]
        對話 ID，用於追蹤同一對話的上下文。
        若提供此 ID，系統會嘗試利用對話歷史來增強回答。

    acl_groups : Optional[list[str]]
        存取控制群組列表，預設為 ["public"]。
        系統只會檢索使用者有權限存取的文件。
        範例：["public", "internal", "medical-staff"]

    tenant_id : str
        租戶識別碼，預設為 "default"。
        用於多租戶環境下的資料隔離。

    stream : bool
        是否啟用串流回應，預設為 True。
        - True: 以 SSE 格式即時串流回應
        - False: 等待完整回答後一次返回

    include_sources : bool
        是否包含來源引用，預設為 True。
        啟用時會在回應中附帶相關文件的引用資訊。

    範例：
    -----
    {
        "question": "糖尿病的治療方式有哪些？",
        "conversation_id": "conv-123",
        "acl_groups": ["public", "patient"],
        "tenant_id": "hospital-a",
        "stream": true,
        "include_sources": true
    }
    """

    question: str = Field(
        ...,  # 必填欄位
        min_length=1,
        max_length=4000,
        description="使用者問題（必填），長度 1-4000 字元"
    )
    conversation_id: Optional[str] = Field(
        None,
        description="對話 ID，用於追蹤對話上下文"
    )
    thread_id: Optional[str] = Field(
        None,
        description="執行緒 ID，用於 HITL 工作流程的狀態持久化。若未提供則自動生成。"
    )
    acl_groups: Optional[list[str]] = Field(
        default=["public"],
        description="存取控制群組列表"
    )
    tenant_id: str = Field(
        default="default",
        description="租戶識別碼"
    )
    stream: bool = Field(
        default=True,
        description="是否啟用串流回應"
    )
    include_sources: bool = Field(
        default=True,
        description="是否包含來源引用"
    )
    enable_hitl: bool = Field(
        default=False,
        description="是否啟用 HITL（Human-in-the-Loop）模式，需要人工審核時會暫停並返回狀態"
    )


class AskResponse(BaseModel):
    """
    問答回應模型（非串流）。

    當 stream=False 時返回此格式的完整回應。

    屬性說明：
    ---------
    answer : str
        系統生成的回答文字。
        回答會基於檢索到的相關文件生成。

    sources : list[dict]
        來源引用列表，包含用於生成回答的文件資訊。
        每個來源包含：
        - chunk_id: 文件片段 ID
        - content: 片段內容（截斷至 200 字元）
        - source_doc: 來源文件名稱
        - relevance_score: 相關性分數 (0-1)

    confidence : float
        回答信心分數 (0-1)。
        分數越高表示系統對回答品質越有信心。

    trace_id : str
        追蹤 ID，用於 Langfuse 可觀測性追蹤。

    回應範例：
    ---------
    {
        "answer": "糖尿病的治療方式包括...",
        "sources": [
            {
                "chunk_id": "chunk-abc123",
                "content": "糖尿病治療包含飲食控制、運動療法...",
                "source_doc": "糖尿病衛教手冊.md",
                "relevance_score": 0.92
            }
        ],
        "confidence": 0.85,
        "trace_id": "550e8400-e29b-41d4-a716-446655440000"
    }
    """

    answer: str = Field(
        ...,
        description="系統生成的回答"
    )
    sources: list[dict] = Field(
        default=[],
        description="來源引用列表"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="回答信心分數 (0-1)"
    )
    trace_id: str = Field(
        default="",
        description="追蹤 ID"
    )
    thread_id: str = Field(
        default="",
        description="執行緒 ID，用於恢復 HITL 工作流程"
    )
    status: str = Field(
        default="completed",
        description="工作流程狀態：completed, pending_review, error"
    )


class HITLResumeRequest(BaseModel):
    """
    HITL 恢復請求模型。

    用於恢復暫停的 HITL 工作流程。

    屬性說明：
    ---------
    thread_id : str
        執行緒 ID，必須是之前暫停的工作流程的 ID。

    approved : bool
        人工審核結果。True 表示批准繼續，False 表示拒絕。

    reviewer_comment : Optional[str]
        審核者的評論或修改建議。

    modified_answer : Optional[str]
        如果需要修改回答，可以在此提供修正後的回答。
    """

    thread_id: str = Field(
        ...,
        description="執行緒 ID，用於恢復暫停的工作流程"
    )
    approved: bool = Field(
        default=True,
        description="人工審核結果：True=批准繼續，False=拒絕"
    )
    reviewer_comment: Optional[str] = Field(
        None,
        description="審核者的評論"
    )
    modified_answer: Optional[str] = Field(
        None,
        description="修正後的回答（如果需要修改）"
    )


class HITLStatusResponse(BaseModel):
    """
    HITL 狀態回應模型。

    用於查詢工作流程的當前狀態。
    """

    thread_id: str = Field(..., description="執行緒 ID")
    status: str = Field(..., description="狀態：pending_review, completed, error")
    question: str = Field(default="", description="原始問題")
    draft_answer: str = Field(default="", description="待審核的回答草稿")
    groundedness_score: float = Field(default=0.0, description="落地性分數")
    needs_review_reason: str = Field(default="", description="需要審核的原因")
    created_at: Optional[str] = Field(None, description="建立時間")
    updated_at: Optional[str] = Field(None, description="更新時間")


# ============================================================================
# SSE 串流生成器
# ============================================================================


async def generate_sse_stream(
    question: str,
    acl_groups: list[str],
    tenant_id: str,
    include_sources: bool,
    session_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    生成 SSE (Server-Sent Events) 串流回應。

    使用統一的 ask_service 執行 GraphRAG 工作流程，確保完整的 Langfuse 追蹤。

    參數說明：
    ---------
    question : str
        使用者問題。

    acl_groups : list[str]
        存取控制群組列表。

    tenant_id : str
        租戶識別碼。

    include_sources : bool
        是否包含來源引用。

    session_id : Optional[str]
        會話 ID，用於 Langfuse 會話追蹤。

    產出事件：
    ---------
    1. response.start - 串流開始事件
    2. response.chunk - 回應文字片段
    3. response.sources - 來源引用資訊
    4. response.done - 串流完成事件
    5. response.error - 錯誤事件
    """
    try:
        async for event in ask_service.ask_stream(
            question=question,
            acl_groups=acl_groups,
            tenant_id=tenant_id,
            session_id=session_id,
            include_sources=include_sources,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    except Exception as e:
        logger.error(f"串流生成錯誤: {e}")
        error_event = {"type": "response.error", "error": str(e)}
        yield f"data: {json.dumps(error_event)}\n\n"


# ============================================================================
# API 端點
# ============================================================================


@router.post("/ask/stream")
async def ask_stream(request: AskRequest):
    """
    串流問答端點 - 以 SSE 格式即時串流回應。

    此端點支援串流和非串流兩種模式：
    - stream=True（預設）：以 SSE 格式即時串流回應
    - stream=False：等待完整回答後返回 JSON

    工作流程：
    ---------
    1. 接收使用者問題和設定參數
    2. 執行 GraphRAG 工作流程（意圖分析、檢索、生成）
    3. 以 SSE 格式串流回應或返回完整 JSON

    SSE 事件序列：
    -------------
    ```
    data: {"type": "response.start", "trace_id": "..."}

    data: {"type": "response.chunk", "content": "糖尿病"}

    data: {"type": "response.chunk", "content": "是一種慢性"}

    data: {"type": "response.chunk", "content": "代謝疾病..."}

    data: {"type": "response.sources", "sources": [...]}

    data: {"type": "response.done", "confidence": 0.85, "trace_id": "..."}

    ```

    HTTP 回應標頭：
    -------------
    - Content-Type: text/event-stream
    - Cache-Control: no-cache
    - Connection: keep-alive
    - X-Accel-Buffering: no（防止 nginx 緩衝）

    錯誤處理：
    ---------
    - 400 Bad Request: 請求參數無效
    - 500 Internal Server Error: 內部錯誤（非串流模式）
    - SSE response.error 事件（串流模式）

    使用範例：
    ---------
    # 使用 curl 測試串流
    curl -N -X POST "http://localhost:8000/api/v1/rag/ask/stream" \\
      -H "Content-Type: application/json" \\
      -H "Accept: text/event-stream" \\
      -d '{"question": "什麼是糖尿病？"}'

    # 使用 Python 讀取 SSE
    import httpx

    with httpx.stream("POST", url, json={"question": "..."}) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                print(event)

    參數：
    -----
    request : AskRequest
        問答請求，包含問題和相關設定

    返回：
    -----
    StreamingResponse | AskResponse
        串流模式返回 SSE 串流，非串流模式返回 JSON
    """
    logger.info(f"問答串流請求: {request.question[:50]}...")

    # ===== 非串流模式處理 =====
    if not request.stream:
        try:
            result = await ask_service.ask(
                question=request.question,
                acl_groups=request.acl_groups or ["public"],
                tenant_id=request.tenant_id,
                session_id=request.conversation_id,
            )

            return AskResponse(
                answer=result.get("answer", ""),
                sources=result.get("sources", []),
                confidence=result.get("confidence", 0.0),
                trace_id=result.get("trace_id", ""),
                thread_id=result.get("thread_id", ""),
                status=result.get("status", "completed"),
            )

        except Exception as e:
            logger.error(f"問答錯誤: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ===== 串流模式處理 =====
    return StreamingResponse(
        generate_sse_stream(
            question=request.question,
            acl_groups=request.acl_groups or ["public"],
            tenant_id=request.tenant_id,
            include_sources=request.include_sources,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",       # 禁用快取
            "Connection": "keep-alive",         # 保持連線
            "X-Accel-Buffering": "no",          # 禁用 nginx 緩衝
        },
    )


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """
    非串流問答端點 - 返回完整 JSON 回應。

    此端點為 /ask/stream 的簡化版本，強制使用非串流模式。
    適用於不需要即時串流的場景，如批次處理或簡單查詢。

    工作流程：
    ---------
    1. 接收使用者問題
    2. 執行 GraphRAG 工作流程
    3. 等待完整回答生成
    4. 返回 JSON 格式回應

    與串流端點的差異：
    -----------------
    - 強制 stream=False，無論請求中的設定
    - 返回完整的 AskResponse JSON
    - 適合需要完整回答的客戶端

    使用範例：
    ---------
    curl -X POST "http://localhost:8000/api/v1/rag/ask" \\
      -H "Content-Type: application/json" \\
      -d '{
        "question": "糖尿病的症狀有哪些？",
        "acl_groups": ["public"],
        "tenant_id": "default"
      }'

    回應範例：
    ---------
    {
        "answer": "糖尿病的常見症狀包括：多喝、多尿、多吃...",
        "sources": [
            {
                "chunk_id": "chunk-123",
                "content": "糖尿病的典型症狀包含三多一少...",
                "source_doc": "糖尿病衛教手冊.md"
            }
        ],
        "confidence": 0.88,
        "trace_id": "550e8400-e29b-41d4-a716-446655440000"
    }

    參數：
    -----
    request : AskRequest
        問答請求，包含問題和相關設定

    返回：
    -----
    AskResponse
        完整的問答回應，包含答案、來源和信心分數

    異常：
    -----
    HTTPException
        - 500: 內部伺服器錯誤
    """
    try:
        # 使用統一的 ask_service，內部已整合 Langfuse tracing
        result = await ask_service.ask(
            question=request.question,
            acl_groups=request.acl_groups or ["public"],
            tenant_id=request.tenant_id,
            user_id=None,  # TODO: 從認證系統取得
            session_id=request.conversation_id,
            conversation_id=request.conversation_id,
        )

        return AskResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            confidence=result.get("confidence", 0.0),
            trace_id=result.get("trace_id", ""),
            thread_id=result.get("thread_id", ""),
            status=result.get("status", "completed"),
        )

    except Exception as e:
        logger.error(f"問答錯誤: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HITL (Human-in-the-Loop) 端點
# ============================================================================


@router.post("/ask/resume", response_model=AskResponse)
async def ask_resume(request: HITLResumeRequest) -> AskResponse:
    """
    恢復暫停的 HITL 工作流程。

    Phase 3: HITL 支援端點。當工作流程因需要人工審核而暫停時，
    使用此端點恢復執行。

    工作流程：
    ---------
    1. 使用 thread_id 載入暫停的工作流程狀態
    2. 根據審核結果更新狀態
    3. 恢復工作流程執行
    4. 返回最終結果

    使用場景：
    ---------
    1. 落地性分數過低需要人工審核
    2. 回答包含敏感資訊需要確認
    3. 系統偵測到潛在風險需要人工介入

    使用範例：
    ---------
    curl -X POST "http://localhost:8000/api/v1/rag/ask/resume" \\
      -H "Content-Type: application/json" \\
      -d '{
        "thread_id": "550e8400-e29b-41d4-a716-446655440000",
        "approved": true,
        "reviewer_comment": "回答內容正確，批准發布"
      }'

    參數：
    -----
    request : HITLResumeRequest
        恢復請求，包含 thread_id 和審核結果

    返回：
    -----
    AskResponse
        完成後的問答回應

    異常：
    -----
    HTTPException
        - 404: 找不到指定的工作流程
        - 400: 工作流程不在暫停狀態
        - 500: 內部伺服器錯誤
    """
    logger.info(f"HITL 恢復請求: thread_id={request.thread_id}")

    try:
        # ===== 步驟 1：載入暫停的工作流程 =====
        workflow = build_graphrag_workflow_with_memory()

        # 設定恢復時的輸入
        resume_input = {
            "hitl_approved": request.approved,
            "hitl_reviewer_comment": request.reviewer_comment,
        }

        # 如果有修改後的回答，使用它
        if request.modified_answer:
            resume_input["final_answer"] = request.modified_answer

        # ===== 步驟 2：恢復工作流程執行 =====
        config = {"configurable": {"thread_id": request.thread_id}}

        # 使用 None 作為輸入恢復工作流程（繼續從中斷點）
        result = await workflow.ainvoke(resume_input, config)

        # ===== 步驟 3：格式化來源引用 =====
        sources = []
        evidence_table = result.get("evidence_table", [])
        for item in evidence_table[:5]:
            source_info = {
                "chunk_id": item.chunk_id if hasattr(item, "chunk_id") else "",
                "content": item.content[:200] if hasattr(item, "content") else "",
                "source_doc": item.source_doc if hasattr(item, "source_doc") else "",
            }
            sources.append(source_info)

        # ===== 步驟 4：返回回應 =====
        return AskResponse(
            answer=result.get("final_answer", ""),
            sources=sources,
            confidence=result.get("confidence", 0.0),
            trace_id=result.get("trace_id", ""),
            thread_id=request.thread_id,
            status="completed",
        )

    except Exception as e:
        logger.error(f"HITL 恢復錯誤: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ask/status/{thread_id}", response_model=HITLStatusResponse)
async def ask_status(thread_id: str) -> HITLStatusResponse:
    """
    查詢工作流程狀態。

    Phase 3: 查詢暫停的 HITL 工作流程狀態，用於前端顯示待審核清單。

    工作流程：
    ---------
    1. 使用 thread_id 載入工作流程狀態
    2. 解析狀態資訊
    3. 返回狀態摘要

    使用範例：
    ---------
    curl "http://localhost:8000/api/v1/rag/ask/status/550e8400-e29b-41d4"

    參數：
    -----
    thread_id : str
        要查詢的工作流程 ID

    返回：
    -----
    HITLStatusResponse
        工作流程狀態資訊

    異常：
    -----
    HTTPException
        - 404: 找不到指定的工作流程
        - 500: 內部伺服器錯誤
    """
    logger.info(f"HITL 狀態查詢: thread_id={thread_id}")

    try:
        # 載入工作流程狀態
        workflow = build_graphrag_workflow_with_memory()
        config = {"configurable": {"thread_id": thread_id}}

        # 獲取當前狀態
        state = await workflow.aget_state(config)

        if not state or not state.values:
            raise HTTPException(status_code=404, detail="找不到指定的工作流程")

        values = state.values

        # 判斷狀態
        status = "pending_review"
        if values.get("final_answer") and not values.get("needs_human_review"):
            status = "completed"
        elif values.get("error"):
            status = "error"

        return HITLStatusResponse(
            thread_id=thread_id,
            status=status,
            question=values.get("question", ""),
            draft_answer=values.get("final_answer", ""),
            groundedness_score=values.get("groundedness_score", 0.0),
            needs_review_reason=values.get("needs_review_reason", ""),
            created_at=values.get("created_at"),
            updated_at=values.get("updated_at"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"HITL 狀態查詢錯誤: {e}")
        raise HTTPException(status_code=500, detail=str(e))
