"""
RAG 相關 API 路由模組。

本模組提供完整的 RAG（Retrieval-Augmented Generation）功能端點，包括：
- 文檔向量化與儲存
- 智能問答（Agentic RAG）
- 文檔檢索測試
- 系統健康檢查
- 向量集合管理
"""
import json
from fastapi import APIRouter, File, HTTPException, UploadFile, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from chatbot_rag.core.config import settings
from chatbot_rag.models.rag import (
    CollectionInfoResponse,
    DocumentSource,
    HealthCheckResponse,
    QuestionRequest,
    RetrievalTestRequest,
    RetrievalTestResponse,
    VectorizeRequest,
    VectorizeResponse
)
from chatbot_rag.models.feedback import FeedbackRequest, FeedbackResponse
from chatbot_rag.services.document_service import document_service
from chatbot_rag.services.embedding_service import embedding_service
from chatbot_rag.services.qdrant_service import qdrant_service
from chatbot_rag.services.retriever_service import retriever_service
from chatbot_rag.services.ask_stream import ask_stream_service
from chatbot_rag.services.feedback_service import feedback_service


# 控制 LLM 串流細節是否輸出到 log
LLM_STREAM_DEBUG = settings.llm_stream_debug or settings.debug


# 建立 RAG 路由器，所有端點前綴為 /api/v1/rag
router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])


@router.post("/vectorize", response_model=VectorizeResponse)
async def vectorize_documents(request: VectorizeRequest):
    """
    文檔向量化端點。

    將指定目錄的文檔轉換為向量並儲存到 Qdrant 向量數據庫。
    這是建立知識庫的第一步，將文本文檔轉換為可搜索的向量。

    ## 處理流程

    1. **文檔讀取**：從指定目錄讀取 Markdown 文件
    2. **文本分塊**：將長文檔切分為適當大小的文本塊
    3. **向量生成**：使用 Embedding 模型生成向量
    4. **向量儲存**：將向量儲存到 Qdrant 數據庫

    ## 處理模式

    - **override**：刪除現有向量，重新建立集合
    - **update**：保留現有向量，只新增或更新新文檔

    Args:
        request (VectorizeRequest): 向量化請求，包含：
            - source (DocumentSource): 文檔來源（預設為 'default'）
            - mode (ProcessMode): 處理模式（'override' 或 'update'）
            - directory (str, optional): 自訂目錄路徑

    Returns:
        VectorizeResponse: 處理結果，包含：
            - status (str): 操作狀態
            - message (str): 詳細訊息
            - mode (str): 使用的處理模式
            - source (str): 文檔來源
            - documents_processed (int): 處理的文檔數量
            - chunks_created (int): 建立的文本塊數量
            - vectors_stored (int): 儲存的向量數量
            - collection_name (str): Qdrant 集合名稱

    Raises:
        HTTPException:
            - 400: 如果使用上傳來源（應使用 /vectorize/upload）
            - 500: 如果向量化處理失敗

    Example:
        **覆蓋模式（重建知識庫）**:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/rag/vectorize" \\
             -H "Content-Type: application/json" \\
             -d '{
               "source": "default",
               "mode": "override"
             }'
        ```

        Response:
        ```json
        {
          "status": "success",
          "message": "Documents vectorized successfully",
          "mode": "override",
          "source": "default",
          "documents_processed": 10,
          "chunks_created": 45,
          "vectors_stored": 45,
          "collection_name": "chatbot_rag"
        }
        ```

        **更新模式（增量更新）**:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/rag/vectorize" \\
             -H "Content-Type: application/json" \\
             -d '{
               "source": "default",
               "mode": "update",
               "directory": "/path/to/custom/docs"
             }'
        ```

    Note:
        - 上傳文件請使用 `/vectorize/upload` 端點
        - 預設目錄為 `rag_test_data/`
        - 只支援 Markdown (.md) 文件
        - 覆蓋模式會刪除所有現有向量，請謹慎使用
    """
    try:
        # 記錄向量化請求資訊
        logger.info(
            f"Vectorization request: source={request.source}, mode={request.mode}, "
            f"use_llm_context={request.use_llm_context}"
        )

        if request.source == DocumentSource.DEFAULT:
            # 從預設或自訂目錄處理文檔
            # 會讀取 Markdown 文件、切分文本、生成向量並儲存到 Qdrant
            result = document_service.process_directory(
                directory=request.directory,
                mode=request.mode.value,
                use_llm=request.use_llm_context,
            )
        else:
            # 上傳文件的情況由 /vectorize/upload 端點處理
            # 此端點只負責處理目錄中的文件
            raise HTTPException(
                status_code=400,
                detail="上傳文件請使用 /vectorize/upload 端點",
            )

        # 構建成功響應，包含處理統計資訊
        return VectorizeResponse(
            status=result["status"],
            message=result.get("message"),
            mode=result.get("mode", request.mode.value),
            source=request.source.value,
            documents_processed=result.get("documents_processed", 0),
            chunks_created=result.get("chunks_created", 0),
            vectors_stored=result.get("vectors_stored", 0),
            collection_name=result.get("collection_name", ""),
        )

    except Exception as e:
        # 記錄錯誤並返回 HTTP 500 錯誤
        logger.error(f"Vectorization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vectorize/upload", response_model=VectorizeResponse)
async def vectorize_uploaded_files(
    files: list[UploadFile] = File(...),
    mode: str = "update",
    use_llm_context: bool = True,
):
    """
    上傳文件向量化端點。

    接受多個文件上傳，處理文件內容並轉換為向量儲存到 Qdrant。
    適用於動態新增知識庫內容的場景。

    ## 處理流程

    1. **文件驗證**：檢查文件格式（只接受 .md 文件）
    2. **內容讀取**：讀取文件內容（UTF-8 編碼）
    3. **文本分塊**：將內容切分為文本塊
    4. **向量生成**：生成 Embedding 向量
    5. **向量儲存**：儲存到 Qdrant 數據庫

    ## 處理模式

    - **override**：刪除現有向量，只保留本次上傳的內容
    - **update**：保留現有向量，新增上傳的內容

    Args:
        files (list[UploadFile]): 上傳的文件列表（支援多文件）
        mode (str): 處理模式，可選值：
            - "update": 更新模式（預設）
            - "override": 覆蓋模式

    Returns:
        VectorizeResponse: 處理結果，包含：
            - status (str): 操作狀態
            - message (str): 詳細訊息
            - mode (str): 使用的處理模式
            - source (str): 固定為 "uploaded"
            - documents_processed (int): 處理的文檔數量
            - chunks_created (int): 建立的文本塊數量
            - vectors_stored (int): 儲存的向量數量
            - collection_name (str): Qdrant 集合名稱

    Raises:
        HTTPException:
            - 400: 如果沒有提供有效的 Markdown 文件
            - 400: 如果 mode 參數無效
            - 500: 如果處理過程失敗

    Example:
        **上傳單個文件**:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/rag/vectorize/upload" \\
             -F "files=@document1.md" \\
             -F "mode=update"
        ```

        **上傳多個文件**:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/rag/vectorize/upload" \\
             -F "files=@doc1.md" \\
             -F "files=@doc2.md" \\
             -F "files=@doc3.md" \\
             -F "mode=update"
        ```

        Response:
        ```json
        {
          "status": "success",
          "message": "Uploaded files vectorized successfully",
          "mode": "update",
          "source": "uploaded",
          "documents_processed": 3,
          "chunks_created": 15,
          "vectors_stored": 15,
          "collection_name": "chatbot_rag"
        }
        ```

    Note:
        - 只接受 Markdown (.md) 文件
        - 非 .md 文件會被自動跳過
        - 文件必須是 UTF-8 編碼
        - 覆蓋模式會刪除所有現有向量
        - 支援一次上傳多個文件
    """
    try:
        # 記錄收到的上傳文件數量
        logger.info(f"Received {len(files)} uploaded files for vectorization")

        # 驗證處理模式參數是否有效
        if mode not in ["override", "update"]:
            raise HTTPException(
                status_code=400,
                detail="處理模式必須是 'override' 或 'update'",
            )

        # 讀取並驗證所有上傳的文件
        file_contents = []
        for file in files:
            # 跳過沒有文件名的文件
            if not file.filename:
                continue

            # 只接受 Markdown 文件，其他格式會被跳過
            if not file.filename.endswith(".md"):
                logger.warning(f"Skipping non-markdown file: {file.filename}")
                continue

            # 讀取文件的二進制內容
            content = await file.read()
            try:
                # 嘗試將二進制內容解碼為 UTF-8 文本
                text = content.decode("utf-8")
                file_contents.append((file.filename, text))
                logger.debug(f"Read file: {file.filename} ({len(text)} chars)")
            except UnicodeDecodeError:
                # 如果解碼失敗（非 UTF-8 編碼），記錄錯誤並跳過該文件
                logger.error(f"Failed to decode file: {file.filename}")
                continue

        # 檢查是否有有效的文件可以處理
        if not file_contents:
            raise HTTPException(
                status_code=400,
                detail="沒有提供有效的 Markdown 文件",
            )

        # 呼叫文檔服務處理上傳的文件
        # 會進行文本切分、向量生成和儲存
        result = document_service.process_uploaded_files(
            file_contents=file_contents,
            mode=mode,
            use_llm=use_llm_context,
        )

        # 構建成功響應，包含處理統計資訊
        return VectorizeResponse(
            status=result["status"],
            message=result.get("message"),
            mode=result.get("mode", mode),
            source="uploaded",
            documents_processed=result.get("documents_processed", 0),
            chunks_created=result.get("chunks_created", 0),
            vectors_stored=result.get("vectors_stored", 0),
            collection_name=result.get("collection_name", ""),
        )

    except HTTPException:
        # 重新拋出 HTTPException（已包含適當的錯誤訊息）
        raise
    except Exception as e:
        # 捕獲其他未預期的異常，記錄並返回 HTTP 500 錯誤
        logger.error(f"Upload vectorization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collection/info", response_model=CollectionInfoResponse)
async def get_collection_info():
    """
    獲取向量集合資訊端點。

    查詢 Qdrant 向量數據庫中集合的詳細資訊，包括向量數量、狀態等。
    用於監控知識庫的規模和健康狀態。

    Returns:
        CollectionInfoResponse: 集合資訊，包含：
            - name (str): 集合名稱
            - points_count (int): 向量點數量（知識庫中的文本塊數量）
            - vectors_count (int, optional): 向量總數
            - status (str): 集合狀態
            - optimizer_status (str): 優化器狀態

    Raises:
        HTTPException:
            - 500: 如果查詢失敗

    Example:
        ```bash
        curl -X GET "http://localhost:8000/api/v1/rag/collection/info"
        ```

        Response:
        ```json
        {
          "name": "chatbot_rag",
          "points_count": 145,
          "vectors_count": 145,
          "status": "green",
          "optimizer_status": "ok"
        }
        ```

    Note:
        - points_count 表示知識庫中的文本塊數量
        - 集合不存在時會返回錯誤
        - 狀態 "green" 表示正常運行
    """
    try:
        # 從 Qdrant 服務獲取集合資訊
        info = qdrant_service.get_collection_info()
        # 將字典資料轉換為 Pydantic 模型並返回
        return CollectionInfoResponse(**info)

    except Exception as e:
        # 記錄錯誤並返回 HTTP 500 錯誤
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collection")
async def delete_collection():
    """
    刪除向量集合端點。

    刪除 Qdrant 中的整個向量集合，清空知識庫。
    這是一個危險操作，會永久刪除所有向量數據。

    Returns:
        dict: 刪除結果，包含：
            - status (str): 操作狀態（"success"）
            - message (str): 結果訊息
            - deleted (bool): 是否成功刪除

    Raises:
        HTTPException:
            - 500: 如果刪除失敗

    Example:
        ```bash
        curl -X DELETE "http://localhost:8000/api/v1/rag/collection"
        ```

        Response:
        ```json
        {
          "status": "success",
          "message": "Collection deleted successfully",
          "deleted": true
        }
        ```

    Warning:
        ⚠️ **危險操作**：此操作會永久刪除所有向量數據，無法恢復！
        - 刪除後需要重新向量化文檔才能使用問答功能
        - 建議在執行前確認是否真的需要刪除
        - 生產環境建議添加額外的權限驗證

    Note:
        - 刪除操作會立即生效
        - 刪除後集合將不存在，需重新建立
        - 可通過重新向量化文檔來恢復知識庫
    """
    try:
        # 記錄警告訊息（這是一個危險操作）
        logger.warning("Deleting collection")
        # 呼叫 Qdrant 服務刪除整個向量集合
        result = qdrant_service.delete_collection()

        # 返回刪除成功的響應
        return {
            "status": "success",
            "message": "Collection deleted successfully",
            "deleted": result,
        }

    except Exception as e:
        # 記錄錯誤並返回 HTTP 500 錯誤
        logger.error(f"Failed to delete collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask/stream")  # 建議開一條新的 SSE endpoint，原本 /ask 保留 JSON
async def ask_question_stream(http_request: Request, request: QuestionRequest):
    """
    Unified Agent 串流端點（Responses backend，預設）。

    - 使用新版 Agentic RAG LangGraph（guard → planner → tools → generation）
    - 事件格式與舊版保持相容，前端只需解析 `stage/channel`
    - 返回統一的 SSE 串流，包含狀態、工具輸出、回答與遙測摘要
    """

    async def event_generator():
        async for event in ask_stream_service.stream_events(
            request=request,
            is_disconnected=http_request.is_disconnected,
        ):
            yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/ask/stream_chat")
async def ask_question_stream_chat(http_request: Request, request: QuestionRequest):
    """
    Unified Agent 串流端點（Chat backend，保留與舊前端相容）。

    - 與 `/ask/stream` 共用 LangGraph，但採用 Chat API
    - 支援 reasoning tokens、reasoning summary 等增強資訊
    - 事件結構相同，可供前端切換觀察不同推論粒度
    """

    async def event_generator():
        async for event in ask_stream_service.stream_events_chat(
            request=request,
            is_disconnected=http_request.is_disconnected,
        ):
            yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/retrieve/test", response_model=RetrievalTestResponse)
async def test_retrieval(request: RetrievalTestRequest):
    """
    文檔檢索測試端點。

    測試向量檢索功能，不生成答案，只返回檢索到的相關文檔。
    用於調試和理解系統會為特定查詢檢索哪些文檔。

    ## 用途

    - **調試檢索效果**：查看查詢會檢索到哪些文檔
    - **評估相關性**：檢查檢索分數和相關性
    - **優化查詢**：測試不同查詢詞的檢索效果
    - **驗證知識庫**：確認知識庫中有相關內容

    ## 檢索邏輯

    1. 將查詢轉換為向量
    2. 在 Qdrant 中進行相似度搜索
    3. 返回相似度最高的 top_k 個文檔
    4. 過濾低於閾值（0.5）的結果

    Args:
        request (RetrievalTestRequest): 檢索測試請求，包含：
            - query (str): 測試查詢（必填）
            - top_k (int): 返回文檔數量（預設 3，範圍 1-10）

    Returns:
        RetrievalTestResponse: 檢索結果，包含：
            - status (str): 測試狀態（"success"）
            - query (str): 原始查詢
            - documents_found (int): 找到的文檔數量
            - documents (list[dict]): 檢索到的文檔列表，每個文檔包含：
                - content (str): 文檔內容
                - score (float): 相似度分數（0-1，越高越相關）
                - metadata (dict): 文檔元數據（來源文件等）

    Raises:
        HTTPException:
            - 500: 如果檢索失敗

    Example:
        **基本檢索測試**:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/rag/retrieve/test" \\
             -H "Content-Type: application/json" \\
             -d '{
               "query": "如何申請退款",
               "top_k": 3
             }'
        ```

        Response:
        ```json
        {
          "status": "success",
          "query": "如何申請退款",
          "documents_found": 3,
          "documents": [
            {
              "content": "退款政策說明...",
              "score": 0.89,
              "metadata": {"source": "refund_policy.md"}
            },
            {
              "content": "退款申請流程...",
              "score": 0.85,
              "metadata": {"source": "refund_process.md"}
            },
            {
              "content": "常見退款問題...",
              "score": 0.72,
              "metadata": {"source": "faq.md"}
            }
          ]
        }
        ```

        **測試不同 top_k 值**:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/rag/retrieve/test" \\
             -H "Content-Type: application/json" \\
             -d '{
               "query": "訂閱方案",
               "top_k": 5
             }'
        ```

    Note:
        - 分數閾值固定為 0.5（低於此分數的文檔會被過濾）
        - 返回的文檔數量可能少於 top_k（如果相關文檔不足）
        - 分數越高表示相關性越強
        - 此端點不會觸發輸入驗證（用於測試目的）
    """
    try:
        # 記錄檢索測試請求（截取前 50 個字元）
        logger.info(f"Testing retrieval for query: '{request.query[:50]}...'")

        # 從 Qdrant 檢索相關文檔
        # score_threshold=0.5 表示只返回相似度大於 0.5 的文檔
        # 相似度分數範圍為 0-1，分數越高表示相關性越強
        documents = retriever_service.retrieve(
            query=request.query,
            top_k=request.top_k,
            score_threshold=0.5,
        )

        # 返回檢索結果，包含找到的文檔和相似度分數
        return RetrievalTestResponse(
            status="success",
            query=request.query,
            documents_found=len(documents),
            documents=documents,
        )

    except Exception as e:
        # 記錄錯誤並返回 HTTP 500 錯誤
        logger.error(f"Retrieval test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    提交用戶對回答的評分。

    將用戶的讚/倒讚評分記錄到 Langfuse，用於追蹤回答品質。

    Args:
        request (FeedbackRequest): 回饋請求，包含：
            - trace_id (str): 從 meta_summary 事件取得的 Langfuse trace ID
            - score (str): "up" (讚) 或 "down" (倒讚)
            - comment (str, optional): 倒讚時的原因說明

    Returns:
        FeedbackResponse: 提交結果，包含：
            - success (bool): 是否成功
            - message (str): 結果訊息
            - score_id (str, optional): Langfuse 評分 ID

    Raises:
        HTTPException:
            - 500: 如果提交到 Langfuse 失敗

    Example:
        **提交讚**:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/rag/feedback" \\
             -H "Content-Type: application/json" \\
             -d '{
               "trace_id": "d2d1e2ddd5ab558f8388c6d9cf510ac8",
               "score": "up"
             }'
        ```

        **提交倒讚（附原因）**:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/rag/feedback" \\
             -H "Content-Type: application/json" \\
             -d '{
               "trace_id": "d2d1e2ddd5ab558f8388c6d9cf510ac8",
               "score": "down",
               "comment": "回答不夠完整"
             }'
        ```

        Response:
        ```json
        {
          "success": true,
          "message": "Feedback submitted",
          "score_id": "abc123"
        }
        ```

    Note:
        - 同一 trace_id 可以重複提交，Langfuse 會更新評分
        - 評分會顯示在 Langfuse Dashboard 的 Scores 區塊
    """
    success, message, score_id = feedback_service.submit_feedback(
        trace_id=request.trace_id,
        score=request.score,
        comment=request.comment,
    )

    if not success:
        raise HTTPException(status_code=500, detail=message)

    return FeedbackResponse(
        success=True,
        message=message,
        score_id=score_id,
    )
