"""
GraphRAG 文檔向量化 API

提供文檔攝取與向量化功能，支援：
- 單一文檔向量化（JSON 內容）
- 檔案上傳向量化（multipart/form-data）
- 批次目錄掃描向量化（指定目錄路徑）
- 工作狀態查詢

端點總覽：
- POST /api/v1/rag/vectorize           - 單一文檔向量化
- POST /api/v1/rag/vectorize/file      - 上傳檔案向量化
- POST /api/v1/rag/vectorize/directory - 批次目錄掃描向量化
- GET  /api/v1/rag/vectorize/status/{job_id} - 查詢工作狀態
"""

import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from chatbot_graphrag.services.ingestion import (
    ingestion_coordinator,
    IngestJobConfig,
    DocumentInput,
)
from chatbot_graphrag.core.constants import PipelineType, ProcessMode, JobStatus
from chatbot_graphrag.core.config import settings

logger = logging.getLogger(__name__)

# ============================================================
# API 路由器配置
# ============================================================

router = APIRouter(prefix="/api/v1/rag", tags=["vectorize"])
"""
GraphRAG 向量化 API 路由器

前綴：/api/v1/rag
標籤：vectorize

所有向量化相關端點都在此路由器下管理。
"""


# ============================================================
# 列舉類型定義
# ============================================================


class PipelineMode(str, Enum):
    """
    處理管道模式。

    GraphRAG 系統支援兩種文檔處理管道：

    Attributes:
        CURATED: 精選管道 - 用於結構化的 YAML+Markdown 文檔
            - 解析 YAML frontmatter 中的 metadata
            - 根據 doc_type 選擇專用 chunker
            - 適用於：醫療指南、程序說明、醫師資料等
        RAW: 原始管道 - 用於非結構化文檔
            - 使用通用 chunker 處理
            - 適用於：PDF、DOCX、HTML、純文字等
    """

    CURATED = "curated"
    RAW = "raw"


class DirectoryProcessMode(str, Enum):
    """
    目錄處理模式。

    控制如何處理目錄中的文檔：

    Attributes:
        OVERRIDE: 覆蓋模式 - 清空現有向量後重新建立
            - 刪除 Qdrant 集合中的所有向量
            - 重建 OpenSearch 索引
            - 重新處理所有文檔
            - ⚠️ 警告：此操作不可逆，請謹慎使用
        UPDATE: 更新模式 - 增量更新（預設）
            - 只處理新增或修改的文檔
            - 保留現有向量
            - 根據文檔雜湊值判斷是否需要重新處理
    """

    OVERRIDE = "override"
    UPDATE = "update"


# ============================================================
# 請求/響應模型定義
# ============================================================


class VectorizeRequest(BaseModel):
    """
    單一文檔向量化請求模型。

    用於將單一文檔內容向量化並儲存到 GraphRAG 系統。
    支援 Curated（YAML+MD）和 Raw（純文字）兩種管道。

    Attributes:
        content: 文檔內容（Markdown 或純文字）
            - Curated 管道：應包含 YAML frontmatter
            - Raw 管道：純文字內容
        doc_type: 文檔類型（可選）
            - procedure.* : 程序說明類
            - guide.*     : 指南類
            - physician   : 醫師資料
            - hospital_team: 醫療團隊
            - 若未指定，系統會自動偵測
        filename: 原始檔名（可選）
            - 用於追蹤來源和日誌記錄
        metadata: 額外 metadata（可選）
            - 會合併到文檔 metadata 中
        acl_groups: 存取控制群組
            - 預設為 ["public"]
            - 用於多租戶權限控制
        tenant_id: 租戶識別碼
            - 預設為 "default"
            - 用於多租戶資料隔離
        async_mode: 是否在背景執行
            - True: 立即返回 job_id，在背景處理
            - False: 同步執行，等待完成後返回
        pipeline: 處理管道類型
            - "curated": 精選管道（預設）
            - "raw": 原始管道

    Example:
        ```json
        {
            "content": "---\\ntitle: 住院須知\\ndoc_type: procedure.admission\\n---\\n# 住院流程...",
            "doc_type": "procedure.admission",
            "filename": "admission_guide.md",
            "pipeline": "curated"
        }
        ```
    """

    content: str = Field(
        ...,
        description="文檔內容（Markdown 或純文字）",
        examples=["---\ntitle: 範例文檔\n---\n# 標題\n內容..."],
    )
    doc_type: Optional[str] = Field(
        None,
        description="文檔類型（如：procedure.admission, guide.health, physician）",
        examples=["procedure.admission", "guide.health", "physician"],
    )
    filename: Optional[str] = Field(
        None,
        description="原始檔名，用於追蹤來源",
        examples=["住院須知.md"],
    )
    metadata: Optional[dict] = Field(
        default_factory=dict,
        description="額外 metadata，會合併到文檔 metadata 中",
    )
    acl_groups: Optional[list[str]] = Field(
        default=["public"],
        description="存取控制群組列表",
        examples=[["public"], ["staff", "admin"]],
    )
    tenant_id: str = Field(
        default="default",
        description="租戶識別碼，用於多租戶資料隔離",
    )
    async_mode: bool = Field(
        default=False,
        description="是否在背景執行。True=非同步返回 job_id，False=同步等待完成",
    )
    pipeline: str = Field(
        default="curated",
        description="處理管道類型：curated（精選）或 raw（原始）",
        examples=["curated", "raw"],
    )


class DirectoryVectorizeRequest(BaseModel):
    """
    批次目錄向量化請求模型。

    用於批次處理指定目錄下的所有文檔。
    支援遞迴掃描子目錄和多種檔案類型。

    Attributes:
        directory: 目錄路徑
            - 支援絕對路徑和相對路徑
            - 相對路徑基於專案根目錄
            - 若為 None，使用預設目錄（rag_test_data/docs）
        mode: 處理模式
            - override: 清空後重建
            - update: 增量更新（預設）
        pipeline: 處理管道類型
            - curated: 精選管道（適用 .md 檔案）
            - raw: 原始管道（適用 .pdf, .docx 等）
        recursive: 是否遞迴掃描子目錄
            - True: 包含所有子目錄
            - False: 只掃描指定目錄
        file_extensions: 要處理的副檔名列表
            - 預設：[".md"]
            - 可擴充：[".md", ".txt", ".pdf", ".docx"]
        async_mode: 是否在背景執行
            - 建議大量文檔時使用 True
        acl_groups: 存取控制群組
        tenant_id: 租戶識別碼
        enable_graph: 是否啟用圖譜抽取
            - True: 抽取實體和關係，建立知識圖譜
            - False: 只進行向量化，不建立圖譜
        enable_community_detection: 是否啟用社群偵測
            - True: 使用 Leiden 演算法偵測社群
            - False: 不進行社群偵測（預設）
            - 注意：需要先啟用 enable_graph

    Example:
        ```json
        {
            "directory": "rag_test_data/docs",
            "mode": "update",
            "pipeline": "curated",
            "recursive": true,
            "file_extensions": [".md"],
            "enable_graph": true
        }
        ```
    """

    directory: Optional[str] = Field(
        None,
        description="目錄路徑。若為 None，使用預設目錄（rag_test_data/docs）",
        examples=["rag_test_data/docs", "/absolute/path/to/docs"],
    )
    mode: DirectoryProcessMode = Field(
        default=DirectoryProcessMode.UPDATE,
        description="處理模式：override（覆蓋）或 update（更新）",
    )
    pipeline: PipelineMode = Field(
        default=PipelineMode.CURATED,
        description="處理管道類型：curated（精選）或 raw（原始）",
    )
    recursive: bool = Field(
        default=True,
        description="是否遞迴掃描子目錄",
    )
    file_extensions: list[str] = Field(
        default=[".md"],
        description="要處理的副檔名列表",
        examples=[[".md"], [".md", ".txt"], [".md", ".pdf", ".docx"]],
    )
    async_mode: bool = Field(
        default=True,
        description="是否在背景執行。建議大量文檔時使用 True",
    )
    acl_groups: list[str] = Field(
        default=["public"],
        description="存取控制群組列表",
    )
    tenant_id: str = Field(
        default="default",
        description="租戶識別碼",
    )
    enable_graph: bool = Field(
        default=True,
        description="是否啟用圖譜抽取（實體、關係抽取）",
    )
    enable_community_detection: bool = Field(
        default=False,
        description="是否啟用社群偵測。需要先啟用 enable_graph",
    )


class VectorizeResponse(BaseModel):
    """
    向量化操作響應模型。

    返回向量化操作的執行結果和狀態資訊。

    Attributes:
        job_id: 工作識別碼
            - 格式：job_<12位十六進制>
            - 用於查詢工作狀態
        status: 工作狀態
            - pending: 等待中
            - running: 執行中
            - completed: 已完成
            - partial: 部分完成（有錯誤）
            - failed: 失敗
            - processing: 處理中（背景任務）
        message: 狀態訊息
            - 描述目前狀態或錯誤資訊
        doc_id: 文檔 ID（單一文檔處理時）
        chunk_count: 建立的 chunk 數量
        documents_processed: 已處理的文檔數量（批次處理時）
        documents_failed: 處理失敗的文檔數量（批次處理時）
        documents_skipped: 跳過的文檔數量（內容未變更）
        entities_extracted: 抽取的實體數量
        relations_extracted: 抽取的關係數量
        progress: 處理進度（0-100）
    """

    job_id: str = Field(
        ...,
        description="工作識別碼，格式：job_<12位十六進制>",
        examples=["job_a1b2c3d4e5f6"],
    )
    status: str = Field(
        ...,
        description="工作狀態：pending, running, completed, partial, failed, processing",
        examples=["completed", "processing", "failed"],
    )
    message: str = Field(
        ...,
        description="狀態訊息或錯誤描述",
        examples=["已成功處理 10 個文檔", "文檔攝取已在背景啟動"],
    )
    doc_id: Optional[str] = Field(
        None,
        description="文檔 ID（單一文檔處理時返回）",
    )
    chunk_count: Optional[int] = Field(
        None,
        description="建立的 chunk 數量",
    )
    documents_processed: Optional[int] = Field(
        None,
        description="已處理的文檔數量（批次處理時）",
    )
    documents_failed: Optional[int] = Field(
        None,
        description="處理失敗的文檔數量（批次處理時）",
    )
    documents_skipped: Optional[int] = Field(
        None,
        description="跳過的文檔數量（內容未變更）",
    )
    entities_extracted: Optional[int] = Field(
        None,
        description="抽取的實體數量",
    )
    relations_extracted: Optional[int] = Field(
        None,
        description="抽取的關係數量",
    )
    progress: Optional[float] = Field(
        None,
        description="處理進度（0-100）",
    )


# ============================================================
# API 端點定義
# ============================================================


@router.post("/vectorize", response_model=VectorizeResponse)
async def vectorize_document(
    request: VectorizeRequest,
    background_tasks: BackgroundTasks,
) -> VectorizeResponse:
    """
    單一文檔向量化端點。

    將單一文檔內容向量化並儲存到 GraphRAG 系統。
    支援同步和非同步兩種執行模式。

    處理流程：
    1. 解析請求參數，決定使用的管道類型
    2. 建立 DocumentInput 和 IngestJobConfig
    3. 根據 async_mode 決定同步或非同步執行
    4. 透過 IngestionCoordinator 執行攝取流程
    5. 返回工作狀態和結果

    Args:
        request (VectorizeRequest): 向量化請求
            - content: 文檔內容（必填）
            - doc_type: 文檔類型（可選）
            - pipeline: 處理管道類型
            - async_mode: 是否非同步執行

    Returns:
        VectorizeResponse: 向量化結果
            - job_id: 工作識別碼
            - status: 執行狀態
            - chunk_count: 建立的 chunk 數量

    Raises:
        HTTPException:
            - 500: 向量化處理過程發生錯誤

    Example:
        **同步處理（等待完成）**:
        ```bash
        curl -X POST "http://localhost:18000/api/v1/rag/vectorize" \\
             -H "Content-Type: application/json" \\
             -d '{
               "content": "---\\ntitle: 住院須知\\n---\\n# 住院流程\\n...",
               "doc_type": "procedure.admission",
               "pipeline": "curated",
               "async_mode": false
             }'
        ```

        **非同步處理（背景執行）**:
        ```bash
        curl -X POST "http://localhost:18000/api/v1/rag/vectorize" \\
             -H "Content-Type: application/json" \\
             -d '{
               "content": "大型文檔內容...",
               "async_mode": true
             }'
        ```

        **響應範例**:
        ```json
        {
          "job_id": "job_a1b2c3d4e5f6",
          "status": "completed",
          "message": "已成功處理 1 個文檔",
          "chunk_count": 15,
          "entities_extracted": 8,
          "relations_extracted": 5
        }
        ```

    Note:
        - Curated 管道會解析 YAML frontmatter 並根據 doc_type 選擇 chunker
        - Raw 管道使用通用 chunker 處理
        - 非同步模式下，使用 GET /vectorize/status/{job_id} 查詢進度
    """
    logger.info(
        f"向量化請求: type={request.doc_type}, "
        f"pipeline={request.pipeline}, async={request.async_mode}"
    )

    try:
        # --------------------------------------------------------
        # 步驟 1: 決定處理管道類型
        # --------------------------------------------------------
        pipeline_type = (
            PipelineType.RAW
            if request.pipeline.lower() == "raw"
            else PipelineType.CURATED
        )

        # --------------------------------------------------------
        # 步驟 2: 建立文檔輸入物件
        # --------------------------------------------------------
        doc_input = DocumentInput(
            content=request.content,
            file_path=request.filename,
            pipeline=pipeline_type,
        )

        # --------------------------------------------------------
        # 步驟 3: 建立工作配置
        # --------------------------------------------------------
        config = IngestJobConfig(
            pipeline=pipeline_type,
            collection_name=settings.qdrant_collection_chunks,
            enable_graph=True,
            enable_community_detection=False,  # 單一文檔不需要社群偵測
            contextual_chunking=True,
        )

        # --------------------------------------------------------
        # 步驟 4: 執行向量化
        # --------------------------------------------------------
        if request.async_mode:
            # 非同步模式：在背景執行
            job = ingestion_coordinator.create_job(config)
            job_id = job.id  # 捕獲 job_id 供背景任務使用

            async def run_ingestion():
                """背景攝取任務。"""
                try:
                    await ingestion_coordinator.ingest(
                        documents=[doc_input],
                        config=config,
                        job_id=job_id,  # 傳入現有 job_id
                    )
                except Exception as e:
                    logger.error(f"[{job_id}] 背景攝取錯誤: {e}")
                    # 更新 job 狀態為失敗
                    existing_job = ingestion_coordinator.get_job(job_id)
                    if existing_job:
                        existing_job.status = JobStatus.FAILED
                        existing_job.error_message = str(e)
                        existing_job.completed_at = datetime.utcnow()

            background_tasks.add_task(run_ingestion)

            return VectorizeResponse(
                job_id=job.id,
                status="processing",
                message="文檔攝取已在背景啟動",
            )

        else:
            # 同步模式：等待完成
            job = await ingestion_coordinator.ingest(
                documents=[doc_input],
                config=config,
            )

            return VectorizeResponse(
                job_id=job.id,
                status=job.status.value,
                message=f"已成功處理 {job.processed_documents} 個文檔",
                chunk_count=job.chunks_created,
                documents_processed=job.processed_documents,
                documents_failed=job.failed_documents,
                documents_skipped=job.skipped_documents,
                entities_extracted=job.entities_extracted,
                relations_extracted=job.relations_extracted,
                progress=job.progress,
            )

    except Exception as e:
        logger.error(f"向量化錯誤: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vectorize/file", response_model=VectorizeResponse)
async def vectorize_file(
    file: UploadFile = File(..., description="要向量化的檔案"),
    doc_type: Optional[str] = Form(None, description="文檔類型"),
    acl_groups: str = Form("public", description="存取控制群組（逗號分隔）"),
    tenant_id: str = Form("default", description="租戶識別碼"),
    async_mode: bool = Form(False, description="是否在背景執行"),
    background_tasks: BackgroundTasks = None,
) -> VectorizeResponse:
    """
    檔案上傳向量化端點。

    接收上傳的檔案，自動偵測類型後進行向量化處理。
    支援多種檔案格式，會根據副檔名自動選擇適當的處理管道。

    支援的檔案類型：
    - 文字檔案（使用 Curated 管道）：.md, .txt, .html
    - 二進位檔案（使用 Raw 管道）：.pdf, .docx

    處理流程：
    1. 讀取上傳的檔案內容
    2. 根據副檔名和 content-type 決定處理管道
    3. 解析 ACL 群組設定
    4. 建立 VectorizeRequest 並呼叫 vectorize_document

    Args:
        file (UploadFile): 上傳的檔案
            - 最大檔案大小依 FastAPI 設定
            - 必須有有效的檔名
        doc_type (str, optional): 文檔類型覆蓋
            - 若未指定，系統會自動偵測
        acl_groups (str): 存取控制群組
            - 多個群組用逗號分隔
            - 例如："staff,admin"
        tenant_id (str): 租戶識別碼
        async_mode (bool): 是否在背景執行

    Returns:
        VectorizeResponse: 向量化結果

    Raises:
        HTTPException:
            - 500: 檔案處理或向量化過程發生錯誤

    Example:
        **上傳 Markdown 檔案**:
        ```bash
        curl -X POST "http://localhost:18000/api/v1/rag/vectorize/file" \\
             -F "file=@住院須知.md" \\
             -F "doc_type=procedure.admission" \\
             -F "acl_groups=public"
        ```

        **上傳 PDF 檔案**:
        ```bash
        curl -X POST "http://localhost:18000/api/v1/rag/vectorize/file" \\
             -F "file=@medical_guide.pdf" \\
             -F "async_mode=true"
        ```

        **響應範例**:
        ```json
        {
          "job_id": "job_x1y2z3w4v5u6",
          "status": "completed",
          "message": "已成功處理 1 個文檔",
          "chunk_count": 25
        }
        ```

    Note:
        - PDF 和 DOCX 檔案會自動使用 Raw 管道
        - 文字檔案（.md, .txt）優先使用 UTF-8 編碼讀取
        - 若 UTF-8 解碼失敗，會回退到 Latin-1 編碼
    """
    logger.info(f"檔案上傳: {file.filename}, size={file.size}")

    # --------------------------------------------------------
    # 步驟 1: 讀取檔案內容
    # --------------------------------------------------------
    content = await file.read()

    # --------------------------------------------------------
    # 步驟 2: 決定內容類型和處理管道
    # --------------------------------------------------------
    filename = file.filename or "upload.txt"
    content_type = file.content_type or "text/plain"

    is_binary = False
    pipeline = "curated"

    # PDF 檔案
    if filename.endswith(".pdf") or content_type == "application/pdf":
        content_str = content.decode("latin-1")  # 二進位安全編碼
        is_binary = True
        pipeline = "raw"

    # DOCX 檔案
    elif filename.endswith(".docx"):
        content_str = content.decode("latin-1")
        is_binary = True
        pipeline = "raw"

    # 文字檔案（.md, .txt, .html 等）
    else:
        try:
            content_str = content.decode("utf-8")
        except UnicodeDecodeError:
            content_str = content.decode("latin-1")

    # --------------------------------------------------------
    # 步驟 3: 解析 ACL 群組
    # --------------------------------------------------------
    groups = [g.strip() for g in acl_groups.split(",") if g.strip()]

    # --------------------------------------------------------
    # 步驟 4: 建立請求並執行向量化
    # --------------------------------------------------------
    request = VectorizeRequest(
        content=content_str,
        doc_type=doc_type,
        filename=filename,
        metadata={"is_binary": is_binary, "content_type": content_type},
        acl_groups=groups,
        tenant_id=tenant_id,
        async_mode=async_mode,
        pipeline=pipeline,
    )

    return await vectorize_document(request, background_tasks)


@router.post("/vectorize/directory", response_model=VectorizeResponse)
async def vectorize_directory(
    request: DirectoryVectorizeRequest,
    background_tasks: BackgroundTasks,
) -> VectorizeResponse:
    """
    批次目錄向量化端點。

    掃描指定目錄中的所有文檔，批次進行向量化處理。
    支援遞迴掃描子目錄，可根據副檔名過濾檔案。

    處理流程：
    1. 驗證目錄路徑是否存在
    2. 遞迴掃描目錄，收集符合條件的檔案
    3. 根據處理模式決定是否清空現有資料
    4. 批次建立 DocumentInput 列表
    5. 透過 IngestionCoordinator 執行攝取

    Args:
        request (DirectoryVectorizeRequest): 目錄向量化請求
            - directory: 目錄路徑（預設使用 rag_test_data/docs）
            - mode: 處理模式（update/override）
            - recursive: 是否遞迴子目錄
            - file_extensions: 副檔名過濾列表

    Returns:
        VectorizeResponse: 向量化結果
            - 包含處理的文檔數量和 chunk 數量

    Raises:
        HTTPException:
            - 400: 目錄不存在
            - 500: 向量化處理過程發生錯誤

    Example:
        **使用預設目錄（增量更新）**:
        ```bash
        curl -X POST "http://localhost:18000/api/v1/rag/vectorize/directory" \\
             -H "Content-Type: application/json" \\
             -d '{
               "mode": "update"
             }'
        ```

        **指定目錄路徑（覆蓋模式）**:
        ```bash
        curl -X POST "http://localhost:18000/api/v1/rag/vectorize/directory" \\
             -H "Content-Type: application/json" \\
             -d '{
               "directory": "/path/to/custom/docs",
               "mode": "override",
               "recursive": true,
               "file_extensions": [".md", ".txt"]
             }'
        ```

        **啟用圖譜抽取和社群偵測**:
        ```bash
        curl -X POST "http://localhost:18000/api/v1/rag/vectorize/directory" \\
             -H "Content-Type: application/json" \\
             -d '{
               "directory": "rag_test_data/docs",
               "enable_graph": true,
               "enable_community_detection": true,
               "async_mode": true
             }'
        ```

        **響應範例**:
        ```json
        {
          "job_id": "job_batch123456",
          "status": "processing",
          "message": "正在處理 50 個文檔...",
          "progress": 0
        }
        ```

    Note:
        - 覆蓋模式（override）會刪除所有現有向量，請謹慎使用
        - 大量文檔建議使用 async_mode=true
        - 使用 GET /vectorize/status/{job_id} 查詢進度
        - 預設目錄為 settings.default_docs_path（通常是 rag_test_data/docs）
    """
    # --------------------------------------------------------
    # 步驟 1: 決定目錄路徑
    # --------------------------------------------------------
    if request.directory:
        dir_path = Path(request.directory)
        # 支援相對路徑
        if not dir_path.is_absolute():
            dir_path = Path.cwd() / dir_path
    else:
        # 使用預設目錄
        dir_path = Path(settings.default_docs_path)

    logger.info(
        f"目錄向量化請求: path={dir_path}, "
        f"mode={request.mode.value}, recursive={request.recursive}"
    )

    # --------------------------------------------------------
    # 步驟 2: 驗證目錄存在
    # --------------------------------------------------------
    if not dir_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"目錄不存在: {dir_path}",
        )

    if not dir_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"路徑不是目錄: {dir_path}",
        )

    # --------------------------------------------------------
    # 步驟 3: 掃描目錄，收集檔案
    # --------------------------------------------------------
    file_paths: list[Path] = []
    extensions = set(ext.lower() for ext in request.file_extensions)

    if request.recursive:
        # 遞迴掃描
        for ext in extensions:
            file_paths.extend(dir_path.rglob(f"*{ext}"))
    else:
        # 只掃描指定目錄
        for ext in extensions:
            file_paths.extend(dir_path.glob(f"*{ext}"))

    # 排序以確保順序一致
    file_paths = sorted(set(file_paths))

    if not file_paths:
        return VectorizeResponse(
            job_id="",
            status="warning",
            message=f"在 {dir_path} 中未找到符合條件的檔案",
            documents_processed=0,
        )

    logger.info(f"找到 {len(file_paths)} 個檔案待處理")

    # --------------------------------------------------------
    # 步驟 4: 建立文檔輸入列表
    # --------------------------------------------------------
    pipeline_type = (
        PipelineType.RAW
        if request.pipeline == PipelineMode.RAW
        else PipelineType.CURATED
    )

    documents: list[DocumentInput] = []
    for fp in file_paths:
        documents.append(
            DocumentInput(
                file_path=str(fp),
                pipeline=pipeline_type,
            )
        )

    # --------------------------------------------------------
    # 步驟 5: 建立工作配置
    # --------------------------------------------------------
    config = IngestJobConfig(
        pipeline=pipeline_type,
        collection_name=settings.qdrant_collection_chunks,
        enable_graph=request.enable_graph,
        enable_community_detection=request.enable_community_detection,
        contextual_chunking=True,
        process_mode=(
            ProcessMode.OVERRIDE
            if request.mode == DirectoryProcessMode.OVERRIDE
            else ProcessMode.UPDATE
        ),
    )

    # --------------------------------------------------------
    # 步驟 6: 執行向量化
    # --------------------------------------------------------
    try:
        if request.async_mode:
            # 非同步模式：在背景執行
            job = ingestion_coordinator.create_job(config)
            job.total_documents = len(documents)
            job_id = job.id  # 捕獲 job_id 供背景任務使用
            docs_to_process = documents  # 捕獲 documents 供背景任務使用

            async def run_batch_ingestion():
                """背景批次攝取任務。"""
                try:
                    await ingestion_coordinator.ingest(
                        documents=docs_to_process,
                        config=config,
                        job_id=job_id,  # 傳入現有 job_id
                    )
                except Exception as e:
                    logger.error(f"[{job_id}] 批次攝取錯誤: {e}")
                    # 更新 job 狀態為失敗
                    existing_job = ingestion_coordinator.get_job(job_id)
                    if existing_job:
                        existing_job.status = JobStatus.FAILED
                        existing_job.error_message = str(e)
                        existing_job.completed_at = datetime.utcnow()

            background_tasks.add_task(run_batch_ingestion)

            return VectorizeResponse(
                job_id=job.id,
                status="processing",
                message=f"正在處理 {len(documents)} 個文檔...",
                documents_processed=0,
                progress=0,
            )

        else:
            # 同步模式：等待完成
            job = await ingestion_coordinator.ingest(
                documents=documents,
                config=config,
            )

            return VectorizeResponse(
                job_id=job.id,
                status=job.status.value,
                message=f"已處理 {job.processed_documents}/{job.total_documents} 個文檔",
                documents_processed=job.processed_documents,
                documents_failed=job.failed_documents,
                documents_skipped=job.skipped_documents,
                chunk_count=job.chunks_created,
                entities_extracted=job.entities_extracted,
                relations_extracted=job.relations_extracted,
                progress=job.progress,
            )

    except Exception as e:
        logger.error(f"目錄向量化錯誤: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vectorize/status/{job_id}", response_model=VectorizeResponse)
async def get_job_status(job_id: str) -> VectorizeResponse:
    """
    查詢向量化工作狀態。

    用於查詢非同步向量化工作的執行狀態和進度。
    可以定期輪詢此端點以追蹤長時間執行的任務。

    Args:
        job_id (str): 工作識別碼
            - 格式：job_<12位十六進制>
            - 由 vectorize 端點返回

    Returns:
        VectorizeResponse: 工作狀態
            - status: 目前狀態
            - progress: 處理進度（0-100）
            - message: 狀態描述

    Raises:
        HTTPException:
            - 404: 找不到指定的工作
            - 500: 查詢過程發生錯誤

    Example:
        **查詢工作狀態**:
        ```bash
        curl "http://localhost:18000/api/v1/rag/vectorize/status/job_a1b2c3d4e5f6"
        ```

        **響應範例（執行中）**:
        ```json
        {
          "job_id": "job_a1b2c3d4e5f6",
          "status": "running",
          "message": "已處理 25/50 個文檔",
          "documents_processed": 25,
          "progress": 50.0
        }
        ```

        **響應範例（已完成）**:
        ```json
        {
          "job_id": "job_a1b2c3d4e5f6",
          "status": "completed",
          "message": "已處理 50/50 個文檔",
          "documents_processed": 50,
          "chunk_count": 320,
          "entities_extracted": 150,
          "relations_extracted": 85,
          "progress": 100.0
        }
        ```

    Note:
        - 工作狀態保存在記憶體中，服務重啟後會遺失
        - 建議在工作開始後定期輪詢（如每 2-5 秒）
        - 長時間執行的工作建議使用 WebSocket 或 SSE 取代輪詢
    """
    try:
        job = ingestion_coordinator.get_job(job_id)

        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"找不到工作: {job_id}",
            )

        # 構建狀態訊息
        status_parts = [f"已處理 {job.processed_documents}/{job.total_documents} 個文檔"]
        if job.skipped_documents > 0:
            status_parts.append(f"跳過 {job.skipped_documents} 個")
        if job.failed_documents > 0:
            status_parts.append(f"失敗 {job.failed_documents} 個")

        return VectorizeResponse(
            job_id=job.id,
            status=job.status.value,
            message=", ".join(status_parts),
            documents_processed=job.processed_documents,
            documents_failed=job.failed_documents,
            documents_skipped=job.skipped_documents,
            chunk_count=job.chunks_created,
            entities_extracted=job.entities_extracted,
            relations_extracted=job.relations_extracted,
            progress=job.progress,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查詢工作狀態錯誤: {e}")
        raise HTTPException(status_code=500, detail=str(e))
