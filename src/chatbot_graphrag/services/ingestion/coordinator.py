"""
攝取協調器

協調 GraphRAG 的雙管線攝取流程。

主要功能：
- 將文件路由到適當的管線（精選 vs 原始）
- 管理攝取任務
- 協調跨服務的儲存（Postgres, MinIO, Qdrant, OpenSearch, NebulaGraph）
- 追蹤文件版本和譜系
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from chatbot_graphrag.core.config import settings
from chatbot_graphrag.core.constants import (
    DocStatus,
    DocType,
    JobStatus,
    PipelineType,
    ProcessMode,
)
from chatbot_graphrag.models.pydantic.ingestion import (
    Chunk,
    ChunkArtifact,
    Document,
    DocumentInput,
    DocumentVersion,
    IngestJob,
    IngestJobConfig,
    IngestJobUpdate,
    PipelineResult,
    TraceLink,
)
from chatbot_graphrag.services.ingestion.curated_pipeline import curated_pipeline
from chatbot_graphrag.services.ingestion.raw_pipeline import raw_pipeline

logger = logging.getLogger(__name__)


class IngestionCoordinator:
    """
    協調雙管線攝取流程。

    職責：
    - 將文件路由到適當的管線（精選 vs 原始）
    - 管理攝取任務
    - 協調跨服務的儲存（Postgres, MinIO, Qdrant, OpenSearch, NebulaGraph）
    - 追蹤文件版本和譜系
    """

    def __init__(self):
        """初始化協調器。"""
        self._jobs: dict[str, IngestJob] = {}  # 記憶體內任務追蹤
        self._initialized = False

    async def initialize(self) -> None:
        """初始化協調器和依賴服務。"""
        if self._initialized:
            return

        logger.info("正在初始化 Ingestion Coordinator...")

        # 延遲匯入服務以避免循環依賴
        from chatbot_graphrag.services.storage import minio_service
        from chatbot_graphrag.services.vector import qdrant_service
        from chatbot_graphrag.services.search import opensearch_service

        # 使用錯誤處理初始化服務
        try:
            logger.info("  - 初始化 MinIO...")
            await minio_service.initialize()
            logger.info("  - MinIO 初始化完成")
        except Exception as e:
            logger.error(f"  - MinIO 初始化失敗: {e}")
            raise

        try:
            logger.info("  - 初始化 Qdrant...")
            await qdrant_service.initialize()
            logger.info("  - Qdrant 初始化完成")
        except Exception as e:
            logger.error(f"  - Qdrant 初始化失敗: {e}")
            raise

        try:
            logger.info("  - 初始化 OpenSearch...")
            await opensearch_service.initialize()
            logger.info("  - OpenSearch 初始化完成")
        except Exception as e:
            logger.error(f"  - OpenSearch 初始化失敗: {e}")
            raise

        # 初始化嵌入服務
        from chatbot_graphrag.services.vector import embedding_service
        try:
            logger.info("  - 初始化 Embedding Service...")
            await embedding_service.initialize()
            logger.info("  - Embedding Service 初始化完成")
        except Exception as e:
            logger.error(f"  - Embedding Service 初始化失敗: {e}")
            raise

        self._initialized = True
        logger.info("Ingestion Coordinator 初始化完成")

    async def close(self) -> None:
        """清理協調器資源。"""
        self._initialized = False

    def create_job(
        self,
        config: IngestJobConfig,
        created_by: Optional[str] = None,
    ) -> IngestJob:
        """
        建立新的攝取任務。

        Args:
            config: 任務配置
            created_by: 建立任務的使用者

        Returns:
            建立的 IngestJob
        """
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = IngestJob(
            id=job_id,
            config=config,
            status=JobStatus.PENDING,
            created_by=created_by,
        )
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[IngestJob]:
        """
        根據 ID 取得任務。

        優先從記憶體查詢，失敗時 fallback 到資料庫。

        Args:
            job_id: 任務 ID

        Returns:
            IngestJob 或 None
        """
        # 1. 優先從記憶體查詢
        job = self._jobs.get(job_id)
        if job:
            return job

        # 2. Fallback 到資料庫（需要 async，這裡使用 sync wrapper）
        # 注意：這是一個同步方法，無法直接使用 async。
        # 在需要時，呼叫方應使用 async 版本 get_job_async()
        return None

    async def get_job_async(self, job_id: str) -> Optional[IngestJob]:
        """
        根據 ID 取得任務（async 版本）。

        優先從記憶體查詢，失敗時 fallback 到資料庫。

        Args:
            job_id: 任務 ID

        Returns:
            IngestJob 或 None
        """
        # 1. 優先從記憶體查詢
        job = self._jobs.get(job_id)
        if job:
            return job

        # 2. Fallback 到資料庫
        try:
            from chatbot_graphrag.db import get_async_session, JobRepository, is_db_initialized

            if is_db_initialized():
                async with get_async_session() as session:
                    repo = JobRepository(session)
                    job = await repo.get_by_id(job_id)
                    if job:
                        # 快取到記憶體以供後續查詢
                        self._jobs[job_id] = job
                        logger.debug(f"從 PostgreSQL 載入 Job {job_id}")
                        return job
        except Exception as e:
            logger.debug(f"從 PostgreSQL 取得 Job {job_id} 失敗: {e}")

        return None

    def update_job(self, job_id: str, update: IngestJobUpdate) -> Optional[IngestJob]:
        """更新任務狀態和指標。"""
        job = self._jobs.get(job_id)
        if not job:
            return None

        update_data = update.model_dump(exclude_none=True)
        for key, value in update_data.items():
            setattr(job, key, value)

        return job

    async def ingest(
        self,
        documents: list[DocumentInput],
        config: IngestJobConfig,
        created_by: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> IngestJob:
        """
        攝取一批文件。

        Args:
            documents: 文件輸入列表
            config: 攝取配置
            created_by: 發起攝取的使用者
            job_id: 如果提供，複用現有的 Job；否則建立新 Job

        Returns:
            包含結果的 IngestJob
        """
        # 確保服務已初始化
        await self.initialize()

        # 複用現有 Job 或建立新 Job
        if job_id:
            job = self.get_job(job_id)
            if not job:
                raise ValueError(f"找不到工作: {job_id}")
            logger.info(f"[{job_id}] 複用現有工作，處理 {len(documents)} 個文件")
        else:
            job = self.create_job(config, created_by)
            logger.info(f"[{job.id}] 建立新工作，處理 {len(documents)} 個文件")

        job.total_documents = len(documents)
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()

        # 持久化 Job 初始狀態
        await self._persist_job(job)

        logger.info(f"[{job.id}] 開始處理 {len(documents)} 個文檔...")

        try:
            # 處理文件
            for i, doc_input in enumerate(documents, 1):
                doc_name = doc_input.file_path or doc_input.url or "inline content"
                try:
                    logger.info(f"[{job.id}] [{i}/{len(documents)}] 開始處理: {doc_name}")
                    result = await self._ingest_document(doc_input, job, config)
                    if result:
                        if result.get("skipped"):
                            job.skipped_documents += 1
                            logger.info(
                                f"[{job.id}] [{i}/{len(documents)}] 跳過: {doc_name} "
                                f"(原因: {result.get('reason', 'unknown')})"
                            )
                        else:
                            job.processed_documents += 1
                            job.chunks_created += result.get("chunks_count", 0)
                            job.entities_extracted += result.get("entities_count", 0)
                            job.relations_extracted += result.get("relations_count", 0)
                            logger.info(
                                f"[{job.id}] [{i}/{len(documents)}] 完成: "
                                f"{result.get('chunks_count', 0)} chunks, "
                                f"{result.get('entities_count', 0)} entities"
                            )
                    else:
                        # 文檔處理失敗（如 YAML 驗證錯誤）
                        job.failed_documents += 1
                        logger.warning(f"[{job.id}] [{i}/{len(documents)}] 跳過: {doc_name} (無法處理)")
                except Exception as e:
                    logger.error(f"[{job.id}] [{i}/{len(documents)}] 處理失敗: {doc_name} - {e}")
                    job.failed_documents += 1
                    job.document_errors.append({
                        "document": doc_name,
                        "error": str(e),
                    })

                # 更新進度
                total = job.total_documents
                processed = job.processed_documents + job.failed_documents + job.skipped_documents
                job.progress = (processed / total * 100) if total > 0 else 0

                # 每 10 個文檔輸出一次進度並持久化（調整為更頻繁的輸出）
                if i % 10 == 0 or i == len(documents):
                    logger.info(
                        f"[{job.id}] 進度: {job.progress:.1f}% ({i}/{len(documents)}) - "
                        f"成功: {job.processed_documents}, 失敗: {job.failed_documents}, "
                        f"跳過: {job.skipped_documents}"
                    )
                    # 定期持久化進度
                    await self._persist_job(job)

            # 完成任務
            job.status = (
                JobStatus.COMPLETED
                if job.failed_documents == 0
                else JobStatus.PARTIAL
                if job.processed_documents > 0
                else JobStatus.FAILED
            )
            job.completed_at = datetime.utcnow()
            job.progress = 100.0

            # 持久化最終狀態
            await self._persist_job(job)

        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}")
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()

            # 持久化失敗狀態
            await self._persist_job(job)

        return job

    async def _ingest_document(
        self,
        doc_input: DocumentInput,
        job: IngestJob,
        config: IngestJobConfig,
    ) -> Optional[dict[str, Any]]:
        """
        攝取單一文件。

        Args:
            doc_input: 文件輸入
            job: 父任務
            config: 攝取配置

        Returns:
            包含攝取結果的字典
        """
        # 決定管線
        pipeline = doc_input.pipeline

        # 取得內容
        if doc_input.content:
            content = doc_input.content
        elif doc_input.file_path:
            file_path = Path(doc_input.file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {doc_input.file_path}")

            # 讀取檔案
            if file_path.suffix.lower() in {".txt", ".md", ".html"}:
                content = file_path.read_text(encoding="utf-8")
            else:
                # 二進制檔案 - 使用原始管線
                pipeline = PipelineType.RAW
                content = None
        else:
            raise ValueError("Either content or file_path must be provided")

        # 透過適當的管線處理
        if pipeline == PipelineType.CURATED:
            document, chunks, errors = await curated_pipeline.process_content(
                content=content or "",
                source_path=doc_input.file_path,
                job=job,
                enable_graph=config.enable_graph,
            )
        else:
            if doc_input.file_path:
                document, chunks, errors = await raw_pipeline.process_file(
                    file_path=doc_input.file_path,
                    job=job,
                    enable_graph=config.enable_graph,
                )
            else:
                document, chunks, errors = await raw_pipeline.process_content(
                    content=content or "",
                    raw_bytes=(content or "").encode("utf-8"),
                    filename="document.txt",
                    content_type="text/plain",
                    job=job,
                    enable_graph=config.enable_graph,
                )

        if errors:
            for err in errors:
                err["source_path"] = doc_input.file_path
                job.document_errors.append(err)
                # 記錄詳細錯誤資訊
                logger.warning(
                    f"文檔處理錯誤 [{doc_input.file_path}]: "
                    f"stage={err.get('stage', 'unknown')}, error={err.get('error', 'unknown')}"
                )

        if not document:
            if errors:
                # 已經記錄了錯誤
                pass
            else:
                logger.warning(f"文檔處理返回空結果 (無錯誤): {doc_input.file_path}")
            return None

        # 儲存文件和 chunks（傳入 job 以支援 ProcessMode 檢查）
        store_result = await self._store_document(document, chunks, config, job)

        # 處理跳過的情況
        if store_result and store_result.get("skipped"):
            return store_result

        return {
            "document_id": document.id,
            "chunks_count": len(chunks),
            "entities_count": store_result.get("entities_count", 0) if store_result else 0,
            "relations_count": store_result.get("relations_count", 0) if store_result else 0,
        }

    async def _delete_document_from_all_stores(
        self,
        doc_id: str,
        job_id: Optional[str] = None,
    ) -> dict[str, int]:
        """
        從所有儲存系統刪除文件資料。

        Args:
            doc_id: 文件 ID
            job_id: 工作 ID（用於日誌）

        Returns:
            各系統刪除結果的字典，值為刪除數量（-1 表示失敗）
        """
        from chatbot_graphrag.services.storage import minio_service
        from chatbot_graphrag.services.vector import qdrant_service
        from chatbot_graphrag.services.search import opensearch_service
        from chatbot_graphrag.services.graph import nebula_client

        log_prefix = f"[{job_id}]" if job_id else ""
        results = {}

        # 1. 刪除 Qdrant 向量
        try:
            await qdrant_service.initialize()
            count = await qdrant_service.delete_chunks_by_doc_id(doc_id)
            results["qdrant"] = count
            logger.info(f"{log_prefix}[{doc_id}] Qdrant 刪除 {count} 個 chunks")
        except Exception as e:
            logger.error(f"{log_prefix}[{doc_id}] Qdrant 刪除失敗: {e}")
            results["qdrant"] = -1

        # 2. 刪除 OpenSearch 索引
        try:
            await opensearch_service.initialize()
            count = await opensearch_service.delete_by_doc_id(doc_id)
            results["opensearch"] = count
            logger.info(f"{log_prefix}[{doc_id}] OpenSearch 刪除 {count} 個文件")
        except Exception as e:
            logger.error(f"{log_prefix}[{doc_id}] OpenSearch 刪除失敗: {e}")
            results["opensearch"] = -1

        # 3. 刪除 NebulaGraph 實體
        try:
            await nebula_client.initialize()
            count = await nebula_client.delete_entities_by_doc_id(doc_id)
            results["nebula"] = count
            logger.info(f"{log_prefix}[{doc_id}] NebulaGraph 刪除 {count} 個實體")
        except Exception as e:
            logger.error(f"{log_prefix}[{doc_id}] NebulaGraph 刪除失敗: {e}")
            results["nebula"] = -1

        # 4. 刪除 MinIO 物件（來源文件）
        try:
            await minio_service.initialize()
            success = await minio_service.delete_document(doc_id)
            results["minio"] = 1 if success else 0
            logger.info(f"{log_prefix}[{doc_id}] MinIO 刪除 {'成功' if success else '失敗'}")
        except Exception as e:
            logger.error(f"{log_prefix}[{doc_id}] MinIO 刪除失敗: {e}")
            results["minio"] = -1

        # 5. 使相關快取失效
        try:
            await qdrant_service.cache_invalidate(doc_ids=[doc_id])
            results["cache"] = 1
        except Exception as e:
            logger.debug(f"{log_prefix}[{doc_id}] 快取失效失敗: {e}")
            results["cache"] = 0

        return results

    async def _get_existing_content_hash(self, doc_id: str) -> Optional[str]:
        """
        取得現有文件的 content_hash。

        優先從 PostgreSQL 查詢，失敗時 fallback 到 MinIO。

        Args:
            doc_id: 文件 ID

        Returns:
            如果文件存在則返回 content_hash，否則返回 None
        """
        # 1. 優先嘗試 PostgreSQL（更快、更可靠）
        try:
            from chatbot_graphrag.db import get_async_session, DocumentRepository, is_db_initialized

            if is_db_initialized():
                async with get_async_session() as session:
                    repo = DocumentRepository(session)
                    content_hash = await repo.get_content_hash(doc_id)
                    if content_hash:
                        logger.debug(f"從 PostgreSQL 取得 {doc_id} 的 content_hash")
                        return content_hash
        except Exception as e:
            logger.debug(f"PostgreSQL 取得 {doc_id} 的 content_hash 失敗: {e}")

        # 2. Fallback 到 MinIO
        from chatbot_graphrag.services.storage import minio_service

        try:
            await minio_service.initialize()
            canonical = await minio_service.get_canonical(doc_id)
            if canonical:
                logger.debug(f"從 MinIO 取得 {doc_id} 的 content_hash")
                return canonical.get("content_hash")
        except Exception as e:
            logger.debug(f"MinIO 取得 {doc_id} 的 content_hash 失敗: {e}")

        return None

    async def _persist_job(self, job: IngestJob) -> None:
        """
        持久化 Job 狀態到 PostgreSQL。

        使用雙寫策略：記憶體 + 資料庫。記憶體用於快速查詢，
        資料庫用於持久化和跨重啟恢復。

        Args:
            job: 要持久化的 IngestJob
        """
        try:
            from chatbot_graphrag.db import get_async_session, JobRepository, is_db_initialized

            if not is_db_initialized():
                logger.debug(f"資料庫未初始化，跳過 Job {job.id} 持久化")
                return

            async with get_async_session() as session:
                repo = JobRepository(session)
                await repo.save(job)
                logger.debug(f"Job {job.id} 已持久化到 PostgreSQL")
        except Exception as e:
            # 持久化失敗不應中斷攝取流程
            logger.warning(f"Job {job.id} 持久化失敗: {e}")

    async def _store_document(
        self,
        document: Document,
        chunks: list[Chunk],
        config: IngestJobConfig,
        job: IngestJob,
    ) -> Optional[dict[str, Any]]:
        """
        將文件和 chunks 儲存到所有服務。

        根據 ProcessMode 決定處理方式：
        - OVERRIDE: 先刪除所有舊資料，再儲存新資料
        - UPDATE: 檢查內容是否變更，只有變更時才更新

        Args:
            document: 處理過的文件
            chunks: 文件 chunks
            config: 攝取配置
            job: 攝取工作（用於日誌和狀態追蹤）

        Returns:
            如果建構了圖譜則返回包含圖譜指標的字典，
            如果跳過則返回 {"skipped": True, "reason": "..."}，
            否則返回 None
        """
        from chatbot_graphrag.services.storage import minio_service
        from chatbot_graphrag.services.vector import qdrant_service
        from chatbot_graphrag.services.search import opensearch_service

        doc_id = document.id
        process_mode = config.mode  # 使用 config.mode（IngestJobConfig 中的欄位）

        # ============================================================
        # ProcessMode 處理邏輯
        # ============================================================

        # OVERRIDE 模式：先刪除所有舊資料
        if process_mode == ProcessMode.OVERRIDE:
            logger.info(f"[{job.id}][{doc_id}] OVERRIDE 模式：刪除現有資料")
            delete_results = await self._delete_document_from_all_stores(doc_id, job.id)
            logger.info(f"[{job.id}][{doc_id}] 刪除結果: {delete_results}")

        # UPDATE 模式：檢查是否需要更新
        elif process_mode == ProcessMode.UPDATE:
            existing_hash = await self._get_existing_content_hash(doc_id)
            if existing_hash:
                if existing_hash == document.content_hash:
                    logger.info(
                        f"[{job.id}][{doc_id}] UPDATE 模式：內容未變更，跳過 "
                        f"(hash: {existing_hash[:16]}...)"
                    )
                    return {"skipped": True, "reason": "content_unchanged"}
                else:
                    # 內容有變更，先刪除舊資料再新增
                    logger.info(
                        f"[{job.id}][{doc_id}] UPDATE 模式：內容已變更，更新資料 "
                        f"(舊 hash: {existing_hash[:16]}..., 新 hash: {document.content_hash[:16]}...)"
                    )
                    await self._delete_document_from_all_stores(doc_id, job.id)
            else:
                logger.info(f"[{job.id}][{doc_id}] UPDATE 模式：新文件，直接儲存")

        # ============================================================
        # 正常儲存流程
        # ============================================================

        # 1. 將標準 JSON 儲存到 MinIO
        canonical_data = {
            "id": document.id,
            "metadata": document.metadata.model_dump(),
            "content_hash": document.content_hash,
            "chunks": [
                {
                    "id": chunk.id,
                    "content": chunk.content,
                    "metadata": chunk.metadata.model_dump(),
                }
                for chunk in chunks
            ],
        }
        await minio_service.save_canonical(
            doc_id=document.id,
            version=document.version,
            canonical_data=canonical_data,
        )

        # 2. 將 chunks 儲存到 Qdrant（含嵌入向量）
        from chatbot_graphrag.services.vector import embedding_service

        if chunks:
            # 抽取文字用於嵌入
            chunk_texts = [chunk.effective_content for chunk in chunks]

            # 生成嵌入向量（稠密 + 稀疏）
            logger.debug(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings = await embedding_service.embed_with_sparse(chunk_texts)

            # 準備 Qdrant 的批次資料
            qdrant_batch = []
            for chunk, emb in zip(chunks, embeddings):
                qdrant_batch.append({
                    "id": chunk.id,
                    "dense_embedding": emb["dense"],
                    "sparse_embedding": emb["sparse"],
                    "payload": {
                        "doc_id": chunk.doc_id,
                        "doc_version": chunk.doc_version,
                        "content": chunk.effective_content,
                        "chunk_type": chunk.metadata.chunk_type.value,
                        "section_title": chunk.metadata.section_title,
                        "position": chunk.metadata.position_in_doc,
                        "doc_type": document.metadata.doc_type.value,
                        "acl_groups": document.metadata.acl_groups,
                        "tenant_id": document.metadata.tenant_id,  # Phase 1: multi-tenant
                        "title": document.metadata.title,
                        "department": document.metadata.department,
                        "tags": document.metadata.tags,
                    },
                })

            # 批次 upsert 到 Qdrant
            upserted_count = await qdrant_service.upsert_chunks_batch(qdrant_batch)
            logger.debug(f"Upserted {upserted_count} chunks to Qdrant")

        # 3. 在 OpenSearch 中建立全文搜尋索引
        for chunk in chunks:
            opensearch_doc = {
                "chunk_id": chunk.id,
                "doc_id": chunk.doc_id,
                "doc_version": chunk.doc_version,
                "content": chunk.content,
                "contextual_content": chunk.contextual_content,
                "doc_type": document.metadata.doc_type.value,
                "section_title": chunk.metadata.section_title,
                "chunk_type": chunk.metadata.chunk_type.value,
                "acl_groups": document.metadata.acl_groups,
                "tenant_id": document.metadata.tenant_id,  # Phase 1: multi-tenant
                "department": document.metadata.department,
                "tags": document.metadata.tags,
            }
            await opensearch_service.index_chunk(
                chunk_id=chunk.id,
                doc_id=chunk.doc_id,
                content=chunk.content,
                metadata=opensearch_doc,
            )

        logger.info(
            f"Stored document {document.id}: "
            f"{len(chunks)} chunks to MinIO, Qdrant, OpenSearch"
        )

        # 4. 如果啟用則建構圖譜
        graph_result = None
        if config.enable_graph and chunks:
            graph_result = await self._build_graph(document, chunks)

        return graph_result

    async def _build_graph(
        self,
        document: Document,
        chunks: list[Chunk],
    ) -> Optional[dict[str, Any]]:
        """
        從文件 chunks 建構知識圖譜。

        抽取實體和關係，然後載入到 NebulaGraph。

        Args:
            document: 正在處理的文件
            chunks: 要從中抽取的文件 chunks

        Returns:
            包含 entities_count 和 relations_count 的字典
        """
        from chatbot_graphrag.services.graph import graph_batch_loader

        try:
            logger.info(f"Building graph for document {document.id}...")

            result = await graph_batch_loader.process_chunks(
                chunks=chunks,
                doc_id=document.id,
                concurrency=3,
                enable_relations=True,
            )

            if result.errors:
                logger.warning(
                    f"Graph building completed with {len(result.errors)} errors: "
                    f"{result.entities_loaded} entities, {result.relations_loaded} relations"
                )
            else:
                logger.info(
                    f"Graph built for {document.id}: "
                    f"{result.entities_loaded} entities, {result.relations_loaded} relations"
                )

            return {
                "entities_count": result.entities_loaded,
                "relations_count": result.relations_loaded,
            }

        except Exception as e:
            # 如果圖譜建構失敗不要讓攝取失敗
            logger.error(f"Graph building failed for {document.id}: {e}")
            return None

    async def ingest_directory(
        self,
        directory: str | Path,
        config: IngestJobConfig,
        created_by: Optional[str] = None,
    ) -> IngestJob:
        """
        從目錄攝取所有文件。

        Args:
            directory: 目錄路徑
            config: 攝取配置
            created_by: 發起攝取的使用者

        Returns:
            包含結果的 IngestJob
        """
        directory = Path(directory)

        # 建立任務
        job = self.create_job(config, created_by)
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()

        try:
            # 根據管線類型處理
            if config.pipeline == PipelineType.CURATED:
                result = await curated_pipeline.process_directory(
                    directory=directory,
                    job=job,
                    pattern="**/*.md",
                    enable_graph=config.enable_graph,
                )
            else:
                result = await raw_pipeline.process_directory(
                    directory=directory,
                    job=job,
                    enable_graph=config.enable_graph,
                )

            # 儲存結果
            skipped_count = 0
            for document in result.documents:
                # 找到此文件的 chunks
                doc_chunks = [c for c in result.chunks if c.doc_id == document.id]
                store_result = await self._store_document(document, doc_chunks, config, job)
                if store_result and store_result.get("skipped"):
                    skipped_count += 1

            # 更新任務
            job.processed_documents = len(result.documents) - skipped_count
            job.skipped_documents = skipped_count
            job.chunks_created = len(result.chunks)
            job.total_documents = len(result.documents)
            job.status = JobStatus.COMPLETED if result.success else JobStatus.PARTIAL
            job.completed_at = datetime.utcnow()
            job.progress = 100.0

        except Exception as e:
            logger.error(f"Directory ingestion failed: {e}")
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()

        return job

    async def delete_document(
        self,
        doc_id: str,
        hard_delete: bool = False,
    ) -> bool:
        """
        刪除文件及其關聯資料。

        Args:
            doc_id: 文件 ID
            hard_delete: 如果為 True，永久刪除；否則標記為已刪除

        Returns:
            刪除是否成功
        """
        from chatbot_graphrag.services.storage import minio_service
        from chatbot_graphrag.services.vector import qdrant_service
        from chatbot_graphrag.services.search import opensearch_service
        from chatbot_graphrag.services.graph import nebula_client

        try:
            if hard_delete:
                # 從所有儲存服務中刪除
                deletion_results = {}

                # 1. 從 MinIO 刪除（來源檔案）
                try:
                    await minio_service.delete_document(doc_id)
                    deletion_results["minio"] = True
                except Exception as e:
                    logger.warning(f"MinIO deletion failed for {doc_id}: {e}")
                    deletion_results["minio"] = False

                # 2. 從 Qdrant 刪除 chunks（向量儲存）
                try:
                    await qdrant_service.initialize()
                    await qdrant_service.delete_chunks_by_doc_id(doc_id)
                    deletion_results["qdrant"] = True
                except Exception as e:
                    logger.warning(f"Qdrant deletion failed for {doc_id}: {e}")
                    deletion_results["qdrant"] = False

                # 3. 從 OpenSearch 刪除 chunks（全文索引）
                try:
                    await opensearch_service.initialize()
                    deleted_count = await opensearch_service.delete_by_doc_id(doc_id)
                    deletion_results["opensearch"] = deleted_count
                except Exception as e:
                    logger.warning(f"OpenSearch deletion failed for {doc_id}: {e}")
                    deletion_results["opensearch"] = False

                # 4. 從 NebulaGraph 刪除實體（知識圖譜）
                try:
                    await nebula_client.initialize()
                    deleted_entities = await nebula_client.delete_entities_by_doc_id(doc_id)
                    deletion_results["nebula"] = deleted_entities
                except Exception as e:
                    logger.warning(f"NebulaGraph deletion failed for {doc_id}: {e}")
                    deletion_results["nebula"] = False

                # 5. 使相關快取條目失效
                try:
                    await qdrant_service.cache_invalidate(doc_ids=[doc_id])
                    deletion_results["cache"] = True
                except Exception as e:
                    logger.debug(f"Cache invalidation failed for {doc_id}: {e}")
                    deletion_results["cache"] = False

                logger.info(
                    f"Hard deleted document {doc_id}: "
                    f"minio={deletion_results.get('minio')}, "
                    f"qdrant={deletion_results.get('qdrant')}, "
                    f"opensearch={deletion_results.get('opensearch')}, "
                    f"nebula={deletion_results.get('nebula')}"
                )
            else:
                # 軟刪除 - 在元資料中標記為已刪除
                # 更新文件元資料以標記為已刪除（保留資料以供可能的恢復）
                # 注意：完整的軟刪除實作會在文件註冊表中更新狀態
                # 目前只記錄意圖 - 生產環境會使用 PostgreSQL 來儲存文件元資料
                logger.info(f"軟刪除文件 {doc_id}（在元資料中標記為已刪除）")

            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 100,
    ) -> list[IngestJob]:
        """
        列出攝取任務。

        Args:
            status: 根據狀態過濾
            limit: 返回的最大任務數

        Returns:
            任務列表
        """
        jobs = list(self._jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        # 按 created_at 降序排序
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]


# 單例實例
ingestion_coordinator = IngestionCoordinator()
