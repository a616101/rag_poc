"""
Job Repository - PostgreSQL CRUD 操作

提供 IngestJob 持久化的封裝層，支援記憶體快取和資料庫儲存雙寫。
"""

from datetime import datetime
from typing import Any, Optional, Sequence

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from chatbot_graphrag.core.constants import (
    JobStatus,
    PipelineType,
    ProcessMode,
)
from chatbot_graphrag.models.pydantic.ingestion import IngestJob, IngestJobConfig
from chatbot_graphrag.models.sqlalchemy.chunk import IngestJob as IngestJobRecord


class JobRepository:
    """
    Job Repository。

    封裝 IngestJob 的資料庫操作，提供 Pydantic ↔ SQLAlchemy 轉換。

    Example:
        async with get_async_session() as session:
            repo = JobRepository(session)
            await repo.save(job)
            job = await repo.get_by_id("job_123")
    """

    def __init__(self, session: AsyncSession):
        """
        初始化 Repository。

        Args:
            session: SQLAlchemy AsyncSession
        """
        self.session = session

    async def save(self, job: IngestJob) -> None:
        """
        儲存或更新 Job（Upsert）。

        使用 PostgreSQL ON CONFLICT 實現原子性的 upsert 操作。

        Args:
            job: Pydantic IngestJob 實例
        """
        stmt = insert(IngestJobRecord).values(
            id=job.id,
            pipeline_type=job.config.pipeline.value,
            process_mode=job.config.mode.value,
            source_path=job.config.source_path,
            source_urls=job.config.source_urls or [],
            collection_name=job.config.collection_name,
            config=job.config.model_dump(),
            status=job.status.value,
            progress=job.progress,
            total_documents=job.total_documents,
            processed_documents=job.processed_documents,
            failed_documents=job.failed_documents,
            skipped_documents=job.skipped_documents,
            chunks_created=job.chunks_created,
            entities_extracted=job.entities_extracted,
            relations_extracted=job.relations_extracted,
            communities_detected=job.communities_detected,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            document_errors=job.document_errors,
            created_by=job.created_by,
        ).on_conflict_do_update(
            index_elements=["id"],
            set_={
                "status": job.status.value,
                "progress": job.progress,
                "processed_documents": job.processed_documents,
                "failed_documents": job.failed_documents,
                "skipped_documents": job.skipped_documents,
                "chunks_created": job.chunks_created,
                "entities_extracted": job.entities_extracted,
                "relations_extracted": job.relations_extracted,
                "communities_detected": job.communities_detected,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "error_message": job.error_message,
                "document_errors": job.document_errors,
            },
        )
        await self.session.execute(stmt)

    async def get_by_id(self, job_id: str) -> Optional[IngestJob]:
        """
        根據 ID 取得 Job。

        Args:
            job_id: 工作 ID

        Returns:
            Pydantic IngestJob 實例或 None
        """
        result = await self.session.execute(
            select(IngestJobRecord).where(IngestJobRecord.id == job_id)
        )
        record = result.scalar_one_or_none()
        if record:
            return self._to_pydantic(record)
        return None

    async def list_by_status(
        self,
        status: JobStatus,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[IngestJob]:
        """
        根據狀態列出 Jobs。

        Args:
            status: 工作狀態
            limit: 最大返回數量
            offset: 起始偏移

        Returns:
            Pydantic IngestJob 列表
        """
        result = await self.session.execute(
            select(IngestJobRecord)
            .where(IngestJobRecord.status == status.value)
            .order_by(IngestJobRecord.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        records = result.scalars().all()
        return [self._to_pydantic(r) for r in records]

    async def list_recent(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[IngestJob]:
        """
        列出最近的 Jobs。

        Args:
            limit: 最大返回數量
            offset: 起始偏移

        Returns:
            Pydantic IngestJob 列表
        """
        result = await self.session.execute(
            select(IngestJobRecord)
            .order_by(IngestJobRecord.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        records = result.scalars().all()
        return [self._to_pydantic(r) for r in records]

    async def update_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: Optional[str] = None,
        completed_at: Optional[datetime] = None,
    ) -> bool:
        """
        更新 Job 狀態。

        Args:
            job_id: 工作 ID
            status: 新狀態
            error_message: 錯誤訊息（可選）
            completed_at: 完成時間（可選）

        Returns:
            True 如果更新成功
        """
        values: dict[str, Any] = {"status": status.value}
        if error_message is not None:
            values["error_message"] = error_message
        if completed_at is not None:
            values["completed_at"] = completed_at

        result = await self.session.execute(
            update(IngestJobRecord)
            .where(IngestJobRecord.id == job_id)
            .values(**values)
        )
        return result.rowcount > 0

    async def update_progress(
        self,
        job_id: str,
        progress: float,
        processed_documents: Optional[int] = None,
        failed_documents: Optional[int] = None,
        skipped_documents: Optional[int] = None,
        chunks_created: Optional[int] = None,
    ) -> bool:
        """
        更新 Job 進度。

        Args:
            job_id: 工作 ID
            progress: 進度百分比
            processed_documents: 已處理文件數（可選）
            failed_documents: 失敗文件數（可選）
            skipped_documents: 跳過文件數（可選）
            chunks_created: 建立的 chunk 數（可選）

        Returns:
            True 如果更新成功
        """
        values: dict[str, Any] = {"progress": progress}
        if processed_documents is not None:
            values["processed_documents"] = processed_documents
        if failed_documents is not None:
            values["failed_documents"] = failed_documents
        if skipped_documents is not None:
            values["skipped_documents"] = skipped_documents
        if chunks_created is not None:
            values["chunks_created"] = chunks_created

        result = await self.session.execute(
            update(IngestJobRecord)
            .where(IngestJobRecord.id == job_id)
            .values(**values)
        )
        return result.rowcount > 0

    async def delete(self, job_id: str) -> bool:
        """
        刪除 Job。

        Args:
            job_id: 工作 ID

        Returns:
            True 如果刪除成功
        """
        result = await self.session.execute(
            update(IngestJobRecord)
            .where(IngestJobRecord.id == job_id)
            .values(status="deleted")
        )
        return result.rowcount > 0

    def _to_pydantic(self, record: IngestJobRecord) -> IngestJob:
        """
        將 SQLAlchemy 記錄轉換為 Pydantic 模型。

        Args:
            record: SQLAlchemy IngestJobRecord 實例

        Returns:
            Pydantic IngestJob 實例
        """
        # 從 config JSON 重建 IngestJobConfig
        config_data = record.config or {}
        config = IngestJobConfig(
            pipeline=PipelineType(config_data.get("pipeline", record.pipeline_type)),
            mode=ProcessMode(config_data.get("mode", record.process_mode)),
            source_path=config_data.get("source_path", record.source_path),
            source_urls=config_data.get("source_urls", record.source_urls),
            collection_name=config_data.get("collection_name", record.collection_name),
            enable_graph=config_data.get("enable_graph", True),
            enable_community_detection=config_data.get("enable_community_detection", True),
            doc_type_filter=config_data.get("doc_type_filter"),
            chunk_size=config_data.get("chunk_size", 500),
            chunk_overlap=config_data.get("chunk_overlap", 50),
            contextual_chunking=config_data.get("contextual_chunking", True),
        )

        return IngestJob(
            id=record.id,
            config=config,
            status=JobStatus(record.status),
            progress=record.progress,
            total_documents=record.total_documents,
            processed_documents=record.processed_documents,
            failed_documents=record.failed_documents,
            skipped_documents=record.skipped_documents,
            chunks_created=record.chunks_created,
            entities_extracted=record.entities_extracted,
            relations_extracted=record.relations_extracted,
            communities_detected=record.communities_detected,
            created_at=record.created_at,
            started_at=record.started_at,
            completed_at=record.completed_at,
            error_message=record.error_message,
            document_errors=record.document_errors or [],
            created_by=record.created_by,
        )
