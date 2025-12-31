"""
精選管線

處理精選 YAML frontmatter + Markdown 文件的管線。

處理階段：
1. 解析 YAML frontmatter
2. 驗證 schema
3. 抽取元資料
4. 分塊內容（類型特定）
5. 生成嵌入向量
6. 儲存到向量資料庫和圖譜
"""

import hashlib
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from chatbot_graphrag.core.constants import DocType, JobStatus, PipelineType
from chatbot_graphrag.models.pydantic.ingestion import (
    Chunk,
    Document,
    DocumentMetadata,
    IngestJob,
    PipelineResult,
    PipelineStageResult,
)
from chatbot_graphrag.services.ingestion.chunkers import get_chunker
from chatbot_graphrag.services.ingestion.schema_validator import (
    SchemaValidationError,
    schema_validator,
)

logger = logging.getLogger(__name__)


class CuratedPipeline:
    """
    處理帶有 YAML frontmatter 的精選 Markdown 文件的管線。

    處理階段：
    1. 解析 YAML frontmatter
    2. 驗證 schema
    3. 抽取元資料
    4. 分塊內容（類型特定）
    5. 生成嵌入向量
    6. 儲存到向量資料庫和圖譜
    """

    def __init__(self):
        """初始化精選管線。"""
        self._initialized = True

    async def process_file(
        self,
        file_path: str | Path,
        job: IngestJob,
        enable_graph: bool = True,
    ) -> tuple[Document | None, list[Chunk], list[dict[str, Any]]]:
        """
        處理單一精選 Markdown 檔案。

        Args:
            file_path: Markdown 檔案路徑
            job: 父攝取任務
            enable_graph: 是否抽取圖譜實體

        Returns:
            (Document, Chunks 列表, 錯誤列表) 元組
        """
        file_path = Path(file_path)
        errors: list[dict[str, Any]] = []

        if not file_path.exists():
            errors.append({
                "stage": "file_read",
                "error": f"File not found: {file_path}",
            })
            return None, [], errors

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            errors.append({
                "stage": "file_read",
                "error": str(e),
            })
            return None, [], errors

        return await self.process_content(
            content=content,
            source_path=str(file_path),
            job=job,
            enable_graph=enable_graph,
        )

    async def process_content(
        self,
        content: str,
        source_path: str | None = None,
        job: IngestJob | None = None,
        enable_graph: bool = True,
    ) -> tuple[Document | None, list[Chunk], list[dict[str, Any]]]:
        """
        處理精選 Markdown 內容。

        Args:
            content: 帶有 YAML frontmatter 的原始 Markdown 內容
            source_path: 原始檔案路徑（供參考）
            job: 父攝取任務
            enable_graph: 是否抽取圖譜實體

        Returns:
            (Document, Chunks 列表, 錯誤列表) 元組
        """
        errors: list[dict[str, Any]] = []
        stages: list[PipelineStageResult] = []

        # 階段 1：解析和驗證 YAML frontmatter
        start_time = datetime.utcnow()
        try:
            curated_schema, body_content = schema_validator.validate_document(
                content,
                strict=True,
            )
            stages.append(PipelineStageResult(
                stage_name="parse_frontmatter",
                success=True,
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            ))
        except SchemaValidationError as e:
            errors.append({
                "stage": "parse_frontmatter",
                "error": str(e),
                "validation_errors": e.errors,
            })
            stages.append(PipelineStageResult(
                stage_name="parse_frontmatter",
                success=False,
                error_message=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            ))
            return None, [], errors

        # 階段 2：轉換為 DocumentMetadata
        start_time = datetime.utcnow()
        try:
            metadata = schema_validator.to_document_metadata(curated_schema)
            stages.append(PipelineStageResult(
                stage_name="extract_metadata",
                success=True,
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            ))
        except Exception as e:
            errors.append({
                "stage": "extract_metadata",
                "error": str(e),
            })
            return None, [], errors

        # 階段 3：建立 Document
        start_time = datetime.utcnow()
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        doc_id = self._generate_doc_id(metadata, content_hash)

        document = Document(
            id=doc_id,
            metadata=metadata,
            content_hash=content_hash,
            raw_content=content,
            pipeline=PipelineType.CURATED,
            version=1,
        )

        stages.append(PipelineStageResult(
            stage_name="create_document",
            success=True,
            duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
        ))

        # 階段 4：分塊內容
        start_time = datetime.utcnow()
        try:
            chunker = get_chunker(metadata.doc_type)
            chunks = chunker.chunk(
                content=body_content,
                doc_id=doc_id,
                metadata=metadata,
                doc_version=1,
            )
            stages.append(PipelineStageResult(
                stage_name="chunk_content",
                success=True,
                output_count=len(chunks),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            ))
        except Exception as e:
            errors.append({
                "stage": "chunk_content",
                "error": str(e),
            })
            return document, [], errors

        logger.info(
            f"Processed document {doc_id}: {len(chunks)} chunks, "
            f"type={metadata.doc_type.value}"
        )

        return document, chunks, errors

    async def process_directory(
        self,
        directory: str | Path,
        job: IngestJob,
        pattern: str = "**/*.md",
        enable_graph: bool = True,
    ) -> PipelineResult:
        """
        處理目錄中的所有 Markdown 檔案。

        Args:
            directory: 目錄路徑
            job: 父攝取任務
            pattern: 檔案的 glob 模式
            enable_graph: 是否抽取圖譜實體

        Returns:
            包含所有處理過的文件和 chunks 的 PipelineResult
        """
        directory = Path(directory)
        started_at = datetime.utcnow()

        all_documents: list[Document] = []
        all_chunks: list[Chunk] = []
        all_errors: list[dict[str, Any]] = []
        stages: list[PipelineStageResult] = []

        # 找到所有匹配的檔案
        files = list(directory.glob(pattern))
        logger.info(f"Found {len(files)} files matching pattern '{pattern}'")

        for file_path in files:
            doc, chunks, errors = await self.process_file(
                file_path=file_path,
                job=job,
                enable_graph=enable_graph,
            )

            if doc:
                all_documents.append(doc)
                all_chunks.extend(chunks)

            if errors:
                for err in errors:
                    err["file"] = str(file_path)
                all_errors.extend(errors)

        stages.append(PipelineStageResult(
            stage_name="process_files",
            success=len(all_errors) == 0,
            input_count=len(files),
            output_count=len(all_documents),
            duration_ms=(datetime.utcnow() - started_at).total_seconds() * 1000,
        ))

        return PipelineResult(
            job_id=job.id,
            pipeline=PipelineType.CURATED,
            success=len(all_errors) == 0,
            stages=stages,
            documents=all_documents,
            chunks=all_chunks,
            total_duration_ms=(datetime.utcnow() - started_at).total_seconds() * 1000,
            started_at=started_at,
            completed_at=datetime.utcnow(),
        )

    def _generate_doc_id(
        self,
        metadata: DocumentMetadata,
        content_hash: str,
    ) -> str:
        """生成確定性的文件 ID。"""
        # 使用 doc_type + title + content hash 前綴作為 ID
        title_slug = metadata.title.lower().replace(" ", "_")[:20]
        hash_prefix = content_hash[:8]
        return f"d_{metadata.doc_type.value}_{title_slug}_{hash_prefix}"


# 單例實例
curated_pipeline = CuratedPipeline()
