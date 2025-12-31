"""
流程分塊器

專門用於流程文件（醫療流程、程序）的分塊器。
"""

import logging
from typing import Any

from chatbot_graphrag.core.constants import ChunkType, DocType
from chatbot_graphrag.models.pydantic.ingestion import (
    Chunk,
    DocumentMetadata,
)
from chatbot_graphrag.services.ingestion.chunkers.base import BaseChunker

logger = logging.getLogger(__name__)


class ProcedureChunker(BaseChunker):
    """
    流程文件的分塊器。

    處理結構化流程內容，包含：
    - 步驟/流程
    - 需求/條件
    - 費用
    - 表單
    - 時程
    """

    SUPPORTED_DOC_TYPES = [DocType.PROCEDURE]

    # 流程文件的章節關鍵字
    STEPS_KEYWORDS = ["步驟", "流程", "step", "procedure", "process"]
    REQUIREMENTS_KEYWORDS = ["需求", "條件", "資格", "requirements", "prerequisites"]
    FEES_KEYWORDS = ["費用", "收費", "價格", "fees", "cost", "price"]
    FORMS_KEYWORDS = ["表單", "表格", "申請書", "forms", "documents"]
    TIMELINE_KEYWORDS = ["時間", "時程", "期限", "timeline", "duration"]
    NOTES_KEYWORDS = ["注意", "備註", "說明", "notes", "remarks"]

    def chunk(
        self,
        content: str,
        doc_id: str,
        metadata: DocumentMetadata,
        doc_version: int = 1,
    ) -> list[Chunk]:
        """
        分塊流程文件內容。

        Args:
            content: 文件主體內容
            doc_id: 父文件 ID
            metadata: 文件元資料
            doc_version: 文件版本號

        Returns:
            Chunk 物件列表
        """
        chunks: list[Chunk] = []
        position = 0

        # 1. 建立元資料 chunk
        meta_chunk = self.create_metadata_chunk(doc_id, doc_version, metadata)
        chunks.append(meta_chunk)
        position += 1

        # 2. 從自訂欄位中抽取結構化章節
        custom = metadata.custom_fields

        # 處理元資料中的步驟
        if "steps" in custom:
            steps_chunks = self._chunk_steps(
                custom["steps"],
                doc_id,
                doc_version,
                position,
            )
            chunks.extend(steps_chunks)
            position += len(steps_chunks)

        # 處理元資料中的條件
        if "requirements" in custom:
            req_chunks = self._chunk_list_field(
                custom["requirements"],
                "申請條件",
                ChunkType.REQUIREMENTS,
                doc_id,
                doc_version,
                position,
            )
            chunks.extend(req_chunks)
            position += len(req_chunks)

        # 處理元資料中的費用
        if "fees" in custom:
            fee_chunks = self._chunk_fees(
                custom["fees"],
                doc_id,
                doc_version,
                position,
            )
            chunks.extend(fee_chunks)
            position += len(fee_chunks)

        # 處理元資料中的表單
        if "forms" in custom:
            form_chunks = self._chunk_list_field(
                custom["forms"],
                "相關表單",
                ChunkType.FORMS,
                doc_id,
                doc_version,
                position,
            )
            chunks.extend(form_chunks)
            position += len(form_chunks)

        # 3. 按章節處理 Markdown 主體內容
        sections = self.split_by_sections(content)

        for section in sections:
            section_chunks = self._chunk_section(
                section,
                doc_id,
                doc_version,
                position,
            )
            chunks.extend(section_chunks)
            position += len(section_chunks)

        return chunks

    def _chunk_steps(
        self,
        steps: list[dict[str, Any]],
        doc_id: str,
        doc_version: int,
        start_position: int,
    ) -> list[Chunk]:
        """分塊流程步驟。"""
        chunks: list[Chunk] = []
        position = start_position

        # 建立整體步驟 chunk
        steps_content = ["## 申請步驟", ""]

        for i, step in enumerate(steps, 1):
            step_num = step.get("step", i)
            title = step.get("title", f"步驟 {i}")
            description = step.get("description", "")
            notes = step.get("notes", "")

            steps_content.append(f"### 步驟 {step_num}: {title}")
            if description:
                steps_content.append(description)
            if notes:
                steps_content.append(f"**注意**: {notes}")
            steps_content.append("")

        full_content = "\n".join(steps_content)

        # 檢查是否需要分割
        if len(full_content) <= self.chunk_size:
            chunk = self.create_chunk(
                content=full_content,
                doc_id=doc_id,
                doc_version=doc_version,
                position=position,
                chunk_type=ChunkType.STEPS,
                section_title="申請步驟",
            )
            chunks.append(chunk)
        else:
            # 建立各個步驟 chunks
            for i, step in enumerate(steps, 1):
                step_num = step.get("step", i)
                title = step.get("title", f"步驟 {i}")
                description = step.get("description", "")
                notes = step.get("notes", "")

                step_content = [f"### 步驟 {step_num}: {title}"]
                if description:
                    step_content.append(description)
                if notes:
                    step_content.append(f"**注意**: {notes}")

                chunk = self.create_chunk(
                    content="\n".join(step_content),
                    doc_id=doc_id,
                    doc_version=doc_version,
                    position=position + i - 1,
                    chunk_type=ChunkType.STEPS,
                    section_title=f"步驟 {step_num}",
                    metadata_extras={"step_number": step_num},
                )
                chunks.append(chunk)

        return chunks

    def _chunk_fees(
        self,
        fees: dict[str, Any],
        doc_id: str,
        doc_version: int,
        position: int,
    ) -> list[Chunk]:
        """分塊費用資訊。"""
        chunks: list[Chunk] = []

        lines = ["## 費用說明", ""]

        # 如果是結構化資料則建立費用表格
        if isinstance(fees, dict):
            if "items" in fees:
                lines.append("| 項目 | 費用 |")
                lines.append("|------|------|")
                for item in fees["items"]:
                    name = item.get("name", "")
                    amount = item.get("amount", "")
                    lines.append(f"| {name} | {amount} |")
                lines.append("")

            if "total" in fees:
                lines.append(f"**總計**: {fees['total']}")

            if "notes" in fees:
                lines.append(f"\n**備註**: {fees['notes']}")

        else:
            lines.append(str(fees))

        content = "\n".join(lines)
        chunk = self.create_chunk(
            content=content,
            doc_id=doc_id,
            doc_version=doc_version,
            position=position,
            chunk_type=ChunkType.FEES,
            section_title="費用說明",
        )
        chunks.append(chunk)

        return chunks

    def _chunk_list_field(
        self,
        items: list[str] | list[dict],
        title: str,
        chunk_type: ChunkType,
        doc_id: str,
        doc_version: int,
        position: int,
    ) -> list[Chunk]:
        """分塊列表類型欄位。"""
        chunks: list[Chunk] = []

        lines = [f"## {title}", ""]

        for item in items:
            if isinstance(item, dict):
                item_str = item.get("name", item.get("title", str(item)))
            else:
                item_str = str(item)
            lines.append(f"- {item_str}")

        content = "\n".join(lines)
        chunk = self.create_chunk(
            content=content,
            doc_id=doc_id,
            doc_version=doc_version,
            position=position,
            chunk_type=chunk_type,
            section_title=title,
        )
        chunks.append(chunk)

        return chunks

    def _chunk_section(
        self,
        section: dict[str, Any],
        doc_id: str,
        doc_version: int,
        start_position: int,
    ) -> list[Chunk]:
        """分塊一個內容章節。"""
        chunks: list[Chunk] = []
        title = section.get("title")
        content = section.get("content", "")

        if not content:
            return chunks

        # 從標題偵測章節類型
        chunk_type = self._detect_section_type(title)

        # 檢查內容是否需要分割
        if len(content) <= self.chunk_size:
            chunk = self.create_chunk(
                content=content,
                doc_id=doc_id,
                doc_version=doc_version,
                position=start_position,
                chunk_type=chunk_type,
                section_title=title,
            )
            chunks.append(chunk)
        else:
            # 按段落或大小分割
            sub_chunks = self.split_by_size(content)
            for i, sub_content in enumerate(sub_chunks):
                chunk = self.create_chunk(
                    content=sub_content,
                    doc_id=doc_id,
                    doc_version=doc_version,
                    position=start_position + i,
                    chunk_type=chunk_type,
                    section_title=title,
                )
                chunks.append(chunk)

        return chunks

    def _detect_section_type(self, title: str | None) -> ChunkType:
        """從章節標題偵測 chunk 類型。"""
        if not title:
            return ChunkType.PARAGRAPH

        title_lower = title.lower()

        if any(kw in title_lower for kw in self.STEPS_KEYWORDS):
            return ChunkType.STEPS
        if any(kw in title_lower for kw in self.REQUIREMENTS_KEYWORDS):
            return ChunkType.REQUIREMENTS
        if any(kw in title_lower for kw in self.FEES_KEYWORDS):
            return ChunkType.FEES
        if any(kw in title_lower for kw in self.FORMS_KEYWORDS):
            return ChunkType.FORMS
        if any(kw in title_lower for kw in self.TIMELINE_KEYWORDS):
            return ChunkType.TIMELINE
        if any(kw in title_lower for kw in self.NOTES_KEYWORDS):
            return ChunkType.PARAGRAPH

        return ChunkType.PARAGRAPH


# 單例實例
procedure_chunker = ProcedureChunker()
