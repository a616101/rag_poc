"""
通用分塊器

未指定文件類型的預設分塊器。

使用基於章節和大小的分割策略。
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


class GenericChunker(BaseChunker):
    """
    未指定文件類型的通用分塊器。

    使用基於章節和大小的分割策略。
    """

    SUPPORTED_DOC_TYPES = [DocType.GENERIC, DocType.FAQ]

    def chunk(
        self,
        content: str,
        doc_id: str,
        metadata: DocumentMetadata,
        doc_version: int = 1,
    ) -> list[Chunk]:
        """
        使用通用策略分塊文件內容。

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

        # 2. 檢查內容是否有章節
        sections = self.split_by_sections(content)

        if sections:
            # 按章節處理
            for section in sections:
                section_chunks = self._chunk_section(
                    section,
                    doc_id,
                    doc_version,
                    position,
                )
                chunks.extend(section_chunks)
                position += len(section_chunks)
        else:
            # 沒有章節，按大小分塊
            text_chunks = self.split_by_size(content)
            for i, text in enumerate(text_chunks):
                chunk_type = self.infer_chunk_type(text, None)
                chunk = self.create_chunk(
                    content=text,
                    doc_id=doc_id,
                    doc_version=doc_version,
                    position=position + i,
                    chunk_type=chunk_type,
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
        level = section.get("level", 0)

        if not content:
            return chunks

        # 首先抽取表格
        tables = self.extract_tables(content)
        for table in tables:
            content = content.replace(table, "")  # Remove table from content

        # 抽取列表
        lists = self.extract_lists(content)

        # 根據內容結構判斷 chunk 類型
        chunk_type = self.infer_chunk_type(content, title)

        # 檢查內容是否需要分割
        if len(content) <= self.chunk_size:
            if content.strip():
                chunk = self.create_chunk(
                    content=content.strip(),
                    doc_id=doc_id,
                    doc_version=doc_version,
                    position=start_position,
                    chunk_type=chunk_type,
                    section_title=title,
                    metadata_extras={"section_level": level},
                )
                chunks.append(chunk)
                start_position += 1
        else:
            # 先按段落分割，如需要再按大小分割
            paragraphs = self.split_by_paragraphs(content)

            current_chunk_content: list[str] = []
            current_size = 0

            for para in paragraphs:
                if current_size + len(para) <= self.chunk_size:
                    current_chunk_content.append(para)
                    current_size += len(para) + 2  # +2 for newlines
                else:
                    # 清空當前 chunk
                    if current_chunk_content:
                        combined = "\n\n".join(current_chunk_content)
                        chunk = self.create_chunk(
                            content=combined,
                            doc_id=doc_id,
                            doc_version=doc_version,
                            position=start_position,
                            chunk_type=chunk_type,
                            section_title=title,
                        )
                        chunks.append(chunk)
                        start_position += 1

                    # 處理大段落
                    if len(para) > self.chunk_size:
                        sub_chunks = self.split_by_size(para)
                        for sub in sub_chunks:
                            chunk = self.create_chunk(
                                content=sub,
                                doc_id=doc_id,
                                doc_version=doc_version,
                                position=start_position,
                                chunk_type=chunk_type,
                                section_title=title,
                            )
                            chunks.append(chunk)
                            start_position += 1
                        current_chunk_content = []
                        current_size = 0
                    else:
                        current_chunk_content = [para]
                        current_size = len(para)

            # 清空剩餘內容
            if current_chunk_content:
                combined = "\n\n".join(current_chunk_content)
                chunk = self.create_chunk(
                    content=combined,
                    doc_id=doc_id,
                    doc_version=doc_version,
                    position=start_position,
                    chunk_type=chunk_type,
                    section_title=title,
                )
                chunks.append(chunk)
                start_position += 1

        # 添加表格 chunks
        for i, table in enumerate(tables):
            table_chunk = self.create_chunk(
                content=table,
                doc_id=doc_id,
                doc_version=doc_version,
                position=start_position + i,
                chunk_type=ChunkType.TABLE,
                section_title=f"{title} - 表格" if title else "表格",
            )
            chunks.append(table_chunk)

        return chunks


# 單例實例
generic_chunker = GenericChunker()
