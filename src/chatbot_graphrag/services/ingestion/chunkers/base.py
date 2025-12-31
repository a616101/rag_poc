"""
基礎分塊器

文件分塊器的抽象基礎類，提供共用功能。

主要功能：
- 定義分塊器介面
- 提供 Markdown 解析工具
- 提供段落和大小分割方法
- 提供 chunk ID 生成
"""

import hashlib
import logging
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any

from chatbot_graphrag.core.constants import ChunkType, DocType
from chatbot_graphrag.models.pydantic.ingestion import (
    Chunk,
    ChunkMetadata,
    DocumentMetadata,
)

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """
    文件分塊器的抽象基礎類。

    提供共用的分塊工具方法並定義
    類型特定分塊器實作的介面。
    """

    # 此分塊器處理的文件類型
    SUPPORTED_DOC_TYPES: list[DocType] = []

    # 預設分塊參數
    DEFAULT_CHUNK_SIZE = 500   # 預設 chunk 大小（字元數）
    DEFAULT_CHUNK_OVERLAP = 50  # 預設重疊大小
    MIN_CHUNK_SIZE = 100        # 最小 chunk 大小
    MAX_CHUNK_SIZE = 2000       # 最大 chunk 大小

    # Markdown 模式
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)  # 標題模式
    TABLE_PATTERN = re.compile(r"^\|.+\|$", re.MULTILINE)             # 表格模式
    LIST_PATTERN = re.compile(r"^[\s]*[-*+]\s+.+$", re.MULTILINE)     # 無序列表模式
    NUMBERED_LIST_PATTERN = re.compile(r"^[\s]*\d+\.\s+.+$", re.MULTILINE)  # 有序列表模式

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """
        初始化分塊器。

        Args:
            chunk_size: 目標 chunk 大小（字元數）
            chunk_overlap: chunks 之間的重疊字元數
        """
        self.chunk_size = max(self.MIN_CHUNK_SIZE, min(chunk_size, self.MAX_CHUNK_SIZE))
        self.chunk_overlap = min(chunk_overlap, chunk_size // 2)

    @abstractmethod
    def chunk(
        self,
        content: str,
        doc_id: str,
        metadata: DocumentMetadata,
        doc_version: int = 1,
    ) -> list[Chunk]:
        """
        將文件內容分割成結構化的 chunks。

        Args:
            content: 文件主體內容（不含 frontmatter）
            doc_id: 父文件 ID
            metadata: 文件元資料
            doc_version: 文件版本號

        Returns:
            Chunk 物件列表
        """
        pass

    def generate_chunk_id(self, doc_id: str, position: int, content: str) -> str:
        """
        生成確定性的 chunk ID。

        Args:
            doc_id: 父文件 ID
            position: 在文件中的位置索引
            content: Chunk 內容

        Returns:
            唯一的 chunk ID
        """
        # 基於內容雜湊建立確定性 ID
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"c_{doc_id[:8]}_{position:04d}_{content_hash}"

    def create_chunk(
        self,
        content: str,
        doc_id: str,
        doc_version: int,
        position: int,
        chunk_type: ChunkType,
        section_title: str | None = None,
        parent_chunk_id: str | None = None,
        metadata_extras: dict[str, Any] | None = None,
    ) -> Chunk:
        """
        建立具有適當元資料的 Chunk 物件。

        Args:
            content: Chunk 文字內容
            doc_id: 父文件 ID
            doc_version: 文件版本
            position: 在文件中的位置
            chunk_type: Chunk 類型
            section_title: 章節標題（如適用）
            parent_chunk_id: 父 chunk ID（用於階層式 chunks）
            metadata_extras: 額外的元資料欄位

        Returns:
            Chunk 物件
        """
        chunk_id = self.generate_chunk_id(doc_id, position, content)

        metadata = ChunkMetadata(
            chunk_type=chunk_type,
            section_title=section_title,
            position_in_doc=position,
            parent_chunk_id=parent_chunk_id,
            custom_fields=metadata_extras or {},
        )

        return Chunk(
            id=chunk_id,
            doc_id=doc_id,
            doc_version=doc_version,
            content=content.strip(),
            metadata=metadata,
            char_count=len(content),
        )

    def split_by_sections(self, content: str) -> list[dict[str, Any]]:
        """
        根據 Markdown 標題將內容分割成章節。

        Args:
            content: Markdown 內容

        Returns:
            包含標題、層級和內容的章節列表
        """
        sections: list[dict[str, Any]] = []
        lines = content.split("\n")
        current_section: dict[str, Any] | None = None
        current_content: list[str] = []

        for line in lines:
            match = self.HEADER_PATTERN.match(line)

            if match:
                # 儲存前一個章節
                if current_section is not None:
                    current_section["content"] = "\n".join(current_content).strip()
                    if current_section["content"]:
                        sections.append(current_section)

                # 開始新章節
                level = len(match.group(1))
                title = match.group(2).strip()
                current_section = {
                    "title": title,
                    "level": level,
                    "content": "",
                }
                current_content = []
            else:
                current_content.append(line)

        # 別忘了最後一個章節
        if current_section is not None:
            current_section["content"] = "\n".join(current_content).strip()
            if current_section["content"]:
                sections.append(current_section)
        elif current_content:
            # 沒有標題的內容
            sections.append({
                "title": None,
                "level": 0,
                "content": "\n".join(current_content).strip(),
            })

        return sections

    def split_by_paragraphs(self, content: str) -> list[str]:
        """
        將內容分割成段落。

        Args:
            content: 文字內容

        Returns:
            段落字串列表
        """
        # 根據雙換行分割
        paragraphs = re.split(r"\n\s*\n", content)
        return [p.strip() for p in paragraphs if p.strip()]

    def split_by_size(
        self,
        text: str,
        max_size: int | None = None,
        overlap: int | None = None,
    ) -> list[str]:
        """
        將文字分割成指定大小的 chunks，帶有重疊。

        Args:
            text: 要分割的文字
            max_size: 最大 chunk 大小（預設為 self.chunk_size）
            overlap: chunks 之間的重疊（預設為 self.chunk_overlap）

        Returns:
            文字 chunks 列表
        """
        max_size = max_size or self.chunk_size
        overlap = overlap or self.chunk_overlap

        if len(text) <= max_size:
            return [text] if text.strip() else []

        chunks: list[str] = []
        start = 0

        while start < len(text):
            # 找到結束位置
            end = start + max_size

            # 嘗試在句子邊界處斷開
            if end < len(text):
                # 在 chunk 最後 20% 範圍內尋找句子結尾
                search_start = end - int(max_size * 0.2)
                search_text = text[search_start:end]

                # 找到最後一個句子邊界
                for pattern in [r"。", r"\. ", r"！", r"？", r"\n"]:
                    matches = list(re.finditer(pattern, search_text))
                    if matches:
                        last_match = matches[-1]
                        end = search_start + last_match.end()
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # 帶重疊移動起始位置
            start = end - overlap

        return chunks

    def extract_tables(self, content: str) -> list[str]:
        """
        從內容中抽取 Markdown 表格。

        Args:
            content: Markdown 內容

        Returns:
            表格字串列表
        """
        tables: list[str] = []
        lines = content.split("\n")
        current_table: list[str] = []
        in_table = False

        for line in lines:
            if self.TABLE_PATTERN.match(line):
                in_table = True
                current_table.append(line)
            elif in_table:
                if line.strip().startswith("|") or line.strip() == "":
                    if line.strip():
                        current_table.append(line)
                else:
                    if current_table:
                        tables.append("\n".join(current_table))
                    current_table = []
                    in_table = False

        if current_table:
            tables.append("\n".join(current_table))

        return tables

    def extract_lists(self, content: str) -> list[str]:
        """
        從內容中抽取列表。

        Args:
            content: Markdown 內容

        Returns:
            列表字串列表
        """
        lists: list[str] = []
        lines = content.split("\n")
        current_list: list[str] = []
        in_list = False

        for line in lines:
            is_list_item = (
                self.LIST_PATTERN.match(line) or self.NUMBERED_LIST_PATTERN.match(line)
            )

            if is_list_item:
                in_list = True
                current_list.append(line)
            elif in_list:
                # 列表項目的延續（縮排）
                if line.startswith("  ") and line.strip():
                    current_list.append(line)
                else:
                    if current_list:
                        lists.append("\n".join(current_list))
                    current_list = []
                    in_list = False

        if current_list:
            lists.append("\n".join(current_list))

        return lists

    def create_metadata_chunk(
        self,
        doc_id: str,
        doc_version: int,
        metadata: DocumentMetadata,
    ) -> Chunk:
        """
        建立元資料摘要 chunk。

        Args:
            doc_id: 文件 ID
            doc_version: 文件版本
            metadata: 文件元資料

        Returns:
            元資料 chunk
        """
        # 建構元資料摘要
        lines = [
            f"# {metadata.title}",
            "",
        ]

        if metadata.description:
            lines.append(f"**說明**: {metadata.description}")
        if metadata.department:
            lines.append(f"**部門**: {metadata.department}")
        if metadata.tags:
            lines.append(f"**標籤**: {', '.join(metadata.tags)}")
        if metadata.effective_date:
            lines.append(f"**生效日期**: {metadata.effective_date.isoformat()}")

        content = "\n".join(lines)

        return self.create_chunk(
            content=content,
            doc_id=doc_id,
            doc_version=doc_version,
            position=0,
            chunk_type=ChunkType.METADATA,
            section_title=metadata.title,
        )

    def infer_chunk_type(self, content: str, section_title: str | None = None) -> ChunkType:
        """
        從內容和章節標題推斷 chunk 類型。

        Args:
            content: Chunk 內容
            section_title: 章節標題（如有）

        Returns:
            推斷的 ChunkType
        """
        content_lower = content.lower()
        title_lower = (section_title or "").lower()

        # 檢查表格
        if self.TABLE_PATTERN.search(content):
            return ChunkType.TABLE

        # 檢查列表
        if self.LIST_PATTERN.search(content) or self.NUMBERED_LIST_PATTERN.search(content):
            return ChunkType.LIST

        # 檢查章節標題關鍵字
        if any(kw in title_lower for kw in ["步驟", "流程", "step", "procedure"]):
            return ChunkType.STEPS

        if any(kw in title_lower for kw in ["費用", "收費", "price", "fee"]):
            return ChunkType.FEES

        if any(kw in title_lower for kw in ["時間", "時程", "schedule", "timeline"]):
            return ChunkType.TIMELINE

        if any(kw in title_lower for kw in ["表單", "表格", "form"]):
            return ChunkType.FORMS

        if any(kw in title_lower for kw in ["聯絡", "聯繫", "contact"]):
            return ChunkType.CONTACT

        if any(kw in title_lower for kw in ["位置", "地點", "location"]):
            return ChunkType.WAYFINDING

        if any(kw in title_lower for kw in ["交通", "transport", "bus", "捷運"]):
            return ChunkType.TRANSPORT_MODE

        if any(kw in title_lower for kw in ["停車", "parking"]):
            return ChunkType.PARKING

        # 預設為段落
        return ChunkType.PARAGRAPH
