"""
醫師分塊器

專門用於醫師/醫生簡介文件的分塊器。
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


class PhysicianChunker(BaseChunker):
    """
    醫師文件的分塊器。

    處理醫師簡介內容，包含：
    - 姓名
    - 專長
    - 門診時間
    - 聯絡資訊
    - 學歷
    - 著作
    """

    SUPPORTED_DOC_TYPES = [DocType.PHYSICIAN]

    # 中文星期
    DAYS_ZH = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"]

    def chunk(
        self,
        content: str,
        doc_id: str,
        metadata: DocumentMetadata,
        doc_version: int = 1,
    ) -> list[Chunk]:
        """
        分塊醫師文件內容。

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

        custom = metadata.custom_fields

        # 抽取醫師姓名以在所有 chunks 中註明
        physician_name = custom.get("name", metadata.title) or ""
        if isinstance(physician_name, dict):
            # 處理字典格式如 {"zh-Hant": "高肇隆", "en": "Kao"}
            physician_name = (
                physician_name.get("zh-Hant")
                or physician_name.get("zh")
                or next(iter(physician_name.values()), "")
            )

        # 1. 簡介/元資料 chunk
        profile_chunk = self._chunk_profile(
            metadata, custom, doc_id, doc_version, position
        )
        chunks.append(profile_chunk)
        position += 1

        # 2. 專長 chunk
        if "specialties" in custom or "expertise" in custom:
            specialties_chunk = self._chunk_specialties(
                custom, doc_id, doc_version, position, physician_name
            )
            chunks.append(specialties_chunk)
            position += 1

        # 3. 門診時間 chunk
        if "schedule" in custom:
            schedule_chunk = self._chunk_schedule(
                custom["schedule"], doc_id, doc_version, position, physician_name
            )
            chunks.append(schedule_chunk)
            position += 1

        # 4. 聯絡資訊 chunk
        if "contact" in custom:
            contact_chunk = self._chunk_contact(
                custom["contact"], doc_id, doc_version, position, physician_name
            )
            chunks.append(contact_chunk)
            position += 1

        # 5. 學經歷 chunk
        if "education" in custom or "experience" in custom:
            edu_chunk = self._chunk_education_experience(
                custom.get("education", []),
                custom.get("experience", []),
                doc_id,
                doc_version,
                position,
                physician_name,
            )
            chunks.append(edu_chunk)
            position += 1

        # 6. 處理額外的 Markdown 內容
        sections = self.split_by_sections(content)
        for section in sections:
            section_chunks = self._chunk_section(
                section, doc_id, doc_version, position, physician_name
            )
            chunks.extend(section_chunks)
            position += len(section_chunks)

        return chunks

    def _chunk_profile(
        self,
        metadata: DocumentMetadata,
        custom: dict[str, Any],
        doc_id: str,
        doc_version: int,
        position: int,
    ) -> Chunk:
        """分塊醫師簡介/元資料。"""
        name = custom.get("name", metadata.title)
        department = metadata.department or custom.get("department", "")

        lines = [
            f"# {name} 醫師",
            "",
        ]

        if department:
            lines.append(f"**科別**: {department}")

        if "title" in custom:
            lines.append(f"**職稱**: {custom['title']}")

        if metadata.description:
            lines.append("")
            lines.append(metadata.description)

        return self.create_chunk(
            content="\n".join(lines),
            doc_id=doc_id,
            doc_version=doc_version,
            position=position,
            chunk_type=ChunkType.METADATA,
            section_title=f"{name} 醫師",
        )

    def _chunk_specialties(
        self,
        custom: dict[str, Any],
        doc_id: str,
        doc_version: int,
        position: int,
        physician_name: str = "",
    ) -> Chunk:
        """分塊醫師專長。"""
        specialties = custom.get("specialties", custom.get("expertise", []))

        # Include physician name for clear attribution
        title = f"## {physician_name}醫師 - 專長/主治項目" if physician_name else "## 專長/主治項目"
        lines = [title, ""]
        for specialty in specialties:
            lines.append(f"- {specialty}")

        section_title = f"{physician_name}醫師 - 專長" if physician_name else "專長"
        return self.create_chunk(
            content="\n".join(lines),
            doc_id=doc_id,
            doc_version=doc_version,
            position=position,
            chunk_type=ChunkType.LIST,
            section_title=section_title,
        )

    def _chunk_schedule(
        self,
        schedule: list[dict[str, Any]],
        doc_id: str,
        doc_version: int,
        position: int,
        physician_name: str = "",
    ) -> Chunk:
        """分塊門診時間資訊。"""
        # 包含醫師姓名以清楚註明
        title = f"## {physician_name}醫師 - 門診時間" if physician_name else "## 門診時間"
        lines = [title, ""]

        # 建構門診時間表格
        lines.append("| 星期 | 上午 | 下午 | 夜間 |")
        lines.append("|------|------|------|------|")

        # 初始化門診時間格子
        grid: dict[str, dict[str, str]] = {
            day: {"上午": "", "下午": "", "夜間": ""} for day in self.DAYS_ZH
        }

        # 填入門診時間
        for entry in schedule:
            day = entry.get("day", "")
            period = entry.get("period", "")
            location = entry.get("location", "✓")

            if day in grid and period in grid[day]:
                grid[day][period] = location

        # 建構表格行
        for day in self.DAYS_ZH:
            row = grid[day]
            lines.append(f"| {day} | {row['上午']} | {row['下午']} | {row['夜間']} |")

        section_title = f"{physician_name}醫師 - 門診時間" if physician_name else "門診時間"
        return self.create_chunk(
            content="\n".join(lines),
            doc_id=doc_id,
            doc_version=doc_version,
            position=position,
            chunk_type=ChunkType.SCHEDULE,
            section_title=section_title,
        )

    def _chunk_contact(
        self,
        contact: dict[str, str],
        doc_id: str,
        doc_version: int,
        position: int,
        physician_name: str = "",
    ) -> Chunk:
        """分塊聯絡資訊。"""
        # 包含醫師姓名以清楚註明
        title = f"## {physician_name}醫師 - 聯絡資訊" if physician_name else "## 聯絡資訊"
        lines = [title, ""]

        if "phone" in contact:
            lines.append(f"**電話**: {contact['phone']}")
        if "email" in contact:
            lines.append(f"**電子郵件**: {contact['email']}")
        if "location" in contact:
            lines.append(f"**診間位置**: {contact['location']}")
        if "assistant" in contact:
            lines.append(f"**助理**: {contact['assistant']}")

        section_title = f"{physician_name}醫師 - 聯絡資訊" if physician_name else "聯絡資訊"
        return self.create_chunk(
            content="\n".join(lines),
            doc_id=doc_id,
            doc_version=doc_version,
            position=position,
            chunk_type=ChunkType.CONTACT,
            section_title=section_title,
        )

    def _chunk_education_experience(
        self,
        education: list[str] | list[dict],
        experience: list[str] | list[dict],
        doc_id: str,
        doc_version: int,
        position: int,
        physician_name: str = "",
    ) -> Chunk:
        """分塊學經歷。"""
        # 包含醫師姓名以清楚註明
        lines = []

        if physician_name:
            lines.append(f"# {physician_name}醫師 - 學經歷")
            lines.append("")

        if education:
            lines.append("## 學歷")
            for edu in education:
                if isinstance(edu, dict):
                    degree = edu.get("degree", "")
                    school = edu.get("school", "")
                    year = edu.get("year", "")
                    lines.append(f"- {degree} - {school} ({year})" if year else f"- {degree} - {school}")
                else:
                    lines.append(f"- {edu}")
            lines.append("")

        if experience:
            lines.append("## 經歷")
            for exp in experience:
                if isinstance(exp, dict):
                    title = exp.get("title", "")
                    org = exp.get("organization", "")
                    lines.append(f"- {title} - {org}" if org else f"- {title}")
                else:
                    lines.append(f"- {exp}")

        section_title = f"{physician_name}醫師 - 學經歷" if physician_name else "學經歷"
        return self.create_chunk(
            content="\n".join(lines),
            doc_id=doc_id,
            doc_version=doc_version,
            position=position,
            chunk_type=ChunkType.PARAGRAPH,
            section_title=section_title,
        )

    def _chunk_section(
        self,
        section: dict[str, Any],
        doc_id: str,
        doc_version: int,
        start_position: int,
        physician_name: str = "",
    ) -> list[Chunk]:
        """分塊一個內容章節。"""
        chunks: list[Chunk] = []
        title = section.get("title")
        content = section.get("content", "")

        if not content:
            return chunks

        # 使用醫師姓名為內容加上前綴以清楚註明
        if physician_name:
            content = f"【{physician_name}醫師】\n{content}"

        chunk_type = self.infer_chunk_type(content, title)

        # 更新 section_title 以包含醫師姓名
        section_title = f"{physician_name}醫師 - {title}" if physician_name and title else title

        if len(content) <= self.chunk_size:
            chunk = self.create_chunk(
                content=content,
                doc_id=doc_id,
                doc_version=doc_version,
                position=start_position,
                chunk_type=chunk_type,
                section_title=section_title,
            )
            chunks.append(chunk)
        else:
            sub_chunks = self.split_by_size(content)
            for i, sub_content in enumerate(sub_chunks):
                chunk = self.create_chunk(
                    content=sub_content,
                    doc_id=doc_id,
                    doc_version=doc_version,
                    position=start_position + i,
                    chunk_type=chunk_type,
                    section_title=section_title,
                )
                chunks.append(chunk)

        return chunks


# 單例實例
physician_chunker = PhysicianChunker()
