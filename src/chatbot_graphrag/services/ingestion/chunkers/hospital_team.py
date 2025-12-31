"""
醫療團隊分塊器

專門用於醫療團隊/部門文件的分塊器。
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


class HospitalTeamChunker(BaseChunker):
    """
    醫療團隊文件的分塊器。

    處理團隊/部門內容，包含：
    - 團隊名稱
    - 成員
    - 服務項目
    - 服務地點
    - 服務時間
    """

    SUPPORTED_DOC_TYPES = [DocType.HOSPITAL_TEAM]

    def chunk(
        self,
        content: str,
        doc_id: str,
        metadata: DocumentMetadata,
        doc_version: int = 1,
    ) -> list[Chunk]:
        """
        分塊醫療團隊文件內容。

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

        # 1. 建立團隊摘要 chunk
        summary_chunk = self._create_summary_chunk(
            metadata,
            custom,
            doc_id,
            doc_version,
            position,
        )
        chunks.append(summary_chunk)
        position += 1

        # 2. 服務 chunk
        if "services" in custom:
            service_chunks = self._chunk_services(
                custom["services"],
                doc_id,
                doc_version,
                position,
            )
            chunks.extend(service_chunks)
            position += len(service_chunks)

        # 3. 成員 chunk
        if "members" in custom:
            member_chunk = self._chunk_members(
                custom["members"],
                doc_id,
                doc_version,
                position,
            )
            chunks.append(member_chunk)
            position += 1

        # 4. 地點 chunk
        if "locations" in custom:
            location_chunk = self._chunk_locations(
                custom["locations"],
                doc_id,
                doc_version,
                position,
            )
            chunks.append(location_chunk)
            position += 1

        # 5. 服務時間 chunk
        if "hours" in custom:
            hours_chunk = self._chunk_hours(
                custom["hours"],
                doc_id,
                doc_version,
                position,
            )
            chunks.append(hours_chunk)
            position += 1

        # 6. 處理額外的 Markdown 內容
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

    def _create_summary_chunk(
        self,
        metadata: DocumentMetadata,
        custom: dict[str, Any],
        doc_id: str,
        doc_version: int,
        position: int,
    ) -> Chunk:
        """建立團隊摘要 chunk。"""
        team_name = custom.get("team_name", metadata.title)

        lines = [
            f"# {team_name}",
            "",
        ]

        if metadata.department:
            lines.append(f"**所屬部門**: {metadata.department}")

        if metadata.description:
            lines.append("")
            lines.append(metadata.description)

        if "mission" in custom:
            lines.append("")
            lines.append(f"**使命**: {custom['mission']}")

        return self.create_chunk(
            content="\n".join(lines),
            doc_id=doc_id,
            doc_version=doc_version,
            position=position,
            chunk_type=ChunkType.METADATA,
            section_title=team_name,
            metadata_extras={
                "team_name": team_name,
                "department": metadata.department,
            },
        )

    def _chunk_services(
        self,
        services: list[str] | list[dict[str, Any]],
        doc_id: str,
        doc_version: int,
        start_position: int,
    ) -> list[Chunk]:
        """分塊服務資訊。"""
        chunks: list[Chunk] = []

        # 建構服務內容
        lines = ["## 服務項目", ""]

        detailed_services: list[dict] = []
        simple_services: list[str] = []

        for service in services:
            if isinstance(service, dict):
                detailed_services.append(service)
            else:
                simple_services.append(str(service))

        # 簡單列表
        if simple_services:
            for svc in simple_services:
                lines.append(f"- {svc}")
            lines.append("")

        # 檢查是否適合放入一個 chunk
        if not detailed_services:
            chunk = self.create_chunk(
                content="\n".join(lines),
                doc_id=doc_id,
                doc_version=doc_version,
                position=start_position,
                chunk_type=ChunkType.SERVICE_SCOPE,
                section_title="服務項目",
            )
            chunks.append(chunk)
            return chunks

        # 建立概覽 chunk
        if simple_services:
            overview_chunk = self.create_chunk(
                content="\n".join(lines),
                doc_id=doc_id,
                doc_version=doc_version,
                position=start_position,
                chunk_type=ChunkType.SERVICE_SCOPE,
                section_title="服務項目",
            )
            chunks.append(overview_chunk)
            start_position += 1

        # 建立詳細服務 chunks
        for i, svc in enumerate(detailed_services):
            svc_lines = []
            svc_name = svc.get("name", svc.get("title", f"服務 {i + 1}"))
            svc_lines.append(f"### {svc_name}")

            if "description" in svc:
                svc_lines.append(svc["description"])
            if "eligibility" in svc:
                svc_lines.append(f"\n**適用對象**: {svc['eligibility']}")
            if "process" in svc:
                svc_lines.append(f"\n**流程**: {svc['process']}")
            if "contact" in svc:
                svc_lines.append(f"\n**聯絡方式**: {svc['contact']}")

            svc_chunk = self.create_chunk(
                content="\n".join(svc_lines),
                doc_id=doc_id,
                doc_version=doc_version,
                position=start_position + i,
                chunk_type=ChunkType.SERVICE_SCOPE,
                section_title=svc_name,
            )
            chunks.append(svc_chunk)

        return chunks

    def _chunk_members(
        self,
        members: list[dict[str, Any]],
        doc_id: str,
        doc_version: int,
        position: int,
    ) -> Chunk:
        """分塊團隊成員資訊。"""
        lines = ["## 團隊成員", ""]

        # 建立成員表格
        lines.append("| 姓名 | 職稱 | 專長 |")
        lines.append("|------|------|------|")

        for member in members:
            name = member.get("name", "")
            title = member.get("title", member.get("role", ""))
            specialty = member.get("specialty", member.get("specialties", ""))
            if isinstance(specialty, list):
                specialty = ", ".join(specialty)
            lines.append(f"| {name} | {title} | {specialty} |")

        return self.create_chunk(
            content="\n".join(lines),
            doc_id=doc_id,
            doc_version=doc_version,
            position=position,
            chunk_type=ChunkType.TABLE,
            section_title="團隊成員",
            metadata_extras={"member_count": len(members)},
        )

    def _chunk_locations(
        self,
        locations: list[dict[str, Any]] | list[str],
        doc_id: str,
        doc_version: int,
        position: int,
    ) -> Chunk:
        """分塊地點資訊。"""
        lines = ["## 服務地點", ""]

        for loc in locations:
            if isinstance(loc, dict):
                name = loc.get("name", loc.get("building", ""))
                floor = loc.get("floor", "")
                room = loc.get("room", "")
                loc_str = name
                if floor:
                    loc_str += f" {floor}"
                if room:
                    loc_str += f" {room}"
                lines.append(f"- {loc_str}")
            else:
                lines.append(f"- {loc}")

        return self.create_chunk(
            content="\n".join(lines),
            doc_id=doc_id,
            doc_version=doc_version,
            position=position,
            chunk_type=ChunkType.LOCATIONS,
            section_title="服務地點",
        )

    def _chunk_hours(
        self,
        hours: dict[str, Any] | list[dict[str, Any]] | str,
        doc_id: str,
        doc_version: int,
        position: int,
    ) -> Chunk:
        """分塊服務時間資訊。"""
        lines = ["## 服務時間", ""]

        if isinstance(hours, str):
            lines.append(hours)
        elif isinstance(hours, list):
            for entry in hours:
                day = entry.get("day", entry.get("days", ""))
                time = entry.get("time", entry.get("hours", ""))
                lines.append(f"- **{day}**: {time}")
        else:
            if "weekday" in hours:
                lines.append(f"**平日**: {hours['weekday']}")
            if "weekend" in hours:
                lines.append(f"**週末**: {hours['weekend']}")
            if "holiday" in hours:
                lines.append(f"**假日**: {hours['holiday']}")
            if "notes" in hours:
                lines.append(f"\n*{hours['notes']}*")

        return self.create_chunk(
            content="\n".join(lines),
            doc_id=doc_id,
            doc_version=doc_version,
            position=position,
            chunk_type=ChunkType.SCHEDULE,
            section_title="服務時間",
        )

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

        chunk_type = self.infer_chunk_type(content, title)

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


# 單例實例
hospital_team_chunker = HospitalTeamChunker()
