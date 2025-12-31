"""
指南分塊器

專門用於指南文件（位置、交通）的分塊器。
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


class GuideChunker(BaseChunker):
    """
    指南文件的分塊器。

    處理兩種子類型：
    - guide.location: 建築物、樓層、房間、座標
    - guide.transport: 交通方式、停車、路線
    """

    SUPPORTED_DOC_TYPES = [DocType.GUIDE_LOCATION, DocType.GUIDE_TRANSPORT]

    def chunk(
        self,
        content: str,
        doc_id: str,
        metadata: DocumentMetadata,
        doc_version: int = 1,
    ) -> list[Chunk]:
        """
        分塊指南文件內容。

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

        # 2. 根據 doc_type 處理
        if metadata.doc_type == DocType.GUIDE_LOCATION:
            type_chunks = self._chunk_location_guide(
                content,
                metadata.custom_fields,
                doc_id,
                doc_version,
                position,
            )
        else:  # GUIDE_TRANSPORT
            type_chunks = self._chunk_transport_guide(
                content,
                metadata.custom_fields,
                doc_id,
                doc_version,
                position,
            )

        chunks.extend(type_chunks)
        position += len(type_chunks)

        # 3. 處理額外的 Markdown 內容
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

    def _chunk_location_guide(
        self,
        content: str,
        custom: dict[str, Any],
        doc_id: str,
        doc_version: int,
        start_position: int,
    ) -> list[Chunk]:
        """分塊位置指南內容。"""
        chunks: list[Chunk] = []
        position = start_position

        # 建構位置摘要
        lines = ["## 位置資訊", ""]

        if "building" in custom:
            lines.append(f"**建築物**: {custom['building']}")
        if "floor" in custom:
            lines.append(f"**樓層**: {custom['floor']}")
        if "room" in custom:
            lines.append(f"**房間**: {custom['room']}")
        if "coordinates" in custom:
            coords = custom["coordinates"]
            lines.append(f"**座標**: ({coords.get('lat', '')}, {coords.get('lng', '')})")

        if len(lines) > 2:  # 不只是標題
            location_chunk = self.create_chunk(
                content="\n".join(lines),
                doc_id=doc_id,
                doc_version=doc_version,
                position=position,
                chunk_type=ChunkType.WAYFINDING,
                section_title="位置資訊",
                metadata_extras={
                    "building": custom.get("building"),
                    "floor": custom.get("floor"),
                },
            )
            chunks.append(location_chunk)
            position += 1

        # 處理地標（如果有）
        if "landmarks" in custom:
            landmarks_lines = ["## 地標與指引", ""]
            for landmark in custom["landmarks"]:
                if isinstance(landmark, dict):
                    name = landmark.get("name", "")
                    desc = landmark.get("description", "")
                    landmarks_lines.append(f"- **{name}**: {desc}")
                else:
                    landmarks_lines.append(f"- {landmark}")

            landmarks_chunk = self.create_chunk(
                content="\n".join(landmarks_lines),
                doc_id=doc_id,
                doc_version=doc_version,
                position=position,
                chunk_type=ChunkType.WAYFINDING,
                section_title="地標與指引",
            )
            chunks.append(landmarks_chunk)

        return chunks

    def _chunk_transport_guide(
        self,
        content: str,
        custom: dict[str, Any],
        doc_id: str,
        doc_version: int,
        start_position: int,
    ) -> list[Chunk]:
        """分塊交通指南內容。"""
        chunks: list[Chunk] = []
        position = start_position

        # 處理交通方式
        if "transport_modes" in custom:
            for mode in custom["transport_modes"]:
                mode_lines = []
                mode_name = mode.get("mode", mode.get("name", "交通方式"))
                mode_lines.append(f"## {mode_name}")
                mode_lines.append("")

                if "description" in mode:
                    mode_lines.append(mode["description"])
                if "routes" in mode:
                    mode_lines.append("")
                    mode_lines.append("### 路線")
                    for route in mode["routes"]:
                        mode_lines.append(f"- {route}")
                if "stops" in mode:
                    mode_lines.append("")
                    mode_lines.append("### 站點")
                    for stop in mode["stops"]:
                        mode_lines.append(f"- {stop}")
                if "frequency" in mode:
                    mode_lines.append(f"\n**班次**: {mode['frequency']}")
                if "duration" in mode:
                    mode_lines.append(f"**車程**: {mode['duration']}")

                mode_chunk = self.create_chunk(
                    content="\n".join(mode_lines),
                    doc_id=doc_id,
                    doc_version=doc_version,
                    position=position,
                    chunk_type=ChunkType.TRANSPORT_MODE,
                    section_title=mode_name,
                    metadata_extras={"transport_mode": mode_name},
                )
                chunks.append(mode_chunk)
                position += 1

        # 處理停車資訊
        if "parking" in custom:
            parking = custom["parking"]
            parking_lines = ["## 停車資訊", ""]

            if isinstance(parking, dict):
                if "location" in parking:
                    parking_lines.append(f"**位置**: {parking['location']}")
                if "capacity" in parking:
                    parking_lines.append(f"**車位數**: {parking['capacity']}")
                if "fees" in parking:
                    parking_lines.append(f"**費用**: {parking['fees']}")
                if "hours" in parking:
                    parking_lines.append(f"**開放時間**: {parking['hours']}")
                if "notes" in parking:
                    parking_lines.append(f"\n**備註**: {parking['notes']}")
            else:
                parking_lines.append(str(parking))

            parking_chunk = self.create_chunk(
                content="\n".join(parking_lines),
                doc_id=doc_id,
                doc_version=doc_version,
                position=position,
                chunk_type=ChunkType.PARKING,
                section_title="停車資訊",
            )
            chunks.append(parking_chunk)

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
guide_chunker = GuideChunker()
