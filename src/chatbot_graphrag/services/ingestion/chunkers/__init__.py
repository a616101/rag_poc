"""
文件分塊策略

提供針對不同文件類型的分塊器：
- GenericChunker: 通用分塊器
- GuideChunker: 指南文件分塊器（位置、交通）
- HospitalTeamChunker: 醫療團隊文件分塊器
- PhysicianChunker: 醫師簡介分塊器
- ProcedureChunker: 流程文件分塊器
"""

from chatbot_graphrag.core.constants import DocType
from chatbot_graphrag.services.ingestion.chunkers.base import BaseChunker
from chatbot_graphrag.services.ingestion.chunkers.generic import (
    GenericChunker,
    generic_chunker,
)
from chatbot_graphrag.services.ingestion.chunkers.guide import (
    GuideChunker,
    guide_chunker,
)
from chatbot_graphrag.services.ingestion.chunkers.hospital_team import (
    HospitalTeamChunker,
    hospital_team_chunker,
)
from chatbot_graphrag.services.ingestion.chunkers.physician import (
    PhysicianChunker,
    physician_chunker,
)
from chatbot_graphrag.services.ingestion.chunkers.procedure import (
    ProcedureChunker,
    procedure_chunker,
)


# 分塊器註冊表：將文件類型映射到分塊器實例
CHUNKER_REGISTRY: dict[DocType, BaseChunker] = {
    DocType.PROCEDURE: procedure_chunker,
    DocType.GUIDE_LOCATION: guide_chunker,
    DocType.GUIDE_TRANSPORT: guide_chunker,
    DocType.PHYSICIAN: physician_chunker,
    DocType.HOSPITAL_TEAM: hospital_team_chunker,
    DocType.FAQ: generic_chunker,
    DocType.GENERIC: generic_chunker,
}


def get_chunker(doc_type: DocType) -> BaseChunker:
    """
    取得適合文件類型的分塊器。

    Args:
        doc_type: 文件類型

    Returns:
        該文件類型的分塊器實例
    """
    return CHUNKER_REGISTRY.get(doc_type, generic_chunker)


__all__ = [
    # 基礎類
    "BaseChunker",
    # 實作
    "GenericChunker",
    "generic_chunker",
    "GuideChunker",
    "guide_chunker",
    "HospitalTeamChunker",
    "hospital_team_chunker",
    "PhysicianChunker",
    "physician_chunker",
    "ProcedureChunker",
    "procedure_chunker",
    # 註冊表
    "CHUNKER_REGISTRY",
    "get_chunker",
]
