"""
文件攝取服務

提供雙管線架構進行文件處理：
- CuratedPipeline: 處理結構化 YAML+Markdown 文件
- RawPipeline: 處理原始文件（PDF, DOCX, HTML）
- IngestionCoordinator: 協調整個攝取流程
- Chunkers: 針對不同文件類型的分塊策略
"""

from chatbot_graphrag.services.ingestion.coordinator import (
    IngestionCoordinator,
    ingestion_coordinator,
)
from chatbot_graphrag.services.ingestion.curated_pipeline import (
    CuratedPipeline,
    curated_pipeline,
)
from chatbot_graphrag.services.ingestion.raw_pipeline import (
    RawPipeline,
    raw_pipeline,
)
from chatbot_graphrag.services.ingestion.schema_validator import (
    SchemaValidator,
    SchemaValidationError,
    schema_validator,
)
from chatbot_graphrag.services.ingestion.chunkers import (
    BaseChunker,
    GenericChunker,
    GuideChunker,
    HospitalTeamChunker,
    PhysicianChunker,
    ProcedureChunker,
    get_chunker,
    CHUNKER_REGISTRY,
)

# 重新匯出 API 路由使用的模型
from chatbot_graphrag.models.pydantic.ingestion import (
    IngestJob,
    IngestJobConfig,
    IngestJobUpdate,
    DocumentInput,
)

__all__ = [
    # 協調器
    "IngestionCoordinator",
    "ingestion_coordinator",
    # 管線
    "CuratedPipeline",
    "curated_pipeline",
    "RawPipeline",
    "raw_pipeline",
    # 驗證
    "SchemaValidator",
    "SchemaValidationError",
    "schema_validator",
    # 分塊器
    "BaseChunker",
    "GenericChunker",
    "GuideChunker",
    "HospitalTeamChunker",
    "PhysicianChunker",
    "ProcedureChunker",
    "get_chunker",
    "CHUNKER_REGISTRY",
    # 模型（為方便重新匯出）
    "IngestJob",
    "IngestJobConfig",
    "IngestJobUpdate",
    "DocumentInput",
]
