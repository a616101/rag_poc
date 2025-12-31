"""
儲存服務

提供各種儲存後端服務：
- MinIO: 物件儲存（文件、規範 JSON、社區報告）
"""

from chatbot_graphrag.services.storage.minio_service import (
    MinIOService,
    minio_service,
)

__all__ = [
    "MinIOService",
    "minio_service",
]
