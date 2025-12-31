"""
MinIO 儲存服務

提供 MinIO 的非同步介面，用於文件和資產儲存。

功能：
- 原始文件儲存（PDF、DOCX 等）
- 規範 JSON 表示
- 文件資產（圖片等）
- 社區報告
"""

import asyncio
import hashlib
import io
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional


class DateTimeEncoder(json.JSONEncoder):
    """用於 datetime 和 date 物件的自訂 JSON 編碼器。"""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        return super().default(obj)

from minio import Minio
from minio.error import S3Error

from chatbot_graphrag.core.config import settings

logger = logging.getLogger(__name__)


class MinIOService:
    """
    MinIO 物件儲存服務。

    管理以下儲存：
    - 原始文件（PDF、DOCX 等）
    - 規範 JSON 表示
    - 文件資產（圖片等）
    - 社區報告
    """

    _instance: Optional["MinIOService"] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> "MinIOService":
        """單例模式以重用客戶端。"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化 MinIO 服務。"""
        if self._initialized:
            return

        self._client: Optional[Minio] = None
        self._buckets = {
            "documents": settings.minio_bucket_documents,
            "chunks": settings.minio_bucket_chunks,
            "canonical": settings.minio_bucket_canonical,
            "assets": settings.minio_bucket_assets,
            "reports": settings.minio_bucket_reports,
        }
        self._initialized = True

    async def initialize(self) -> None:
        """初始化 MinIO 客戶端和儲存桶。"""
        if self._client is not None:
            return

        self._client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )

        logger.info(f"MinIO client initialized: {settings.minio_endpoint}")

        # 確保儲存桶存在
        await self._ensure_buckets()

    async def close(self) -> None:
        """關閉 MinIO 客戶端。"""
        # MinIO 客戶端不需要顯式關閉
        self._client = None
        logger.info("MinIO client closed")

    async def _ensure_buckets(self) -> None:
        """確保所有需要的儲存桶存在。"""
        loop = asyncio.get_event_loop()

        for bucket_name in self._buckets.values():
            try:
                exists = await loop.run_in_executor(
                    None,
                    self._client.bucket_exists,
                    bucket_name,
                )
                if not exists:
                    await loop.run_in_executor(
                        None,
                        self._client.make_bucket,
                        bucket_name,
                    )
                    logger.info(f"Created bucket: {bucket_name}")
            except S3Error as e:
                logger.error(f"Error creating bucket {bucket_name}: {e}")
                raise

    def _get_bucket(self, bucket_type: str) -> str:
        """根據類型獲取儲存桶名稱。"""
        if bucket_type not in self._buckets:
            raise ValueError(f"Unknown bucket type: {bucket_type}")
        return self._buckets[bucket_type]

    # ==================== 文件操作 ====================

    async def upload_document(
        self,
        doc_id: str,
        content: bytes,
        filename: str,
        content_type: str,
        metadata: Optional[dict[str, str]] = None,
    ) -> str:
        """上傳原始文件。"""
        bucket = self._get_bucket("documents")
        object_name = f"{doc_id}/{filename}"

        loop = asyncio.get_event_loop()

        # 計算內容雜湊
        content_hash = hashlib.sha256(content).hexdigest()

        # 準備元資料
        minio_metadata = metadata or {}
        minio_metadata["content-hash"] = content_hash

        await loop.run_in_executor(
            None,
            lambda: self._client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=io.BytesIO(content),
                length=len(content),
                content_type=content_type,
                metadata=minio_metadata,
            ),
        )

        logger.debug(f"Uploaded document: {bucket}/{object_name}")
        return f"{bucket}/{object_name}"

    async def download_document(
        self,
        doc_id: str,
        filename: str,
    ) -> Optional[bytes]:
        """下載原始文件。"""
        bucket = self._get_bucket("documents")
        object_name = f"{doc_id}/{filename}"

        loop = asyncio.get_event_loop()

        try:
            response = await loop.run_in_executor(
                None,
                lambda: self._client.get_object(bucket, object_name),
            )
            content = response.read()
            response.close()
            response.release_conn()
            return content
        except S3Error as e:
            if e.code == "NoSuchKey":
                return None
            raise

    async def delete_document(self, doc_id: str) -> bool:
        """刪除文件的所有物件。"""
        bucket = self._get_bucket("documents")
        prefix = f"{doc_id}/"

        loop = asyncio.get_event_loop()

        try:
            # 列出所有帶有前綴的物件
            objects = await loop.run_in_executor(
                None,
                lambda: list(self._client.list_objects(bucket, prefix=prefix)),
            )

            # 刪除每個物件
            for obj in objects:
                await loop.run_in_executor(
                    None,
                    lambda o=obj: self._client.remove_object(bucket, o.object_name),
                )

            return True
        except S3Error as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False

    # ==================== 規範 JSON 操作 ====================

    async def save_canonical(
        self,
        doc_id: str,
        version: int,
        canonical_data: dict[str, Any],
    ) -> str:
        """儲存文件的規範 JSON 表示。"""
        bucket = self._get_bucket("canonical")
        object_name = f"{doc_id}/v{version}/canonical.json"

        loop = asyncio.get_event_loop()

        content = json.dumps(canonical_data, ensure_ascii=False, indent=2, cls=DateTimeEncoder).encode("utf-8")
        content_hash = hashlib.sha256(content).hexdigest()

        await loop.run_in_executor(
            None,
            lambda: self._client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=io.BytesIO(content),
                length=len(content),
                content_type="application/json",
                metadata={"content-hash": content_hash},
            ),
        )

        logger.debug(f"Saved canonical: {bucket}/{object_name}")
        return f"{bucket}/{object_name}"

    async def load_canonical(
        self,
        doc_id: str,
        version: int,
    ) -> Optional[dict[str, Any]]:
        """載入規範 JSON 表示。"""
        bucket = self._get_bucket("canonical")
        object_name = f"{doc_id}/v{version}/canonical.json"

        loop = asyncio.get_event_loop()

        try:
            response = await loop.run_in_executor(
                None,
                lambda: self._client.get_object(bucket, object_name),
            )
            content = response.read()
            response.close()
            response.release_conn()
            return json.loads(content.decode("utf-8"))
        except S3Error as e:
            if e.code == "NoSuchKey":
                return None
            raise

    async def list_canonical_versions(self, doc_id: str) -> list[int]:
        """列出文件的所有規範版本。"""
        bucket = self._get_bucket("canonical")
        prefix = f"{doc_id}/v"

        loop = asyncio.get_event_loop()

        objects = await loop.run_in_executor(
            None,
            lambda: list(self._client.list_objects(bucket, prefix=prefix)),
        )

        versions = []
        for obj in objects:
            # 從路徑中提取版本，如 "doc_id/v1/canonical.json"
            parts = obj.object_name.split("/")
            if len(parts) >= 2:
                version_str = parts[1].lstrip("v")
                if version_str.isdigit():
                    versions.append(int(version_str))

        return sorted(set(versions), reverse=True)

    # ==================== 資產操作 ====================

    async def upload_asset(
        self,
        doc_id: str,
        asset_id: str,
        content: bytes,
        filename: str,
        content_type: str,
    ) -> str:
        """上傳文件資產（圖片、PDF 等）。"""
        bucket = self._get_bucket("assets")
        # 保留檔案副檔名
        ext = Path(filename).suffix
        object_name = f"{doc_id}/{asset_id}{ext}"

        loop = asyncio.get_event_loop()

        content_hash = hashlib.sha256(content).hexdigest()

        await loop.run_in_executor(
            None,
            lambda: self._client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=io.BytesIO(content),
                length=len(content),
                content_type=content_type,
                metadata={"content-hash": content_hash, "original-filename": filename},
            ),
        )

        logger.debug(f"Uploaded asset: {bucket}/{object_name}")
        return f"{bucket}/{object_name}"

    async def download_asset(
        self,
        doc_id: str,
        asset_id: str,
        extension: str = "",
    ) -> Optional[bytes]:
        """下載文件資產。"""
        bucket = self._get_bucket("assets")
        object_name = f"{doc_id}/{asset_id}{extension}"

        loop = asyncio.get_event_loop()

        try:
            response = await loop.run_in_executor(
                None,
                lambda: self._client.get_object(bucket, object_name),
            )
            content = response.read()
            response.close()
            response.release_conn()
            return content
        except S3Error as e:
            if e.code == "NoSuchKey":
                return None
            raise

    async def get_asset_url(
        self,
        doc_id: str,
        asset_id: str,
        extension: str = "",
        expires: timedelta = timedelta(hours=1),
    ) -> str:
        """獲取資產的預簽名 URL。"""
        bucket = self._get_bucket("assets")
        object_name = f"{doc_id}/{asset_id}{extension}"

        loop = asyncio.get_event_loop()

        url = await loop.run_in_executor(
            None,
            lambda: self._client.presigned_get_object(
                bucket_name=bucket,
                object_name=object_name,
                expires=expires,
            ),
        )

        return url

    async def list_assets(self, doc_id: str) -> list[dict[str, Any]]:
        """列出文件的所有資產。"""
        bucket = self._get_bucket("assets")
        prefix = f"{doc_id}/"

        loop = asyncio.get_event_loop()

        objects = await loop.run_in_executor(
            None,
            lambda: list(self._client.list_objects(bucket, prefix=prefix)),
        )

        assets = []
        for obj in objects:
            assets.append({
                "object_name": obj.object_name,
                "size": obj.size,
                "last_modified": obj.last_modified,
                "etag": obj.etag,
            })

        return assets

    # ==================== 社區報告操作 ====================

    async def save_community_report(
        self,
        community_id: str,
        level: int,
        report_data: dict[str, Any],
    ) -> str:
        """儲存社區報告。"""
        bucket = self._get_bucket("reports")
        object_name = f"communities/level_{level}/{community_id}.json"

        loop = asyncio.get_event_loop()

        content = json.dumps(report_data, ensure_ascii=False, indent=2, cls=DateTimeEncoder).encode("utf-8")

        await loop.run_in_executor(
            None,
            lambda: self._client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=io.BytesIO(content),
                length=len(content),
                content_type="application/json",
            ),
        )

        logger.debug(f"Saved community report: {bucket}/{object_name}")
        return f"{bucket}/{object_name}"

    async def load_community_report(
        self,
        community_id: str,
        level: int,
    ) -> Optional[dict[str, Any]]:
        """載入社區報告。"""
        bucket = self._get_bucket("reports")
        object_name = f"communities/level_{level}/{community_id}.json"

        loop = asyncio.get_event_loop()

        try:
            response = await loop.run_in_executor(
                None,
                lambda: self._client.get_object(bucket, object_name),
            )
            content = response.read()
            response.close()
            response.release_conn()
            return json.loads(content.decode("utf-8"))
        except S3Error as e:
            if e.code == "NoSuchKey":
                return None
            raise

    async def list_community_reports(
        self,
        level: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """列出社區報告，可選按層級過濾。"""
        bucket = self._get_bucket("reports")

        if level is not None:
            prefix = f"communities/level_{level}/"
        else:
            prefix = "communities/"

        loop = asyncio.get_event_loop()

        objects = await loop.run_in_executor(
            None,
            lambda: list(self._client.list_objects(bucket, prefix=prefix, recursive=True)),
        )

        reports = []
        for obj in objects:
            if obj.object_name.endswith(".json"):
                reports.append({
                    "object_name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified,
                })

        return reports

    # ==================== Chunk 儲存操作 ====================

    async def save_chunks(
        self,
        doc_id: str,
        version: int,
        chunks: list[dict[str, Any]],
    ) -> str:
        """儲存文件版本的 chunks。"""
        bucket = self._get_bucket("chunks")
        object_name = f"{doc_id}/v{version}/chunks.json"

        loop = asyncio.get_event_loop()

        content = json.dumps(chunks, ensure_ascii=False, cls=DateTimeEncoder).encode("utf-8")

        await loop.run_in_executor(
            None,
            lambda: self._client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=io.BytesIO(content),
                length=len(content),
                content_type="application/json",
            ),
        )

        logger.debug(f"Saved chunks: {bucket}/{object_name}")
        return f"{bucket}/{object_name}"

    async def load_chunks(
        self,
        doc_id: str,
        version: int,
    ) -> Optional[list[dict[str, Any]]]:
        """載入文件版本的 chunks。"""
        bucket = self._get_bucket("chunks")
        object_name = f"{doc_id}/v{version}/chunks.json"

        loop = asyncio.get_event_loop()

        try:
            response = await loop.run_in_executor(
                None,
                lambda: self._client.get_object(bucket, object_name),
            )
            content = response.read()
            response.close()
            response.release_conn()
            return json.loads(content.decode("utf-8"))
        except S3Error as e:
            if e.code == "NoSuchKey":
                return None
            raise

    # ==================== 健康檢查 ====================

    async def health_check(self) -> dict[str, Any]:
        """檢查 MinIO 連線健康狀態。"""
        try:
            loop = asyncio.get_event_loop()

            buckets = await loop.run_in_executor(
                None,
                self._client.list_buckets,
            )

            bucket_names = [b.name for b in buckets]

            return {
                "status": "healthy",
                "endpoint": settings.minio_endpoint,
                "buckets": bucket_names,
                "connected": True,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connected": False,
            }

    # ==================== 工具方法 ====================

    async def object_exists(
        self,
        bucket_type: str,
        object_name: str,
    ) -> bool:
        """檢查物件是否存在。"""
        bucket = self._get_bucket(bucket_type)
        loop = asyncio.get_event_loop()

        try:
            await loop.run_in_executor(
                None,
                lambda: self._client.stat_object(bucket, object_name),
            )
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            raise

    async def get_object_info(
        self,
        bucket_type: str,
        object_name: str,
    ) -> Optional[dict[str, Any]]:
        """獲取物件元資料。"""
        bucket = self._get_bucket(bucket_type)
        loop = asyncio.get_event_loop()

        try:
            stat = await loop.run_in_executor(
                None,
                lambda: self._client.stat_object(bucket, object_name),
            )
            return {
                "bucket": bucket,
                "object_name": object_name,
                "size": stat.size,
                "etag": stat.etag,
                "content_type": stat.content_type,
                "last_modified": stat.last_modified,
                "metadata": stat.metadata,
            }
        except S3Error as e:
            if e.code == "NoSuchKey":
                return None
            raise


# 單例實例
minio_service = MinIOService()
