"""
Qdrant 向量服務

提供 Qdrant 的非同步介面，支援混合向量搜索（密集 + 稀疏）。

功能：
- 密集嵌入（文本嵌入模型）
- 稀疏嵌入（SPLADE）
- 命名向量用於多向量儲存
- 負載過濾
"""

import asyncio
import logging
import uuid
from typing import Any, Optional

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    ScoredPoint,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
)

from chatbot_graphrag.core.config import settings
from chatbot_graphrag.models.pydantic.ingestion import Chunk

logger = logging.getLogger(__name__)

# 用於從 chunk ID 生成確定性 UUID 的命名空間 UUID
CHUNK_UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def string_to_uuid(s: str) -> str:
    """將字串轉換為確定性 UUID5 字串。"""
    return str(uuid.uuid5(CHUNK_UUID_NAMESPACE, s))


class QdrantService:
    """
    Qdrant 混合向量搜索服務。

    支援：
    - 密集嵌入（文本嵌入模型）
    - 稀疏嵌入（SPLADE）
    - 命名向量用於多向量儲存
    - 負載過濾
    """

    _instance: Optional["QdrantService"] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> "QdrantService":
        """單例模式以重用客戶端。"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化 Qdrant 服務。"""
        if self._initialized:
            return

        self._client: Optional[AsyncQdrantClient] = None
        self._chunks_collection = settings.qdrant_collection_chunks
        self._cache_collection = settings.qdrant_collection_cache
        self._communities_collection = settings.qdrant_collection_communities
        self._initialized = True

    async def initialize(self) -> None:
        """初始化 Qdrant 客戶端和集合。"""
        if self._client is not None:
            return

        self._client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=60,
        )

        logger.info(f"Qdrant client initialized: {settings.qdrant_url}")

        # 確保集合存在
        await self._ensure_collections()

    async def close(self) -> None:
        """關閉 Qdrant 客戶端。"""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Qdrant client closed")

    async def _ensure_collections(self) -> None:
        """確保所有需要的集合存在。"""
        collections = await self._client.get_collections()
        existing = {c.name for c in collections.collections}

        # Chunks 集合（混合：密集 + 稀疏）
        if self._chunks_collection not in existing:
            await self._create_chunks_collection()

        # 快取集合（語義快取）
        if self._cache_collection not in existing:
            await self._create_cache_collection()

        # 社區集合（社區報告嵌入）
        if self._communities_collection not in existing:
            await self._create_communities_collection()

    async def _create_chunks_collection(self) -> None:
        """建立帶有混合向量的 chunks 集合。"""
        await self._client.create_collection(
            collection_name=self._chunks_collection,
            vectors_config={
                "dense": VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,
                    ),
                ),
            },
        )

        # 建立負載索引
        await self._client.create_payload_index(
            collection_name=self._chunks_collection,
            field_name="doc_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        await self._client.create_payload_index(
            collection_name=self._chunks_collection,
            field_name="doc_type",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        await self._client.create_payload_index(
            collection_name=self._chunks_collection,
            field_name="chunk_type",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        await self._client.create_payload_index(
            collection_name=self._chunks_collection,
            field_name="acl_groups",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        await self._client.create_payload_index(
            collection_name=self._chunks_collection,
            field_name="tenant_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        logger.info(f"Created chunks collection: {self._chunks_collection}")

    async def _create_cache_collection(self) -> None:
        """建立語義快取集合。"""
        await self._client.create_collection(
            collection_name=self._cache_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dimension,
                distance=Distance.COSINE,
            ),
        )

        await self._client.create_payload_index(
            collection_name=self._cache_collection,
            field_name="query_hash",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        await self._client.create_payload_index(
            collection_name=self._cache_collection,
            field_name="tenant_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        logger.info(f"Created cache collection: {self._cache_collection}")

    async def _create_communities_collection(self) -> None:
        """建立社區集合。"""
        await self._client.create_collection(
            collection_name=self._communities_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dimension,
                distance=Distance.COSINE,
            ),
        )

        await self._client.create_payload_index(
            collection_name=self._communities_collection,
            field_name="level",
            field_schema=models.PayloadSchemaType.INTEGER,
        )
        await self._client.create_payload_index(
            collection_name=self._communities_collection,
            field_name="community_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        logger.info(f"Created communities collection: {self._communities_collection}")

    # ==================== Chunk 操作 ====================

    async def upsert_chunk(
        self,
        chunk_id: str,
        dense_embedding: list[float],
        sparse_embedding: Optional[dict[int, float]],
        payload: dict[str, Any],
    ) -> str:
        """使用混合嵌入 upsert 一個 chunk。"""
        vectors = {"dense": dense_embedding}

        if sparse_embedding and settings.sparse_encoder_enabled:
            # 將稀疏字典轉換為 Qdrant 格式
            indices = list(sparse_embedding.keys())
            values = list(sparse_embedding.values())
            vectors["sparse"] = models.SparseVector(
                indices=indices,
                values=values,
            )

        # Store original chunk_id in payload
        payload["chunk_id"] = chunk_id

        # Convert string ID to valid UUID for Qdrant
        point_id = string_to_uuid(chunk_id)

        point = PointStruct(
            id=point_id,
            vector=vectors,
            payload=payload,
        )

        await self._client.upsert(
            collection_name=self._chunks_collection,
            points=[point],
        )

        return chunk_id

    async def upsert_chunks_batch(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """批次 upsert chunks。"""
        total_upserted = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            points = []

            for chunk_data in batch:
                vectors = {"dense": chunk_data["dense_embedding"]}

                if chunk_data.get("sparse_embedding") and settings.sparse_encoder_enabled:
                    sparse = chunk_data["sparse_embedding"]
                    vectors["sparse"] = models.SparseVector(
                        indices=list(sparse.keys()),
                        values=list(sparse.values()),
                    )

                # Store original chunk_id in payload
                payload = chunk_data["payload"].copy()
                original_id = chunk_data["id"]
                payload["chunk_id"] = original_id

                # Convert string ID to valid UUID for Qdrant
                point_id = string_to_uuid(original_id)

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vectors,
                        payload=payload,
                    )
                )

            await self._client.upsert(
                collection_name=self._chunks_collection,
                points=points,
            )
            total_upserted += len(points)

        return total_upserted

    async def delete_chunks_by_doc_id(self, doc_id: str) -> int:
        """刪除文件的所有 chunks。"""
        result = await self._client.delete(
            collection_name=self._chunks_collection,
            points_selector=models.FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
        )
        return result.status == models.UpdateStatus.COMPLETED

    async def update_chunk_metadata(
        self,
        chunk_id: str,
        metadata_updates: dict[str, Any],
    ) -> bool:
        """
        更新特定 chunk 的元資料（負載）。

        Args:
            chunk_id: 要更新的 chunk ID
            metadata_updates: 要更新的欄位字典

        Returns:
            如果成功則為 True
        """
        point_id = string_to_uuid(chunk_id)

        try:
            await self._client.set_payload(
                collection_name=self._chunks_collection,
                payload=metadata_updates,
                points=[point_id],
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to update chunk metadata for {chunk_id}: {e}")
            return False

    async def update_chunks_metadata_batch(
        self,
        updates: list[dict[str, Any]],
    ) -> int:
        """
        批次更新多個 chunks 的元資料。

        Args:
            updates: 包含 "chunk_id" 和 "metadata" 鍵的字典列表

        Returns:
            成功更新的 chunks 數量
        """
        success_count = 0

        for update in updates:
            chunk_id = update.get("chunk_id")
            metadata = update.get("metadata", {})

            if chunk_id and metadata:
                if await self.update_chunk_metadata(chunk_id, metadata):
                    success_count += 1

        return success_count

    async def get_chunks_by_ids(
        self,
        chunk_ids: list[str],
    ) -> list[dict[str, Any]]:
        """
        根據 ID 檢索 chunks。

        此方法用於 GraphRAG 在圖譜遍歷後獲取與實體相關的 chunks。

        Args:
            chunk_ids: 要檢索的 chunk ID 列表

        Returns:
            包含 id、content 和 metadata 的 chunk 資料字典列表
        """
        if not chunk_ids:
            return []

        # 將字串 ID 轉換為 Qdrant 的 UUID
        point_ids = [string_to_uuid(cid) for cid in chunk_ids]

        try:
            # 根據 ID 檢索點
            points = await self._client.retrieve(
                collection_name=self._chunks_collection,
                ids=point_ids,
                with_payload=True,
                with_vectors=False,  # Don't need vectors for this use case
            )

            results = []
            for point in points:
                payload = point.payload or {}
                results.append({
                    "id": str(point.id),
                    "chunk_id": payload.get("chunk_id", ""),
                    "doc_id": payload.get("doc_id", ""),
                    "content": payload.get("content", ""),
                    "metadata": {
                        k: v for k, v in payload.items()
                        if k not in ("chunk_id", "doc_id", "content")
                    },
                })

            logger.debug(f"Retrieved {len(results)}/{len(chunk_ids)} chunks by ID")
            return results

        except Exception as e:
            logger.warning(f"Failed to retrieve chunks by IDs: {e}")
            return []

    # ==================== 混合搜索 ====================

    async def search_dense(
        self,
        query_embedding: list[float],
        limit: int = 20,
        score_threshold: float = 0.5,
        filter_conditions: Optional[dict[str, Any]] = None,
    ) -> list[ScoredPoint]:
        """僅使用密集嵌入進行搜索。"""
        qdrant_filter = self._build_filter(filter_conditions)

        results = await self._client.search(
            collection_name=self._chunks_collection,
            query_vector=("dense", query_embedding),
            limit=limit,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return results

    async def search_sparse(
        self,
        sparse_embedding: dict[int, float],
        limit: int = 20,
        filter_conditions: Optional[dict[str, Any]] = None,
    ) -> list[ScoredPoint]:
        """僅使用稀疏嵌入進行搜索。"""
        qdrant_filter = self._build_filter(filter_conditions)

        sparse_vector = models.SparseVector(
            indices=list(sparse_embedding.keys()),
            values=list(sparse_embedding.values()),
        )

        results = await self._client.search(
            collection_name=self._chunks_collection,
            query_vector=models.NamedSparseVector(
                name="sparse",
                vector=sparse_vector,
            ),
            limit=limit,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return results

    async def search_hybrid(
        self,
        query_embedding: list[float],
        sparse_embedding: Optional[dict[int, float]],
        limit: int = 40,
        score_threshold: float = 0.35,
        filter_conditions: Optional[dict[str, Any]] = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        混合搜索，結合密集和稀疏嵌入。

        使用查詢融合來合併兩種搜索的結果。
        """
        qdrant_filter = self._build_filter(filter_conditions)

        # 密集搜索
        dense_results = await self._client.search(
            collection_name=self._chunks_collection,
            query_vector=("dense", query_embedding),
            limit=limit * 2,  # Fetch more for merging
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        # 稀疏搜索（如果可用）
        sparse_results = []
        if sparse_embedding and settings.sparse_encoder_enabled:
            sparse_vector = models.SparseVector(
                indices=list(sparse_embedding.keys()),
                values=list(sparse_embedding.values()),
            )
            sparse_results = await self._client.search(
                collection_name=self._chunks_collection,
                query_vector=models.NamedSparseVector(
                    name="sparse",
                    vector=sparse_vector,
                ),
                limit=limit * 2,
                query_filter=qdrant_filter,
                with_payload=True,
            )

        # 使用加權評分合併結果
        return self._merge_search_results(
            dense_results,
            sparse_results,
            dense_weight,
            sparse_weight,
            limit,
        )

    def _merge_search_results(
        self,
        dense_results: list[ScoredPoint],
        sparse_results: list[ScoredPoint],
        dense_weight: float,
        sparse_weight: float,
        limit: int,
    ) -> list[dict[str, Any]]:
        """使用加權評分合併密集和稀疏結果。"""
        merged = {}

        # 添加密集結果
        for rank, result in enumerate(dense_results):
            point_id = str(result.id)
            if point_id not in merged:
                merged[point_id] = {
                    "id": point_id,
                    "payload": result.payload,
                    "dense_score": result.score,
                    "sparse_score": 0.0,
                    "dense_rank": rank + 1,
                    "sparse_rank": len(sparse_results) + 1,
                }
            else:
                merged[point_id]["dense_score"] = result.score
                merged[point_id]["dense_rank"] = rank + 1

        # 添加稀疏結果
        for rank, result in enumerate(sparse_results):
            point_id = str(result.id)
            if point_id not in merged:
                merged[point_id] = {
                    "id": point_id,
                    "payload": result.payload,
                    "dense_score": 0.0,
                    "sparse_score": result.score,
                    "dense_rank": len(dense_results) + 1,
                    "sparse_rank": rank + 1,
                }
            else:
                merged[point_id]["sparse_score"] = result.score
                merged[point_id]["sparse_rank"] = rank + 1

        # 計算組合分數
        for point_id, data in merged.items():
            # 加權分數組合
            data["combined_score"] = (
                dense_weight * data["dense_score"]
                + sparse_weight * data["sparse_score"]
            )

        # 按組合分數排序並限制
        sorted_results = sorted(
            merged.values(),
            key=lambda x: x["combined_score"],
            reverse=True,
        )[:limit]

        return sorted_results

    def _build_filter(
        self,
        conditions: Optional[dict[str, Any]],
    ) -> Optional[Filter]:
        """從條件字典建立 Qdrant 過濾器。"""
        if not conditions:
            return None

        must_conditions = []

        for key, value in conditions.items():
            if isinstance(value, list):
                must_conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchAny(any=value),
                    )
                )
            elif isinstance(value, dict) and "range" in value:
                must_conditions.append(
                    FieldCondition(
                        key=key,
                        range=Range(**value["range"]),
                    )
                )
            else:
                must_conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )

        return Filter(must=must_conditions) if must_conditions else None

    # ==================== 語義快取操作 ====================

    async def cache_lookup(
        self,
        query_embedding: list[float],
        threshold: float = 0.90,
        tenant_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """透過語義相似度查詢快取的回應。"""
        filter_conditions = {}
        if tenant_id:
            filter_conditions["tenant_id"] = tenant_id

        results = await self._client.search(
            collection_name=self._cache_collection,
            query_vector=query_embedding,
            limit=1,
            score_threshold=threshold,
            query_filter=self._build_filter(filter_conditions) if filter_conditions else None,
            with_payload=True,
        )

        if results:
            return {
                "id": str(results[0].id),
                "score": results[0].score,
                "payload": results[0].payload,
            }
        return None

    async def cache_store(
        self,
        query_embedding: list[float],
        payload: dict[str, Any],
    ) -> str:
        """
        在語義快取中儲存查詢-回應對。

        使用 query_hash 作為點 ID 進行去重複 - 相同的問題
        會更新同一個快取條目而不是建立重複項。
        """
        # 使用 query_hash 作為點 ID 進行去重複
        # 如果沒有提供 query_hash，回退到 UUID（正常流程不應發生）
        query_hash = payload.get("query_hash")
        if query_hash:
            # 使用雜湊的前 32 個字元作為 ID（Qdrant 支援字串 ID）
            point_id = query_hash[:32]
        else:
            point_id = str(uuid.uuid4())
            logger.warning("cache_store called without query_hash - using random UUID")

        await self._client.upsert(
            collection_name=self._cache_collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=query_embedding,
                    payload=payload,
                )
            ],
        )

        return point_id

    async def cache_invalidate(
        self,
        query_hash: Optional[str] = None,
        doc_ids: Optional[list[str]] = None,
        invalidate_all: bool = False,
    ) -> int:
        """使快取條目失效。"""
        if invalidate_all:
            # 刪除所有點
            info = await self._client.get_collection(self._cache_collection)
            count = info.points_count

            # 重新建立集合
            await self._client.delete_collection(self._cache_collection)
            await self._create_cache_collection()

            return count

        conditions = []
        if query_hash:
            conditions.append(
                FieldCondition(
                    key="query_hash",
                    match=MatchValue(value=query_hash),
                )
            )
        if doc_ids:
            # 使用這些文件的快取條目失效
            conditions.append(
                FieldCondition(
                    key="source_doc_ids",
                    match=MatchAny(any=doc_ids),
                )
            )

        if not conditions:
            return 0

        await self._client.delete(
            collection_name=self._cache_collection,
            points_selector=models.FilterSelector(
                filter=Filter(should=conditions)
            ),
        )

        # 近似計數（Qdrant 不會直接返回刪除計數）
        return len(conditions)

    async def cache_invalidate_by_version(
        self,
        index_version: Optional[str] = None,
        prompt_version: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> int:
        """
        根據版本使快取條目失效。

        階段 6：當索引或提示詞變更時啟用自動快取失效。

        Args:
            index_version: 使不匹配此索引版本的條目失效
            prompt_version: 使不匹配此提示詞版本的條目失效
            tenant_id: 限制失效到特定租戶

        Returns:
            失效的條目近似數量
        """
        # 建立不匹配當前版本的條目過濾器
        # （即應該失效的舊版本條目）
        must_not_conditions = []
        must_conditions = []

        if index_version:
            must_not_conditions.append(
                FieldCondition(
                    key="index_version",
                    match=MatchValue(value=index_version),
                )
            )

        if prompt_version:
            must_not_conditions.append(
                FieldCondition(
                    key="prompt_version",
                    match=MatchValue(value=prompt_version),
                )
            )

        if tenant_id:
            must_conditions.append(
                FieldCondition(
                    key="tenant_id",
                    match=MatchValue(value=tenant_id),
                )
            )

        if not must_not_conditions:
            logger.warning("cache_invalidate_by_version called without version filters")
            return 0

        # 獲取刪除前的計數以供報告
        info = await self._client.get_collection(self._cache_collection)
        initial_count = info.points_count

        # 刪除舊版本的條目
        # 使用 must_not 找到沒有當前版本的條目
        filter_obj = Filter(
            must=must_conditions if must_conditions else None,
            must_not=must_not_conditions,
        )

        await self._client.delete(
            collection_name=self._cache_collection,
            points_selector=models.FilterSelector(filter=filter_obj),
        )

        # 獲取刪除後的計數
        info = await self._client.get_collection(self._cache_collection)
        final_count = info.points_count

        invalidated = initial_count - final_count

        logger.info(
            f"Cache invalidation by version: removed {invalidated} entries "
            f"(index_version={index_version}, prompt_version={prompt_version})"
        )

        return invalidated

    async def get_cache_count(self) -> int:
        """
        獲取快取條目的總數。

        階段 6：由快取管理端點用於統計。

        Returns:
            快取集合中的條目總數
        """
        try:
            info = await self._client.get_collection(self._cache_collection)
            return info.points_count
        except Exception as e:
            logger.warning(f"Failed to get cache count: {e}")
            return 0

    # ==================== 社區操作 ====================

    async def upsert_community_report(
        self,
        community_id: str,
        embedding: list[float],
        payload: dict[str, Any],
    ) -> str:
        """Upsert 一個社區報告嵌入。"""
        await self._client.upsert(
            collection_name=self._communities_collection,
            points=[
                PointStruct(
                    id=community_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )
        return community_id

    async def search_communities(
        self,
        query_embedding: list[float],
        level: Optional[int] = None,
        limit: int = 10,
        score_threshold: float = 0.5,
    ) -> list[ScoredPoint]:
        """透過語義相似度搜索社區報告。"""
        filter_conditions = {}
        if level is not None:
            filter_conditions["level"] = level

        results = await self._client.search(
            collection_name=self._communities_collection,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=self._build_filter(filter_conditions) if filter_conditions else None,
            with_payload=True,
        )

        return results

    # ==================== 健康檢查 ====================

    async def health_check(self) -> dict[str, Any]:
        """檢查 Qdrant 連線健康狀態。"""
        try:
            collections = await self._client.get_collections()
            collection_names = [c.name for c in collections.collections]

            chunks_info = None
            if self._chunks_collection in collection_names:
                info = await self._client.get_collection(self._chunks_collection)
                chunks_info = {
                    "points_count": info.points_count,
                    "indexed_vectors_count": info.indexed_vectors_count,
                }

            return {
                "status": "healthy",
                "collections": collection_names,
                "chunks_collection": chunks_info,
                "connected": True,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connected": False,
            }

    # ==================== 集合管理 ====================

    async def get_collection_info(
        self,
        collection_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """獲取集合資訊。"""
        name = collection_name or self._chunks_collection
        info = await self._client.get_collection(name)

        return {
            "name": name,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status.name,
            "config": {
                "vectors": str(info.config.params.vectors),
            },
        }

    async def delete_collection(self, collection_name: str) -> bool:
        """刪除一個集合。"""
        result = await self._client.delete_collection(collection_name)
        return result


# 單例實例
qdrant_service = QdrantService()
