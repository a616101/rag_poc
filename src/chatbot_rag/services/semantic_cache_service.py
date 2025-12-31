"""
語意快取服務模組

此模組提供基於 Qdrant 向量資料庫的語意快取功能，用於：
- 快取 RAG 問答對，避免重複 LLM 呼叫
- 根據問題語意相似度查詢快取
- 在文件更新時自動清除相關快取
"""

import hashlib
import time
from typing import Any, Optional, TypedDict

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    VectorParams,
)

from chatbot_rag.core.config import settings
from chatbot_rag.services.embedding_service import embedding_service


# 高信心度閾值：高於此閾值直接使用，無需跨語言搜尋
HIGH_CONFIDENCE_THRESHOLD = 0.95


class CacheHit(TypedDict):
    """快取命中結果結構"""

    question: str  # 快取的問題
    answer: str  # 快取的回答
    answer_meta: dict  # 回答的 metadata
    score: float  # 相似度分數
    cache_id: str  # 快取項目 ID
    created_at: float  # 建立時間
    access_count: int  # 存取次數


class SemanticCacheService:
    """
    語意快取服務類別

    使用 Qdrant 向量資料庫儲存問答對，
    根據問題的語意相似度來查詢和返回快取的回答。
    """

    def __init__(self):
        """初始化語意快取服務"""
        self.client: Optional[QdrantClient] = None
        self.collection_name = settings.semantic_cache_collection_name
        self.similarity_threshold = settings.semantic_cache_similarity_threshold
        self.ttl_seconds = settings.semantic_cache_ttl_seconds
        self.enabled = settings.semantic_cache_enabled

    def connect(self) -> QdrantClient:
        """
        連接到 Qdrant 資料庫

        Returns:
            QdrantClient: 已連接的 Qdrant 客戶端實例
        """
        if self.client is None:
            try:
                logger.info(f"[SemanticCache] Connecting to Qdrant at {settings.qdrant_url}")
                if settings.qdrant_api_key:
                    self.client = QdrantClient(
                        url=settings.qdrant_url,
                        api_key=settings.qdrant_api_key,
                    )
                else:
                    self.client = QdrantClient(url=settings.qdrant_url)
                logger.info("[SemanticCache] Successfully connected to Qdrant")
            except Exception as e:
                logger.error(f"[SemanticCache] Failed to connect to Qdrant: {e}")
                raise
        return self.client

    def ensure_collection(self, recreate: bool = False) -> bool:
        """
        確保快取集合存在

        Args:
            recreate: 是否重新建立集合

        Returns:
            bool: 集合是否被建立
        """
        try:
            client = self.connect()
            collections = client.get_collections().collections
            collection_exists = any(
                col.name == self.collection_name for col in collections
            )

            if collection_exists and recreate:
                logger.info(f"[SemanticCache] Deleting existing collection: {self.collection_name}")
                client.delete_collection(collection_name=self.collection_name)
                collection_exists = False

            if not collection_exists:
                logger.info(
                    f"[SemanticCache] Creating collection: {self.collection_name} "
                    f"(dimension: {settings.embedding_dimension})"
                )
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"[SemanticCache] Collection created: {self.collection_name}")
                return True
            else:
                logger.debug(f"[SemanticCache] Collection already exists: {self.collection_name}")
                return False

        except Exception as e:
            logger.error(f"[SemanticCache] Failed to ensure collection: {e}")
            raise

    def _generate_cache_id(self, question: str) -> int:
        """
        根據問題生成快取 ID

        Args:
            question: 正規化後的問題

        Returns:
            int: 64 位元整數 ID
        """
        hash_input = question.strip().lower()
        hash_hex = hashlib.md5(hash_input.encode("utf-8")).hexdigest()
        return int(hash_hex[:16], 16)

    def lookup(
        self,
        question: str,
        *,
        user_language: Optional[str] = None,
        search_all_languages: bool = True,
    ) -> Optional[CacheHit]:
        """
        查詢相似問題的快取（支援跨語言搜尋）

        策略：
        1. 首先進行全域搜尋
        2. 如果找到高信心度結果（>= HIGH_CONFIDENCE_THRESHOLD），直接返回
        3. 如果信心度不足但超過基本閾值，仍然返回

        Args:
            question: 正規化後的問題
            user_language: 使用者語言（用於 metadata 記錄）
            search_all_languages: 是否跨語言搜尋（預設 True）

        Returns:
            CacheHit: 如果找到相似度 >= 閾值的快取，返回快取結果；否則返回 None
        """
        if not self.enabled:
            logger.debug("[SemanticCache] Cache is disabled, skipping lookup")
            return None

        try:
            # 確保集合存在
            self.ensure_collection()

            # 生成問題的 embedding
            question_embedding = embedding_service.embed_text(question)
            if not question_embedding:
                logger.warning("[SemanticCache] Failed to generate question embedding")
                return None

            client = self.connect()

            # 進行相似度搜尋（搜尋多個結果以便選擇最佳）
            search_result = client.query_points(
                collection_name=self.collection_name,
                query=question_embedding,
                limit=3,  # 搜尋多個結果
                score_threshold=self.similarity_threshold,
            )

            if not search_result.points:
                logger.debug(
                    f"[SemanticCache] No cache hit for question (threshold={self.similarity_threshold})"
                )
                return None

            # 選擇最佳結果
            best_point = None
            best_score = 0.0

            for point in search_result.points:
                payload = point.payload or {}

                # 檢查 TTL
                if self.ttl_seconds > 0:
                    created_at = payload.get("created_at", 0)
                    if time.time() - created_at > self.ttl_seconds:
                        continue

                # 語言匹配加分
                cached_lang = payload.get("user_language", "")
                lang_bonus = 0.01 if user_language and cached_lang == user_language else 0

                effective_score = point.score + lang_bonus

                if effective_score > best_score:
                    best_score = effective_score
                    best_point = point

            if not best_point:
                logger.debug(
                    f"[SemanticCache] All candidates expired or filtered out"
                )
                return None

            payload = best_point.payload or {}

            cache_hit: CacheHit = {
                "question": payload.get("question", ""),
                "answer": payload.get("answer", ""),
                "answer_meta": payload.get("answer_meta", {}),
                "score": best_point.score,
                "cache_id": str(best_point.id),
                "created_at": payload.get("created_at", 0),
                "access_count": payload.get("access_count", 0),
            }

            # 記錄是否跨語言命中
            cached_lang = payload.get("user_language", "")
            cross_lang_hit = user_language and cached_lang and cached_lang != user_language

            logger.info(
                f"[SemanticCache] Cache hit! score={best_point.score:.4f}, "
                f"cache_id={best_point.id}, cross_lang={cross_lang_hit}"
            )

            # 更新存取統計（異步，不阻塞回應）
            self._update_access_stats_async(str(best_point.id))

            return cache_hit

        except Exception as e:
            logger.error(f"[SemanticCache] Lookup failed: {e}")
            return None

    def _update_access_stats_async(self, cache_id: str) -> None:
        """
        異步更新快取存取統計

        Args:
            cache_id: 快取項目 ID
        """
        try:
            client = self.connect()
            # 取得當前 payload
            points = client.retrieve(
                collection_name=self.collection_name,
                ids=[int(cache_id)],
                with_payload=True,
            )
            if points:
                payload = points[0].payload or {}
                access_count = payload.get("access_count", 0) + 1
                # 更新 payload
                client.set_payload(
                    collection_name=self.collection_name,
                    payload={
                        "last_accessed_at": time.time(),
                        "access_count": access_count,
                    },
                    points=[int(cache_id)],
                )
        except Exception as e:
            # 不阻塞主流程
            logger.debug(f"[SemanticCache] Failed to update access stats: {e}")

    def store(
        self,
        question: str,
        answer: str,
        *,
        original_question: Optional[str] = None,
        answer_meta: Optional[dict] = None,
        source_filenames: Optional[list[str]] = None,
        user_language: str = "zh-hant",
        intent: str = "",
    ) -> Optional[str]:
        """
        儲存問答對到快取（支援雙向量儲存）

        當 original_question 與 question（正規化問題）不同時，
        會同時建立兩個快取條目，提高語意相似問題的命中率。

        Args:
            question: 正規化後的問題（主要儲存向量）
            answer: 生成的回答
            original_question: 原始問題（可選，若提供且與 question 不同則額外儲存）
            answer_meta: 回答的 metadata
            source_filenames: 來源文件名稱列表（用於快取清除）
            user_language: 使用者語言
            intent: 問題意圖

        Returns:
            str: 主快取項目 ID，如果儲存失敗則返回 None
        """
        if not self.enabled:
            logger.debug("[SemanticCache] Cache is disabled, skipping store")
            return None

        try:
            # 確保集合存在
            self.ensure_collection()

            now = time.time()
            base_payload = {
                "answer": answer,
                "answer_meta": answer_meta or {},
                "source_filenames": source_filenames or [],
                "user_language": user_language,
                "language_variants": [user_language],  # 追蹤所有語言版本
                "intent": intent,
                "created_at": now,
                "last_accessed_at": now,
                "access_count": 0,
                "ttl_seconds": self.ttl_seconds,
            }

            points_to_store = []
            primary_cache_id = None

            # 1. 儲存主要問題（正規化後的問題）
            question_embedding = embedding_service.embed_text(question)
            if question_embedding:
                cache_id = self._generate_cache_id(question)
                question_hash = hashlib.md5(question.strip().lower().encode()).hexdigest()

                payload = {
                    **base_payload,
                    "question": question,
                    "question_hash": question_hash,
                    "original_question": original_question or question,
                    "is_normalized": True,
                }

                points_to_store.append(
                    PointStruct(
                        id=cache_id,
                        vector=question_embedding,
                        payload=payload,
                    )
                )
                primary_cache_id = cache_id
                logger.debug(f"[SemanticCache] Prepared normalized entry: {question[:50]}...")

            # 2. 若原始問題與正規化問題不同，額外儲存原始問題向量
            if original_question and original_question.strip().lower() != question.strip().lower():
                original_embedding = embedding_service.embed_text(original_question)
                if original_embedding:
                    original_cache_id = self._generate_cache_id(original_question)
                    original_hash = hashlib.md5(
                        original_question.strip().lower().encode()
                    ).hexdigest()

                    original_payload = {
                        **base_payload,
                        "question": original_question,
                        "question_hash": original_hash,
                        "original_question": original_question,
                        "normalized_question": question,
                        "is_normalized": False,
                    }

                    points_to_store.append(
                        PointStruct(
                            id=original_cache_id,
                            vector=original_embedding,
                            payload=original_payload,
                        )
                    )
                    logger.debug(
                        f"[SemanticCache] Prepared original entry: {original_question[:50]}..."
                    )

            # 執行儲存
            if points_to_store:
                client = self.connect()
                client.upsert(
                    collection_name=self.collection_name,
                    points=points_to_store,
                )
                logger.info(
                    f"[SemanticCache] Stored {len(points_to_store)} cache entries, "
                    f"primary_id={primary_cache_id}, sources={source_filenames}"
                )
                return str(primary_cache_id) if primary_cache_id else None
            else:
                logger.warning("[SemanticCache] Failed to generate any embeddings")
                return None

        except Exception as e:
            logger.error(f"[SemanticCache] Store failed: {e}")
            return None

    def invalidate_by_filenames(self, filenames: list[str]) -> int:
        """
        根據來源文件清除相關快取

        當文件更新時，清除所有使用該文件的快取項目。

        Args:
            filenames: 要清除快取的文件名稱列表

        Returns:
            int: 被清除的快取項目數量
        """
        if not filenames:
            return 0

        try:
            client = self.connect()

            # 檢查集合是否存在
            collections = client.get_collections().collections
            if not any(col.name == self.collection_name for col in collections):
                logger.debug("[SemanticCache] Collection does not exist, nothing to invalidate")
                return 0

            # 使用 scroll 找出包含這些檔案的快取項目
            # Qdrant 的 MatchAny 可以匹配陣列中的任意元素
            filter_condition = Filter(
                should=[
                    FieldCondition(
                        key="source_filenames",
                        match=MatchAny(any=filenames),
                    )
                ]
            )

            # 先查詢符合條件的點
            results, _ = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=1000,
                with_payload=False,
            )

            if not results:
                logger.debug(
                    f"[SemanticCache] No cache entries found for files: {filenames}"
                )
                return 0

            # 刪除這些點
            point_ids = [point.id for point in results]
            client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids,
            )

            logger.info(
                f"[SemanticCache] Invalidated {len(point_ids)} cache entries "
                f"for files: {filenames}"
            )

            return len(point_ids)

        except Exception as e:
            logger.error(f"[SemanticCache] Invalidation failed: {e}")
            return 0

    def clear_all(self) -> bool:
        """
        清除所有快取

        Returns:
            bool: 是否成功清除
        """
        try:
            client = self.connect()

            # 檢查集合是否存在
            collections = client.get_collections().collections
            if not any(col.name == self.collection_name for col in collections):
                logger.debug("[SemanticCache] Collection does not exist")
                return True

            # 刪除並重建集合
            client.delete_collection(collection_name=self.collection_name)
            logger.info(f"[SemanticCache] Cleared all cache entries")

            return True

        except Exception as e:
            logger.error(f"[SemanticCache] Clear failed: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """
        取得快取統計資訊

        Returns:
            dict: 快取統計資訊
        """
        try:
            client = self.connect()

            # 檢查集合是否存在
            collections = client.get_collections().collections
            if not any(col.name == self.collection_name for col in collections):
                return {
                    "enabled": self.enabled,
                    "collection_exists": False,
                    "total_entries": 0,
                    "similarity_threshold": self.similarity_threshold,
                    "ttl_seconds": self.ttl_seconds,
                }

            collection_info = client.get_collection(
                collection_name=self.collection_name
            )

            return {
                "enabled": self.enabled,
                "collection_exists": True,
                "collection_name": self.collection_name,
                "total_entries": collection_info.points_count,
                "similarity_threshold": self.similarity_threshold,
                "ttl_seconds": self.ttl_seconds,
                "status": str(collection_info.status),
            }

        except Exception as e:
            logger.error(f"[SemanticCache] Get stats failed: {e}")
            return {
                "enabled": self.enabled,
                "error": str(e),
            }


# 建立全域語意快取服務實例
semantic_cache_service = SemanticCacheService()
