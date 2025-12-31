"""
OpenSearch 服務

提供 OpenSearch 的非同步介面，用於全文搜尋功能。

主要功能：
- BM25 全文搜尋
- 中文語言分析（CJK 分詞）
- 模糊匹配
- 高亮顯示
- 批次索引
- More Like This 相似文件搜尋
"""

import asyncio
import logging
from typing import Any, Optional

from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import NotFoundError, RequestError

from chatbot_graphrag.core.config import settings

logger = logging.getLogger(__name__)


class OpenSearchService:
    """
    OpenSearch 全文搜尋服務。

    提供：
    - 基於 BM25 的全文搜尋
    - 中文語言分析
    - 模糊匹配
    - 高亮顯示
    """

    _instance: Optional["OpenSearchService"] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> "OpenSearchService":
        """單例模式以重複使用客戶端。"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化 OpenSearch 服務。"""
        if self._initialized:
            return

        self._client: Optional[AsyncOpenSearch] = None
        self._index_name = settings.opensearch_index_chunks
        self._initialized = True

    async def initialize(self) -> None:
        """初始化 OpenSearch 客戶端和索引。"""
        if self._client is not None:
            return

        self._client = AsyncOpenSearch(
            hosts=[settings.opensearch_url],
            http_compress=True,
            timeout=settings.opensearch_timeout,
        )

        logger.info(f"OpenSearch client initialized: {settings.opensearch_url}")

        # 確保索引存在
        await self._ensure_index()

    async def close(self) -> None:
        """關閉 OpenSearch 客戶端。"""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("OpenSearch client closed")

    async def _ensure_index(self) -> None:
        """確保 chunks 索引存在並具有正確的映射。"""
        try:
            exists = await self._client.indices.exists(index=self._index_name)
            if not exists:
                await self._create_index()
        except Exception as e:
            logger.error(f"Error checking/creating index: {e}")
            raise

    async def _create_index(self) -> None:
        """建立帶有中文分析器的 chunks 索引。

        使用 CJK 分析器（內建），為中文、日文和韓文文本提供 bi-gram 分詞。
        這不需要額外的插件如 IK Analysis。
        """
        index_settings = {
            "settings": {
                "number_of_shards": 2,
                "number_of_replicas": 1,
                "analysis": {
                    "analyzer": {
                        "chinese_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "cjk_width", "cjk_bigram"],
                        },
                        "chinese_search_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "cjk_width", "cjk_bigram"],
                        },
                    },
                },
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "doc_type": {"type": "keyword"},
                    "chunk_type": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "analyzer": "chinese_analyzer",
                        "search_analyzer": "chinese_search_analyzer",
                        "fields": {
                            "raw": {"type": "keyword"},
                        },
                    },
                    "contextual_content": {
                        "type": "text",
                        "analyzer": "chinese_analyzer",
                        "search_analyzer": "chinese_search_analyzer",
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "chinese_analyzer",
                        "boost": 2.0,
                    },
                    "section_title": {
                        "type": "text",
                        "analyzer": "chinese_analyzer",
                        "boost": 1.5,
                    },
                    "language": {"type": "keyword"},
                    "acl_groups": {"type": "keyword"},
                    "tenant_id": {"type": "keyword"},
                    "entity_names": {
                        "type": "text",
                        "analyzer": "chinese_analyzer",
                    },
                    "tags": {"type": "keyword"},
                    "department": {"type": "keyword"},
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"},
                },
            },
        }

        try:
            await self._client.indices.create(
                index=self._index_name,
                body=index_settings,
            )
            logger.info(f"Created OpenSearch index: {self._index_name}")
        except RequestError as e:
            if "resource_already_exists_exception" in str(e):
                logger.info(f"Index {self._index_name} already exists")
            else:
                raise

    # ==================== 索引操作 ====================

    async def index_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        content: str,
        metadata: dict[str, Any],
    ) -> bool:
        """索引單一 chunk。"""
        document = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "content": content,
            **metadata,
        }

        try:
            await self._client.index(
                index=self._index_name,
                id=chunk_id,
                body=document,
                refresh=False,  # 不等待重新整理
            )
            return True
        except Exception as e:
            logger.error(f"Error indexing chunk {chunk_id}: {e}")
            return False

    async def index_chunks_bulk(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> tuple[int, int]:
        """批次索引 chunks。"""
        success_count = 0
        error_count = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # 準備批次動作
            actions = []
            for chunk in batch:
                actions.append({
                    "index": {
                        "_index": self._index_name,
                        "_id": chunk["chunk_id"],
                    }
                })
                actions.append({
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "content": chunk["content"],
                    **chunk.get("metadata", {}),
                })

            try:
                # 使用批次 API
                response = await self._client.bulk(
                    body=actions,
                    refresh=False,
                )

                # 計算成功/錯誤數
                for item in response.get("items", []):
                    if "error" in item.get("index", {}):
                        error_count += 1
                        logger.warning(f"Bulk index error: {item['index']['error']}")
                    else:
                        success_count += 1

            except Exception as e:
                logger.error(f"Bulk indexing error: {e}")
                error_count += len(batch)

        # 批次操作後重新整理索引
        await self._client.indices.refresh(index=self._index_name)

        return success_count, error_count

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """刪除文件的所有 chunks。"""
        query = {
            "query": {
                "term": {"doc_id": doc_id}
            }
        }

        try:
            response = await self._client.delete_by_query(
                index=self._index_name,
                body=query,
                refresh=True,
            )
            return response.get("deleted", 0)
        except Exception as e:
            logger.error(f"Error deleting chunks for doc {doc_id}: {e}")
            return 0

    # ==================== 搜尋操作 ====================

    async def search(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
        filters: Optional[dict[str, Any]] = None,
        highlight: bool = True,
        min_score: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        使用 BM25 評分的全文搜尋。

        Args:
            query: 搜尋查詢文字
            limit: 返回的最大結果數
            offset: 分頁偏移量
            filters: 欄位過濾器（doc_type, acl_groups 等）
            highlight: 是否返回高亮片段
            min_score: 最低分數閾值

        Returns:
            帶有分數的匹配 chunks 列表
        """
        # 建構查詢
        search_query = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "content^1.0",
                                "contextual_content^0.8",
                                "title^2.0",
                                "section_title^1.5",
                                "entity_names^1.2",
                            ],
                            "type": "best_fields",
                            "fuzziness": "AUTO",
                        }
                    }
                ],
                "filter": self._build_filters(filters),
            }
        }

        # 建構請求主體
        body = {
            "query": search_query,
            "from": offset,
            "size": limit,
            "min_score": min_score,
            "_source": True,
        }

        # 添加高亮
        if highlight:
            body["highlight"] = {
                "fields": {
                    "content": {
                        "fragment_size": 200,
                        "number_of_fragments": 3,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                    },
                    "contextual_content": {
                        "fragment_size": 200,
                        "number_of_fragments": 2,
                    },
                },
            }

        try:
            response = await self._client.search(
                index=self._index_name,
                body=body,
            )

            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "chunk_id": hit["_id"],
                    "score": hit["_score"],
                    **hit["_source"],
                }
                if "highlight" in hit:
                    result["highlights"] = hit["highlight"]
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    async def search_phrase(
        self,
        phrase: str,
        limit: int = 20,
        slop: int = 2,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        短語搜尋，用於精確或近似精確匹配。

        Args:
            phrase: 要搜尋的短語
            limit: 最大結果數
            slop: 詞語可移動的位置數
            filters: 欄位過濾器

        Returns:
            匹配 chunks 列表
        """
        search_query = {
            "bool": {
                "must": [
                    {
                        "match_phrase": {
                            "content": {
                                "query": phrase,
                                "slop": slop,
                            }
                        }
                    }
                ],
                "filter": self._build_filters(filters),
            }
        }

        body = {
            "query": search_query,
            "size": limit,
            "_source": True,
        }

        try:
            response = await self._client.search(
                index=self._index_name,
                body=body,
            )

            return [
                {
                    "chunk_id": hit["_id"],
                    "score": hit["_score"],
                    **hit["_source"],
                }
                for hit in response["hits"]["hits"]
            ]

        except Exception as e:
            logger.error(f"Phrase search error: {e}")
            return []

    async def search_entities(
        self,
        entity_names: list[str],
        limit: int = 20,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        搜尋包含特定實體的 chunks。

        Args:
            entity_names: 要搜尋的實體名稱列表
            limit: 最大結果數
            filters: 欄位過濾器

        Returns:
            匹配 chunks 列表
        """
        search_query = {
            "bool": {
                "should": [
                    {"match": {"entity_names": name}}
                    for name in entity_names
                ],
                "minimum_should_match": 1,
                "filter": self._build_filters(filters),
            }
        }

        body = {
            "query": search_query,
            "size": limit,
            "_source": True,
        }

        try:
            response = await self._client.search(
                index=self._index_name,
                body=body,
            )

            return [
                {
                    "chunk_id": hit["_id"],
                    "score": hit["_score"],
                    **hit["_source"],
                }
                for hit in response["hits"]["hits"]
            ]

        except Exception as e:
            logger.error(f"Entity search error: {e}")
            return []

    def _build_filters(
        self,
        filters: Optional[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """建構 OpenSearch 過濾子句。"""
        if not filters:
            return []

        filter_clauses = []

        for field, value in filters.items():
            if isinstance(value, list):
                filter_clauses.append({
                    "terms": {field: value}
                })
            elif isinstance(value, dict) and "range" in value:
                filter_clauses.append({
                    "range": {field: value["range"]}
                })
            else:
                filter_clauses.append({
                    "term": {field: value}
                })

        return filter_clauses

    # ==================== 聚合操作 ====================

    async def get_doc_type_counts(
        self,
        filters: Optional[dict[str, Any]] = None,
    ) -> dict[str, int]:
        """按文件類型取得 chunk 計數。"""
        body = {
            "size": 0,
            "query": {
                "bool": {
                    "filter": self._build_filters(filters),
                }
            } if filters else {"match_all": {}},
            "aggs": {
                "doc_types": {
                    "terms": {
                        "field": "doc_type",
                        "size": 50,
                    }
                }
            },
        }

        try:
            response = await self._client.search(
                index=self._index_name,
                body=body,
            )

            return {
                bucket["key"]: bucket["doc_count"]
                for bucket in response["aggregations"]["doc_types"]["buckets"]
            }

        except Exception as e:
            logger.error(f"Aggregation error: {e}")
            return {}

    async def get_department_counts(
        self,
        filters: Optional[dict[str, Any]] = None,
    ) -> dict[str, int]:
        """按部門取得 chunk 計數。"""
        body = {
            "size": 0,
            "query": {
                "bool": {
                    "filter": self._build_filters(filters),
                }
            } if filters else {"match_all": {}},
            "aggs": {
                "departments": {
                    "terms": {
                        "field": "department",
                        "size": 100,
                    }
                }
            },
        }

        try:
            response = await self._client.search(
                index=self._index_name,
                body=body,
            )

            return {
                bucket["key"]: bucket["doc_count"]
                for bucket in response["aggregations"]["departments"]["buckets"]
            }

        except Exception as e:
            logger.error(f"Aggregation error: {e}")
            return {}

    # ==================== More Like This 相似搜尋 ====================

    async def more_like_this(
        self,
        chunk_id: str,
        limit: int = 10,
        min_term_freq: int = 1,
        min_doc_freq: int = 1,
    ) -> list[dict[str, Any]]:
        """
        使用 More Like This 查詢找到相似的 chunks。

        Args:
            chunk_id: 來源 chunk ID
            limit: 最大結果數
            min_term_freq: 來源中的最低詞頻
            min_doc_freq: 最低文件頻率

        Returns:
            相似 chunks 列表
        """
        body = {
            "query": {
                "more_like_this": {
                    "fields": ["content", "contextual_content"],
                    "like": [
                        {
                            "_index": self._index_name,
                            "_id": chunk_id,
                        }
                    ],
                    "min_term_freq": min_term_freq,
                    "min_doc_freq": min_doc_freq,
                    "max_query_terms": 25,
                }
            },
            "size": limit,
            "_source": True,
        }

        try:
            response = await self._client.search(
                index=self._index_name,
                body=body,
            )

            return [
                {
                    "chunk_id": hit["_id"],
                    "score": hit["_score"],
                    **hit["_source"],
                }
                for hit in response["hits"]["hits"]
                if hit["_id"] != chunk_id  # Exclude source
            ]

        except Exception as e:
            logger.error(f"MLT error: {e}")
            return []

    # ==================== 健康檢查 ====================

    async def health_check(self) -> dict[str, Any]:
        """檢查 OpenSearch 連接健康狀態。"""
        try:
            info = await self._client.info()
            cluster_health = await self._client.cluster.health()

            index_stats = None
            try:
                stats = await self._client.indices.stats(index=self._index_name)
                index_stats = {
                    "docs_count": stats["indices"][self._index_name]["total"]["docs"]["count"],
                    "size_bytes": stats["indices"][self._index_name]["total"]["store"]["size_in_bytes"],
                }
            except NotFoundError:
                index_stats = {"status": "index_not_found"}

            return {
                "status": "healthy",
                "cluster_name": info["cluster_name"],
                "cluster_status": cluster_health["status"],
                "index": self._index_name,
                "index_stats": index_stats,
                "connected": True,
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connected": False,
            }

    # ==================== 索引管理 ====================

    async def refresh_index(self) -> bool:
        """重新整理索引以使最近的變更可搜尋。"""
        try:
            await self._client.indices.refresh(index=self._index_name)
            return True
        except Exception as e:
            logger.error(f"Refresh error: {e}")
            return False

    async def get_index_info(self) -> dict[str, Any]:
        """取得索引資訊。"""
        try:
            settings = await self._client.indices.get_settings(index=self._index_name)
            mappings = await self._client.indices.get_mapping(index=self._index_name)
            stats = await self._client.indices.stats(index=self._index_name)

            return {
                "name": self._index_name,
                "settings": settings[self._index_name]["settings"],
                "mappings": mappings[self._index_name]["mappings"],
                "docs_count": stats["indices"][self._index_name]["total"]["docs"]["count"],
                "size_bytes": stats["indices"][self._index_name]["total"]["store"]["size_in_bytes"],
            }
        except Exception as e:
            logger.error(f"Get index info error: {e}")
            return {"error": str(e)}

    async def delete_index(self) -> bool:
        """刪除索引（謹慎使用）。"""
        try:
            await self._client.indices.delete(index=self._index_name)
            logger.warning(f"Deleted OpenSearch index: {self._index_name}")
            return True
        except Exception as e:
            logger.error(f"Delete index error: {e}")
            return False


# 單例實例
opensearch_service = OpenSearchService()
