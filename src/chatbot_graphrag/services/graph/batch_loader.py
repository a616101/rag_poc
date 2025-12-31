"""
圖譜批次載入服務

協調實體抽取、關係抽取，以及批次載入到 NebulaGraph。

主要功能：
- 從 chunks 中抽取實體
- 抽取實體間的關係
- 合併重複的實體/關係
- 將實體和關係載入到 NebulaGraph
- 建立 chunks 與實體的關聯
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from chatbot_graphrag.models.pydantic.graph import Entity, Relation
from chatbot_graphrag.models.pydantic.ingestion import Chunk

logger = logging.getLogger(__name__)


@dataclass
class BatchLoadResult:
    """批次載入操作的結果。"""

    entities_extracted: int = 0  # 抽取的實體數
    relations_extracted: int = 0  # 抽取的關係數
    entities_loaded: int = 0  # 載入的實體數
    relations_loaded: int = 0  # 載入的關係數
    chunks_processed: int = 0  # 處理的 chunks 數
    errors: list[dict[str, Any]] = field(default_factory=list)  # 錯誤列表
    execution_time_ms: float = 0.0  # 執行時間（毫秒）
    started_at: Optional[datetime] = None  # 開始時間
    completed_at: Optional[datetime] = None  # 完成時間

    @property
    def success(self) -> bool:
        """檢查批次載入是否成功。"""
        return len(self.errors) == 0 and self.entities_loaded > 0


class GraphBatchLoader:
    """
    協調完整的圖譜建構管線。

    1. 從 chunks 中抽取實體
    2. 抽取實體間的關係
    3. 合併重複的實體/關係
    4. 將實體和關係載入到 NebulaGraph
    5. 建立 chunks 與實體的關聯
    """

    def __init__(self):
        """初始化批次載入器。"""
        self._initialized = False

    @staticmethod
    def _build_content_with_section_title(chunk: Chunk) -> str:
        """
        構建包含 section_title 的完整內容。

        用於實體和關係提取，確保 LLM 能看到完整的上下文。
        這解決了實體提取時缺少章節標題導致的資訊遺失問題。

        Args:
            chunk: Chunk 物件

        Returns:
            包含 section_title 的完整內容字串
        """
        base_content = chunk.effective_content
        section_title = chunk.metadata.section_title
        if section_title and section_title not in base_content:
            return f"## {section_title}\n\n{base_content}"
        return base_content

    async def initialize(self) -> None:
        """初始化依賴服務。"""
        if self._initialized:
            return

        from chatbot_graphrag.services.graph.entity_extractor import entity_extractor
        from chatbot_graphrag.services.graph.relation_extractor import relation_extractor
        from chatbot_graphrag.services.graph.nebula_client import nebula_client

        self._entity_extractor = entity_extractor
        self._relation_extractor = relation_extractor
        self._nebula_client = nebula_client

        # 初始化所有服務
        await self._entity_extractor.initialize()
        await self._relation_extractor.initialize()
        await self._nebula_client.initialize()

        self._initialized = True
        logger.info("圖譜批次載入器已初始化")

    async def process_chunks(
        self,
        chunks: list[Chunk],
        doc_id: str,
        concurrency: int = 3,
        enable_relations: bool = True,
    ) -> BatchLoadResult:
        """
        透過完整的圖譜建構管線處理 chunks。

        Args:
            chunks: 文件 chunks 列表
            doc_id: 文件 ID
            concurrency: 最大並發 LLM 呼叫數
            enable_relations: 是否抽取關係

        Returns:
            包含統計資訊的 BatchLoadResult
        """
        if not self._initialized:
            await self.initialize()

        import time
        start_time = time.time()
        result = BatchLoadResult(
            started_at=datetime.utcnow(),
            chunks_processed=len(chunks),
        )

        if not chunks:
            result.completed_at = datetime.utcnow()
            return result

        try:
            # 步驟 1：從所有 chunks 中抽取實體
            logger.info(f"Extracting entities from {len(chunks)} chunks...")
            chunk_dicts = [
                {
                    "id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "content": self._build_content_with_section_title(chunk),
                }
                for chunk in chunks
            ]

            entities_by_chunk = await self._entity_extractor.extract_entities_batch(
                chunk_dicts, concurrency=concurrency
            )

            # 展平並合併實體
            all_entities: list[Entity] = []
            for chunk_entities in entities_by_chunk.values():
                all_entities.extend(chunk_entities)

            merged_entities = self._entity_extractor.merge_entities(all_entities)
            result.entities_extracted = len(merged_entities)
            logger.info(f"Extracted {len(all_entities)} entities, merged to {len(merged_entities)}")

            # 步驟 2：抽取關係（如果啟用）
            all_relations: list[Relation] = []
            if enable_relations and len(merged_entities) >= 2:
                logger.info("Extracting relations...")

                # 為關係抽取建構實體映射
                entity_id_map = {e.id: e for e in merged_entities}

                # 處理每個 chunk 及其實體
                chunks_with_entities = []
                for chunk in chunks:
                    chunk_entity_ids = entities_by_chunk.get(chunk.id, [])
                    chunk_entities = [
                        entity_id_map.get(e.id)
                        for e in chunk_entity_ids
                        if e.id in entity_id_map
                    ]
                    # 過濾掉 None 值
                    chunk_entities = [e for e in chunk_entities if e is not None]

                    if len(chunk_entities) >= 2:
                        chunks_with_entities.append({
                            "chunk_id": chunk.id,
                            "doc_id": chunk.doc_id,
                            "content": self._build_content_with_section_title(chunk),
                            "entities": chunk_entities,
                        })

                if chunks_with_entities:
                    relations_by_chunk = await self._relation_extractor.extract_relations_batch(
                        chunks_with_entities, concurrency=concurrency
                    )

                    for chunk_relations in relations_by_chunk.values():
                        all_relations.extend(chunk_relations)

                    # 合併重複的關係
                    merged_relations = self._relation_extractor.merge_relations(all_relations)
                    result.relations_extracted = len(merged_relations)
                    all_relations = merged_relations
                    logger.info(f"Extracted {len(all_relations)} relations")

            # 步驟 3：將實體載入到 NebulaGraph
            logger.info(f"Loading {len(merged_entities)} entities into NebulaGraph...")
            entity_load_count = 0
            for entity in merged_entities:
                try:
                    await self._nebula_client.upsert_entity(entity)
                    entity_load_count += 1
                except Exception as e:
                    result.errors.append({
                        "stage": "entity_load",
                        "entity_id": entity.id,
                        "entity_name": entity.name,
                        "error": str(e),
                    })
                    logger.warning(f"Failed to load entity {entity.name}: {e}")

            result.entities_loaded = entity_load_count
            logger.info(f"Loaded {entity_load_count} entities")

            # 步驟 4：將關係載入到 NebulaGraph
            if all_relations:
                logger.info(f"Loading {len(all_relations)} relations into NebulaGraph...")
                relation_load_count = 0
                for relation in all_relations:
                    try:
                        await self._nebula_client.upsert_relation(relation)
                        relation_load_count += 1
                    except Exception as e:
                        result.errors.append({
                            "stage": "relation_load",
                            "relation_id": relation.id,
                            "source_id": relation.source_id,
                            "target_id": relation.target_id,
                            "error": str(e),
                        })
                        logger.warning(f"Failed to load relation {relation.id}: {e}")

                result.relations_loaded = relation_load_count
                logger.info(f"Loaded {relation_load_count} relations")

            # 步驟 5：建立 chunks 與實體的關聯（建立 chunk 頂點和 mentions 邊）
            await self._link_chunks_to_entities(
                chunks, entities_by_chunk, doc_id, result
            )

            # 步驟 6：使用 entity_ids 更新 Qdrant chunks（啟用 GraphRAG 檢索）
            await self._update_chunk_entity_ids_in_qdrant(
                entities_by_chunk, result
            )

        except Exception as e:
            logger.error(f"Batch load failed: {e}")
            result.errors.append({
                "stage": "batch_load",
                "error": str(e),
            })

        result.completed_at = datetime.utcnow()
        result.execution_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Batch load completed: {result.entities_loaded} entities, "
            f"{result.relations_loaded} relations, "
            f"{len(result.errors)} errors, "
            f"{result.execution_time_ms:.1f}ms"
        )

        return result

    async def _link_chunks_to_entities(
        self,
        chunks: list[Chunk],
        entities_by_chunk: dict[str, list[Entity]],
        doc_id: str,
        result: BatchLoadResult,
    ) -> None:
        """建立 chunk 頂點並將它們連結到實體。"""
        from chatbot_graphrag.core.constants import NEBULA_CHUNK_TAG

        for chunk in chunks:
            chunk_entities = entities_by_chunk.get(chunk.id, [])
            if not chunk_entities:
                continue

            chunk_vid = f"chunk_{chunk.id}"

            try:
                # 建立 chunk 頂點
                # 為 nGQL 字串跳脫特殊字元：先跳脫反斜線，然後是引號和換行
                content_preview = chunk.content[:200]
                content_preview = content_preview.replace('\\', '\\\\')  # Escape backslashes first
                content_preview = content_preview.replace('"', '\\"')    # Escape double quotes
                content_preview = content_preview.replace('\n', ' ')     # Replace newlines with space
                content_preview = content_preview.replace('\r', ' ')     # Replace carriage returns
                content_preview = content_preview.replace('\t', ' ')     # Replace tabs
                chunk_query = f"""
                INSERT VERTEX {NEBULA_CHUNK_TAG} (
                    doc_id, chunk_type, content_preview, position, created_at
                ) VALUES "{chunk_vid}": (
                    "{doc_id}",
                    "{chunk.metadata.chunk_type.value}",
                    "{content_preview}",
                    {chunk.metadata.position_in_doc},
                    datetime()
                );
                """
                await self._nebula_client.execute(chunk_query)

                # 將 chunk 連結到其實體
                entity_vids = [e.id for e in chunk_entities]
                await self._nebula_client.link_chunk_to_entities(chunk_vid, entity_vids)

            except Exception as e:
                result.errors.append({
                    "stage": "chunk_link",
                    "chunk_id": chunk.id,
                    "error": str(e),
                })
                logger.warning(f"Failed to link chunk {chunk.id}: {e}")

    async def _update_chunk_entity_ids_in_qdrant(
        self,
        entities_by_chunk: dict[str, list[Entity]],
        result: BatchLoadResult,
    ) -> None:
        """
        使用 entity_ids 更新 Qdrant chunk 元資料。

        這使 GraphRAG 檢索能夠直接從搜尋結果中存取實體 ID。

        Args:
            entities_by_chunk: chunk_id 到實體的映射
            result: 用於追蹤錯誤的 BatchLoadResult
        """
        from chatbot_graphrag.services.vector import qdrant_service

        await qdrant_service.initialize()

        updates = []
        for chunk_id, entities in entities_by_chunk.items():
            entity_ids = [e.id for e in entities]
            if entity_ids:
                updates.append({
                    "chunk_id": chunk_id,
                    "metadata": {"entity_ids": entity_ids},
                })

        if not updates:
            return

        try:
            updated_count = await qdrant_service.update_chunks_metadata_batch(updates)
            logger.info(f"Updated {updated_count}/{len(updates)} chunks with entity_ids in Qdrant")
        except Exception as e:
            result.errors.append({
                "stage": "qdrant_entity_ids_update",
                "error": str(e),
            })
            logger.warning(f"Failed to update entity_ids in Qdrant: {e}")

    async def process_document(
        self,
        chunks: list[Chunk],
        doc_id: str,
        **kwargs,
    ) -> BatchLoadResult:
        """
        處理單一文件的 chunks。

        process_chunks 的別名，帶有文件上下文。
        """
        return await self.process_chunks(chunks, doc_id, **kwargs)

    async def rebuild_graph_for_document(
        self,
        doc_id: str,
        chunks: list[Chunk],
        **kwargs,
    ) -> BatchLoadResult:
        """
        重建文件的圖譜（刪除現有的，然後重新建立）。

        透過先移除文件的所有現有實體、關係和 chunk 連結，
        然後再處理新的 chunks，確保乾淨的重建。

        Args:
            doc_id: 文件 ID
            chunks: 要處理的新 chunks

        Returns:
            BatchLoadResult
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Rebuilding graph for document {doc_id}")

        # 步驟 1：刪除此文件的現有實體和關係
        try:
            deleted_count = await self._nebula_client.delete_entities_by_doc_id(doc_id)
            logger.info(f"Deleted {deleted_count} existing entities for doc {doc_id}")
        except Exception as e:
            logger.warning(f"Failed to delete existing entities for {doc_id}: {e}")
            # 即使刪除失敗也繼續重建

        # 步驟 2：處理新的 chunks
        return await self.process_chunks(chunks, doc_id, **kwargs)


# 單例實例
graph_batch_loader = GraphBatchLoader()
