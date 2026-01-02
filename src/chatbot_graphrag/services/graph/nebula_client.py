"""
NebulaGraph 客戶端服務

提供 NebulaGraph 的非同步介面，用於實體/關係儲存和圖譜遍歷。

主要功能：
- 連接池管理
- 實體 CRUD 操作
- 關係管理
- 多跳圖譜遍歷
- 社區操作
- Chunk 與實體的關聯
"""

import asyncio
import hashlib
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config as NebulaConfig
from nebula3.data.ResultSet import ResultSet

from chatbot_graphrag.core.config import settings
from chatbot_graphrag.core.constants import (
    EntityType,
    RelationType,
    NEBULA_ENTITY_TAG,
    NEBULA_COMMUNITY_TAG,
    NEBULA_CHUNK_TAG,
    NEBULA_EDGE_TYPES,
)
from chatbot_graphrag.models.pydantic.graph import (
    Entity,
    Relation,
    Community,
    Subgraph,
    GraphTraversalResult,
)

logger = logging.getLogger(__name__)


class NebulaGraphClient:
    """
    GraphRAG 操作的 NebulaGraph 客戶端。

    管理連接池並提供以下方法：
    - 實體 CRUD 操作
    - 關係管理
    - 圖譜遍歷
    - 社區操作
    """

    _instance: Optional["NebulaGraphClient"] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> "NebulaGraphClient":
        """單例模式以重複使用連接池。"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化 NebulaGraph 客戶端。"""
        if self._initialized:
            return

        self._pool: Optional[ConnectionPool] = None
        self._space = settings.nebula_space
        self._initialized = True

    async def initialize(self, max_retries: int = 5, retry_delay: float = 2.0) -> None:
        """使用重試邏輯初始化連接池。

        Args:
            max_retries: 最大連接嘗試次數
            retry_delay: 重試之間的延遲（秒）
        """
        if self._pool is not None:
            return

        config = NebulaConfig()
        config.max_connection_pool_size = settings.nebula_pool_size
        config.timeout = settings.nebula_timeout

        last_error: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                self._pool = ConnectionPool()
                ok = self._pool.init(
                    [(settings.nebula_host, settings.nebula_port)],
                    config,
                )
                if ok:
                    logger.info(
                        f"NebulaGraph connection pool initialized: "
                        f"{settings.nebula_host}:{settings.nebula_port}"
                    )
                    # 確保 space 存在
                    await self._ensure_space_and_schema()
                    return
                else:
                    raise ConnectionError("Connection pool init returned False")
            except Exception as e:
                last_error = e
                self._pool = None
                if attempt < max_retries:
                    logger.warning(
                        f"NebulaGraph connection attempt {attempt}/{max_retries} failed: {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)

        raise ConnectionError(
            f"Failed to connect to NebulaGraph at {settings.nebula_host}:{settings.nebula_port} "
            f"after {max_retries} attempts: {last_error}"
        )

    async def connect(self) -> None:
        """initialize() 的別名 - 連接到 NebulaGraph。"""
        await self.initialize()

    async def is_connected(self) -> bool:
        """檢查客戶端是否已連接。"""
        if not self._pool:
            return False
        try:
            # 嘗試取得 session 以驗證連接
            session = self._pool.get_session(
                settings.nebula_user,
                settings.nebula_password,
            )
            session.release()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """關閉連接池。"""
        if self._pool:
            self._pool.close()
            self._pool = None
            logger.info("NebulaGraph connection pool closed")

    @asynccontextmanager
    async def get_session(self):
        """從連接池取得 session。"""
        if not self._pool:
            await self.initialize()

        session = self._pool.get_session(
            settings.nebula_user,
            settings.nebula_password,
        )
        try:
            # Switch to space
            result = session.execute(f"USE {self._space}")
            if not result.is_succeeded():
                raise RuntimeError(f"Failed to use space {self._space}: {result.error_msg()}")
            yield session
        finally:
            session.release()

    async def execute(self, query: str) -> ResultSet:
        """執行 nGQL 查詢。"""
        async with self.get_session() as session:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, session.execute, query)
            if not result.is_succeeded():
                logger.error(f"Query failed: {query[:200]}... Error: {result.error_msg()}")
                raise RuntimeError(f"NebulaGraph query failed: {result.error_msg()}")
            return result

    async def execute_json(self, query: str) -> list[dict[str, Any]]:
        """執行查詢並以 JSON 格式返回結果。"""
        result = await self.execute(query)
        return self._result_to_dicts(result)

    def _result_to_dicts(self, result: ResultSet) -> list[dict[str, Any]]:
        """將 ResultSet 轉換為字典列表。"""
        if result.is_empty():
            return []

        rows = []
        col_names = result.keys()

        for row_idx in range(result.row_size()):
            row_dict = {}
            for col_idx, col_name in enumerate(col_names):
                value = result.row_values(row_idx)[col_idx]
                row_dict[col_name] = self._convert_value(value)
            rows.append(row_dict)

        return rows

    def _convert_value(self, value: Any) -> Any:
        """將 NebulaGraph 值轉換為 Python 類型。"""
        if value.is_null():
            return None
        if value.is_bool():
            return value.as_bool()
        if value.is_int():
            return value.as_int()
        if value.is_double():
            return value.as_double()
        if value.is_string():
            return value.as_string()
        if value.is_list():
            return [self._convert_value(v) for v in value.as_list()]
        if value.is_map():
            return {k: self._convert_value(v) for k, v in value.as_map().items()}
        if value.is_vertex():
            vertex = value.as_node()
            return {
                "vid": vertex.get_id().as_string(),
                "tags": [tag for tag in vertex.tags()],
                "properties": {
                    tag: {
                        prop: self._convert_value(vertex.properties(tag).get(prop))
                        for prop in vertex.prop_names(tag)
                    }
                    for tag in vertex.tags()
                },
            }
        if value.is_edge():
            edge = value.as_relationship()
            # 直接從 properties() 字典取得屬性，而非使用 prop_names()
            props = edge.properties()
            return {
                "src": edge.start_vertex_id().as_string(),
                "dst": edge.end_vertex_id().as_string(),
                "type": edge.edge_name(),
                "rank": edge.ranking(),
                "properties": {
                    prop: self._convert_value(props.get(prop))
                    for prop in props.keys()
                },
            }
        if value.is_path():
            path = value.as_path()
            return {
                "nodes": [self._convert_value(n) for n in path.nodes()],
                "relationships": [self._convert_value(r) for r in path.relationships()],
            }
        return str(value)

    async def _ensure_space_and_schema(self) -> None:
        """確保圖譜 space 和 schema 存在。"""
        session = self._pool.get_session(
            settings.nebula_user,
            settings.nebula_password,
        )
        try:
            # 建立 space（如果不存在）
            create_space = f"""
            CREATE SPACE IF NOT EXISTS {self._space} (
                vid_type = FIXED_STRING(128),
                partition_num = 10,
                replica_factor = 1
            );
            """
            result = session.execute(create_space)
            if not result.is_succeeded():
                error_msg = result.error_msg()
                # "Host not enough!" 是警告，不是錯誤
                if "Host not enough" not in error_msg:
                    logger.warning(f"Create space result: {error_msg}")
                else:
                    logger.debug(f"Create space warning (可忽略): {error_msg}")

            # 等待 space 準備就緒 - NebulaGraph 需要時間同步
            logger.debug(f"等待 space '{self._space}' 準備就緒...")
            await asyncio.sleep(5)

            # Switch to space（帶重試）
            max_use_retries = 5
            for attempt in range(1, max_use_retries + 1):
                use_result = session.execute(f"USE {self._space}")
                if use_result.is_succeeded():
                    logger.debug(f"成功切換到 space: {self._space}")
                    break
                elif attempt < max_use_retries:
                    logger.warning(
                        f"USE {self._space} 失敗 (嘗試 {attempt}/{max_use_retries}): "
                        f"{use_result.error_msg()}，等待重試..."
                    )
                    await asyncio.sleep(2)
                else:
                    raise RuntimeError(
                        f"無法切換到 space {self._space}: {use_result.error_msg()}"
                    )

            # 建立 schema（tags 和 edges）
            await self._create_schema(session)

            # 驗證 schema 已建立
            await self._verify_schema(session)

        finally:
            session.release()

    async def _create_schema(self, session) -> None:
        """建立圖譜 schema（tags 和邊類型）。"""

        def _execute_and_log(query: str, description: str) -> bool:
            """執行查詢並記錄結果。"""
            result = session.execute(query)
            if not result.is_succeeded():
                error_msg = result.error_msg()
                # "Existed!" 表示已存在，不是錯誤
                if "Existed!" not in error_msg:
                    logger.warning(f"{description} 失敗: {error_msg}")
                    return False
            return True

        # Entity tag（實體標籤）
        entity_tag = f"""
        CREATE TAG IF NOT EXISTS {NEBULA_ENTITY_TAG} (
            name string,
            entity_type string,
            description string,
            properties string,
            source_chunk_ids string,
            doc_ids string,
            mention_count int,
            created_at datetime,
            updated_at datetime
        );
        """
        _execute_and_log(entity_tag, f"建立 {NEBULA_ENTITY_TAG} tag")

        # Community tag（社區標籤）
        community_tag = f"""
        CREATE TAG IF NOT EXISTS {NEBULA_COMMUNITY_TAG} (
            level int,
            title string,
            summary string,
            entity_count int,
            edge_count int,
            importance_score double,
            created_at datetime,
            updated_at datetime
        );
        """
        _execute_and_log(community_tag, f"建立 {NEBULA_COMMUNITY_TAG} tag")

        # Chunk tag（chunk 標籤）
        chunk_tag = f"""
        CREATE TAG IF NOT EXISTS {NEBULA_CHUNK_TAG} (
            doc_id string,
            chunk_type string,
            content_preview string,
            position int,
            created_at datetime
        );
        """
        _execute_and_log(chunk_tag, f"建立 {NEBULA_CHUNK_TAG} tag")

        # 等待 schema 同步
        await asyncio.sleep(3)

        # 建立邊類型
        for edge_type in NEBULA_EDGE_TYPES:
            edge_schema = f"""
            CREATE EDGE IF NOT EXISTS {edge_type} (
                description string,
                weight double,
                source_chunk_ids string,
                mention_count int,
                created_at datetime
            );
            """
            _execute_and_log(edge_schema, f"建立 {edge_type} edge")

        # 等待 schema 同步後再建立索引
        await asyncio.sleep(5)

        # 建立索引
        _execute_and_log(
            f"CREATE TAG INDEX IF NOT EXISTS idx_entity_type ON {NEBULA_ENTITY_TAG}(entity_type(50));",
            "建立 idx_entity_type 索引",
        )
        _execute_and_log(
            f"CREATE TAG INDEX IF NOT EXISTS idx_entity_name ON {NEBULA_ENTITY_TAG}(name(255));",
            "建立 idx_entity_name 索引",
        )
        _execute_and_log(
            f"CREATE TAG INDEX IF NOT EXISTS idx_community_level ON {NEBULA_COMMUNITY_TAG}(level);",
            "建立 idx_community_level 索引",
        )

        logger.info("NebulaGraph schema 建立完成")

    async def _verify_schema(self, session) -> None:
        """驗證 schema 已正確建立。"""
        max_retries = 10
        retry_delay = 2

        for attempt in range(1, max_retries + 1):
            # 檢查 tags
            result = session.execute("SHOW TAGS;")
            if not result.is_succeeded():
                if attempt < max_retries:
                    logger.debug(f"SHOW TAGS 失敗，重試 {attempt}/{max_retries}")
                    await asyncio.sleep(retry_delay)
                    continue
                raise RuntimeError(f"無法驗證 schema: {result.error_msg()}")

            existing_tags = set()
            for row_idx in range(result.row_size()):
                tag_name = result.row_values(row_idx)[0].as_string()
                existing_tags.add(tag_name)

            required_tags = {NEBULA_ENTITY_TAG, NEBULA_COMMUNITY_TAG, NEBULA_CHUNK_TAG}
            missing_tags = required_tags - existing_tags

            if missing_tags:
                if attempt < max_retries:
                    logger.debug(
                        f"等待 schema 同步 ({attempt}/{max_retries})，"
                        f"缺少 tags: {missing_tags}"
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.warning(f"Schema 驗證超時，缺少 tags: {missing_tags}")
                    # 不拋出錯誤，讓程式繼續嘗試
                    break

            # 檢查 edges
            result = session.execute("SHOW EDGES;")
            if result.is_succeeded():
                existing_edges = set()
                for row_idx in range(result.row_size()):
                    edge_name = result.row_values(row_idx)[0].as_string()
                    existing_edges.add(edge_name)

                missing_edges = set(NEBULA_EDGE_TYPES) - existing_edges
                if missing_edges:
                    if attempt < max_retries:
                        logger.debug(
                            f"等待 schema 同步 ({attempt}/{max_retries})，"
                            f"缺少 edges: {missing_edges}"
                        )
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        logger.warning(f"Schema 驗證超時，缺少 edges: {missing_edges}")
                        break

            # Schema 驗證通過
            logger.info(
                f"NebulaGraph schema 驗證通過: "
                f"{len(existing_tags)} tags, {len(existing_edges)} edges"
            )
            return

        logger.warning("NebulaGraph schema 驗證未完全通過，但繼續執行")

    # ==================== 實體操作 ====================

    def _generate_entity_vid(self, name: str, entity_type: EntityType) -> str:
        """為實體生成唯一的頂點 ID。"""
        normalized = name.lower().strip()
        hash_input = f"{entity_type.value}:{normalized}"
        return f"e_{hashlib.sha256(hash_input.encode()).hexdigest()[:16]}"

    def _escape_ngql_string(self, value: str) -> str:
        """
        為 nGQL 字串值跳脫特殊字元。

        順序很重要：先跳脫反斜線，然後再跳脫其他字元。
        """
        if not value:
            return ""
        escaped = value.replace('\\', '\\\\')   # Escape backslashes first
        escaped = escaped.replace('"', '\\"')    # Escape double quotes
        escaped = escaped.replace('\n', ' ')     # Replace newlines with space
        escaped = escaped.replace('\r', ' ')     # Replace carriage returns
        escaped = escaped.replace('\t', ' ')     # Replace tabs
        return escaped

    async def upsert_entity(self, entity: Entity) -> str:
        """插入或更新實體頂點。"""
        vid = entity.nebula_vid or self._generate_entity_vid(entity.name, entity.entity_type)

        # 為 nGQL 跳脫字串
        name = self._escape_ngql_string(entity.name)
        description = self._escape_ngql_string(entity.description or "")
        properties_json = self._escape_ngql_string(str(entity.properties))
        source_chunks = ",".join(entity.source_chunk_ids)
        doc_ids = ",".join(entity.doc_ids)

        query = f"""
        INSERT VERTEX {NEBULA_ENTITY_TAG} (
            name, entity_type, description, properties,
            source_chunk_ids, doc_ids, mention_count, created_at, updated_at
        ) VALUES "{vid}": (
            "{name}",
            "{entity.entity_type.value}",
            "{description}",
            "{properties_json}",
            "{source_chunks}",
            "{doc_ids}",
            {entity.mention_count},
            datetime(),
            datetime()
        );
        """
        await self.execute(query)
        return vid

    async def get_entity(self, vid: str) -> Optional[dict[str, Any]]:
        """根據頂點 ID 取得實體。"""
        query = f'FETCH PROP ON {NEBULA_ENTITY_TAG} "{vid}" YIELD properties(vertex);'
        results = await self.execute_json(query)
        return results[0] if results else None

    async def find_entities_by_type(
        self,
        entity_type: EntityType,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """按類型查找實體。"""
        query = f"""
        LOOKUP ON {NEBULA_ENTITY_TAG}
        WHERE {NEBULA_ENTITY_TAG}.entity_type == "{entity_type.value}"
        YIELD id(vertex) as vid, properties(vertex) as props
        LIMIT {limit};
        """
        return await self.execute_json(query)

    async def search_entities_by_name(
        self,
        name_pattern: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """按名稱模式搜尋實體。"""
        # 注意：對於全文搜尋，建議使用 OpenSearch
        # 這裡使用索引查找，需要精確匹配或前綴匹配
        query = f"""
        LOOKUP ON {NEBULA_ENTITY_TAG}
        WHERE {NEBULA_ENTITY_TAG}.name STARTS WITH "{name_pattern}"
        YIELD id(vertex) as vid, properties(vertex) as props
        LIMIT {limit};
        """
        return await self.execute_json(query)

    # ==================== 關係操作 ====================

    async def upsert_relation(self, relation: Relation) -> None:
        """插入或更新關係邊。"""
        edge_type = relation.relation_type.value
        description = self._escape_ngql_string(relation.description or "")
        source_chunks = ",".join(relation.source_chunk_ids)

        query = f"""
        INSERT EDGE {edge_type} (
            description, weight, source_chunk_ids, mention_count, created_at
        ) VALUES "{relation.source_id}" -> "{relation.target_id}": (
            "{description}",
            {relation.weight},
            "{source_chunks}",
            {relation.mention_count},
            datetime()
        );
        """
        await self.execute(query)

    async def get_relations(
        self,
        source_vid: str,
        relation_types: Optional[list[RelationType]] = None,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """取得實體的關係。"""
        if relation_types:
            edge_types = ",".join(rt.value for rt in relation_types)
        else:
            edge_types = ",".join(NEBULA_EDGE_TYPES)

        if direction == "out":
            query = f'GO FROM "{source_vid}" OVER {edge_types} YIELD edge as e;'
        elif direction == "in":
            query = f'GO FROM "{source_vid}" OVER {edge_types} REVERSELY YIELD edge as e;'
        else:
            query = f'GO FROM "{source_vid}" OVER {edge_types} BIDIRECT YIELD edge as e;'

        return await self.execute_json(query)

    # ==================== 圖譜遍歷 ====================

    async def traverse(
        self,
        seed_vids: list[str],
        max_hops: int = 2,
        relation_types: Optional[list[RelationType]] = None,
        max_vertices: int = 100,
        tenant_id: Optional[str] = None,  # 第 1 階段：多租戶過濾
        acl_groups: Optional[list[str]] = None,  # 第 1 階段：ACL 過濾
    ) -> GraphTraversalResult:
        """
        從種子頂點遍歷圖譜。

        Args:
            seed_vids: 起始頂點 ID 列表
            max_hops: 最大遍歷深度
            relation_types: 按關係類型過濾（None = 全部）
            max_vertices: 返回的最大頂點數
            tenant_id: 多租戶過濾的租戶 ID（第 1 階段）
            acl_groups: 存取控制的 ACL 群組（第 1 階段）

        Returns:
            包含子圖的 GraphTraversalResult
        """
        import time
        start_time = time.time()

        # 驗證 seed_vids - 如果沒有種子則返回空結果
        if not seed_vids:
            logger.warning("Graph traverse called with empty seed_vids")
            return GraphTraversalResult(
                subgraph=Subgraph(
                    entities=[],
                    relations=[],
                    seed_entity_ids=[],
                    hop_count=0,
                ),
                traversal_path=[],
                total_entities_visited=0,
                total_edges_traversed=0,
                execution_time_ms=0.0,
            )

        # 第 1 階段：記錄 ACL 上下文用於除錯
        if tenant_id:
            logger.debug(f"Graph traverse with tenant_id: {tenant_id}")
        if acl_groups:
            logger.debug(f"Graph traverse with acl_groups: {acl_groups}")

        if relation_types:
            edge_types = ",".join(rt.value for rt in relation_types)
        else:
            edge_types = ",".join(NEBULA_EDGE_TYPES)

        seed_vids_str = ",".join(f'"{vid}"' for vid in seed_vids)

        logger.debug(f"Graph traverse query: FROM {seed_vids_str[:100]}... BOTH {edge_types}")

        # 使用 GET SUBGRAPH 進行多跳遍歷
        # 注意：GET SUBGRAPH 使用 BOTH/IN/OUT 子句（不是用於 GO 語句的 OVER）
        # 使用 rels 作為別名而非 edges，避免與 NebulaGraph 關鍵字衝突
        query = (
            f"GET SUBGRAPH WITH PROP {max_hops} STEPS FROM {seed_vids_str} "
            f"BOTH {edge_types} "
            f"YIELD VERTICES AS nodes, EDGES AS rels;"
        )

        results = await self.execute_json(query)

        # 將結果解析為子圖
        entities = []
        relations = []
        traversal_path = [seed_vids]
        visited_vids = set(seed_vids)

        for row in results:
            nodes = row.get("nodes", [])
            rels = row.get("rels", [])

            for node in nodes[:max_vertices]:
                if isinstance(node, dict) and node.get("vid"):
                    vid = node["vid"]
                    if vid not in visited_vids:
                        visited_vids.add(vid)
                        # 轉換為 Entity（簡化版）
                        props = node.get("properties", {}).get(NEBULA_ENTITY_TAG, {})
                        if props:
                            entities.append(
                                Entity(
                                    id=vid,
                                    name=props.get("name", ""),
                                    entity_type=EntityType(props.get("entity_type", "generic")),
                                    description=props.get("description"),
                                    mention_count=props.get("mention_count", 1),
                                    source_chunk_ids=props.get("source_chunk_ids", "").split(","),
                                    doc_ids=props.get("doc_ids", "").split(","),
                                )
                            )

            for rel in rels:
                if isinstance(rel, dict):
                    relations.append(
                        Relation(
                            id=f"{rel['src']}_{rel['type']}_{rel['dst']}",
                            source_id=rel["src"],
                            target_id=rel["dst"],
                            relation_type=RelationType(rel["type"]),
                            description=rel.get("properties", {}).get("description"),
                            weight=rel.get("properties", {}).get("weight", 1.0),
                            mention_count=rel.get("properties", {}).get("mention_count", 1),
                        )
                    )

        execution_time_ms = (time.time() - start_time) * 1000

        return GraphTraversalResult(
            subgraph=Subgraph(
                entities=entities,
                relations=relations,
                seed_entity_ids=seed_vids,
                hop_count=max_hops,
            ),
            traversal_path=traversal_path,
            total_entities_visited=len(visited_vids),
            total_edges_traversed=len(relations),
            execution_time_ms=execution_time_ms,
        )

    async def find_shortest_path(
        self,
        source_vid: str,
        target_vid: str,
        max_hops: int = 5,
    ) -> Optional[list[dict[str, Any]]]:
        """找到兩個實體之間的最短路徑。"""
        edge_types = ",".join(NEBULA_EDGE_TYPES)

        query = f"""
        FIND SHORTEST PATH FROM "{source_vid}" TO "{target_vid}"
        OVER {edge_types}
        WHERE $$.{NEBULA_ENTITY_TAG}.name IS NOT NULL
        YIELD path AS p
        LIMIT 1;
        """
        results = await self.execute_json(query)
        return results[0] if results else None

    # ==================== 社區操作 ====================

    async def upsert_community(self, community: Community) -> str:
        """插入或更新社區頂點。"""
        vid = community.nebula_vid or f"c_{community.level}_{community.id}"

        title = self._escape_ngql_string(community.title or "")
        summary = self._escape_ngql_string(community.summary or "")

        query = f"""
        INSERT VERTEX {NEBULA_COMMUNITY_TAG} (
            level, title, summary, entity_count, edge_count,
            importance_score, created_at, updated_at
        ) VALUES "{vid}": (
            {community.level},
            "{title}",
            "{summary}",
            {community.entity_count},
            {community.edge_count},
            {community.modularity_score or 0.0},
            datetime(),
            datetime()
        );
        """
        await self.execute(query)

        # 添加從實體到社區的 member_of 邊
        for entity_id in community.entity_ids:
            member_query = f"""
            INSERT EDGE member_of (description, weight, source_chunk_ids, mention_count, created_at)
            VALUES "{entity_id}" -> "{vid}": ("", 1.0, "", 1, datetime());
            """
            try:
                await self.execute(member_query)
            except Exception as e:
                logger.warning(f"Failed to create member_of edge: {e}")

        return vid

    async def get_community(self, vid: str) -> Optional[dict[str, Any]]:
        """根據頂點 ID 取得社區。"""
        query = f'FETCH PROP ON {NEBULA_COMMUNITY_TAG} "{vid}" YIELD properties(vertex);'
        results = await self.execute_json(query)
        return results[0] if results else None

    async def get_communities_at_level(
        self,
        level: int,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """取得特定層級的所有社區。"""
        query = f"""
        LOOKUP ON {NEBULA_COMMUNITY_TAG}
        WHERE {NEBULA_COMMUNITY_TAG}.level == {level}
        YIELD id(vertex) as vid, properties(vertex) as props
        LIMIT {limit};
        """
        return await self.execute_json(query)

    async def get_community_members(
        self,
        community_vid: str,
    ) -> list[dict[str, Any]]:
        """取得社區的所有實體成員。"""
        query = f"""
        GO FROM "{community_vid}" OVER member_of REVERSELY
        YIELD $$.{NEBULA_ENTITY_TAG}.name as name,
              $$.{NEBULA_ENTITY_TAG}.entity_type as type,
              id($$) as vid;
        """
        return await self.execute_json(query)

    # ==================== Chunk 操作 ====================

    async def link_chunk_to_entities(
        self,
        chunk_vid: str,
        entity_vids: list[str],
    ) -> None:
        """建立從 chunk 到實體的 mentions 邊。"""
        for entity_vid in entity_vids:
            query = f"""
            INSERT EDGE mentions (description, weight, source_chunk_ids, mention_count, created_at)
            VALUES "{chunk_vid}" -> "{entity_vid}": ("", 1.0, "", 1, datetime());
            """
            await self.execute(query)

    async def get_chunks_mentioning_entity(
        self,
        entity_vid: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """取得提及某個實體的 chunks。"""
        query = f"""
        GO FROM "{entity_vid}" OVER mentions REVERSELY
        YIELD $$.{NEBULA_CHUNK_TAG}.doc_id as doc_id,
              $$.{NEBULA_CHUNK_TAG}.content_preview as preview,
              id($$) as chunk_vid
        LIMIT {limit};
        """
        return await self.execute_json(query)

    # ==================== 刪除操作 ====================

    async def delete_entities_by_doc_id(self, doc_id: str) -> int:
        """
        刪除與文件關聯的所有實體和關係。

        此方法：
        1. 找到所有 doc_ids 欄位中包含該 doc_id 的實體
        2. 刪除連接到這些實體的邊
        3. 刪除實體頂點
        4. 刪除該文件的 chunk 頂點

        Args:
            doc_id: 要刪除實體的文件 ID

        Returns:
            刪除的實體數量
        """
        deleted_count = 0

        try:
            # 步驟 1：找到所有包含此 doc_id 的實體
            # 注意：doc_ids 以逗號分隔的字串儲存
            find_query = f"""
            LOOKUP ON {NEBULA_ENTITY_TAG}
            YIELD id(vertex) as vid, {NEBULA_ENTITY_TAG}.doc_ids as doc_ids
            | WHERE $-.doc_ids CONTAINS "{doc_id}"
            """

            try:
                results = await self.execute_json(find_query)
                entity_vids = [r.get("vid") for r in results if r.get("vid")]
            except Exception as e:
                logger.warning(f"LOOKUP query failed (index may not exist): {e}")
                entity_vids = []

            if entity_vids:
                # 步驟 2：刪除連接到這些實體的邊
                edge_types = ",".join(NEBULA_EDGE_TYPES)
                for vid in entity_vids:
                    # 刪除出邊
                    delete_out = f'DELETE EDGE {edge_types} FROM "{vid}" -> *;'
                    try:
                        await self.execute(delete_out)
                    except Exception as e:
                        logger.debug(f"Delete outgoing edges for {vid}: {e}")

                    # 刪除入邊
                    delete_in = f'DELETE EDGE {edge_types} * -> "{vid}";'
                    try:
                        await self.execute(delete_in)
                    except Exception as e:
                        logger.debug(f"Delete incoming edges for {vid}: {e}")

                # 步驟 3：刪除實體頂點
                vids_str = ",".join(f'"{vid}"' for vid in entity_vids)
                delete_vertices = f"DELETE VERTEX {vids_str};"
                await self.execute(delete_vertices)
                deleted_count = len(entity_vids)

            # 步驟 4：刪除此文件的 chunk 頂點
            chunk_find_query = f"""
            LOOKUP ON {NEBULA_CHUNK_TAG}
            WHERE {NEBULA_CHUNK_TAG}.doc_id == "{doc_id}"
            YIELD id(vertex) as vid;
            """

            try:
                chunk_results = await self.execute_json(chunk_find_query)
                chunk_vids = [r.get("vid") for r in chunk_results if r.get("vid")]

                if chunk_vids:
                    # 刪除從 chunks 發出的 mentions 邊
                    for vid in chunk_vids:
                        delete_mentions = f'DELETE EDGE mentions FROM "{vid}" -> *;'
                        try:
                            await self.execute(delete_mentions)
                        except Exception as e:
                            logger.debug(f"Delete mentions edges for {vid}: {e}")

                    # 刪除 chunk 頂點
                    chunk_vids_str = ",".join(f'"{vid}"' for vid in chunk_vids)
                    delete_chunks = f"DELETE VERTEX {chunk_vids_str};"
                    await self.execute(delete_chunks)
                    logger.debug(f"Deleted {len(chunk_vids)} chunk vertices for doc {doc_id}")

            except Exception as e:
                logger.debug(f"Chunk deletion query: {e}")

            logger.info(f"Deleted {deleted_count} entities for document {doc_id}")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete entities for doc {doc_id}: {e}")
            return 0

    # ==================== 健康檢查 ====================

    async def health_check(self) -> dict[str, Any]:
        """檢查 NebulaGraph 連接健康狀態。"""
        try:
            result = await self.execute("SHOW SPACES;")
            return {
                "status": "healthy",
                "space": self._space,
                "connected": True,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connected": False,
            }


# 單例實例
nebula_client = NebulaGraphClient()
