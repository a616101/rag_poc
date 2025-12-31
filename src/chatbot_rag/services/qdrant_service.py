"""
Qdrant 向量資料庫服務模組

此模組提供與 Qdrant 向量資料庫互動的功能，包括：
- 資料庫連接管理
- 向量集合的建立和管理
- 向量的儲存和更新
- 相似度搜尋
"""

from typing import Optional, Set

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, MatchText, PointStruct, VectorParams

from chatbot_rag.core.config import settings


class QdrantService:
    """
    Qdrant 向量資料庫服務類別

    此類別負責管理 Qdrant 向量資料庫的所有操作，
    包括連接、集合管理、向量儲存和搜尋等功能。
    Qdrant 是一個高效能的向量搜尋引擎，專門用於相似度搜尋。
    """

    def __init__(self):
        """
        初始化 Qdrant 服務

        建立服務實例並設定集合名稱，但不立即建立連線。
        採用延遲初始化策略。
        """
        self.client: Optional[QdrantClient] = None  # Qdrant 客戶端實例（延遲初始化）
        self.collection_name = settings.qdrant_collection_name  # 向量集合名稱

    def connect(self) -> QdrantClient:
        """
        連接到 Qdrant 資料庫

        如果尚未建立連線，則建立新的連線。
        支援使用 API 金鑰進行身份驗證。

        Returns:
            QdrantClient: 已連接的 Qdrant 客戶端實例

        Raises:
            Exception: 當連接失敗時拋出異常
        """
        if self.client is None:
            try:
                logger.info(f"Connecting to Qdrant at {settings.qdrant_url}")

                # 初始化客戶端，根據是否有 API 金鑰選擇不同的連接方式
                if settings.qdrant_api_key:
                    # 使用 API 金鑰連接（適用於雲端服務）
                    self.client = QdrantClient(
                        url=settings.qdrant_url,
                        api_key=settings.qdrant_api_key,
                    )
                else:
                    # 不使用身份驗證連接（適用於本地部署）
                    self.client = QdrantClient(
                        url=settings.qdrant_url,
                    )

                logger.info("Successfully connected to Qdrant")
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {e}")
                raise

        return self.client

    def ensure_collection(self, recreate: bool = False) -> bool:
        """
        確保向量集合存在，如不存在則建立

        檢查集合是否存在，如果不存在則建立新集合。
        可選擇重新建立集合，這會刪除所有現有數據。

        Args:
            recreate: 是否重新建立集合（會刪除現有集合和所有數據）

        Returns:
            bool: 如果集合被建立或重新建立則返回 True，如果已存在則返回 False

        Raises:
            Exception: 當集合建立失敗時拋出異常
        """
        try:
            client = self.connect()

            # 檢查集合是否存在
            collections = client.get_collections().collections
            collection_exists = any(
                col.name == self.collection_name for col in collections
            )

            # 如果集合存在且需要重新建立，先刪除現有集合
            if collection_exists and recreate:
                logger.info(f"Deleting existing collection: {self.collection_name}")
                client.delete_collection(collection_name=self.collection_name)
                collection_exists = False

            # 如果集合不存在，建立新集合
            if not collection_exists:
                logger.info(
                    f"Creating collection: {self.collection_name} "
                    f"(dimension: {settings.embedding_dimension})"
                )
                # 建立集合並配置向量參數
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.embedding_dimension,  # 向量維度
                        distance=Distance.COSINE,  # 使用餘弦相似度計算距離
                    ),
                )
                logger.info(f"Collection created: {self.collection_name}")
                return True
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
                return False

        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    def upsert_vectors(
        self,
        points: list[PointStruct],
    ) -> dict:
        """
        插入或更新向量集合中的向量

        將向量點插入到集合中，如果點的 ID 已存在則更新。
        這個操作是原子性的，確保數據一致性。

        Args:
            points: 要插入或更新的點列表，每個點包含 ID、向量和元數據

        Returns:
            dict: 操作結果字典，包含：
                - status: 操作狀態
                - count: 處理的點數量
                - operation_id: 操作 ID（如果有）

        Raises:
            Exception: 當插入或更新操作失敗時拋出異常
        """
        try:
            client = self.connect()

            logger.info(f"Upserting {len(points)} points to {self.collection_name}")

            # 執行插入或更新操作
            operation_info = client.upsert(
                collection_name=self.collection_name,
                points=points,  # 點列表，包含向量和元數據
            )

            logger.info(f"Successfully upserted {len(points)} points")

            return {
                "status": "success",
                "count": len(points),
                "operation_id": operation_info.operation_id if operation_info else None,
            }

        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            raise

    def search_similar(
        self,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
    ) -> list[dict]:
        """
        搜尋相似的向量

        使用查詢向量在集合中搜尋最相似的向量點。
        基於餘弦相似度進行排序，返回最相似的結果。

        Args:
            query_vector: 查詢向量，用於尋找相似的向量
            limit: 返回結果的最大數量，預設為 5
            score_threshold: 相似度分數閾值，只返回分數高於此值的結果（可選）

        Returns:
            list[dict]: 搜尋結果列表，每個結果包含：
                - id: 點的唯一識別符
                - score: 相似度分數（0-1之間）
                - payload: 點的元數據

        Raises:
            Exception: 當搜尋操作失敗時拋出異常
        """
        try:
            client = self.connect()

            # 執行相似度搜尋
            search_result = client.query_points(
                collection_name=self.collection_name,
                query=query_vector,  # 查詢向量
                limit=limit,  # 最大結果數
                score_threshold=score_threshold,  # 分數閾值
            )

            # 格式化搜尋結果
            results = [
                {
                    "id": point.id,
                    "score": point.score,  # 相似度分數
                    "payload": point.payload,  # 元數據（如文件內容、來源等）
                }
                for point in search_result.points
            ]

            logger.info(f"Found {len(results)} similar vectors")
            return results

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            raise

    def get_collection_info(self) -> dict:
        """
        取得集合資訊

        查詢集合的詳細資訊，包括點數量、向量數量和狀態。

        Returns:
            dict: 集合資訊字典，包含：
                - name: 集合名稱
                - points_count: 點的總數
                - vectors_count: 向量的總數
                - status: 集合狀態
                - optimizer_status: 優化器狀態

        Raises:
            Exception: 當操作失敗時拋出異常
        """
        try:
            client = self.connect()

            # 取得集合詳細資訊
            collection_info = client.get_collection(
                collection_name=self.collection_name
            )

            return {
                "name": self.collection_name,
                "points_count": collection_info.points_count,  # 儲存的點數量
                "vectors_count": collection_info.vectors_count,  # 向量數量
                "status": collection_info.status,  # 集合狀態
                "optimizer_status": collection_info.optimizer_status,  # 優化器狀態
            }

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise

    def delete_collection(self) -> bool:
        """
        刪除集合

        永久刪除指定的向量集合及其所有數據。
        此操作不可逆，請謹慎使用。

        Returns:
            bool: 如果集合成功刪除則返回 True

        Raises:
            Exception: 當刪除操作失敗時拋出異常
        """
        try:
            client = self.connect()

            logger.info(f"Deleting collection: {self.collection_name}")
            # 執行刪除操作
            result = client.delete_collection(collection_name=self.collection_name)

            logger.info(f"Collection deleted: {self.collection_name}")
            return result

        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise

    def fetch_neighboring_chunks(
        self,
        filename: str,
        chunk_index: int,
        window: int = 1,
    ) -> list[dict]:
        """
        根據檔案名稱和 chunk_index 取得相鄰的 chunks。

        用於擴充檢索結果的上下文，當匹配到某個 chunk 時，
        可以同時取得前後的 chunks 以提供更完整的資訊。

        Args:
            filename: 檔案名稱（用於過濾同一文件的 chunks）
            chunk_index: 目標 chunk 的索引
            window: 要取得的相鄰 chunk 數量（前後各 window 個）

        Returns:
            list[dict]: 相鄰 chunks 的列表，按 chunk_index 排序
        """
        try:
            client = self.connect()

            # 計算要查詢的 chunk_index 範圍
            min_index = max(0, chunk_index - window)
            max_index = chunk_index + window

            # 使用 scroll 查詢同一檔案的相鄰 chunks
            # 注意：Qdrant scroll 不支援範圍查詢，所以我們查詢同檔案所有 chunks 再過濾
            results, _ = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="filename",
                            match=MatchValue(value=filename),
                        ),
                    ]
                ),
                limit=100,  # 假設單一文件不會超過 100 個 chunks
                with_payload=True,
                with_vectors=False,
            )

            # 過濾出在範圍內的 chunks
            neighboring_chunks = []
            for point in results:
                payload = point.payload or {}
                idx = payload.get("chunk_index", -1)
                if min_index <= idx <= max_index:
                    neighboring_chunks.append({
                        "id": point.id,
                        "chunk_index": idx,
                        "payload": payload,
                    })

            # 按 chunk_index 排序
            neighboring_chunks.sort(key=lambda x: x["chunk_index"])

            logger.debug(
                f"Fetched {len(neighboring_chunks)} neighboring chunks for {filename} "
                f"(index {chunk_index}, window {window})"
            )

            return neighboring_chunks

        except Exception as e:
            logger.error(f"Failed to fetch neighboring chunks: {e}")
            return []

    def list_by_filter(
        self,
        filters: dict[str, str],
        limit: int = 50,
        return_content: bool = False,
    ) -> list[dict]:
        """
        按 metadata filter 列出匹配的文件。

        用於聚合型查詢場景（如「心臟血管科有哪些醫師？」），
        直接透過 metadata filter 取得所有符合條件的文件，
        而不使用向量相似度搜尋。

        Args:
            filters: metadata 過濾條件，例如 {"department": "心臟血管科"}
            limit: 返回結果的最大數量，預設為 50
            return_content: 是否返回文件內容（chunk_content），預設為 False

        Returns:
            list[dict]: 匹配文件的列表，每個文件包含：
                - filename: 檔案名稱
                - department: 科別（如有）
                - doctor: 醫師名稱（如有）
                - title: 標題
                - entry_type: 文件類型
                - schedule_count: 門診筆數（如有）
                - content: 文件內容（僅當 return_content=True）

        Example:
            >>> qdrant_service.list_by_filter({"department": "心臟血管科"})
            [
                {"filename": "吳榮州.md", "department": "心臟血管科",
                 "doctor": "吳榮州", "schedule_count": 42, ...},
                ...
            ]
        """
        try:
            client = self.connect()

            # 建構 Filter 條件
            must_conditions = [
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
                for key, value in filters.items()
            ]

            # 執行 scroll 查詢
            results, _ = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=must_conditions),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            # 整理結果，按 filename 去重（同一文件可能有多個 chunks）
            seen_files: dict[str, dict] = {}
            for point in results:
                payload = point.payload or {}
                filename = payload.get("filename", "")

                # 跳過已處理的文件（只取第一個 chunk 的 metadata）
                if filename in seen_files:
                    # 如果需要內容，合併 chunks
                    if return_content and "chunk_content" in payload:
                        existing = seen_files[filename]
                        chunk_idx = payload.get("chunk_index", 0)
                        existing["_chunks"].append({
                            "index": chunk_idx,
                            "content": payload.get("chunk_content", ""),
                        })
                    continue

                # 建立文件記錄
                file_record = {
                    "filename": filename,
                    "title": payload.get("title", ""),
                    "entry_type": payload.get("entry_type", ""),
                    "department": payload.get("department", ""),
                    "doctor": payload.get("doctor", ""),
                    "category": payload.get("category", ""),
                    "schedule_count": payload.get("schedule_count"),
                }

                if return_content:
                    file_record["_chunks"] = [{
                        "index": payload.get("chunk_index", 0),
                        "content": payload.get("chunk_content", ""),
                    }]

                seen_files[filename] = file_record

            # 如果有內容，組合 chunks
            file_list = list(seen_files.values())
            if return_content:
                for file_record in file_list:
                    chunks = sorted(file_record.pop("_chunks", []), key=lambda x: x["index"])
                    file_record["content"] = "\n".join(c["content"] for c in chunks)

            logger.info(
                f"list_by_filter found {len(file_list)} unique files "
                f"with filters: {filters}"
            )

            return file_list

        except Exception as e:
            logger.error(f"Failed to list by filter: {e}")
            return []

    def fetch_chunks_by_indices(
        self,
        filename: str,
        indices: set[int],
    ) -> list[dict]:
        """
        取得指定文件中特定索引的 chunks。

        用於 Adaptive Chunk Expansion：只取得需要的 chunks，
        避免取得整個文件造成 context 過大。

        Args:
            filename: 檔案名稱
            indices: 要取得的 chunk_index 集合

        Returns:
            list[dict]: 指定索引的 chunks 列表，按 chunk_index 排序，
                       每個 chunk 包含 id、chunk_index、payload
        """
        if not indices:
            return []

        try:
            client = self.connect()

            # 查詢同一檔案的所有 chunks，再過濾出需要的索引
            # 注意：Qdrant 不支援 IN 查詢，所以先取全部再過濾
            results, _ = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="filename",
                            match=MatchValue(value=filename),
                        ),
                    ]
                ),
                limit=200,
                with_payload=True,
                with_vectors=False,
            )

            # 過濾出指定索引的 chunks
            chunks = []
            for point in results:
                payload = point.payload or {}
                chunk_idx = payload.get("chunk_index", -1)
                if chunk_idx in indices:
                    chunks.append({
                        "id": point.id,
                        "chunk_index": chunk_idx,
                        "payload": payload,
                    })

            # 按 chunk_index 排序
            chunks.sort(key=lambda x: x["chunk_index"])

            logger.debug(
                f"Fetched {len(chunks)}/{len(indices)} chunks for {filename} "
                f"(requested indices: {sorted(indices)})"
            )

            return chunks

        except Exception as e:
            logger.error(f"Failed to fetch chunks by indices: {e}")
            return []

    def fetch_all_chunks_by_filename(self, filename: str) -> list[dict]:
        """
        取得指定文件的所有 chunks。

        用於 Agentic RAG 場景：當 Agent 判斷檢索結果不完整時，
        可呼叫此方法取得完整文件內容。

        Args:
            filename: 檔案名稱

        Returns:
            list[dict]: 該文件所有 chunks 的列表，按 chunk_index 排序，
                       每個 chunk 包含 id、chunk_index、payload
        """
        try:
            client = self.connect()

            # 查詢同一檔案的所有 chunks
            results, _ = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="filename",
                            match=MatchValue(value=filename),
                        ),
                    ]
                ),
                limit=200,  # 支援較大文件（最多 200 個 chunks）
                with_payload=True,
                with_vectors=False,
            )

            # 整理結果
            chunks = []
            for point in results:
                payload = point.payload or {}
                chunks.append({
                    "id": point.id,
                    "chunk_index": payload.get("chunk_index", 0),
                    "payload": payload,
                })

            # 按 chunk_index 排序
            chunks.sort(key=lambda x: x["chunk_index"])

            logger.info(
                f"Fetched {len(chunks)} chunks for filename: {filename}"
            )

            return chunks

        except Exception as e:
            logger.error(f"Failed to fetch chunks by filename: {e}")
            return []

    def get_unique_field_values(
        self,
        field_name: str,
        limit: int = 1000,
    ) -> Set[str]:
        """
        取得某個 metadata 欄位的所有唯一值。

        用於建立動態的欄位值索引，避免硬編碼映射表。

        Args:
            field_name: metadata 欄位名稱（如 "department"）
            limit: 最大掃描數量

        Returns:
            Set[str]: 該欄位的所有唯一值
        """
        try:
            client = self.connect()

            # 使用 scroll 掃描所有 points
            results, _ = client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=[field_name],  # 只取需要的欄位
                with_vectors=False,
            )

            unique_values: Set[str] = set()
            for point in results:
                payload = point.payload or {}
                value = payload.get(field_name)
                if value and isinstance(value, str):
                    unique_values.add(value)

            logger.debug(
                f"get_unique_field_values: found {len(unique_values)} unique values for '{field_name}'"
            )

            return unique_values

        except Exception as e:
            logger.error(f"Failed to get unique field values: {e}")
            return set()

    def list_by_fuzzy_filter(
        self,
        field_name: str,
        query_value: str,
        entry_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """
        使用模糊匹配進行 metadata filter 查詢。

        當用戶輸入的值可能是簡稱或別名時使用。
        策略：使用 MatchText 進行包含匹配，找出所有可能的結果。

        Args:
            field_name: metadata 欄位名稱（如 "department"）
            query_value: 用戶輸入的查詢值（可能是簡稱）
            entry_type: 額外的文件類型過濾
            limit: 返回結果的最大數量

        Returns:
            list[dict]: 匹配文件的列表

        Example:
            >>> qdrant_service.list_by_fuzzy_filter("department", "心臟")
            [{"department": "心臟血管科", "doctor": "王醫師", ...}, ...]
        """
        try:
            client = self.connect()

            # 建構 Filter 條件 - 使用 MatchText 進行包含匹配
            must_conditions = [
                FieldCondition(
                    key=field_name,
                    match=MatchText(text=query_value),
                ),
            ]

            # 額外的 entry_type 過濾
            if entry_type:
                must_conditions.append(
                    FieldCondition(
                        key="entry_type",
                        match=MatchValue(value=entry_type),
                    )
                )

            # 執行 scroll 查詢
            results, _ = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=must_conditions),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            # 整理結果，按 filename 去重
            seen_files: dict[str, dict] = {}
            for point in results:
                payload = point.payload or {}
                filename = payload.get("filename", "")

                if filename in seen_files:
                    continue

                file_record = {
                    "filename": filename,
                    "title": payload.get("title", ""),
                    "entry_type": payload.get("entry_type", ""),
                    "department": payload.get("department", ""),
                    "doctor": payload.get("doctor", ""),
                    "category": payload.get("category", ""),
                    "schedule_count": payload.get("schedule_count"),
                }
                seen_files[filename] = file_record

            file_list = list(seen_files.values())

            logger.info(
                f"list_by_fuzzy_filter found {len(file_list)} files "
                f"for {field_name} contains '{query_value}'"
            )

            return file_list

        except Exception as e:
            logger.error(f"Failed to list by fuzzy filter: {e}")
            # Fallback: 嘗試精確匹配
            return self.list_by_filter(
                filters={field_name: query_value},
                limit=limit,
            )


# 建立全域 Qdrant 服務實例，供整個應用程式使用
qdrant_service = QdrantService()
