"""
文件處理服務模組

此模組提供了 RAG (Retrieval-Augmented Generation) 系統的文件處理功能，包括：
- 從目錄或檔案載入文件
- 將文件分割成適當大小的文字塊（chunks）
- 生成文字塊的向量嵌入（embeddings）
- 將向量儲存到 Qdrant 向量資料庫

主要類別：
    DocumentService: 文件處理服務的核心類別

全域實例：
    document_service: DocumentService 的單例實例，可直接導入使用
"""

import hashlib
from pathlib import Path
from typing import Optional

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from qdrant_client.models import PointStruct

from chatbot_rag.core.config import settings
from chatbot_rag.services.contextual_chunking_service import (
    contextual_chunking_service,
)
from chatbot_rag.services.embedding_service import embedding_service
from chatbot_rag.services.qdrant_service import qdrant_service
from chatbot_rag.services.semantic_cache_service import semantic_cache_service


class DocumentService:
    """
    文件處理服務類別

    此類別負責處理文件的載入、分割、向量化和儲存等操作。
    主要功能包括：
    1. 從目錄或單一檔案載入 Markdown 文件
    2. 使用 RecursiveCharacterTextSplitter 將長文件分割成較小的文字塊
    3. 為每個文字塊生成唯一的 ID
    4. 與 embedding_service 和 qdrant_service 協作，生成向量並儲存到向量資料庫

    屬性：
        text_splitter: RecursiveCharacterTextSplitter 實例，用於分割文件
    """

    def __init__(self):
        """
        初始化文件處理服務

        建立一個 RecursiveCharacterTextSplitter 實例，使用以下設定：
        - chunk_size: 每個文字塊的目標字元數（從 settings 讀取）
        - chunk_overlap: 相鄰文字塊之間的重疊字元數（從 settings 讀取）
        - length_function: 用於計算文字長度的函數（使用內建的 len）
        - is_separator_regex: 分隔符是否為正則表達式（設為 False）
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def _parse_markdown_with_frontmatter(self, file_path: Path) -> dict:
        """
        解析含 YAML frontmatter 的 Markdown 檔案。

        - 若檔案開頭為 `---`，則視為有 frontmatter：
            - 解析 `---` 與下一個 `---` 之間為 YAML frontmatter
            - 將 frontmatter 轉成 dict 並放入 metadata
        - 其餘內容則視為 Markdown 主要內容

        回傳：
            dict: {
                "content": <純 Markdown 內容（不含 frontmatter）>,
                "metadata": <從 frontmatter 與檔案路徑整理出的 metadata>,
            }
        """
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        frontmatter_data: dict = {}
        content_text = raw_text

        # 檢查是否有 YAML frontmatter
        if raw_text.startswith("---"):
            # 找到第二個 '---' 的位置
            # 允許第一行為 '---' 或 '---\n'
            parts = raw_text.split("\n")
            end_index = None
            for i in range(1, len(parts)):
                if parts[i].strip() == "---":
                    end_index = i
                    break

            if end_index is not None:
                frontmatter_str = "\n".join(parts[1:end_index])
                # 剩餘內容為真正的 Markdown 本文
                content_text = "\n".join(parts[end_index + 1 :])

                # 使用 PyYAML 解析 frontmatter，若失敗則忽略錯誤
                try:
                    loaded = yaml.safe_load(frontmatter_str) or {}
                    if isinstance(loaded, dict):
                        frontmatter_data = loaded
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to parse frontmatter for {file_path}: {e}")
                    frontmatter_data = {"frontmatter_raw": frontmatter_str}

                # 不論解析成功與否，都保留原始 frontmatter 字串以利除錯
                if "frontmatter_raw" not in frontmatter_data:
                    frontmatter_data["frontmatter_raw"] = frontmatter_str

        # 基礎 metadata（路徑相關）
        base_metadata = {
            "source": str(file_path),
            "filename": file_path.name,
        }

        # 若無法計算相對路徑，不會拋例外，只是少一個欄位
        try:
            # 嘗試以設定中的 default_docs_path 為基準計算相對路徑
            default_base = Path(settings.default_docs_path).resolve()
            relative_path = None
            try:
                relative_path = str(file_path.resolve().relative_to(default_base))
            except ValueError:
                # 若不在 default_docs_path 之下，再退回以傳入目錄為基準計算
                pass

            if relative_path is None:
                # 最簡單情況：直接使用檔名
                relative_path = file_path.name

            base_metadata["relative_path"] = relative_path
        except Exception:
            # 相對路徑計算失敗時不影響主流程
            pass

        metadata = {**frontmatter_data, **base_metadata}

        return {
            "content": content_text,
            "metadata": metadata,
        }

    def load_documents_from_directory(self, directory: str | Path) -> list[dict]:
        """
        從指定目錄（含子目錄）載入所有 Markdown 文件。

        - 遞迴掃描指定目錄下的所有 `.md` 檔案
        - 若檔案包含 YAML frontmatter，會解析並寫入 metadata
        - 每個文件都會包含：
            - content: Markdown 主體內容（不含 frontmatter）
            - metadata: 檔案路徑與 frontmatter 中的結構化欄位

        參數：
            directory: 包含文件的根目錄路徑，可以是字串或 Path 物件

        回傳：
            list[dict]: 文件列表，每個文件是一個包含以下鍵值的字典：
                - content (str)
                - metadata (dict)

        例外：
            FileNotFoundError: 當指定的目錄不存在時拋出此例外
        """
        directory = Path(directory)

        # 檢查目錄是否存在
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        logger.info(f"Loading documents recursively from {directory}")

        documents: list[dict] = []
        # 遞迴取得目錄中所有的 .md 檔案
        md_files = list(directory.rglob("*.md"))

        # 逐一讀取每個 Markdown 檔案
        for file_path in md_files:
            try:
                parsed = self._parse_markdown_with_frontmatter(file_path)
                documents.append(parsed)
                logger.debug(f"Loaded document: {file_path}")
            except Exception as e:  # noqa: BLE001
                # 記錄錯誤但不中斷處理其他檔案
                logger.error(f"Failed to load document {file_path}: {e}")

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def load_document_from_file(self, file_path: str | Path) -> dict:
        """
        從單一檔案載入文件

        此方法讀取指定檔案的內容，並建立包含內容和元資料的文件字典。

        參數：
            file_path: 文件檔案的路徑，可以是字串或 Path 物件

        回傳：
            dict: 包含以下鍵值的文件字典：
                - content (str): 檔案的完整內容
                - metadata (dict): 包含以下資訊的元資料：
                    - source (str): 檔案的完整路徑
                    - filename (str): 檔案名稱

        例外：
            FileNotFoundError: 當指定的檔案不存在時拋出此例外
        """
        file_path = Path(file_path)

        # 檢查檔案是否存在
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Loading document from {file_path}")

        # 讀取檔案內容
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 建立並回傳文件字典
        return {
            "content": content,
            "metadata": {
                "source": str(file_path),
                "filename": file_path.name,
            },
        }

    def chunk_documents(
        self,
        documents: list[dict],
        use_contextual_chunking: bool = True,
        use_llm: Optional[bool] = None,
        progress_callback: Optional[callable] = None,
    ) -> list[dict]:
        """
        將文件分割成文字塊

        此方法使用 text_splitter 將長文件分割成較小的文字塊，以便進行向量化處理。
        每個文字塊都會保留原始文件的元資料，並額外添加塊索引和總塊數資訊。
        分割時會考慮 chunk_overlap 設定，使相鄰的塊之間有重疊，以保持語義連貫性。

        若啟用 Contextual Chunking，會為每個 chunk 添加文檔脈絡（來源、章節等），
        提升向量檢索的精準度。

        **重要**：文件會序列化處理，每個文件的所有 chunks 完成脈絡化後，
        才會開始處理下一個文件，避免 LLM 多工導致後端超載。

        參數：
            documents: 要分割的文件列表，每個文件是包含 content 和 metadata 的字典
            use_contextual_chunking: 是否啟用脈絡化分塊（預設為 True）
            use_llm: 是否使用 LLM 生成 Level 2 脈絡。None 表示使用系統預設值
            progress_callback: 進度回調函數，接收 (current_doc, total_docs, doc_filename) 參數

        回傳：
            list[dict]: 文字塊列表，每個塊是一個包含以下鍵值的字典：
                - text (str): 文字塊的原始內容
                - contextualized_text (str): 脈絡化後的內容（若啟用）
                - section_path (str): 章節路徑（若啟用）
                - metadata (dict): 包含原始文件的元資料，並額外包含：
                    - chunk_index (int): 此塊在文件中的索引（從 0 開始）
                    - total_chunks (int): 該文件總共分割成幾個塊
        """
        total_docs = len(documents)
        logger.info(f"Chunking {total_docs} documents (sequential processing)")

        # 判斷是否實際啟用 contextual chunking
        enable_contextual = (
            use_contextual_chunking and settings.contextual_chunking_enabled
        )

        # 決定是否使用 LLM（優先使用參數，否則使用系統設定）
        actual_use_llm = use_llm if use_llm is not None else settings.contextual_chunking_use_llm

        if enable_contextual:
            logger.info(
                f"Contextual chunking is enabled (use_llm={actual_use_llm})"
            )

        all_chunks = []
        # 序列化處理每個文件（一個完成後才處理下一個）
        for doc_index, doc in enumerate(documents):
            doc_content = doc["content"]
            doc_metadata = doc["metadata"]
            doc_filename = doc_metadata.get("filename", f"document_{doc_index}")

            # 進度回調
            if progress_callback:
                progress_callback(doc_index + 1, total_docs, doc_filename)

            logger.info(
                f"Processing document {doc_index + 1}/{total_docs}: {doc_filename}"
            )

            # 使用 text_splitter 分割文件內容
            text_chunks = self.text_splitter.split_text(doc_content)
            chunk_count = len(text_chunks)

            logger.debug(f"  Split into {chunk_count} chunks")

            # 為每個文字塊建立字典，包含內容和完整的元資料
            doc_chunks = []
            for i, chunk_text in enumerate(text_chunks):
                doc_chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            **doc_metadata,  # 保留原始文件的所有元資料
                            "chunk_index": i,  # 添加塊索引
                            "total_chunks": chunk_count,  # 添加總塊數
                        },
                    }
                )

            # 若啟用 contextual chunking，為這個文件的所有 chunks 添加脈絡
            # 注意：這裡會等待所有 chunks 完成後才繼續下一個文件
            if enable_contextual:
                logger.debug(
                    f"  Contextualizing {chunk_count} chunks (use_llm={actual_use_llm})..."
                )
                doc_chunks = contextual_chunking_service.contextualize_chunks(
                    chunks=doc_chunks,
                    doc_content=doc_content,
                    doc_metadata=doc_metadata,
                    use_llm=actual_use_llm,
                )
                logger.debug(f"  Contextualization complete for {doc_filename}")
            else:
                # 未啟用時，contextualized_text 就是原始 text
                for chunk in doc_chunks:
                    chunk["contextualized_text"] = chunk["text"]
                    chunk["section_path"] = ""

            all_chunks.extend(doc_chunks)

            logger.info(
                f"Completed document {doc_index + 1}/{total_docs}: {doc_filename} "
                f"({chunk_count} chunks)"
            )

        logger.info(f"Created {len(all_chunks)} chunks from {total_docs} documents")
        return all_chunks

    def generate_chunk_id(self, chunk: dict) -> str:
        """
        為文字塊生成唯一的 ID

        此方法根據文字塊的來源、索引和內容生成一個唯一的 MD5 雜湊值作為 ID。
        這確保了相同內容在相同位置的塊會產生相同的 ID，可用於去重或更新。

        參數：
            chunk: 文字塊字典，必須包含以下資訊：
                - metadata['source']: 來源檔案路徑
                - metadata['chunk_index']: 塊索引
                - text: 文字塊內容

        回傳：
            str: 32 位元的十六進位 MD5 雜湊字串
        """
        # 組合來源、索引和內容建立唯一字串
        unique_string = f"{chunk['metadata']['source']}:{chunk['metadata']['chunk_index']}:{chunk['text']}"

        # 生成 MD5 雜湊值
        return hashlib.md5(unique_string.encode()).hexdigest()

    def create_vector_points(
        self, chunks: list[dict], embeddings: list[list[float]]
    ) -> list[PointStruct]:
        """
        從文字塊和嵌入向量建立 Qdrant 向量點

        此方法將文字塊和對應的嵌入向量組合成 Qdrant 可以儲存的 PointStruct 物件。
        每個點包含：
        - 唯一的整數 ID（從雜湊值轉換而來）
        - 嵌入向量
        - 包含文字內容和元資料的 payload

        參數：
            chunks: 文字塊列表，每個塊包含 text 和 metadata
            embeddings: 嵌入向量列表，每個向量是一個浮點數列表

        回傳：
            list[PointStruct]: Qdrant 向量點列表，可直接用於插入向量資料庫

        例外：
            ValueError: 當文字塊數量與嵌入向量數量不一致時拋出此例外
        """
        # 驗證輸入的文字塊和嵌入向量數量必須相同
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match"
            )

        logger.info(f"Creating {len(chunks)} vector points")

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            # 生成唯一的 ID（MD5 雜湊值）
            point_id = self.generate_chunk_id(chunk)

            # 將雜湊值轉換為整數（Qdrant 要求 ID 為整數）
            # 只使用前 16 個字元以避免數值過大
            point_id_int = int(point_id[:16], 16)

            # 建立 payload，將文字內容與完整 metadata 一併寫入
            metadata = chunk.get("metadata", {}) or {}
            payload = {
                "text": chunk["text"],  # 原始文字（用於顯示）
                "contextualized_text": chunk.get(
                    "contextualized_text", chunk["text"]
                ),  # 脈絡化文字（用於 embedding）
                "section_path": chunk.get("section_path", ""),  # 章節路徑
            }

            # 將 metadata 展開寫入 payload，避免遺漏 entry_type / module 等欄位
            # 若有 key 與基礎欄位衝突，以 metadata 為主，確保來源資訊一致
            payload.update(metadata)

            # 確保必需欄位存在
            if "chunk_index" not in payload:
                payload["chunk_index"] = metadata.get("chunk_index", 0)
            if "total_chunks" not in payload:
                payload["total_chunks"] = metadata.get("total_chunks", 1)

            point = PointStruct(
                id=point_id_int,
                vector=embedding,
                payload=payload,
            )
            points.append(point)

        logger.info(f"Created {len(points)} vector points")
        return points

    def process_and_store_documents(
        self,
        documents: list[dict],
        mode: str = "update",
        use_llm: Optional[bool] = None,
    ) -> dict:
        """
        處理文件並將其儲存到 Qdrant 向量資料庫

        這是文件處理的核心方法，執行完整的處理流程：
        1. 確保 Qdrant 集合存在（根據 mode 決定是否重建）
        2. 將文件分割成文字塊（序列化處理，避免 LLM 超載）
        3. 為所有文字塊生成嵌入向量
        4. 建立 Qdrant 向量點
        5. 將向量點儲存到資料庫

        參數：
            documents: 要處理的文件列表，每個文件包含 content 和 metadata
            mode: 處理模式，可選值：
                - 'override': 重建集合，刪除所有現有向量後再儲存新向量
                - 'update': 更新模式，新增或更新向量（預設值）
            use_llm: 是否使用 LLM 生成 chunk 脈絡。None 表示使用系統預設值

        回傳：
            dict: 處理結果字典，包含以下鍵值：
                - status (str): 處理狀態 ('success' 或 'warning')
                - mode (str): 使用的處理模式
                - documents_processed (int): 處理的文件數量
                - chunks_created (int): 建立的文字塊數量
                - vectors_stored (int): 儲存的向量數量
                - collection_name (str): Qdrant 集合名稱

        例外：
            Exception: 當處理過程中發生任何錯誤時拋出，錯誤訊息會被記錄
        """
        try:
            total_docs = len(documents)
            actual_use_llm = use_llm if use_llm is not None else settings.contextual_chunking_use_llm

            logger.info(
                f"Processing {total_docs} documents (mode: {mode}, use_llm: {actual_use_llm})"
            )

            # 確保 Qdrant 集合存在
            # 如果 mode 是 'override'，則重建集合
            recreate = mode == "override"
            qdrant_service.ensure_collection(recreate=recreate)

            # 定義進度回調函數
            def log_progress(current: int, total: int, filename: str):
                logger.info(f"[Progress] Document {current}/{total}: {filename}")

            # 將文件分割成文字塊（序列化處理）
            chunks = self.chunk_documents(
                documents,
                use_llm=use_llm,
                progress_callback=log_progress,
            )

            # 檢查是否成功建立文字塊
            if not chunks:
                logger.warning("No chunks created from documents")
                return {
                    "status": "warning",
                    "message": "No chunks created from documents",
                    "documents_processed": 0,
                    "chunks_created": 0,
                    "vectors_stored": 0,
                }

            # 生成嵌入向量
            # 注意：使用原始 text 生成 embedding，而非 contextualized_text
            # 原因：<context> 前綴會稀釋語義匹配，降低檢索精準度
            # contextualized_text 保留給 LLM 生成答案時使用
            logger.info("Generating embeddings...")
            texts = [chunk["text"] for chunk in chunks]
            embeddings = embedding_service.embed_texts(texts)

            # 建立 Qdrant 向量點
            points = self.create_vector_points(chunks, embeddings)

            # 將向量點儲存到 Qdrant
            result = qdrant_service.upsert_vectors(points)

            # 清除受影響文件的語意快取
            # 提取所有被處理文件的檔名
            affected_filenames = list(
                {doc["metadata"].get("filename") for doc in documents if doc.get("metadata", {}).get("filename")}
            )
            cache_invalidated = 0
            if affected_filenames and settings.semantic_cache_enabled:
                cache_invalidated = semantic_cache_service.invalidate_by_filenames(
                    affected_filenames
                )
                if cache_invalidated > 0:
                    logger.info(
                        f"Invalidated {cache_invalidated} cache entries for files: {affected_filenames}"
                    )

            # 組合最終結果
            final_result = {
                "status": "success",
                "mode": mode,
                "documents_processed": len(documents),
                "chunks_created": len(chunks),
                "vectors_stored": result["count"],
                "cache_entries_invalidated": cache_invalidated,
                "collection_name": settings.qdrant_collection_name,
            }

            logger.info(f"Document processing completed: {final_result}")
            return final_result

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise

    def process_directory(
        self,
        directory: Optional[str] = None,
        mode: str = "update",
        use_llm: Optional[bool] = None,
    ) -> dict:
        """
        處理目錄中的所有文件

        這是一個便利方法，用於處理指定目錄（或預設目錄）中的所有 Markdown 文件。
        它會載入目錄中的所有文件，然後呼叫 process_and_store_documents 進行處理。

        參數：
            directory: 包含文件的目錄路徑
                - 如果為 None，則使用 settings 中設定的預設文件路徑
                - 可以是字串或 Path 物件
            mode: 處理模式，可選值：
                - 'override': 重建集合，刪除所有現有向量
                - 'update': 更新模式，新增或更新向量（預設值）
            use_llm: 是否使用 LLM 生成 chunk 脈絡。None 表示使用系統預設值

        回傳：
            dict: 處理結果字典，與 process_and_store_documents 回傳格式相同

        例外：
            Exception: 當處理過程中發生任何錯誤時拋出
        """
        # 如果未指定目錄，使用預設路徑
        if directory is None:
            directory = settings.default_docs_path

        logger.info(f"Processing documents from directory: {directory}")

        # 載入目錄中的所有文件
        documents = self.load_documents_from_directory(directory)

        # 檢查是否找到文件
        if not documents:
            logger.warning(f"No documents found in {directory}")
            return {
                "status": "warning",
                "message": f"No documents found in {directory}",
                "documents_processed": 0,
                "chunks_created": 0,
                "vectors_stored": 0,
            }

        # 處理並儲存文件
        return self.process_and_store_documents(documents, mode=mode, use_llm=use_llm)

    def process_uploaded_files(
        self,
        file_contents: list[tuple[str, str]],
        mode: str = "update",
        use_llm: Optional[bool] = None,
    ) -> dict:
        """
        處理上傳的檔案

        這是一個便利方法，用於處理從 API 或其他來源上傳的檔案內容。
        它會將檔案內容轉換成文件格式，然後呼叫 process_and_store_documents 進行處理。

        參數：
            file_contents: 檔案內容列表，每個元素是一個 (檔案名稱, 內容) 的元組
                例如：[("doc1.md", "內容1"), ("doc2.md", "內容2")]
            mode: 處理模式，可選值：
                - 'override': 重建集合，刪除所有現有向量
                - 'update': 更新模式，新增或更新向量（預設值）
            use_llm: 是否使用 LLM 生成 chunk 脈絡。None 表示使用系統預設值

        回傳：
            dict: 處理結果字典，與 process_and_store_documents 回傳格式相同

        例外：
            Exception: 當處理過程中發生任何錯誤時拋出
        """
        logger.info(f"Processing {len(file_contents)} uploaded files")

        # 將上傳的檔案內容轉換成文件格式
        # 來源路徑標記為 "uploaded/"，表示這些是上傳的檔案
        documents = [
            {
                "content": content,
                "metadata": {
                    "source": f"uploaded/{filename}",
                    "filename": filename,
                },
            }
            for filename, content in file_contents
        ]

        # 處理並儲存文件
        return self.process_and_store_documents(documents, mode=mode, use_llm=use_llm)


# 全域實例
# 建立 DocumentService 的單例實例，可在其他模組中直接導入使用
# 例如：from chatbot_rag.services.document_service import document_service
document_service = DocumentService()
