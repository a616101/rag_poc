"""
文件檢索服務模組

此模組提供 RAG (Retrieval-Augmented Generation) 系統的核心檢索功能。
主要負責根據使用者查詢，從 Qdrant 向量資料庫中檢索相關文件片段。

功能特點：
    - 語義搜尋：使用向量相似度檢索相關文件
    - 相似度過濾：支援設定最低相似度閾值
    - 結果格式化：將檢索結果格式化為可讀的上下文字串
    - 測試功能：提供檢索功能的測試介面

使用範例：
    >>> retriever_service.retrieve("如何申請退款", top_k=5)
    >>> context = retriever_service.format_context(documents)
"""

from typing import Optional

from loguru import logger

from chatbot_rag.core.config import settings
from chatbot_rag.services.embedding_service import embedding_service
from chatbot_rag.services.qdrant_service import qdrant_service


class RetrieverService:
    """
    文件檢索服務類別

    此類別封裝了從向量資料庫檢索相關文件的所有功能。
    它整合了文字嵌入服務和 Qdrant 向量搜尋服務，提供完整的檢索流程。

    工作流程：
        1. 接收使用者查詢文字
        2. 將查詢文字轉換為向量嵌入
        3. 在 Qdrant 中搜尋相似向量
        4. 格式化並返回檢索結果

    屬性：
        無特殊屬性，所有服務依賴都透過全域實例注入
    """

    def __init__(self):
        """
        初始化檢索服務

        目前不需要特殊初始化邏輯，保留此方法以便未來擴展。
        所有依賴的服務（embedding_service, qdrant_service）都使用全域實例。
        """
        pass

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = 0.5,
        expand_context: bool = False,
        context_window: int = 1,
    ) -> list[dict]:
        """
        檢索與查詢相關的文件片段

        此方法是檢索服務的核心功能，執行完整的語義搜尋流程：
        1. 將查詢文字轉換為向量嵌入
        2. 在向量資料庫中搜尋相似文件
        3. 根據相似度閾值過濾結果
        4. （可選）擴充上下文：取得相鄰的 chunks
        5. 格式化並返回檢索結果

        參數：
            query (str): 使用者的查詢文字，用於搜尋相關文件
            top_k (int, optional): 要檢索的文件數量上限。預設為 5
            score_threshold (Optional[float], optional): 最低相似度分數閾值 (0-1)。
                低於此分數的文件會被過濾掉。預設為 0.5
            expand_context (bool, optional): 是否擴充上下文（取得相鄰 chunks）。預設為 True
            context_window (int, optional): 上下文窗口大小（前後各取幾個 chunks）。預設為 1

        返回：
            list[dict]: 檢索到的文件列表，每個文件包含以下欄位：
                - content (str): 文件內容文字（若擴充上下文則包含相鄰 chunks）
                - source (str): 來源類型（如 "markdown"）
                - filename (str): 檔案名稱
                - chunk_index (int): 文件分塊索引
                - score (float): 相似度分數 (0-1)

        拋出異常：
            Exception: 當檢索過程發生錯誤時，如：
                - 向量嵌入生成失敗
                - Qdrant 搜尋失敗
                - 資料格式錯誤

        使用範例：
            >>> docs = retriever_service.retrieve("如何退款", top_k=3, score_threshold=0.7)
            >>> print(f"找到 {len(docs)} 個相關文件")
        """
        try:
            # 記錄檢索開始，截斷過長的查詢文字以避免日誌過長
            logger.info(f"Retrieving documents for query: '{query[:50]}...'")

            # 步驟 1: 將查詢文字轉換為向量嵌入
            # 使用 embedding_service 生成與文件相同維度的查詢向量
            query_embedding = embedding_service.embed_text(query)

            # 步驟 2: 在 Qdrant 向量資料庫中搜尋相似文件
            # 使用餘弦相似度找出最相關的文件片段
            search_results = qdrant_service.search_similar(
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
            )

            # 步驟 3: 格式化搜尋結果為統一的文件格式
            documents = []
            for result in search_results:
                payload = result.get("payload", {}) or {}
                filename = payload.get("filename", "unknown")
                chunk_index = payload.get("chunk_index", 0)
                original_text = payload.get("text", "")

                # 優先使用 contextualized_text（包含脈絡資訊），讓 LLM 能看到文件脈絡
                # 若沒有則退回使用原始 text
                base_content = payload.get("contextualized_text") or original_text

                # 步驟 4: 擴充上下文（取得相鄰 chunks）
                expanded_content = base_content
                if expand_context and context_window > 0:
                    expanded_content = self._expand_context_with_neighbors(
                        filename=filename,
                        chunk_index=chunk_index,
                        fallback_text=base_content,
                        window=context_window,
                    )

                # 從 Qdrant 的 payload 中提取文件資訊
                # 使用 .get() 方法提供預設值，避免欄位缺失時出錯
                doc = {
                    "content": expanded_content,
                    "source": payload.get("source", "unknown"),
                    "filename": filename,
                    "chunk_index": chunk_index,
                    "score": result["score"],  # 相似度分數
                    # 將完整 payload 保留在 metadata 中，讓上層可以存取
                    # 例如 entry_type、module、required_forms、download_paths 等
                    "metadata": payload,
                }
                documents.append(doc)

            # 記錄檢索成功，顯示文件數量和相似度分數
            logger.info(
                f"Retrieved {len(documents)} documents with scores: "
                f"{[d['score'] for d in documents]}"
            )

            return documents

        except Exception as e:
            # 捕獲所有異常，記錄錯誤並重新拋出
            logger.error(f"Document retrieval failed: {e}")
            raise

    def _expand_context_with_neighbors(
        self,
        filename: str,
        chunk_index: int,
        fallback_text: str,
        window: int = 1,
    ) -> str:
        """
        擴充上下文：取得相鄰 chunks 並合併。

        Args:
            filename: 檔案名稱
            chunk_index: 目標 chunk 的索引
            fallback_text: 找不到相鄰 chunks 時的備用文字
            window: 上下文窗口大小（前後各取幾個 chunks）

        Returns:
            str: 合併後的文字（包含相鄰 chunks，優先使用 contextualized_text）
        """
        try:
            neighbors = qdrant_service.fetch_neighboring_chunks(
                filename=filename,
                chunk_index=chunk_index,
                window=window,
            )

            if not neighbors or len(neighbors) <= 1:
                # 沒有相鄰 chunks 或只有自己
                return fallback_text

            # 按 chunk_index 排序並合併文字
            # 優先使用 contextualized_text（包含脈絡資訊）
            texts = []
            for neighbor in neighbors:
                payload = neighbor.get("payload", {})
                # 優先使用 contextualized_text，否則退回 text
                text = payload.get("contextualized_text") or payload.get("text", "")
                if text:
                    texts.append(text)

            if not texts:
                return fallback_text

            # 合併時使用空行分隔（因為 chunk_overlap 已處理重疊）
            merged_text = "\n".join(texts)

            logger.debug(
                f"Expanded context for {filename}[{chunk_index}]: "
                f"{len(fallback_text)} -> {len(merged_text)} chars"
            )

            return merged_text

        except Exception as e:
            logger.warning(f"Failed to expand context: {e}")
            return fallback_text

    def format_context(self, documents: list[dict]) -> str:
        """
        將檢索到的文件格式化為上下文字串

        此方法將檢索結果轉換為結構化的文字格式，方便 LLM 理解和使用。
        每個文件都會標註來源、相似度分數等資訊，幫助追蹤資訊來源。

        參數：
            documents (list[dict]): 檢索到的文件列表，每個文件應包含：
                - content: 文件內容
                - filename: 檔案名稱
                - score: 相似度分數

        返回：
            str: 格式化的上下文字串，包含所有文件內容和元資訊。
                如果沒有文件，返回提示訊息。

        格式範例：
            [Document 1] (來源: refund_policy.md, 相似度: 0.85)
            退款申請需在購買後 7 天內提出...

            [Document 2] (來源: faq.md, 相似度: 0.72)
            常見退款問題...

        使用範例：
            >>> docs = retriever_service.retrieve("退款政策")
            >>> context = retriever_service.format_context(docs)
            >>> print(context)
        """
        # 檢查是否有檢索到文件
        if not documents:
            return "No relevant documents found."

        # 將每個文件格式化為帶編號的區塊
        context_parts = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.get("metadata", {}) or {}
            entry_type = metadata.get("entry_type")
            module = metadata.get("module")

            # 基本標題行：包含來源檔名、類型與相似度
            header_parts: list[str] = [f"[Document {i}]"]
            header_parts.append(f"(來源: {doc['filename']}")
            if entry_type:
                header_parts.append(f" 類型: {entry_type}")
            if module:
                header_parts.append(f" 模組: {module}")
            header_parts.append(f" 相似度: {doc['score']:.2f})")
            header = " ".join(header_parts)

            extra_lines: list[str] = []

            # 若為表單相關文件，盡量把可下載路徑整理出來，讓 LLM 能直接使用
            # 1) form_catalog：通常會有 required_forms + download_paths
            if entry_type == "form_catalog":
                required_forms = metadata.get("required_forms") or []
                if isinstance(required_forms, list):
                    for form in required_forms:
                        if not isinstance(form, dict):
                            continue
                        form_name = form.get("name") or form.get("form_id") or ""
                        formats = form.get("format") or []
                        if isinstance(formats, list):
                            formats_str = ",".join(str(f) for f in formats)
                        else:
                            formats_str = str(formats)
                        download_paths = form.get("download_paths") or []
                        if isinstance(download_paths, list):
                            for path in download_paths:
                                extra_lines.append(
                                    f"表單下載：{form_name} (格式: {formats_str}) 下載路徑: {path}"
                                )
                        elif download_paths:
                            extra_lines.append(
                                f"表單下載：{form_name} (格式: {formats_str}) 下載路徑: {download_paths}"
                            )

            # 2) form_template：通常會有 file_templates[].path
            if entry_type == "form_template":
                file_templates = metadata.get("file_templates") or []
                if isinstance(file_templates, list):
                    for tpl in file_templates:
                        if not isinstance(tpl, dict):
                            continue
                        fmt = tpl.get("format") or tpl.get("type") or ""
                        path = tpl.get("path") or ""
                        if path:
                            extra_lines.append(
                                f"表單範本下載：格式: {fmt} 下載路徑: {path}"
                            )

            # 組合額外資訊與內容
            body_parts = []
            if extra_lines:
                body_parts.append("\n".join(extra_lines))
            body_parts.append(doc["content"])
            body = "\n".join(body_parts)

            context_parts.append(f"{header}\n{body}\n")

        # 使用換行符連接所有文件區塊
        context = "\n".join(context_parts)

        # 記錄格式化後的上下文長度，用於監控和除錯
        logger.debug(f"Formatted context: {len(context)} chars")

        return context

    def test_retrieval(self, test_query: str = "如何申請退款") -> dict:
        """
        測試檢索功能是否正常運作

        此方法提供一個簡單的測試介面，用於驗證檢索服務的各個環節：
        - 向量嵌入生成
        - Qdrant 搜尋
        - 結果格式化
        可用於開發除錯、系統健康檢查或示範用途。

        參數：
            test_query (str, optional): 用於測試的查詢文字。
                預設值為 "如何申請退款"，這是一個常見的測試查詢。

        返回：
            dict: 測試結果，包含以下欄位：
                - status (str): 測試狀態 ("success" 表示成功)
                - query (str): 使用的測試查詢
                - documents_found (int): 找到的文件數量
                - documents (list): 檢索到的文件列表

        拋出異常：
            Exception: 當測試過程中發生任何錯誤時，如：
                - 檢索服務不可用
                - 向量資料庫連接失敗
                - 資料格式錯誤

        使用範例：
            >>> result = retriever_service.test_retrieval()
            >>> print(f"測試狀態: {result['status']}")
            >>> print(f"找到 {result['documents_found']} 個文件")

            >>> # 使用自訂查詢測試
            >>> result = retriever_service.test_retrieval("學習規範")
            >>> for doc in result['documents']:
            >>>     print(f"- {doc['filename']}: {doc['score']:.2f}")
        """
        try:
            # 記錄測試開始
            logger.info(f"Testing retrieval with query: {test_query}")

            # 執行檢索，限制返回 3 個文件以加快測試速度
            documents = self.retrieve(query=test_query, top_k=3)

            # 組裝測試結果
            result = {
                "status": "success",
                "query": test_query,
                "documents_found": len(documents),
                "documents": documents,
            }

            # 記錄測試成功
            logger.info("Retrieval test completed successfully")
            return result

        except Exception as e:
            # 捕獲測試過程中的異常，記錄錯誤並重新拋出
            logger.error(f"Retrieval test failed: {e}")
            raise


# 全域實例
# 建立 RetrieverService 的單例實例，供整個應用程式使用
# 這種模式確保所有模組使用同一個檢索服務實例，避免重複初始化
retriever_service = RetrieverService()
