"""
文字嵌入服務模組

此模組提供使用 OpenAI 相容 API 生成文字嵌入向量的功能。
嵌入向量是文字的數值表示，用於語義相似度計算和向量檢索。
支援單個文字和批次處理多個文字。
"""

from typing import Optional

from loguru import logger
from openai import OpenAI

from chatbot_rag.core.config import settings


class EmbeddingService:
    """
    文字嵌入服務類別

    此類別負責與 OpenAI 相容的 API（如 LMStudio）連接，
    將文字轉換為高維度的向量表示。這些向量可用於：
    - 文件相似度計算
    - 語義搜尋
    - 文件檢索和排序
    """

    def __init__(self):
        """
        初始化嵌入服務

        建立服務實例，設定模型名稱，但不立即建立連線。
        連線將在首次需要時建立（延遲初始化）。
        """
        self.client: Optional[OpenAI] = None  # OpenAI 客戶端實例（延遲初始化）
        self.model = settings.embedding_model  # 使用的嵌入模型名稱

    def connect(self) -> OpenAI:
        """
        連接到 OpenAI 相容的 API

        如果尚未建立連線，則建立新的連線。
        使用單例模式，確保只建立一個客戶端實例。

        Returns:
            OpenAI: 已連接的 OpenAI 客戶端實例

        Raises:
            Exception: 當連接失敗時拋出異常
        """
        if self.client is None:
            try:
                logger.info(
                    f"Connecting to OpenAI-compatible API at {settings.openai_api_base}"
                )
                # 建立 OpenAI 客戶端，連接到設定的 API 端點
                self.client = OpenAI(
                    api_key="lmstudio",  # API 金鑰
                    base_url="http://192.168.50.152:1234/v1",  # API 基礎 URL
                )
                logger.info("Successfully connected to embedding service")
            except Exception as e:
                logger.error(f"Failed to connect to embedding service: {e}")
                raise

        return self.client

    def embed_text(self, text: str) -> list[float]:
        """
        為單個文字生成嵌入向量

        將輸入的文字轉換為固定維度的數值向量表示。
        這個向量捕捉了文字的語義資訊。

        Args:
            text: 要轉換為嵌入向量的文字

        Returns:
            list[float]: 嵌入向量，一個浮點數列表

        Raises:
            Exception: 當嵌入向量生成失敗時拋出異常
        """
        try:
            # 確保已連接到 API
            client = self.connect()

            # 呼叫 API 生成嵌入向量
            response = client.embeddings.create(
                model=self.model,  # 使用設定的模型
                input=[text],  # 輸入文字
            )

            # 提取嵌入向量（API 回應的第一個元素）
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding of dimension {len(embedding)}")

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def embed_texts(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        """
        批次處理多個文字並生成嵌入向量

        為了提高效率，將多個文字分批處理。
        這對於處理大量文件時特別有用，可以減少 API 呼叫次數。

        Args:
            texts: 要轉換的文字列表
            batch_size: 每批次處理的文字數量，預設為 10

        Returns:
            list[list[float]]: 嵌入向量列表，每個元素對應一個輸入文字的向量

        Raises:
            Exception: 當嵌入向量生成失敗時拋出異常
        """
        try:
            # 確保已連接到 API
            client = self.connect()
            embeddings = []

            logger.info(f"Generating embeddings for {len(texts)} texts")

            # 分批處理文字
            for i in range(0, len(texts), batch_size):
                # 取得當前批次的文字
                batch = texts[i : i + batch_size]
                logger.debug(
                    f"Processing batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}"
                )

                # 為當前批次生成嵌入向量
                response = client.embeddings.create(
                    model=self.model,
                    input=batch,  # 批次輸入
                )

                # 提取批次中所有文字的嵌入向量
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def test_connection(self) -> dict:
        """
        測試嵌入服務的連接

        通過生成一個測試嵌入向量來驗證服務是否正常運作，
        並檢查向量維度是否符合預期。

        Returns:
            dict: 測試結果字典，包含以下鍵值：
                - status: 測試狀態
                - model: 使用的模型名稱
                - api_base: API 基礎 URL
                - embedding_dimension: 實際向量維度
                - expected_dimension: 預期向量維度
                - dimension_match: 維度是否匹配

        Raises:
            Exception: 當測試失敗時拋出異常
        """
        try:
            logger.info("Testing embedding service connection")

            # 生成測試嵌入向量
            test_text = "This is a test sentence."
            embedding = self.embed_text(test_text)

            # 組裝測試結果
            result = {
                "status": "success",
                "model": self.model,
                "api_base": settings.openai_api_base,
                "embedding_dimension": len(embedding),
                "expected_dimension": settings.embedding_dimension,
                "dimension_match": len(embedding) == settings.embedding_dimension,
            }

            # 如果向量維度不匹配，記錄警告
            if not result["dimension_match"]:
                logger.warning(
                    f"Embedding dimension mismatch: got {len(embedding)}, "
                    f"expected {settings.embedding_dimension}"
                )

            logger.info("Embedding service test completed successfully")
            return result

        except Exception as e:
            logger.error(f"Embedding service test failed: {e}")
            raise


# 建立全域嵌入服務實例，供整個應用程式使用
embedding_service = EmbeddingService()
