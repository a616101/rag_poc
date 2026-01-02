"""
GraphRAG 嵌入向量服務

提供非同步介面生成密集和稀疏嵌入向量。
支援 OpenAI 相容 API 的密集嵌入和 SPLADE 稀疏嵌入。

功能：
- 密集嵌入：透過 OpenAI 相容 API
- 稀疏嵌入：透過 SPLADE（可選）
- 批次處理以提高效率
"""

import asyncio
import logging
from typing import Any, Optional

from openai import AsyncOpenAI

from chatbot_graphrag.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    GraphRAG 非同步嵌入服務。

    支援：
    - 透過 OpenAI 相容 API 生成密集嵌入
    - 透過 SPLADE 生成稀疏嵌入（可選）
    - 批次處理以提高效率
    """

    _instance: Optional["EmbeddingService"] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> "EmbeddingService":
        """單例模式以重用客戶端。"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化嵌入服務。"""
        if self._initialized:
            return

        self._client: Optional[AsyncOpenAI] = None
        self._sparse_encoder: Optional[Any] = None
        self._sparse_tokenizer: Optional[Any] = None
        self._model = settings.embedding_model
        self._dimension = settings.embedding_dimension
        self._sparse_enabled = settings.sparse_encoder_enabled
        self._sparse_model = settings.sparse_encoder_model
        self._initialized = True

    async def initialize(self) -> None:
        """初始化嵌入客戶端。"""
        if self._client is not None:
            return

        # 初始化 OpenAI 客戶端用於密集嵌入
        # 使用獨立的 Embedding API 設定（如果有設定），否則使用預設的 Chat API 設定
        self._client = AsyncOpenAI(
            api_key=settings.effective_embedding_api_key,
            base_url=settings.effective_embedding_api_base,
            timeout=60.0,
        )

        logger.info(f"Embedding client initialized: {settings.effective_embedding_api_base}")
        logger.info(f"Model: {self._model}, Dimension: {self._dimension}")

        # 如果啟用則初始化 SPLADE 編碼器
        if self._sparse_enabled:
            await self._init_sparse_encoder()

    async def _init_sparse_encoder(self) -> None:
        """初始化 SPLADE 稀疏編碼器。"""
        try:
            # 延遲導入以避免啟動開銷
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            import torch

            logger.info(f"Loading sparse encoder: {self._sparse_model}")

            # 在執行器中運行以避免阻塞
            loop = asyncio.get_event_loop()

            def load_splade():
                tokenizer = AutoTokenizer.from_pretrained(self._sparse_model)
                model = AutoModelForMaskedLM.from_pretrained(self._sparse_model)
                model.eval()
                # 如果可用則移至 GPU
                if torch.cuda.is_available():
                    model = model.cuda()
                return tokenizer, model

            self._sparse_tokenizer, self._sparse_encoder = await loop.run_in_executor(
                None, load_splade
            )

            logger.info("Sparse encoder initialized successfully")

        except ImportError as e:
            logger.warning(f"Could not import transformers/torch: {e}")
            logger.warning("Sparse embeddings will be disabled")
            self._sparse_enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize sparse encoder: {e}")
            self._sparse_enabled = False

    async def close(self) -> None:
        """關閉嵌入服務。"""
        if self._client:
            await self._client.close()
            self._client = None

        self._sparse_encoder = None
        self._sparse_tokenizer = None
        logger.info("Embedding service closed")

    # ==================== 密集嵌入 ====================

    async def embed_text(self, text: str) -> list[float]:
        """
        為單一文本生成密集嵌入向量。

        Args:
            text: 要嵌入的文本

        Returns:
            密集嵌入向量
        """
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 20,
    ) -> list[list[float]]:
        """
        為多個文本生成密集嵌入向量。

        Args:
            texts: 要嵌入的文本列表
            batch_size: 每次 API 呼叫的文本數量

        Returns:
            密集嵌入向量列表
        """
        if not texts:
            return []

        if self._client is None:
            await self.initialize()

        # 使用並發控制進行嵌入 API 呼叫
        from chatbot_graphrag.core.concurrency import llm_concurrency

        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size + 1

            try:
                # 使用嵌入信號量進行 API 呼叫
                async with llm_concurrency.acquire("embedding"):
                    response = await self._client.embeddings.create(
                        model=self._model,
                        input=batch,
                    )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                if batch_num % 10 == 0 or batch_num == total_batches:
                    logger.debug(
                        f"Embedding progress: {batch_num}/{total_batches} batches"
                    )

            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {batch_num}: {e}")
                raise

        return all_embeddings

    # ==================== 稀疏嵌入 ====================

    async def sparse_embed_text(self, text: str) -> Optional[dict[int, float]]:
        """
        使用 SPLADE 為單一文本生成稀疏嵌入向量。

        Args:
            text: 要嵌入的文本

        Returns:
            稀疏嵌入為 {token_id: weight} 字典，如果停用則為 None
        """
        if not self._sparse_enabled or self._sparse_encoder is None:
            return None

        results = await self.sparse_embed_texts([text])
        return results[0] if results else None

    async def sparse_embed_texts(
        self,
        texts: list[str],
        batch_size: int = 8,
    ) -> list[Optional[dict[int, float]]]:
        """
        使用 SPLADE 為多個文本生成稀疏嵌入向量。

        Args:
            texts: 要嵌入的文本列表
            batch_size: 每批次的文本數量

        Returns:
            稀疏嵌入為 {token_id: weight} 字典的列表
        """
        if not self._sparse_enabled or self._sparse_encoder is None:
            return [None] * len(texts)

        if not texts:
            return []

        loop = asyncio.get_event_loop()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            try:
                # 在執行器中運行 SPLADE 編碼
                batch_embeddings = await loop.run_in_executor(
                    None,
                    self._encode_sparse_batch,
                    batch,
                )
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"Failed to generate sparse embeddings: {e}")
                # 對失敗的批次返回 None
                all_embeddings.extend([None] * len(batch))

        return all_embeddings

    def _encode_sparse_batch(self, texts: list[str]) -> list[dict[int, float]]:
        """使用 SPLADE 編碼文本批次（同步）。"""
        import torch

        results = []

        # 分詞
        inputs = self._sparse_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # 移至設備
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # 獲取 SPLADE 分數
        with torch.no_grad():
            outputs = self._sparse_encoder(**inputs)
            # SPLADE: 序列最大池化，然後 ReLU + log(1+x)
            logits = outputs.logits
            attention_mask = inputs["attention_mask"].unsqueeze(-1)

            # 使用注意力遮罩的最大池化
            logits = logits.masked_fill(~attention_mask.bool(), float("-inf"))
            max_logits, _ = logits.max(dim=1)

            # SPLADE 激活函數: log(1 + ReLU(x))
            sparse_vecs = torch.log1p(torch.relu(max_logits))

        # 轉換為稀疏字典格式
        for vec in sparse_vecs:
            # 獲取非零索引和值
            nonzero_mask = vec > 0
            indices = nonzero_mask.nonzero().squeeze(-1).cpu().tolist()
            values = vec[nonzero_mask].cpu().tolist()

            if isinstance(indices, int):
                indices = [indices]
                values = [values]

            sparse_dict = {idx: val for idx, val in zip(indices, values)}
            results.append(sparse_dict)

        return results

    # ==================== 組合嵌入 ====================

    async def embed_with_sparse(
        self,
        texts: list[str],
        dense_batch_size: int = 20,
        sparse_batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        """
        為文本同時生成密集和稀疏嵌入向量。

        Args:
            texts: 要嵌入的文本列表
            dense_batch_size: 密集嵌入的批次大小
            sparse_batch_size: 稀疏嵌入的批次大小

        Returns:
            包含 'dense' 和 'sparse' 鍵的字典列表
        """
        if not texts:
            return []

        # 並行生成嵌入
        dense_task = self.embed_texts(texts, batch_size=dense_batch_size)

        if self._sparse_enabled:
            sparse_task = self.sparse_embed_texts(texts, batch_size=sparse_batch_size)
            dense_embeddings, sparse_embeddings = await asyncio.gather(
                dense_task, sparse_task
            )
        else:
            dense_embeddings = await dense_task
            sparse_embeddings = [None] * len(texts)

        # 組合結果
        results = []
        for dense, sparse in zip(dense_embeddings, sparse_embeddings):
            results.append({
                "dense": dense,
                "sparse": sparse,
            })

        return results

    # ==================== 健康檢查 ====================

    async def health_check(self) -> dict[str, Any]:
        """檢查嵌入服務健康狀態。"""
        try:
            if self._client is None:
                await self.initialize()

            # 測試密集嵌入
            test_embedding = await self.embed_text("test")

            result = {
                "status": "healthy",
                "model": self._model,
                "dimension": len(test_embedding),
                "expected_dimension": self._dimension,
                "dimension_match": len(test_embedding) == self._dimension,
                "sparse_enabled": self._sparse_enabled,
            }

            # 如果啟用則測試稀疏嵌入
            if self._sparse_enabled:
                sparse_test = await self.sparse_embed_text("test")
                result["sparse_working"] = sparse_test is not None
                if sparse_test:
                    result["sparse_vocab_size"] = len(sparse_test)

            return result

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }


# 單例實例
embedding_service = EmbeddingService()
