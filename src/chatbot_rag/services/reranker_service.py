"""
Reranker 服務模組

此模組提供文件重新排序功能，使用 Cross-Encoder 模型對檢索結果進行精確排序。
支援多種 Reranker 提供者：
- OpenAI 相容端點（LMStudio、vLLM 等）- 使用 LLM 進行 Pointwise Reranking
- Jina AI Reranker API
- Cohere Rerank API
- 本地 Cross-Encoder（sentence-transformers）

主要功能：
- 對檢索到的文件進行重新排序
- 基於分數閾值過濾低相關性結果
- 支援批次處理多個查詢

注意：Langfuse trace 整合由 LangGraph reranker 節點負責（見 graph/nodes/reranker.py）
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx
from loguru import logger

from chatbot_rag.core.config import settings

class RerankerService:
    """
    Reranker 服務類別

    使用 Cross-Encoder 模型對檢索結果進行重新排序，
    提高檢索結果的相關性和準確度。
    """

    def __init__(self):
        """初始化 Reranker 服務"""
        self._http_client: Optional[httpx.Client] = None
        self._openai_client: Optional["OpenAI"] = None
        self._executor = ThreadPoolExecutor(max_workers=5)

    @property
    def http_client(self) -> httpx.Client:
        """取得或建立 HTTP client（延遲載入）"""
        if self._http_client is None:
            self._http_client = httpx.Client(
                timeout=settings.reranker_timeout,
            )
        return self._http_client

    @property
    def openai_client(self) -> "OpenAI":
        """取得或建立 OpenAI client（延遲載入）"""
        if self._openai_client is None:
            from openai import OpenAI

            # 使用 reranker 專用設定，若無則使用通用 LLM 設定
            api_base = settings.reranker_api_base or settings.openai_api_base
            api_key = settings.reranker_api_key or settings.openai_api_key or "lm-studio"

            self._openai_client = OpenAI(
                api_key=api_key,
                base_url=api_base,
                timeout=settings.reranker_timeout,
            )
            logger.info(f"[RERANKER] OpenAI client connected to {api_base}")
        return self._openai_client

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> list[dict]:
        """
        對文件列表進行重新排序。

        Args:
            query: 查詢文字
            documents: 文件列表，每個文件需包含 'content' 欄位
            top_k: 返回前 k 個結果（None 表示不限制，由閾值決定）
            score_threshold: 分數閾值，低於此分數的結果會被過濾

        Returns:
            list[dict]: 重新排序後的文件列表，新增 'rerank_score' 欄位
        """
        if not settings.reranker_enabled:
            logger.debug("[RERANKER] Reranker disabled, returning original order")
            return documents

        if not documents:
            return []

        if not query.strip():
            logger.warning("[RERANKER] Empty query, returning original order")
            return documents

        # 使用設定值或參數值
        threshold = score_threshold if score_threshold is not None else settings.reranker_score_threshold
        provider = settings.reranker_provider.lower()

        # 開始計時
        start_time = time.monotonic()

        try:
            if provider == "openai":
                reranked = self._rerank_with_openai(query, documents)
            elif provider == "jina":
                reranked = self._rerank_with_jina(query, documents)
            elif provider == "cohere":
                reranked = self._rerank_with_cohere(query, documents)
            elif provider == "local":
                reranked = self._rerank_with_local(query, documents)
            else:
                logger.warning(f"[RERANKER] Unknown provider: {provider}, returning original order")
                return documents

            # 按 rerank_score 降序排序
            reranked.sort(key=lambda d: d.get("rerank_score", 0), reverse=True)

            # 記錄排序後的分數（供 Langfuse 顯示）
            scores_summary = [
                {
                    "filename": d.get("filename", "unknown")[:30],
                    "score": round(d.get("rerank_score", 0), 3),
                }
                for d in reranked[:10]  # 只記錄前 10 個
            ]

            # 應用分數閾值過濾
            filtered_count = 0
            if threshold > 0 and reranked:
                before_count = len(reranked)

                # 策略 1: 絕對閾值過濾
                reranked = [d for d in reranked if d.get("rerank_score", 0) >= threshold]

                # 策略 2: 相對分數過濾 - 如果最高分與其他分數差距太大，只保留高分文件
                # 這可以處理「全部分數都在 0.5-0.7 之間但都不相關」的情況
                if len(reranked) > 1:
                    max_score = reranked[0].get("rerank_score", 0)
                    # 保留分數在最高分 80% 以上的文件
                    relative_threshold = max_score * 0.8
                    reranked_relative = [
                        d for d in reranked
                        if d.get("rerank_score", 0) >= relative_threshold
                    ]
                    # 只有當相對過濾確實減少了文件數量時才使用
                    if len(reranked_relative) < len(reranked):
                        logger.info(
                            f"[RERANKER] Relative filtering: keeping docs with score >= {relative_threshold:.3f} "
                            f"(80% of max {max_score:.3f})"
                        )
                        reranked = reranked_relative

                filtered_count = before_count - len(reranked)
                if filtered_count > 0:
                    logger.info(
                        f"[RERANKER] Filtered by threshold {threshold}: {before_count} -> {len(reranked)}"
                    )

            # 應用 top_k 限制
            if top_k is not None and len(reranked) > top_k:
                reranked = reranked[:top_k]

            duration_ms = (time.monotonic() - start_time) * 1000

            logger.info(
                f"[RERANKER] Reranked {len(documents)} docs -> {len(reranked)} results "
                f"(provider={provider}, threshold={threshold}, duration={duration_ms:.1f}ms)"
            )

            return reranked

        except Exception as e:
            logger.error(f"[RERANKER] Rerank failed: {e}")
            # 失敗時返回原始順序
            return documents

    def _rerank_with_openai(
        self,
        query: str,
        documents: list[dict],
    ) -> list[dict]:
        """
        使用 OpenAI 相容端點進行 LLM-based Reranking（Pointwise Reranking）。

        此方法透過 LLM 為每個文件評估與查詢的相關性分數（0-1）。
        適用於 LMStudio、vLLM、Ollama 等 OpenAI 相容的本地部署服務。

        Pointwise Reranking 策略：
        - 對每個文件獨立評分，不進行文件間比較
        - 使用結構化輸出確保分數格式正確
        - 支援批次處理以提高效率

        Args:
            query: 查詢文字
            documents: 文件列表

        Returns:
            list[dict]: 包含 rerank_score 的文件列表
        """
        # 批次大小：每次請求評估的文件數
        batch_size = settings.reranker_batch_size

        # 建立評分 prompt
        system_prompt = """你是一個文件相關性評估專家。你的任務是評估文件與查詢的相關程度。

請為每個文件評分，分數範圍 0.0 到 1.0：
- 1.0: 完全相關，直接回答查詢
- 0.7-0.9: 高度相關，包含查詢所需的主要資訊
- 0.4-0.6: 部分相關，包含一些有用資訊
- 0.1-0.3: 低度相關，只有間接關聯
- 0.0: 完全不相關

輸出格式：每行一個分數，對應文件順序，例如：
0.85
0.42
0.91"""

        reranked = []
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(documents), batch_size):
            batch_docs = documents[batch_idx:batch_idx + batch_size]
            current_batch = batch_idx // batch_size + 1

            # 建立文件列表
            doc_list = []
            for i, doc in enumerate(batch_docs):
                content = doc.get("content", "")[:1000]  # 限制內容長度
                doc_list.append(f"[文件 {i + 1}]\n{content}")

            user_prompt = f"""查詢：{query}

{chr(10).join(doc_list)}

請為上述 {len(batch_docs)} 個文件評分（每行一個 0.0-1.0 的分數）："""

            try:
                response = self.openai_client.chat.completions.create(
                    model=settings.reranker_model or settings.chat_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,  # 確保評分穩定
                    max_tokens=100,  # 分數輸出不需要太多 tokens
                )

                # 解析分數
                response_text = response.choices[0].message.content.strip()
                scores = self._parse_scores(response_text, len(batch_docs))

                # 更新文件分數
                for i, doc in enumerate(batch_docs):
                    new_doc = dict(doc)
                    new_doc["rerank_score"] = scores[i] if i < len(scores) else 0.0
                    reranked.append(new_doc)

                logger.debug(
                    f"[RERANKER] OpenAI batch {current_batch}/{total_batches}: "
                    f"scores={scores}"
                )

            except Exception as e:
                logger.warning(
                    f"[RERANKER] OpenAI batch {current_batch} failed: {e}, "
                    f"assigning default scores"
                )
                # 失敗時使用原始相似度分數或預設值
                for doc in batch_docs:
                    new_doc = dict(doc)
                    # 使用原始 embedding 分數作為 fallback
                    new_doc["rerank_score"] = doc.get("score", 0.5)
                    reranked.append(new_doc)

        return reranked

    def _parse_scores(self, response_text: str, expected_count: int) -> list[float]:
        """
        解析 LLM 回應中的分數。

        支援多種格式：
        - 純數字每行一個：0.85
        - 帶標籤：1. 0.85 或 文件1: 0.85
        - JSON 格式：[0.85, 0.42]

        Args:
            response_text: LLM 回應文字
            expected_count: 預期的分數數量

        Returns:
            list[float]: 解析出的分數列表
        """
        scores = []

        # 嘗試 JSON 解析
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, list):
                scores = [float(s) for s in parsed]
                if len(scores) >= expected_count:
                    return scores[:expected_count]
        except (json.JSONDecodeError, ValueError):
            pass

        # 使用正則表達式提取數字
        pattern = r"(?:^|\s|:|\]|\[|,)(\d+\.?\d*)"
        matches = re.findall(pattern, response_text)

        for match in matches:
            try:
                score = float(match)
                # 確保分數在合理範圍
                if 0.0 <= score <= 1.0:
                    scores.append(score)
                elif 0 <= score <= 100:
                    # 處理百分比格式
                    scores.append(score / 100)
            except ValueError:
                continue

            if len(scores) >= expected_count:
                break

        # 如果解析的分數不足，用 0.5 補齊
        while len(scores) < expected_count:
            scores.append(0.5)

        return scores[:expected_count]

    def _rerank_with_jina(
        self,
        query: str,
        documents: list[dict],
    ) -> list[dict]:
        """
        使用 Jina AI Reranker API 進行重新排序。

        Jina Reranker API 文件：https://jina.ai/reranker/
        """
        api_key = settings.reranker_api_key
        if not api_key:
            logger.warning("[RERANKER] Jina API key not set, returning original order")
            return documents

        # 準備文件內容
        doc_texts = [d.get("content", "") for d in documents]

        try:
            response = self.http_client.post(
                "https://api.jina.ai/v1/rerank",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.reranker_model,
                    "query": query,
                    "documents": doc_texts,
                    "top_n": len(doc_texts),  # 取得所有結果的分數
                },
            )
            response.raise_for_status()
            result = response.json()

            # 解析結果並更新文件
            results = result.get("results", [])
            reranked = []
            for item in results:
                idx = item.get("index", 0)
                score = item.get("relevance_score", 0)
                if 0 <= idx < len(documents):
                    doc = dict(documents[idx])
                    doc["rerank_score"] = score
                    reranked.append(doc)

            return reranked

        except httpx.HTTPStatusError as e:
            logger.error(f"[RERANKER] Jina API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"[RERANKER] Jina rerank failed: {e}")
            raise

    def _rerank_with_cohere(
        self,
        query: str,
        documents: list[dict],
    ) -> list[dict]:
        """
        使用 Cohere Rerank API 進行重新排序。

        Cohere Rerank API 文件：https://docs.cohere.com/reference/rerank
        """
        api_key = settings.reranker_api_key
        if not api_key:
            logger.warning("[RERANKER] Cohere API key not set, returning original order")
            return documents

        # 準備文件內容
        doc_texts = [d.get("content", "") for d in documents]

        try:
            response = self.http_client.post(
                "https://api.cohere.ai/v1/rerank",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.reranker_model or "rerank-multilingual-v3.0",
                    "query": query,
                    "documents": doc_texts,
                    "top_n": len(doc_texts),
                    "return_documents": False,
                },
            )
            response.raise_for_status()
            result = response.json()

            # 解析結果並更新文件
            results = result.get("results", [])
            reranked = []
            for item in results:
                idx = item.get("index", 0)
                score = item.get("relevance_score", 0)
                if 0 <= idx < len(documents):
                    doc = dict(documents[idx])
                    doc["rerank_score"] = score
                    reranked.append(doc)

            return reranked

        except httpx.HTTPStatusError as e:
            logger.error(f"[RERANKER] Cohere API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"[RERANKER] Cohere rerank failed: {e}")
            raise

    def _rerank_with_local(
        self,
        query: str,
        documents: list[dict],
    ) -> list[dict]:
        """
        使用本地 Cross-Encoder 模型進行重新排序。

        需要安裝 sentence-transformers：pip install sentence-transformers

        注意：Cross-Encoder 的 predict() 輸出是原始 logits（可能是負數到正數）。
        我們使用 sigmoid 將其轉換為 0-1 的機率值，以便與閾值過濾相容。
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.error(
                "[RERANKER] sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            return documents

        try:
            import math

            # 延遲載入模型（第一次使用時載入）
            if not hasattr(self, "_local_model"):
                model_name = settings.reranker_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
                logger.info(f"[RERANKER] Loading local model: {model_name}")
                self._local_model = CrossEncoder(model_name)

            # 準備 query-document pairs
            doc_texts = [d.get("content", "") for d in documents]
            pairs = [(query, text) for text in doc_texts]

            # Debug: 記錄查詢和文件內容摘要
            logger.debug(f"[RERANKER] Query: {query[:100]}...")
            for i, text in enumerate(doc_texts):
                filename = documents[i].get("filename", "unknown")
                preview = text[:80].replace("\n", " ") if text else "(empty)"
                logger.debug(f"[RERANKER] Doc {i+1} ({filename}): {preview}...")

            # 計算分數（原始 logits）
            raw_scores = self._local_model.predict(pairs)

            # 根據模型類型決定是否需要 sigmoid 正規化
            # BGE reranker 模型輸出的是經過 sigmoid 的分數，範圍已經是 0-1
            # MS-MARCO 模型輸出的是原始 logits，需要 sigmoid 轉換
            model_name = settings.reranker_model or ""
            is_bge_model = "bge" in model_name.lower()

            def sigmoid(x: float) -> float:
                try:
                    return 1 / (1 + math.exp(-x))
                except OverflowError:
                    return 0.0 if x < 0 else 1.0

            # 更新文件分數
            reranked = []
            for i, doc in enumerate(documents):
                new_doc = dict(doc)
                raw_score = float(raw_scores[i])

                # BGE 模型已經輸出 0-1 分數，不需要 sigmoid
                # 其他模型（如 MS-MARCO）輸出 logits，需要 sigmoid
                if is_bge_model:
                    normalized_score = raw_score
                else:
                    normalized_score = sigmoid(raw_score)

                new_doc["rerank_score"] = normalized_score
                new_doc["rerank_raw_score"] = raw_score  # 保留原始分數供 debug
                reranked.append(new_doc)

            # 詳細記錄每個文件的分數（方便 debug）
            for i, doc in enumerate(reranked):
                filename = doc.get("filename", "unknown")[:30]
                raw = doc.get("rerank_raw_score", 0)
                norm = doc.get("rerank_score", 0)
                logger.debug(
                    f"[RERANKER] Doc {i+1}: {filename}... raw={raw:.3f} -> norm={norm:.3f}"
                )

            logger.info(
                f"[RERANKER] Local scores summary: "
                f"min={min(d.get('rerank_score', 0) for d in reranked):.3f}, "
                f"max={max(d.get('rerank_score', 0) for d in reranked):.3f}, "
                f"model={settings.reranker_model or 'default'}"
            )

            return reranked

        except Exception as e:
            logger.error(f"[RERANKER] Local rerank failed: {e}")
            raise


# 全域實例
reranker_service = RerankerService()
