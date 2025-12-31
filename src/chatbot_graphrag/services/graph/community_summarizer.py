"""
社區摘要服務

為偵測到的社區生成基於 LLM 的摘要。

主要功能：
- 為社區生成標題和摘要
- 識別關鍵實體和關係
- 提取主題和重要性分數
- 生成嵌入向量用於語意搜尋
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from chatbot_graphrag.core.config import settings
from chatbot_graphrag.models.pydantic.graph import (
    Community,
    CommunityReport,
    Entity,
    Relation,
)

logger = logging.getLogger(__name__)

# 社區摘要提示詞模板
COMMUNITY_SUMMARY_PROMPT = """你是一個知識圖譜分析專家。請根據以下社區（community）中的實體和關係，生成一個簡潔的摘要報告。

社區層級: {level}
實體數量: {entity_count}

社區成員（實體）:
{entities_list}

社區內關係:
{relations_list}

請以 JSON 格式回覆，格式如下：
```json
{{
  "title": "社區標題（簡短、描述性的名稱）",
  "summary": "社區摘要（2-3句話描述這個社區的主題和特點）",
  "key_entities": ["最重要的實體1", "最重要的實體2", "最重要的實體3"],
  "key_relations": ["關鍵關係描述1", "關鍵關係描述2"],
  "themes": ["主題1", "主題2"],
  "importance_score": 0.8
}}
```

注意事項：
1. title 應該反映社區的核心主題
2. summary 應該解釋這個社區為什麼重要，它代表什麼概念
3. key_entities 最多列出5個最重要的實體
4. themes 是這個社區涵蓋的主題領域
5. importance_score 是 0.0-1.0 的數值，表示這個社區的重要性"""


@dataclass
class CommunitySummaryResult:
    """社區摘要結果。"""

    reports: list[CommunityReport] = field(default_factory=list)  # 摘要報告列表
    total_summarized: int = 0  # 摘要總數
    failed_count: int = 0  # 失敗數
    total_tokens: int = 0  # token 總數
    execution_time_ms: float = 0.0  # 執行時間（毫秒）


class CommunitySummarizer:
    """
    為圖譜社區生成 LLM 摘要。

    建立包含以下內容的 CommunityReport 物件：
    - 標題和摘要
    - 關鍵實體和關係
    - 主題和重要性分數
    - 用於語意搜尋的嵌入向量
    """

    def __init__(self):
        """初始化社區摘要器。"""
        self._initialized = False
        self._llm_client = None

    async def initialize(self) -> None:
        """初始化 LLM 客戶端。"""
        if self._initialized:
            return

        from openai import AsyncOpenAI

        self._llm_client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
        )

        self._initialized = True
        logger.info("社區摘要器已初始化")

    async def summarize_community(
        self,
        community: Community,
        entities: list[Entity],
        relations: list[Relation],
        max_retries: int = 2,
    ) -> Optional[CommunityReport]:
        """
        為單一社區生成摘要。

        Args:
            community: 要摘要的社區
            entities: 社區中的實體
            relations: 社區內的關係
            max_retries: 最大重試次數

        Returns:
            CommunityReport 或失敗時返回 None
        """
        if not self._initialized:
            await self.initialize()

        if not entities:
            return None

        # 建構實體列表
        entities_list = "\n".join(
            f"- {e.name} ({e.entity_type.value}): {e.description or '無描述'}"
            for e in entities[:30]  # Limit to avoid token overflow
        )

        # 建構關係列表
        entity_name_map = {e.id: e.name for e in entities}
        relations_list = "\n".join(
            f"- {entity_name_map.get(r.source_id, r.source_id)} "
            f"--[{r.relation_type.value}]--> "
            f"{entity_name_map.get(r.target_id, r.target_id)}"
            for r in relations[:20]  # Limit relations
        ) or "無明確關係"

        prompt = COMMUNITY_SUMMARY_PROMPT.format(
            level=community.level,
            entity_count=len(entities),
            entities_list=entities_list,
            relations_list=relations_list,
        )

        # 匯入並發控制
        from chatbot_graphrag.core.concurrency import llm_concurrency

        for attempt in range(max_retries + 1):
            try:
                # 對 LLM API 呼叫使用並發控制
                async with llm_concurrency.acquire("chat"):
                    response = await self._llm_client.chat.completions.create(
                        model=settings.chat_model,
                        messages=[
                            {
                                "role": "system",
                                "content": "你是一個知識圖譜分析專家，請以 JSON 格式回覆。",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.3,
                        max_tokens=1000,
                    )

                result_text = response.choices[0].message.content
                report = self._parse_report(result_text, community)

                if report:
                    # 計算 token 數
                    report.token_count = response.usage.total_tokens if response.usage else 0
                    report.model_used = settings.chat_model
                    return report

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Community summary attempt {attempt + 1} failed: {e}"
                    )
                    await asyncio.sleep(1)
                else:
                    logger.error(
                        f"Community summary failed after {max_retries + 1} attempts: {e}"
                    )

        return None

    def _parse_report(
        self,
        llm_response: str,
        community: Community,
    ) -> Optional[CommunityReport]:
        """將 LLM 回應解析為 CommunityReport。"""
        # 從回應中抽取 JSON
        json_match = re.search(
            r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', llm_response
        )
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{[\s\S]*"title"[\s\S]*\}', llm_response)
            if json_match:
                json_str = json_match.group(0)
            else:
                logger.warning("No JSON found in community summary response")
                return None

        try:
            data = json.loads(json_str)

            importance_score = data.get("importance_score", 0.5)
            if not isinstance(importance_score, (int, float)):
                importance_score = 0.5
            importance_score = max(0.0, min(1.0, float(importance_score)))

            return CommunityReport(
                community_id=community.id,
                level=community.level,
                title=data.get("title", f"Community {community.id}"),
                summary=data.get("summary", ""),
                key_entities=data.get("key_entities", [])[:5],
                key_relations=data.get("key_relations", [])[:5],
                themes=data.get("themes", [])[:5],
                importance_score=importance_score,
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse community summary JSON: {e}")
            return None

    async def summarize_communities(
        self,
        communities: list[Community],
        entities_by_community: dict[str, list[Entity]],
        relations_by_community: dict[str, list[Relation]],
        concurrency: int = 3,
    ) -> CommunitySummaryResult:
        """
        為多個社區生成摘要。

        Args:
            communities: 要摘要的社區列表
            entities_by_community: community_id 到實體的映射
            relations_by_community: community_id 到關係的映射
            concurrency: 最大並發 LLM 呼叫數

        Returns:
            CommunitySummaryResult
        """
        if not self._initialized:
            await self.initialize()

        import time
        start_time = time.time()
        result = CommunitySummaryResult()

        if not communities:
            return result

        semaphore = asyncio.Semaphore(concurrency)

        async def process_community(
            community: Community,
        ) -> Optional[CommunityReport]:
            async with semaphore:
                entities = entities_by_community.get(community.id, [])
                relations = relations_by_community.get(community.id, [])
                return await self.summarize_community(community, entities, relations)

        tasks = [process_community(c) for c in communities]
        reports = await asyncio.gather(*tasks, return_exceptions=True)

        for report in reports:
            if isinstance(report, Exception):
                logger.error(f"Community summarization error: {report}")
                result.failed_count += 1
            elif report is not None:
                result.reports.append(report)
                result.total_tokens += report.token_count

        result.total_summarized = len(result.reports)
        result.execution_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Summarized {result.total_summarized}/{len(communities)} communities "
            f"in {result.execution_time_ms:.1f}ms, {result.total_tokens} tokens"
        )

        return result

    async def summarize_and_store(
        self,
        communities: list[Community],
        entities_by_community: dict[str, list[Entity]],
        relations_by_community: dict[str, list[Relation]],
        concurrency: int = 3,
    ) -> CommunitySummaryResult:
        """
        生成摘要並儲存社區報告。

        Args:
            communities: 要摘要的社區
            entities_by_community: community_id 到實體的映射
            relations_by_community: community_id 到關係的映射
            concurrency: 最大並發 LLM 呼叫數

        Returns:
            CommunitySummaryResult
        """
        result = await self.summarize_communities(
            communities,
            entities_by_community,
            relations_by_community,
            concurrency,
        )

        if not result.reports:
            return result

        # 使用摘要更新社區並儲存到 NebulaGraph
        from chatbot_graphrag.services.graph.nebula_client import nebula_client

        try:
            await nebula_client.initialize()

            for report in result.reports:
                # 查找並更新社區
                community = next(
                    (c for c in communities if c.id == report.community_id),
                    None,
                )
                if community:
                    community.title = report.title
                    community.summary = report.summary

                    try:
                        await nebula_client.upsert_community(community)
                    except Exception as e:
                        logger.warning(
                            f"Failed to update community {community.id}: {e}"
                        )

            logger.info(
                f"Updated {len(result.reports)} communities with summaries"
            )

        except Exception as e:
            logger.error(f"Failed to store community summaries: {e}")

        return result

    async def generate_embeddings_for_reports(
        self,
        reports: list[CommunityReport],
    ) -> list[CommunityReport]:
        """
        為社區報告摘要生成嵌入向量。

        Args:
            reports: 要生成嵌入向量的報告

        Returns:
            填入嵌入向量的報告
        """
        from chatbot_graphrag.services.vector import embedding_service

        await embedding_service.initialize()

        # 為摘要生成嵌入向量
        texts = [
            f"{report.title}. {report.summary}"
            for report in reports
        ]

        embeddings = await embedding_service.embed_batch(texts)

        for report, embedding in zip(reports, embeddings):
            report.embedding = embedding

        logger.info(f"Generated embeddings for {len(reports)} community reports")

        return reports


# 單例實例
community_summarizer = CommunitySummarizer()
