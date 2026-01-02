"""
關係抽取服務

使用 LLM 從文件 chunks 中抽取實體之間的關係。

主要功能：
- 從文本中識別實體間的關係
- 支援多種關係類型（歸屬、工作、執行、位置等）
- 批次處理和去重合併
"""

import asyncio
import hashlib
import json
import logging
import re
from typing import Any, Optional

from chatbot_graphrag.core.config import settings
from chatbot_graphrag.core.constants import RelationType
from chatbot_graphrag.models.pydantic.graph import Entity, Relation

logger = logging.getLogger(__name__)

# 關係抽取提示詞模板
RELATION_EXTRACTION_PROMPT = """你是一個專業的關係提取系統。請根據以下文本和已識別的實體，提取實體之間的關係。

文本內容：
{content}

已識別的實體：
{entities_list}

請提取以下類型的關係：
- belongs_to: 歸屬關係（例如：醫生歸屬於科室）
- works_in: 工作地點關係（例如：某人在某部門工作）
- performs: 執行關係（例如：醫生執行某項手術）
- located_at: 位置關係（例如：某科室位於某樓層）
- requires: 需求關係（例如：某程序需要某文件）
- mentions: 提及關係（例如：文件提及某服務）
- related_to: 一般關聯關係
- treats: 治療關係（例如：某藥物治療某疾病）
- connects_to: 連接關係（例如：某樓層連接某區域）
- part_of: 部分-整體關係
- member_of: 成員關係

請以 JSON 格式回覆，格式如下：
```json
{{
  "relations": [
    {{
      "source": "來源實體名稱",
      "target": "目標實體名稱",
      "relation_type": "關係類型",
      "description": "關係的簡短描述",
      "weight": 0.8
    }}
  ]
}}
```

注意事項：
1. 只提取文本中明確表達的關係，不要推測
2. source 和 target 必須是上述已識別實體中的名稱
3. weight 表示關係的確信度（0.0-1.0）
4. 每個關係都應該有合適的 description
5. 確保 JSON 格式正確"""


class RelationExtractor:
    """
    使用 LLM 從文本中抽取實體之間的關係。

    分析文本內容和已識別的實體，抽取
    有意義的關係，如 works_in、belongs_to 等。
    """

    def __init__(self):
        """初始化關係抽取器。"""
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
        logger.info("關係抽取器已初始化")

    async def extract_relations(
        self,
        content: str,
        entities: list[Entity],
        chunk_id: str,
        doc_id: str,
        max_retries: int = 2,
    ) -> list[Relation]:
        """
        根據實體列表從文本中抽取關係。

        Args:
            content: 要抽取關係的文本內容
            entities: 在內容中找到的實體列表
            chunk_id: 來源 chunk 的 ID
            doc_id: 來源文件的 ID
            max_retries: 失敗時的最大重試次數

        Returns:
            抽取到的 Relation 物件列表
        """
        if not self._initialized:
            await self.initialize()

        if not content or not entities or len(entities) < 2:
            return []

        # 為提示詞建構實體列表
        entities_list = "\n".join(
            f"- {e.name} ({e.entity_type.value})"
            for e in entities
        )

        # 建立實體名稱到 ID 的映射
        entity_map = {e.name.lower(): e for e in entities}

        prompt = RELATION_EXTRACTION_PROMPT.format(
            content=content[:4000],
            entities_list=entities_list,
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
                                "content": "你是一個專業的關係提取系統，請以 JSON 格式回覆。確保 JSON 格式完整且正確。",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,
                        max_tokens=4000,  # 增加以避免截斷
                    )

                result_text = response.choices[0].message.content
                if not result_text:
                    logger.warning(f"LLM returned empty response for relations in chunk {chunk_id}")
                    return []

                # 記錄 LLM 返回的 JSON 內容
                logger.info(f"[Relation Extraction] chunk={chunk_id[:16]}... LLM response:\n{result_text[:500]}{'...' if len(result_text) > 500 else ''}")

                relations = self._parse_relations(
                    result_text, entity_map, chunk_id, doc_id
                )

                logger.info(
                    f"[Relation Extraction] chunk={chunk_id[:16]}... extracted {len(relations)} relations: "
                    f"{[(r.source_id[:12] + '->' + r.target_id[:12]) for r in relations[:3]]}{'...' if len(relations) > 3 else ''}"
                )
                return relations

            except Exception as e:
                error_str = str(e)
                is_model_error = "model_not_found" in error_str or "No matching loaded model" in error_str

                if attempt < max_retries:
                    # 指數退避：2s, 4s, 8s...
                    wait_time = 2 ** (attempt + 1)
                    if is_model_error:
                        logger.warning(
                            f"Relation extraction attempt {attempt + 1} failed (model issue): {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                    else:
                        logger.warning(
                            f"Relation extraction attempt {attempt + 1} failed: {e}"
                        )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Relation extraction failed after {max_retries + 1} attempts: {e}"
                    )
                    return []

        return []

    def _parse_relations(
        self,
        llm_response: str,
        entity_map: dict[str, Entity],
        chunk_id: str,
        doc_id: str,
    ) -> list[Relation]:
        """將 LLM 回應解析為 Relation 物件。"""
        relations = []

        if not llm_response:
            logger.warning(f"Empty LLM response for relations in chunk {chunk_id}")
            return []

        json_str = self._extract_json(llm_response)
        if not json_str:
            logger.warning("No JSON found in LLM response for relations")
            return []

        try:
            data = json.loads(json_str)
            raw_relations = data.get("relations", [])

            for raw in raw_relations:
                relation = self._create_relation(
                    raw, entity_map, chunk_id, doc_id
                )
                if relation:
                    relations.append(relation)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse relation JSON: {e}")
            # 嘗試修復並重新解析
            repaired = self._repair_json(json_str)
            if repaired:
                try:
                    data = json.loads(repaired)
                    raw_relations = data.get("relations", [])
                    for raw in raw_relations:
                        relation = self._create_relation(
                            raw, entity_map, chunk_id, doc_id
                        )
                        if relation:
                            relations.append(relation)
                    logger.info(f"Recovered {len(relations)} relations after JSON repair")
                except json.JSONDecodeError:
                    logger.warning("JSON repair failed")

        return relations

    def _extract_json(self, text: str) -> Optional[str]:
        """從 LLM 回應中抽取 JSON，使用改進的匹配邏輯。"""
        # 方法 1：匹配 markdown 程式碼區塊並平衡大括號
        code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*)', text)
        if code_block_match:
            json_candidate = code_block_match.group(1)
            # Find closing ``` and trim
            end_marker = json_candidate.find('```')
            if end_marker > 0:
                json_candidate = json_candidate[:end_marker].strip()
            # Balance braces
            return self._balance_braces(json_candidate)

        # 方法 2：找到包含 "relations" 鍵的 JSON 物件
        start_idx = text.find('{"relations"')
        if start_idx == -1:
            start_idx = text.find('{ "relations"')
        if start_idx == -1:
            start_idx = text.find('{')

        if start_idx >= 0:
            return self._balance_braces(text[start_idx:])

        return None

    def _balance_braces(self, text: str) -> str:
        """從文本中抽取平衡的 JSON 物件。"""
        if not text.startswith('{'):
            return text

        depth = 0
        in_string = False
        escape_next = False
        end_idx = 0

        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break

        if end_idx > 0:
            return text[:end_idx]

        # 如果不平衡，返回原始內容並讓修復處理
        return text

    def _repair_json(self, json_str: str) -> Optional[str]:
        """嘗試修復格式錯誤的 JSON。"""
        # Remove trailing content after last valid array/object close
        repaired = json_str.rstrip()

        # Fix common issues: trailing commas before closing brackets
        repaired = re.sub(r',\s*]', ']', repaired)
        repaired = re.sub(r',\s*}', '}', repaired)

        # Try to close unclosed structures
        open_braces = repaired.count('{') - repaired.count('}')
        open_brackets = repaired.count('[') - repaired.count(']')

        # If inside a string that wasn't closed, try to close it
        if repaired.count('"') % 2 == 1:
            repaired += '"'

        # Close arrays first, then objects
        repaired += ']' * max(0, open_brackets)
        repaired += '}' * max(0, open_braces)

        return repaired if repaired != json_str else None

    def _create_relation(
        self,
        raw: dict[str, Any],
        entity_map: dict[str, Entity],
        chunk_id: str,
        doc_id: str,
    ) -> Optional[Relation]:
        """從原始抽取資料建立 Relation 物件。"""
        source_name = raw.get("source", "").strip().lower()
        target_name = raw.get("target", "").strip().lower()

        # 驗證來源和目標實體存在
        source_entity = entity_map.get(source_name)
        target_entity = entity_map.get(target_name)

        if not source_entity or not target_entity:
            # 嘗試部分匹配
            source_entity = self._find_entity_by_name(source_name, entity_map)
            target_entity = self._find_entity_by_name(target_name, entity_map)

        if not source_entity or not target_entity:
            logger.debug(
                f"Skipping relation: entity not found "
                f"(source={source_name}, target={target_name})"
            )
            return None

        # 將類型字串映射到 RelationType 列舉
        type_str = raw.get("relation_type", "").lower()
        relation_type = self._map_relation_type(type_str)

        # 生成一致的 ID
        relation_id = self._generate_relation_id(
            source_entity.id, target_entity.id, relation_type
        )

        weight = raw.get("weight", 1.0)
        if not isinstance(weight, (int, float)):
            weight = 1.0
        weight = max(0.0, min(1.0, float(weight)))

        return Relation(
            id=relation_id,
            source_id=source_entity.id,
            target_id=target_entity.id,
            relation_type=relation_type,
            description=raw.get("description"),
            weight=weight,
            properties={"original_type": type_str},
            source_chunk_ids=[chunk_id],
            doc_ids=[doc_id],
            mention_count=1,
        )

    def _find_entity_by_name(
        self,
        name: str,
        entity_map: dict[str, Entity],
    ) -> Optional[Entity]:
        """根據部分名稱匹配查找實體。"""
        name_lower = name.lower()

        # 首先嘗試精確匹配
        if name_lower in entity_map:
            return entity_map[name_lower]

        # 嘗試部分匹配
        for key, entity in entity_map.items():
            if name_lower in key or key in name_lower:
                return entity

        return None

    def _map_relation_type(self, type_str: str) -> RelationType:
        """將字串類型映射到 RelationType 列舉。"""
        type_mapping = {
            "belongs_to": RelationType.BELONGS_TO,
            "works_in": RelationType.WORKS_IN,
            "performs": RelationType.PERFORMS,
            "located_at": RelationType.LOCATED_AT,
            "requires": RelationType.REQUIRES,
            "mentions": RelationType.MENTIONS,
            "related_to": RelationType.RELATED_TO,
            "treats": RelationType.TREATS,
            "connects_to": RelationType.CONNECTS_TO,
            "part_of": RelationType.PART_OF,
            "member_of": RelationType.MEMBER_OF,
        }
        return type_mapping.get(type_str, RelationType.RELATED_TO)

    def _generate_relation_id(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
    ) -> str:
        """根據來源、目標和類型生成唯一的關係 ID。"""
        hash_input = f"{source_id}:{relation_type.value}:{target_id}"
        hash_hex = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        return f"r_{relation_type.value}_{hash_hex}"

    async def extract_relations_batch(
        self,
        chunks_with_entities: list[dict[str, Any]],
        concurrency: int = 3,
    ) -> dict[str, list[Relation]]:
        """
        並行從多個 chunks 中抽取關係。

        Args:
            chunks_with_entities: 字典列表，包含：
                - 'chunk_id': str
                - 'doc_id': str
                - 'content': str
                - 'entities': list[Entity]
            concurrency: 最大並發抽取數

        Returns:
            chunk_id 到關係列表的映射字典
        """
        semaphore = asyncio.Semaphore(concurrency)
        results = {}
        total_chunks = len(chunks_with_entities)
        completed_count = 0
        lock = asyncio.Lock()

        logger.info(f"[Relation Extraction] Starting batch extraction: {total_chunks} chunks, concurrency={concurrency}")

        async def process_chunk(chunk: dict, index: int) -> tuple[str, list[Relation]]:
            nonlocal completed_count
            async with semaphore:
                chunk_id = chunk.get("chunk_id", "")
                doc_id = chunk.get("doc_id", "")
                content = chunk.get("content", "")
                entities = chunk.get("entities", [])

                logger.info(f"[Relation Extraction] Processing chunk {index + 1}/{total_chunks}: {chunk_id[:20]}... ({len(entities)} entities)")

                relations = await self.extract_relations(
                    content, entities, chunk_id, doc_id
                )

                async with lock:
                    completed_count += 1
                    logger.info(f"[Relation Extraction] Progress: {completed_count}/{total_chunks} ({100*completed_count//total_chunks}%)")

                return chunk_id, relations

        tasks = [process_chunk(chunk, i) for i, chunk in enumerate(chunks_with_entities)]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for result in completed:
            if isinstance(result, Exception):
                logger.error(f"Batch relation extraction error: {result}")
            else:
                chunk_id, relations = result
                results[chunk_id] = relations

        total_relations = sum(len(r) for r in results.values())
        logger.info(f"[Relation Extraction] Batch complete: {total_chunks} chunks processed, {total_relations} relations extracted")

        return results

    def merge_relations(self, relations: list[Relation]) -> list[Relation]:
        """
        根據來源、目標和類型合併重複的關係。

        合併 source_chunk_ids、doc_ids，並增加 mention_count。
        """
        relation_map: dict[str, Relation] = {}

        for relation in relations:
            if relation.id in relation_map:
                existing = relation_map[relation.id]
                # 合併來源 chunks
                existing.source_chunk_ids = list(
                    set(existing.source_chunk_ids + relation.source_chunk_ids)
                )
                # 合併文件 IDs
                existing.doc_ids = list(
                    set(existing.doc_ids + relation.doc_ids)
                )
                # 增加提及計數
                existing.mention_count += relation.mention_count
                # 平均權重
                existing.weight = (
                    existing.weight + relation.weight
                ) / 2
            else:
                relation_map[relation.id] = relation

        return list(relation_map.values())


# 單例實例
relation_extractor = RelationExtractor()
