"""
實體抽取服務

使用 LLM 從文件 chunks 中抽取實體。

主要功能：
- 從文本中識別實體（人物、科室、程序、位置等）
- 使用結構化提示詞進行抽取
- 支援批次處理
- 實體去重與合併
"""

import asyncio
import hashlib
import json
import logging
import re
from typing import Any, Optional

from chatbot_graphrag.core.config import settings
from chatbot_graphrag.core.constants import EntityType
from chatbot_graphrag.models.pydantic.graph import Entity

logger = logging.getLogger(__name__)

# 實體抽取提示詞模板
ENTITY_EXTRACTION_PROMPT = """你是一個專業的實體提取系統。請從以下文本中提取所有相關的實體。

文本內容：
{content}

請提取以下類型的實體：
- person: 人名（醫生、護士、員工等）
- department: 科室/部門名稱
- procedure: 醫療程序/服務/檢查項目
- location: 地點/位置（樓層、區域、建築物）
- building: 建築物名稱
- floor: 樓層
- room: 房間/診間
- form: 表單/文件名稱
- medication: 藥物名稱
- equipment: 設備名稱
- service: 服務項目
- condition: 疾病/症狀
- contact: 聯繫方式（電話、分機等）

請以 JSON 格式回覆，格式如下：
```json
{{
  "entities": [
    {{
      "name": "實體名稱",
      "type": "實體類型",
      "description": "簡短描述（可選）",
      "aliases": ["別名1", "別名2"]
    }}
  ]
}}
```

注意事項：
1. 只提取明確提到的實體，不要推測
2. 實體名稱要完整準確
3. 如果同一實體有多種稱呼，請放在 aliases 中
4. 確保 JSON 格式正確"""


class EntityExtractor:
    """
    使用 LLM 從文本中抽取實體。

    使用結構化提示詞來識別和分類實體，
    例如人物、科室、程序、位置等。
    """

    def __init__(self):
        """初始化實體抽取器。"""
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
        logger.info("實體抽取器已初始化")

    async def extract_entities(
        self,
        content: str,
        chunk_id: str,
        doc_id: str,
        max_retries: int = 2,
    ) -> list[Entity]:
        """
        從文本 chunk 中抽取實體。

        Args:
            content: 要抽取實體的文本內容
            chunk_id: 來源 chunk 的 ID
            doc_id: 來源文件的 ID
            max_retries: 失敗時的最大重試次數

        Returns:
            抽取到的 Entity 物件列表
        """
        if not self._initialized:
            await self.initialize()

        if not content or len(content.strip()) < 10:
            return []

        prompt = ENTITY_EXTRACTION_PROMPT.format(content=content[:4000])

        # 匯入並發控制
        from chatbot_graphrag.core.concurrency import llm_concurrency

        for attempt in range(max_retries + 1):
            try:
                # 對 LLM API 呼叫使用並發控制
                async with llm_concurrency.acquire("chat"):
                    response = await self._llm_client.chat.completions.create(
                        model=settings.chat_model,
                        messages=[
                            {"role": "system", "content": "你是一個專業的實體提取系統，請以 JSON 格式回覆。確保 JSON 格式完整且正確。"},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,
                        max_tokens=4000,  # 增加以避免截斷
                    )

                result_text = response.choices[0].message.content
                if not result_text:
                    logger.warning(f"LLM returned empty response for chunk {chunk_id}")
                    return []
                entities = self._parse_entities(result_text, chunk_id, doc_id)

                logger.debug(f"Extracted {len(entities)} entities from chunk {chunk_id}")
                return entities

            except Exception as e:
                error_str = str(e)
                is_model_error = "model_not_found" in error_str or "No matching loaded model" in error_str

                if attempt < max_retries:
                    # 指數退避：2s, 4s, 8s...
                    wait_time = 2 ** (attempt + 1)
                    if is_model_error:
                        logger.warning(
                            f"Entity extraction attempt {attempt + 1} failed (model issue): {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                    else:
                        logger.warning(f"Entity extraction attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Entity extraction failed after {max_retries + 1} attempts: {e}")
                    return []

        return []

    def _parse_entities(
        self,
        llm_response: str,
        chunk_id: str,
        doc_id: str,
    ) -> list[Entity]:
        """將 LLM 回應解析為 Entity 物件。"""
        entities = []

        if not llm_response:
            logger.warning(f"Empty LLM response for chunk {chunk_id}")
            return []

        json_str = self._extract_json(llm_response)
        if not json_str:
            logger.warning("No JSON found in LLM response for entities")
            return []

        try:
            data = json.loads(json_str)
            raw_entities = data.get("entities", [])

            for raw in raw_entities:
                entity = self._create_entity(raw, chunk_id, doc_id)
                if entity:
                    entities.append(entity)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse entity JSON: {e}")
            # 嘗試修復並重新解析
            repaired = self._repair_json(json_str)
            if repaired:
                try:
                    data = json.loads(repaired)
                    raw_entities = data.get("entities", [])
                    for raw in raw_entities:
                        entity = self._create_entity(raw, chunk_id, doc_id)
                        if entity:
                            entities.append(entity)
                    logger.info(f"Recovered {len(entities)} entities after JSON repair")
                except json.JSONDecodeError:
                    logger.warning("JSON repair failed")

        return entities

    def _extract_json(self, text: str) -> Optional[str]:
        """從 LLM 回應中抽取 JSON，使用改進的匹配邏輯。"""
        # 方法 1：匹配 markdown 程式碼區塊並平衡大括號
        code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*)', text)
        if code_block_match:
            json_candidate = code_block_match.group(1)
            # 找到結束的 ``` 並修剪
            end_marker = json_candidate.find('```')
            if end_marker > 0:
                json_candidate = json_candidate[:end_marker].strip()
            # 平衡大括號
            return self._balance_braces(json_candidate)

        # 方法 2：找到包含 "entities" 鍵的 JSON 物件
        start_idx = text.find('{"entities"')
        if start_idx == -1:
            start_idx = text.find('{ "entities"')
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
        # 移除最後一個有效陣列/物件關閉後的尾隨內容
        repaired = json_str.rstrip()

        # 修復常見問題：結束括號前的尾隨逗號
        repaired = re.sub(r',\s*]', ']', repaired)
        repaired = re.sub(r',\s*}', '}', repaired)

        # 嘗試關閉未關閉的結構
        open_braces = repaired.count('{') - repaired.count('}')
        open_brackets = repaired.count('[') - repaired.count(']')

        # 如果在未關閉的字串內，嘗試關閉它
        if repaired.count('"') % 2 == 1:
            repaired += '"'

        # 先關閉陣列，然後關閉物件
        repaired += ']' * max(0, open_brackets)
        repaired += '}' * max(0, open_braces)

        return repaired if repaired != json_str else None

    def _create_entity(
        self,
        raw: dict[str, Any],
        chunk_id: str,
        doc_id: str,
    ) -> Optional[Entity]:
        """從原始抽取資料建立 Entity 物件。"""
        name = raw.get("name", "").strip()
        if not name:
            return None

        # 將類型字串映射到 EntityType 列舉
        type_str = raw.get("type", "").lower()
        entity_type = self._map_entity_type(type_str)

        # 生成一致的 ID
        entity_id = self._generate_entity_id(name, entity_type)

        return Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            description=raw.get("description"),
            aliases=raw.get("aliases", []),
            properties={"original_type": type_str},
            source_chunk_ids=[chunk_id],
            doc_ids=[doc_id],
            mention_count=1,
        )

    def _map_entity_type(self, type_str: str) -> EntityType:
        """將字串類型映射到 EntityType 列舉。"""
        type_mapping = {
            "person": EntityType.PERSON,
            "department": EntityType.DEPARTMENT,
            "procedure": EntityType.PROCEDURE,
            "location": EntityType.LOCATION,
            "building": EntityType.BUILDING,
            "floor": EntityType.FLOOR,
            "room": EntityType.ROOM,
            "form": EntityType.FORM,
            "medication": EntityType.MEDICATION,
            "equipment": EntityType.EQUIPMENT,
            "service": EntityType.SERVICE,
            "condition": EntityType.CONDITION,
            "transport": EntityType.TRANSPORT,
            "contact": EntityType.CONTACT,
        }
        return type_mapping.get(type_str, EntityType.SERVICE)

    def _generate_entity_id(self, name: str, entity_type: EntityType) -> str:
        """根據名稱和類型生成唯一的實體 ID。"""
        normalized = name.lower().strip()
        hash_input = f"{entity_type.value}:{normalized}"
        hash_hex = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        return f"e_{entity_type.value}_{hash_hex}"

    async def extract_entities_batch(
        self,
        chunks: list[dict[str, Any]],
        concurrency: int = 3,
    ) -> dict[str, list[Entity]]:
        """
        並行從多個 chunks 中抽取實體。

        Args:
            chunks: chunk 字典列表，包含 'id', 'doc_id', 'content' 鍵
            concurrency: 最大並發抽取數

        Returns:
            chunk_id 到實體列表的映射字典
        """
        semaphore = asyncio.Semaphore(concurrency)
        results = {}

        async def process_chunk(chunk: dict) -> tuple[str, list[Entity]]:
            async with semaphore:
                chunk_id = chunk.get("id", "")
                doc_id = chunk.get("doc_id", "")
                content = chunk.get("content", "")

                entities = await self.extract_entities(content, chunk_id, doc_id)
                return chunk_id, entities

        tasks = [process_chunk(chunk) for chunk in chunks]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for result in completed:
            if isinstance(result, Exception):
                logger.error(f"Batch extraction error: {result}")
            else:
                chunk_id, entities = result
                results[chunk_id] = entities

        return results

    def merge_entities(self, entities: list[Entity]) -> list[Entity]:
        """
        根據名稱和類型合併重複的實體。

        合併 source_chunk_ids、doc_ids，並增加 mention_count。
        """
        entity_map: dict[str, Entity] = {}

        for entity in entities:
            if entity.id in entity_map:
                existing = entity_map[entity.id]
                # 合併來源 chunks
                existing.source_chunk_ids = list(set(
                    existing.source_chunk_ids + entity.source_chunk_ids
                ))
                # 合併文件 IDs
                existing.doc_ids = list(set(
                    existing.doc_ids + entity.doc_ids
                ))
                # 合併別名
                existing.aliases = list(set(
                    existing.aliases + entity.aliases
                ))
                # 增加提及計數
                existing.mention_count += entity.mention_count
            else:
                entity_map[entity.id] = entity

        return list(entity_map.values())


# 單例實例
entity_extractor = EntityExtractor()
