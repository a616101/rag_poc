"""
查詢分解服務

使用 LLM 分析查詢並生成子查詢以改善檢索效果。
不使用硬編碼模式 - LLM 理解意圖並生成適當的子查詢。
支援對話歷史以處理後續問題。

主要功能：
- 分析使用者查詢的意圖和實體
- 生成多個子查詢以提高召回率
- 處理對話上下文（補充代詞指代）
- 支援實體查詢、反向查詢和一般查詢
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


@dataclass
class DecomposedQuery:
    """查詢分解結果。"""

    original_query: str  # 原始查詢
    primary_query: str  # 主要查詢（可能是解析後的）
    sub_queries: list[str] = field(default_factory=list)  # 生成的子查詢列表

    # 填入上下文後的解析查詢（用於後續問題）
    resolved_query: Optional[str] = None

    # 抽取的實體
    physician_name: Optional[str] = None  # 醫師姓名
    department: Optional[str] = None  # 科別
    property_type: Optional[str] = None  # 屬性類型

    # 查詢類型
    query_type: str = "general"  # "entity_lookup", "reverse_lookup", "general"

    # 建議的 Qdrant 元資料過濾器
    metadata_filters: dict[str, Any] = field(default_factory=dict)

    # 分解推理說明
    reasoning: str = ""


# 查詢分解的系統提示詞
DECOMPOSITION_SYSTEM_PROMPT = """你是一個專門分析醫療查詢的助手。你的任務是分析用戶的問題，並生成多個子查詢來幫助檢索相關文檔。

## 最重要：resolved_query 必填！

`resolved_query` 是用於生成最終答案的問題，**必須是完整且獨立可理解的句子**。
這個欄位**絕對不能留空或填 null**！

規則：
1. **有對話歷史時**：必須將追問問題與上下文融合，生成完整問題
   - 「那李醫師呢？」→「李醫師什麼時候看診？」
   - 「他的專長？」→「王醫師的專長是什麼？」
2. **沒有對話歷史時**：直接填入原始問題（可稍作潤飾使其更完整）

## 對話上下文處理

如果提供了對話歷史，你必須：
1. **理解上下文**：根據對話歷史理解當前問題的完整含義
2. **補充指代**：如果當前問題包含代詞（如「它」「這個」「他」「那」）或省略主詞，必須從歷史中補充
3. **生成完整的 resolved_query**：這是最終答案生成時使用的問題，必須獨立可理解
4. **生成完整子查詢**：所有子查詢都必須包含完整的上下文資訊

例如：
- 歷史：「什麼是糖尿病？」→ 當前：「怎麼治療？」
  → resolved_query：「糖尿病怎麼治療？」
  → 子查詢：["糖尿病治療", "糖尿病治療方法", "糖尿病如何治療"]

- 歷史：「王醫師的專長是什麼？」→ 當前：「他的門診時間？」
  → resolved_query：「王醫師的門診時間是什麼時候？」
  → 子查詢：["王醫師門診時間", "王醫師門診", "王醫師看診時間"]

- 歷史：「王醫師什麼時候看診？」→ 當前：「那李醫師呢？」
  → resolved_query：「李醫師什麼時候看診？」
  → 子查詢：["李醫師門診時間", "李醫師門診", "李醫師看診時間"]

## 你的核心任務

分析用戶問題，生成多個不同角度的子查詢，確保能從向量資料庫中檢索到相關內容。
不要猜測文檔類型或進行過濾 - 讓語義搜索自己找到相關文檔。

## 查詢類型

1. **entity_lookup** (實體查詢): 查詢特定實體的屬性
   - 例如: "吳明昇醫師的專長是什麼？" → 找特定醫師的專長

2. **reverse_lookup** (反向查詢): 根據屬性查找符合條件的實體
   - 例如: "哪些醫師當過主任？" → 找有主任經歷的醫師
   - 例如: "心臟科有哪些醫師？" → 找心臟科的所有醫師

3. **general** (一般查詢): 其他類型的問題
   - 例如: "如何掛號？" → 查詢掛號流程

## 輸出格式

請以 JSON 格式輸出分析結果：
```json
{
  "query_type": "entity_lookup|reverse_lookup|general",
  "reasoning": "簡短說明為什麼這樣分類，以及如何從歷史中補充上下文（如適用）",
  "resolved_query": "完整且獨立可理解的問題（必填！不能為 null）",
  "entities": {
    "physician_name": "醫師姓名（如有，null 如果沒有）",
    "department": "科別（如有，null 如果沒有）",
    "property_type": "查詢的屬性類型，如 specialty/schedule/experience/education（null 如果不適用）"
  },
  "sub_queries": [
    "子查詢1（必須包含完整上下文）",
    "子查詢2",
    "..."
  ]
}
```

## 子查詢生成指南 (最重要!)

子查詢的目的是從不同角度搜索，確保找到相關文檔。請生成多樣化的子查詢：

1. **entity_lookup**: 生成包含實體名稱和不同屬性關鍵詞的子查詢
   - 例如 "吳明昇的專長" → ["吳明昇", "吳明昇 醫師", "吳明昇 專長", "吳明昇 主治項目"]

2. **reverse_lookup**: 生成包含屬性關鍵詞的子查詢，用於廣泛搜索
   - 例如 "哪些醫師當過主任" → ["主任 經歷", "曾任主任", "學經歷 主任", "科主任", "擔任主任"]
   - 例如 "心臟科有哪些醫師" → ["心臟科 醫師", "心臟內科", "心臟血管科", "心臟科 專長"]

3. **general**: 生成問題的不同表述方式，包含同義詞
   - 例如 "如何掛號" → ["掛號流程", "掛號方式", "門診掛號", "預約掛號", "就醫流程"]
   - 例如 "怎麼看診" → ["看診流程", "就醫流程", "門診流程", "就診須知"]

請確保子查詢：
- 覆蓋不同的表述方式和同義詞
- 包含用戶可能沒提到但相關的關鍵詞
- **必須包含從歷史中提取的關鍵實體/主題（如果有對話歷史）**
- 數量在 4-6 個之間
- 不要重複相同的內容
"""


def _format_chat_history(
    messages: list[BaseMessage],
    current_query: str,
    max_turns: int = 5,
) -> str:
    """
    為分解提示格式化對話歷史。

    Args:
        messages: LangChain 訊息列表（可能包含當前問題）
        current_query: 正在處理的當前查詢（要從歷史中排除）
        max_turns: 要包含的最大最近輪數

    Returns:
        格式化的對話歷史字串（不包含當前問題）
    """
    if not messages:
        return ""

    # 從訊息末端過濾掉當前問題
    # 當前問題作為 `query` 參數單獨傳遞
    filtered_messages = []
    for msg in messages:
        # 如果最後一個 HumanMessage 與當前查詢匹配則跳過
        if isinstance(msg, HumanMessage) and msg.content.strip() == current_query.strip():
            continue
        filtered_messages.append(msg)

    if not filtered_messages:
        return ""

    # 取得最後 N 輪（1 輪 = 1 個使用者 + 1 個助理 = 2 則訊息）
    max_messages = max_turns * 2
    recent_messages = filtered_messages[-max_messages:] if len(filtered_messages) > max_messages else filtered_messages

    history_parts = []
    for msg in recent_messages:
        if isinstance(msg, HumanMessage):
            history_parts.append(f"用戶: {msg.content}")
        elif isinstance(msg, AIMessage):
            # 截斷長 AI 回應以節省 token
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            history_parts.append(f"助理: {content}")

    return "\n".join(history_parts)


async def decompose_query_with_llm(
    query: str,
    chat_history: Optional[list[BaseMessage]] = None,
    llm=None,
    callbacks: Optional[list] = None,
) -> DecomposedQuery:
    """
    使用 LLM 分析查詢並生成子查詢。

    Args:
        query: 使用者的原始查詢
        chat_history: 可選的先前訊息列表作為上下文
        llm: 可選的 LLM 實例（如未提供將使用工廠）
        callbacks: 可選的 Langfuse 追蹤回呼列表

    Returns:
        包含分析結果和子查詢的 DecomposedQuery
    """
    if llm is None:
        from chatbot_graphrag.services.llm.factory import llm_factory
        llm = llm_factory.get_fast_model(temperature=0.0, max_tokens=800)

    try:
        # 建構帶有可選對話歷史的使用者訊息
        if chat_history:
            formatted_history = _format_chat_history(chat_history, current_query=query)
            if formatted_history:
                user_content = f"""## 對話歷史
{formatted_history}

## 當前問題
{query}

請分析當前問題，並根據對話歷史補充上下文來生成子查詢。"""
                logger.info(f"Query decomposition with history: {len(chat_history)} messages")
            else:
                # 沒有先前歷史（第一則訊息）
                user_content = f"請分析以下查詢：\n\n{query}"
        else:
            user_content = f"請分析以下查詢：\n\n{query}"

        messages = [
            SystemMessage(content=DECOMPOSITION_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        # 如果提供了 Langfuse 追蹤的回呼則傳遞
        invoke_kwargs = {}
        if callbacks:
            invoke_kwargs["config"] = {"callbacks": callbacks}

        # 對 LLM 呼叫使用並發控制
        from chatbot_graphrag.core.concurrency import with_chat_semaphore

        response = await with_chat_semaphore(
            lambda: llm.ainvoke(messages, **invoke_kwargs)
        )
        response_text = response.content.strip()

        # 解析 JSON 回應
        # 如果存在 markdown 程式碼區塊則處理
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)

        # 抽取實體
        entities = result.get("entities", {})
        physician_name = entities.get("physician_name")
        department = entities.get("department")
        property_type = entities.get("property_type")

        # 清理 None 字串
        if physician_name in ("null", "None", ""):
            physician_name = None
        if department in ("null", "None", ""):
            department = None
        if property_type in ("null", "None", ""):
            property_type = None

        # 取得子查詢
        sub_queries = result.get("sub_queries", [])

        # 取得解析後的查詢（必填欄位，用於最終答案生成）
        resolved_query = result.get("resolved_query")
        if resolved_query in ("null", "None", "", None):
            # LLM 沒有返回 resolved_query，使用原始查詢作為 fallback
            resolved_query = query
            logger.warning(
                f"LLM did not return resolved_query, using original query as fallback: {query[:50]}"
            )

        # resolved_query 現在保證有值，用於 primary_query 和最終答案生成
        primary = resolved_query

        # 建構分解後的查詢
        decomposed = DecomposedQuery(
            original_query=query,
            primary_query=primary,
            resolved_query=resolved_query,  # 現在保證有值
            sub_queries=sub_queries,
            physician_name=physician_name,
            department=department,
            property_type=property_type,
            query_type=result.get("query_type", "general"),
            metadata_filters={},  # 不使用硬編碼過濾器 - 信任語意搜尋
            reasoning=result.get("reasoning", ""),
        )

        logger.info(
            f"LLM decomposition: type={decomposed.query_type}, "
            f"resolved_query={resolved_query[:50]}..., sub_queries={len(sub_queries)}"
        )
        logger.debug(f"Decomposition details: {decomposed}")

        return decomposed

    except Exception as e:
        logger.warning(f"LLM decomposition failed, falling back to basic: {e}")
        return _basic_decomposition(query)


def _basic_decomposition(query: str) -> DecomposedQuery:
    """
    不使用 LLM 的基本後備分解。

    只返回原始查詢，進行最小處理。
    """
    return DecomposedQuery(
        original_query=query,
        primary_query=query,
        sub_queries=[],
        reasoning="基本後備（LLM 不可用）",
    )


def decompose_query(query: str) -> DecomposedQuery:
    """
    基本分解的同步包裝器。

    對於非同步 LLM 分解，請使用 decompose_query_with_llm()。
    """
    return _basic_decomposition(query)
