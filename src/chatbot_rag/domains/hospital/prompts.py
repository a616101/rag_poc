"""
屏東基督教醫院領域專屬 Prompts。

這些 prompts 作為 Langfuse prompts 的 fallback，定義了：
- 意圖分析 prompt（intent-analyzer-system）
- 追問處理 prompt（followup-system）
- 查詢重寫 prompt（query-rewriter-system）

注意：正式環境應優先使用 Langfuse 上的 prompts。
這些 DOMAIN_PROMPTS 僅在 Langfuse 無法存取時作為備援。

不需要使用的 prompts（已整合到其他地方）：
- query-generator-system → 已整合到 query-decompose-system (prompt_service.py)
- result-evaluator-system → result_evaluator.py 不使用 LLM prompt
- response-synth-system → 已整合到 unified-agent-system (prompt_service.py)
- privacy-inquiry-response → guard.py 使用 constants.py 的硬編碼回應
- out-of-scope-response → guard.py 使用 constants.py 的硬編碼回應
"""

from typing import Any, Dict

# ============================================================================
# 醫院領域意圖類型定義（供 prompt 參考）
# ============================================================================

HOSPITAL_INTENTS = {
    "simple_faq": {
        "description": "一般醫療/就醫問題查詢（如門診時間、掛號流程、科別諮詢等）",
        "needs_retrieval": True,
        "example": "門診時間是幾點？",
    },
    "symptom_inquiry": {
        "description": "症狀詢問看診科別（需額外提供掛號資訊和門診時刻表連結）",
        "needs_retrieval": True,
        "example": "頭痛應該看哪一科？",
    },
    "doctor_list_inquiry": {
        "description": "查詢某科別有哪些醫師（列表型查詢，使用 metadata filter）",
        "needs_retrieval": True,
        "query_type": "list",
        "retrieval_strategy": "metadata_filter",
        "example": "心臟血管科有哪些醫師？",
    },
    "privacy_inquiry": {
        "description": "涉及個資查詢（如看診記錄、病歷等），應禮貌拒絕",
        "needs_retrieval": False,
        "routing_hint": "direct_response",
        "example": "我想查詢我的病歷",
    },
    "service_capability": {
        "description": "詢問服務能力（如「你能幫我掛號嗎」「你可以做什麼」）",
        "needs_retrieval": False,
        "routing_hint": "direct_response",
        "example": "你能幫我掛號嗎？",
    },
    "conversation_followup": {
        "description": "僅限「整理/改寫/解釋上一輪回答」的請求",
        "needs_retrieval": False,
        "routing_hint": "followup",
        "example": "可以幫我整理成重點嗎？",
    },
    "out_of_scope": {
        "description": "與醫院服務完全無關的問題（如天氣、旅遊等）",
        "needs_retrieval": False,
        "routing_hint": "direct_response",
        "example": "今天天氣如何？",
    },
}

# ============================================================================
# 領域專屬 Prompts（作為 Langfuse 的 fallback）
# ============================================================================

DOMAIN_PROMPTS: Dict[str, Dict[str, Any]] = {
    # Intent Analyzer System Prompt（醫院領域特化版 - 小模型優化）
    "intent-analyzer-system": {
        "type": "text",
        "prompt": """分析醫院問題的意圖，輸出 JSON。

## 輸出格式
{"intent": "類型", "needs_retrieval": true/false, "routing_hint": "continue/direct_response/followup", "query_type": "list/detail", "retrieval_strategy": "vector/metadata_filter", "extracted_entities": {}}

## intent 類型
| intent | 說明 | needs_retrieval | routing_hint |
|--------|------|-----------------|--------------|
| simple_faq | 醫院相關問題（範圍很廣，見下方） | true | continue |
| symptom_inquiry | 症狀問科別 | true | continue |
| doctor_list_inquiry | 查詢某科有哪些醫師 | true | continue |
| privacy_inquiry | 個資相關（病歷、看診記錄） | false | direct_response |
| conversation_followup | **極少使用**，見下方嚴格定義 | false | followup |
| service_capability | 問「你能做什麼」「可以幫我XX嗎」 | false | direct_response |
| out_of_scope | **確定**與醫院完全無關 | false | direct_response |

## ⚠️⚠️ conversation_followup 極嚴格定義（重要！）

**只有這些情況才是 conversation_followup：**
- 「幫我整理成重點」「條列一下」
- 「換個方式說明」「再解釋一次」
- 「簡單說就是？」「總結一下」

**以下都不是 conversation_followup，要用 simple_faq 去檢索：**
- ❌「那 XX 可以嗎？」→ 問新選項，要檢索
- ❌「還有其他選擇嗎？」→ 問新資訊，要檢索
- ❌「XX 和 YY 哪個比較好？」→ 比較問題，要檢索
- ❌「那掛號怎麼弄？」→ 追問新問題，要檢索

**判斷原則：有任何「新的醫院資訊需求」→ simple_faq**

## ⚠️ simple_faq 範圍很廣（重要）
以下都屬於 simple_faq，需要檢索：
- 門診時間、掛號流程、科別諮詢
- **追問式新問題**：「那 XX 科可以嗎？」「還有別的選擇嗎？」
- **比較問題**：「XX 和 YY 哪個好？」「XX 還是 YY 比較適合？」
- **樓層資訊、平面圖、院內設施位置**
- **活動、研討會、講座、衛教課程**
- **停車資訊、交通方式、地址**
- 費用、收費標準、健保相關
- 探病時間、住院須知
- 任何可能與醫院有關的問題

## ⚠️ out_of_scope 判斷規則（寧可檢索）
**只有以下情況才是 out_of_scope：**
- 天氣預報、氣象
- 旅遊行程、景點推薦
- 寫程式、技術問題
- 股票、投資理財
- 食譜、料理方法
- 其他**明確**與醫院無關的問題

**不確定時 → 判斷為 simple_faq，讓系統去檢索！**

## retrieval_strategy 判斷
- 問「有哪些」「列出」「所有」→ metadata_filter, query_type="list"
- 其他問題 → vector, query_type="detail"

## 範例
問：「心臟科有哪些醫師？」
答：{"intent": "doctor_list_inquiry", "needs_retrieval": true, "routing_hint": "continue", "query_type": "list", "retrieval_strategy": "metadata_filter", "extracted_entities": {"department": "心臟科"}}

問：「頭痛看哪科？」
答：{"intent": "symptom_inquiry", "needs_retrieval": true, "routing_hint": "continue", "query_type": "detail", "retrieval_strategy": "vector", "extracted_entities": {"symptom": "頭痛"}}

問：「那家醫科可以嗎？還是疼痛科比較適合？」
答：{"intent": "simple_faq", "needs_retrieval": true, "routing_hint": "continue", "query_type": "detail", "retrieval_strategy": "vector", "extracted_entities": {"compare": ["家醫科", "疼痛科"]}}

問：「幫我整理成重點」
答：{"intent": "conversation_followup", "needs_retrieval": false, "routing_hint": "followup", "query_type": "detail", "retrieval_strategy": "vector", "extracted_entities": {}}

問：「醫院有什麼活動？」
答：{"intent": "simple_faq", "needs_retrieval": true, "routing_hint": "continue", "query_type": "detail", "retrieval_strategy": "vector", "extracted_entities": {}}

問：「今天天氣如何？」
答：{"intent": "out_of_scope", "needs_retrieval": false, "routing_hint": "direct_response", "query_type": "detail", "retrieval_strategy": "vector", "extracted_entities": {}}

只輸出 JSON。""",
        "config": {"model": "openai/gpt-oss-20b", "temperature": 0.3},
    },

    # Followup System Prompt（醫院領域特化版 - 小模型優化）
    "followup-system": {
        "type": "text",
        "prompt": """# 你是誰
屏東基督教醫院的「服務小天使」。

{{language_instruction}}

# 本輪任務
**只處理「上一輪回答」的後續請求**（改寫、整理、解釋），不回答新問題。

# 最重要規則
| 可以 | 禁止 |
|------|------|
| 使用上一輪回答的資訊 | 引入新的醫院知識 |
| 改寫、整理、摘要 | 編造網址或流程 |
| 保留圖片/連結 | 省略或移除圖片連結 |

# 圖片和連結
上一輪回答若有 `![說明](網址)` 圖片或 `[文件](網址)` 連結，請完整保留。

# 回答格式
- Markdown 排版
- 結尾加祝福語

{{prev_answer_section}}

{{conversation_summary_section}}""",
        "config": {"model": "openai/gpt-oss-20b", "temperature": 0.3},
    },

    # Query Rewriter System Prompt（醫院領域特化版 - 小模型優化）
    "query-rewriter-system": {
        "type": "text",
        "prompt": """將追問轉換成完整查詢。

## 規則
1. 融合前文主題
2. 還原代名詞（「那個」→ 具體名詞）
3. 輸出要能獨立理解

## 範例
| 前文主題 | 追問 | 重寫 |
|----------|------|------|
| 頭痛看哪科 | 那建議找誰？ | 頭痛應該找哪位醫師？ |
| 掛號流程 | 那時間呢？ | 掛號的時間是什麼時候？ |

只輸出重寫後的查詢。""",
        "config": {"model": "openai/gpt-oss-20b", "temperature": 0.1},
    },
}
