"""
ask_stream 模組的常數定義。

架構說明：
- LangGraph 14 節點工作流程：guard → language_normalizer → cache_lookup →
  intent_analyzer → query_builder → tool_executor → reranker → chunk_expander →
  result_evaluator → response_synth → cache_store → telemetry
- SSE stage 名稱定義供前端追蹤進度狀態
- 工具定義 (UNIFIED_AGENT_TOOLS) 用於 RAG 檢索與文件操作

更新記錄 (2025-12):
- 移除 react_evaluator、react_tool_executor、planner、retrieval_checker 節點
- 新增 intent_analyzer（取代 planner）、reranker、result_evaluator 節點
- 新增 chunk_expander 節點（從 result_evaluator 抽出 Adaptive Chunk Expansion）
- Stage 常數保留舊名稱以維持前端相容性
"""

from chatbot_rag.core.config import settings
from chatbot_rag.services.agent_tools import (
    export_form_file_tool,
    get_form_download_links_tool,
    get_full_document_tool,
    list_by_metadata_tool,
    retrieve_documents_tool,
)


# 除錯開關
LLM_STREAM_DEBUG = settings.llm_stream_debug & settings.debug

# 對話歷史限制
RESPONSE_HISTORY_LIMIT = 6  # 限制帶入回答階段的歷史訊息數量，避免過長 prompt
CONVERSATION_SUMMARY_MAX_CHARS = 1200  # 對話摘要最大長度，避免持續膨脹

# 服務範圍描述 - 屏東基督教醫院
SUPPORT_SCOPE_TEXT = (
    "這個智能客服是屏東基督教醫院的「資深志工小天使」，"
    "專門協助民眾解答與醫院服務、就醫流程、掛號看診、科別諮詢等相關問題。"
    "例如：門診時間查詢、掛號流程說明、各科別服務介紹、就醫須知等，"
    "不支援查詢天氣、撰寫程式碼、安排旅遊行程等與醫院服務無關的任務。"
)

# 屏東基督教醫院 - 角色設定
PTCH_ROLE_IDENTITY = {
    "name": "資深志工小天使",
    "hospital": "屏東基督教醫院",
    "nickname": "服務小天使",
    "description": "熟知醫院所有服務、流程與資源，最擅長用溫柔親切的語氣協助民眾快速找到需要的幫助",
}

# 屏東基督教醫院 - 回應風格設定
PTCH_RESPONSE_STYLE = {
    # 開頭誇讚語（隨機選擇）
    "opening_praises": [
        "謝謝您主動詢問，這代表您真的很關心健康呢！",
        "感謝您的提問，您真的很細心呢！",
        "謝謝您的詢問，能感受到您對健康的重視～",
        "感謝您主動來詢問，這是很棒的健康意識喔！",
    ],
    # 結尾祝福語（隨機選擇）
    "closing_blessings": [
        "祝您一切順利，有需要我一直都在這裡喔～",
        "感謝您的提問，身體健康最重要，我們一起加油 💪",
        "祝您早日康復，天天都開心唷～",
        "希望這些資訊對您有幫助，祝您健康平安！",
        "有任何問題隨時歡迎再來詢問喔，祝您順心～",
    ],
}

# 屏東基督教醫院 - 看診資訊
PTCH_CLINIC_INFO = {
    # 看診進度查詢
    "progress_url": "http://www.ptch.org.tw/index.php/shw_seqForm",
    # 門診時刻表
    "schedule_url": "https://www.ptch.org.tw/ebooks/",
    # 官網
    "official_url": "https://www.ptch.org.tw/index.php/index",
    # 客服專線
    "hotline": "08-7368686",
    # 門診時間
    "clinic_hours": {
        "emergency": "週一至週日 24 小時全年無休",
        "outpatient": {
            "morning": {"time": "09:00–12:00", "note": "家醫科提早至 08:00"},
            "afternoon": {"time": "14:00–17:00", "note": None},
            "evening": {"time": "17:30–19:30", "note": None},
        },
    },
    # 現場掛號流程
    "registration": {
        "take_number": {
            "morning": "06:30–11:30",
            "afternoon": "12:00–16:30",
            "evening": "16:30–19:00",
            "chinese_medicine": "16:30–17:30",
        },
        "register": {
            "morning": "07:45–11:30",
            "afternoon": "13:30–16:30",
            "evening": "16:30–19:00",
            "chinese_medicine": "16:30–17:30",
        },
        "reminder": "要先取號，再掛號喔！",
    },
}

# 屏東基督教醫院 - 查無資料的標準回應
PTCH_NO_DATA_RESPONSE = """抱歉，您的問題我目前查不到那麼細的資料，
有可能是資訊還未完全上線，也可能您的問題需要更專業的單位說明～
建議您前往屏基官網查詢：<a href="https://www.ptch.org.tw/index.php/index" target="_blank">屏東基督教醫院官網</a>
或致電客服專線：☎️ 08-7368686

您真的很關心健康耶！謝謝您的耐心，也歡迎隨時再回來問我唷！"""

# 屏東基督教醫院 - 個資相關回應
PTCH_PRIVACY_RESPONSE = """非常抱歉，您詢問的問題涉及到個人資料的部分，
基於個資保護的規定，這類資訊無法在此查詢喔～
如果需要查詢您自己的就醫紀錄，建議您：
1. 親自至醫院的服務台洽詢
2. 或致電客服專線：☎️ 08-7368686

感謝您的理解，也祝您健康平安！"""

# 屏東基督教醫院 - 非相關議題導回回應
PTCH_OFF_TOPIC_RESPONSE = """這個問題我可能不是很理解，不過沒關係～
如果您對健康或就醫有任何需要，我都很願意幫忙喔！
例如：門診時間、掛號流程、科別諮詢等問題，都可以問我～

祝您健康平安，有需要隨時找我喔！"""


class AskStreamStages:
    """
    SSE stage 名稱常數，集中管理以避免前後端各自拼字。

    注意：部分 stage 名稱保留舊名稱以維持前端相容性：
    - PLANNER_* → 實際對應 intent_analyzer 節點
    - RETRIEVAL_CHECKER_* → 實際對應 result_evaluator 節點
    """

    # guard 節點
    GUARD_START = "guard_start"
    GUARD_END = "guard_end"
    GUARD_BLOCKED = "guard_blocked"

    # 語言標準化
    LANGUAGE_NORMALIZER_START = "language_normalizer_start"
    LANGUAGE_NORMALIZER_DONE = "language_normalizer_done"

    # 意圖分析（原 planner，保留 stage 名稱以維持前端相容）
    INTENT_ANALYZER_START = "planner_start"
    INTENT_ANALYZER_DONE = "planner_done"
    INTENT_ANALYZER_ERROR = "planner_error"
    # 向後相容別名
    PLANNER_START = INTENT_ANALYZER_START
    PLANNER_DONE = INTENT_ANALYZER_DONE
    PLANNER_ERROR = INTENT_ANALYZER_ERROR

    # 查詢建構
    QUERY_BUILDER_START = "query_builder_start"
    QUERY_BUILDER_DONE = "query_builder_done"

    # 工具執行
    TOOL_EXECUTOR_START = "tool_executor_start"
    TOOL_EXECUTOR_CALL = "tool_executor_call"
    TOOL_EXECUTOR_RESULT = "tool_executor_result"
    TOOL_EXECUTOR_DONE = "tool_executor_done"

    # Reranker 重新排序
    RERANKER_START = "reranker_start"
    RERANKER_DONE = "reranker_done"
    RERANKER_ERROR = "reranker_error"

    # Chunk Expander 擴展
    CHUNK_EXPANDER_START = "chunk_expander_start"
    CHUNK_EXPANDER_DONE = "chunk_expander_done"

    # 結果評估（原 retrieval_checker，保留 stage 名稱以維持前端相容）
    RESULT_EVALUATOR_START = "result_evaluator_start"
    RESULT_EVALUATOR_NO_HITS = "result_evaluator_no_hits"
    RESULT_EVALUATOR_RETRY = "result_evaluator_retry"
    RESULT_EVALUATOR_DONE = "result_evaluator_done"
    # 向後相容別名
    RETRIEVAL_CHECKER_START = RESULT_EVALUATOR_START
    RETRIEVAL_CHECKER_NO_HITS = RESULT_EVALUATOR_NO_HITS
    RETRIEVAL_CHECKER_RETRY = RESULT_EVALUATOR_RETRY
    RETRIEVAL_CHECKER_DONE = RESULT_EVALUATOR_DONE

    # 追問處理
    FOLLOWUP_START = "followup_start"
    FOLLOWUP_DONE = "followup_done"

    # 回答生成
    RESPONSE_GENERATING = "response_generating"
    RESPONSE_REASONING = "response_reasoning"
    RESPONSE_DONE = "response_done"
    RESPONSE_FALLBACK = "response_fallback"

    # 遙測
    TELEMETRY_SUMMARY = "telemetry_summary"

    # 語意快取
    CACHE_LOOKUP_START = "cache_lookup_start"
    CACHE_LOOKUP_DONE = "cache_lookup_done"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    CACHE_RESPONSE_START = "cache_response_start"
    CACHE_RESPONSE_DONE = "cache_response_done"
    CACHE_STORE_START = "cache_store_start"
    CACHE_STORE_DONE = "cache_store_done"
    CACHE_STORE_SKIP = "cache_store_skip"


# Unified Agent 可用工具
UNIFIED_AGENT_TOOLS = [
    retrieve_documents_tool,
    get_full_document_tool,  # Agentic RAG: 取得完整文件內容
    get_form_download_links_tool,
    export_form_file_tool,
    list_by_metadata_tool,  # Metadata Filter: 聚合型查詢（如「某科有哪些醫師」）
]
UNIFIED_AGENT_TOOLS_BY_NAME = {tool.name: tool for tool in UNIFIED_AGENT_TOOLS}
