"""
Agent Tools 的 OpenAI Function Calling Schema 定義。

此模組提供：
1. TOOLS_SCHEMA: OpenAI 格式的工具定義，用於 Langfuse Playground 測試
2. get_tools_schema_json(): 匯出 JSON 字串的輔助函數
3. TOOLS_SCHEMA_BY_NAME: 按工具名稱索引的字典

使用場景：
- Langfuse Playground 測試時貼上工具定義
- SDK-based Experiments 中綁定工具
- 文件產生與同步
"""

import json
from typing import Any

TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_documents_tool",
            "description": (
                "文檔檢索工具 - 從知識庫中檢索與查詢相關的文檔。"
                "用於回答平台操作、政策、流程、課程相關等問題。"
                "當使用者詢問「如何...」「怎麼...」「什麼是...」等問題時使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用於檢索的查詢文字，應該是完整的問題或關鍵概念",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_full_document_tool",
            "description": (
                "取得指定文件的完整內容。"
                "當 retrieve_documents 返回的內容不完整時使用（如表格被截斷、只有標題沒有詳細內容）。"
                "檢索結果若顯示「共 N 個片段，目前顯示第 M 個」，代表需要使用此工具取得完整文件。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "文件名稱，從 retrieve_documents 結果的來源欄位取得，例如 '劉柏屏.md'",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_form_download_links_tool",
            "description": (
                "表單下載連結查詢工具 - 查詢申請表、範本的下載連結。"
                "用於使用者需要下載表格、申請書、範本檔案時。"
                "常見問題如「申請表在哪下載」「範本檔案連結」等。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "表單相關的查詢文字，如「開放式課程掛置申請表」「測驗題範本」",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_by_metadata_tool",
            "description": (
                "Metadata 過濾列表工具 - 列出符合特定條件的所有文件摘要。"
                "適用於聚合型查詢，如「心臟血管科有哪些醫師？」「骨科的門診排班？」。"
                "使用 metadata filter 精確查詢，確保不遺漏任何符合條件的文件。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filter_field": {
                        "type": "string",
                        "description": (
                            "要過濾的 metadata 欄位名稱，常用值：\n"
                            "- 'department': 按科別過濾（如「心臟血管科」「骨科」）\n"
                            "- 'entry_type': 按文件類型過濾（如「門診時刻表」「衛教單」）\n"
                            "- 'doctor': 按醫師名稱過濾"
                        ),
                    },
                    "filter_value": {
                        "type": "string",
                        "description": "過濾條件的值，必須精確匹配（如「心臟血管科」而非「心臟科」）",
                    },
                    "entry_type": {
                        "type": "string",
                        "description": (
                            "額外的文件類型過濾（可選），用於限定結果範圍，"
                            "如「門診時刻表」「衛教單」「自動爬取」"
                        ),
                    },
                },
                "required": ["filter_field", "filter_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "export_form_file_tool",
            "description": (
                "表單匯出工具 - 根據使用者提供的資料產生填好的表單檔案（xlsx/csv）。"
                "使用前需先確認所有必填欄位都已收集完整。"
                "若資訊不足，應先向使用者追問，不要急著呼叫此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "form_id": {
                        "type": "string",
                        "description": (
                            "表單 ID，對應 RAG 中的 form_template.form_id，"
                            "例如 'open-course-suspension-application'、"
                            "'spoc-course-suspension-application'"
                        ),
                    },
                    "format": {
                        "type": "string",
                        "enum": ["xlsx", "csv"],
                        "default": "xlsx",
                        "description": "輸出格式，支援 xlsx 或 csv，預設為 xlsx",
                    },
                    "data": {
                        "type": "object",
                        "description": (
                            "欄位填寫內容的 dict，key 需對應 form_template.fields[].key，"
                            "例如 {'org_unit': '某機關', 'applicant_name': '王小明'}"
                        ),
                        "additionalProperties": True,
                    },
                },
                "required": ["form_id"],
            },
        },
    },
]

# 按工具名稱索引
TOOLS_SCHEMA_BY_NAME: dict[str, dict[str, Any]] = {
    tool["function"]["name"]: tool for tool in TOOLS_SCHEMA
}


def get_tools_schema_json(indent: int = 2) -> str:
    """
    回傳 Tools Schema 的 JSON 字串。

    可直接貼到 Langfuse Playground 的 Tools 區塊使用。

    Args:
        indent: JSON 縮排空格數，預設 2

    Returns:
        格式化的 JSON 字串
    """
    return json.dumps(TOOLS_SCHEMA, ensure_ascii=False, indent=indent)


def get_tool_names() -> list[str]:
    """回傳所有工具名稱列表"""
    return [tool["function"]["name"] for tool in TOOLS_SCHEMA]
