"""
Agent 工具輸出訊息常數。

這些是工具回傳給 LLM 的結構化訊息，包含格式標籤和行為指引。
獨立成模組以避免與 ask_stream/constants.py 的循環引用。
"""

# 屏東基督教醫院聯絡資訊（避免從 constants.py 循環引用）
PTCH_OFFICIAL_URL = "https://www.ptch.org.tw/index.php/index"
PTCH_HOTLINE = "08-7368686"


class ToolOutputLabels:
    """工具輸出的結構化標籤。"""

    RETRIEVAL_RESULT = "[檢索結果]"
    FULL_DOCUMENT_RESULT = "[完整文件結果]"
    FORM_DOWNLOAD_RESULT = "[表單下載結果]"
    METADATA_FILTER_RESULT = "[Metadata 過濾結果]"
    FORM_EXPORT_RESULT = "[表單匯出結果]"
    RETRIEVAL_ERROR = "[檢索錯誤]"
    FULL_DOCUMENT_ERROR = "[完整文件錯誤]"
    FORM_DOWNLOAD_ERROR = "[表單下載錯誤]"
    METADATA_FILTER_ERROR = "[Metadata 過濾錯誤]"
    FORM_EXPORT_ERROR = "[表單匯出錯誤]"


class ToolOutputHints:
    """工具輸出的提示訊息，用於引導 LLM 的後續行為。"""

    # 文件片段提示
    INCOMPLETE_DOCUMENT_HEADER = "[提示] 以下文件內容可能不完整（被切分為多個片段）："
    INCOMPLETE_DOCUMENT_INSTRUCTION = (
        "如需完整內容，請使用 get_full_document 工具並提供 filename。"
    )
    GET_FULL_DOCUMENT_HINT = (
        "[提示] 若需要特定文件的詳細內容，請使用 get_full_document_tool 並提供 filename。"
    )

    # 檢索失敗時的行為指引
    NO_RETRIEVAL_WARNING = """⚠️ 知識庫中沒有找到與此問題相關的資訊。

[重要提示]
由於沒有找到相關文檔，你**不應該**憑空編造答案。
請誠實地告訴使用者目前知識庫中沒有這方面的資訊，
並建議他們改用其他官方管道查詢或洽詢客服。"""

    RETRIEVAL_ERROR_WARNING = """[重要提示]
由於檢索失敗，你暫時無法取得相關資訊。
請誠實告知使用者系統目前無法檢索資料，並建議稍後再試或聯繫客服。"""

    # 表單下載失敗時的行為指引
    NO_FORM_WARNING = f"""⚠️ 知識庫中沒有找到與此問題對應的表單下載資訊。

[重要提示]
你**不應該**隨意編造檔案下載連結或網址。
請誠實告知使用者目前找不到對應表單下載資訊，並建議：
- 前往屏東基督教醫院官網查詢：{PTCH_OFFICIAL_URL}
- 或致電客服專線 {PTCH_HOTLINE} 詢問正確的表單取得方式。"""

    FORM_DOWNLOAD_ERROR_WARNING = """[重要提示]
由於查詢失敗，你暫時無法提供正確的下載連結。
請誠實地告訴使用者系統目前無法查詢表單下載資訊，並建議他們稍後再試或聯繫官方客服取得表單。"""

    # 表單匯出失敗時的行為指引
    FORM_EXPORT_ERROR_WARNING = """[重要提示]
目前無法自動產生表單檔案，請先建議使用者改為下載空白範本（透過 get_form_download_links 工具取得），
再手動填寫，以確保申請流程不中斷。"""

    # Metadata 過濾失敗時的提示
    NO_METADATA_RESULT_HINT = """可能原因：
- 欄位值不精確（如「心臟科」應為「心臟血管科」）
- 該分類下確實沒有相關文件

建議：
- 使用 retrieve_documents_tool 進行語義搜尋
- 或調整查詢條件後重試"""
