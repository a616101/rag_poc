"""
Agent 用工具定義模組。

此模組提供可給 LLM 綁定使用的 LangChain Tool：
- retrieve_documents_tool: 一般知識庫文檔檢索
- get_full_document_tool: 取得指定文件的完整內容
- get_form_download_links_tool: 表單下載連結查詢

這些工具都會回傳「已整理好、可直接給 LLM 當 context 的文字」，方便在
Workflow + Agent 架構中使用 `llm.bind_tools([...])` 進行自動工具調用。

注意：工具輸出訊息的常數定義在 ask_stream/constants.py 中的
ToolOutputLabels 和 ToolOutputHints 類別。
"""

from langchain_core.tools import tool
from loguru import logger

from chatbot_rag.llm.graph_nodes import retrieval_top_k_var
from chatbot_rag.services.retriever_service import retriever_service
from chatbot_rag.services.qdrant_service import qdrant_service
from chatbot_rag.services.form_tool_service import form_download_service
from chatbot_rag.services.form_export_service import form_export_service
from chatbot_rag.services.tool_output_messages import (
    ToolOutputLabels,
    ToolOutputHints,
)
# 漸進式檢索閾值設定
RETRIEVAL_THRESHOLDS = [0.65, 0.50, 0.35]


@tool
def retrieve_documents_tool(query: str) -> str:
    """
    文檔檢索工具 - 從知識庫中檢索與查詢相關的文檔。

    參數：
        query (str): 用於檢索的查詢文字，通常由 Agent 根據使用者問題生成。

    行為：
        1. 使用漸進式閾值策略進行語義檢索（0.65 → 0.50 → 0.35）
        2. 高閾值優先確保品質，低閾值作為後備擴大召回
        3. 將檢索結果格式化為可讀文字，包含來源檔案與相似度
        4. 回傳「[檢索結果] + 各文件內容」的文字摘要

    注意：
        - 若未找到任何文檔，會明確說明「找到的文檔數量: 0」，並提醒不要亂編造答案。
    """
    logger.info("Tool[retrieve_documents] called with query: {}", query[:80])

    try:
        # 從 context variable 取得 top_k 參數（由 API 端點傳入，預設 3）
        top_k = retrieval_top_k_var.get()
        logger.info("Tool[retrieve_documents] using top_k={}", top_k)

        logger.info("Tool[retrieve_documents] final query for retrieval: {}", query)

        # 漸進式閾值檢索：從高到低嘗試
        documents = []
        used_threshold = RETRIEVAL_THRESHOLDS[0]

        for threshold in RETRIEVAL_THRESHOLDS:
            documents = retriever_service.retrieve(
                query=query,
                top_k=top_k,
                score_threshold=threshold,
            )
            used_threshold = threshold

            if documents:
                logger.info(
                    "Tool[retrieve_documents] found {} docs at threshold={}",
                    len(documents),
                    threshold,
                )
                break
            else:
                logger.info(
                    "Tool[retrieve_documents] no docs at threshold={}, trying lower",
                    threshold,
                )

        # 構建結果前綴，包含檢索資訊
        result_lines = [ToolOutputLabels.RETRIEVAL_RESULT]
        result_lines.append(f"找到的文檔數量: {len(documents)}")
        result_lines.append(f"使用的相似度閾值: {used_threshold}")

        # 收集文件的 chunk 資訊，提示 Agent 可能需要取得完整文件
        incomplete_docs = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            total_chunks = metadata.get("total_chunks", 1)
            if total_chunks > 1:
                incomplete_docs.append({
                    "filename": doc.get("filename", "unknown"),
                    "total_chunks": total_chunks,
                    "current_chunk": metadata.get("chunk_index", 0),
                })

        if incomplete_docs:
            result_lines.append("")
            result_lines.append(ToolOutputHints.INCOMPLETE_DOCUMENT_HEADER)
            for info in incomplete_docs:
                result_lines.append(
                    f"  - {info['filename']} (共 {info['total_chunks']} 個片段，"
                    f"目前顯示第 {info['current_chunk'] + 1} 個)"
                )
            result_lines.append(ToolOutputHints.INCOMPLETE_DOCUMENT_INSTRUCTION)

        result_prefix = "\n".join(result_lines) + "\n"

        if not documents:
            logger.warning("Tool[retrieve_documents] found no documents after all thresholds")
            return f"{result_prefix}\n{ToolOutputHints.NO_RETRIEVAL_WARNING}"

        context = retriever_service.format_context(documents)
        logger.info(
            "Tool[retrieve_documents] retrieved {} documents (threshold={})",
            len(documents),
            used_threshold,
        )

        return f"{result_prefix}\n{context}"

    except Exception as e:  # noqa: BLE001
        logger.error("Tool[retrieve_documents] error: {}", e)
        return (
            f"{ToolOutputLabels.RETRIEVAL_ERROR}\n"
            f"檢索文檔時發生錯誤: {str(e)}\n\n"
            f"{ToolOutputHints.RETRIEVAL_ERROR_WARNING}"
        )


@tool
def get_full_document_tool(filename: str) -> str:
    """
    取得指定文件的完整內容。

    當 retrieve_documents 返回的內容不完整時（例如：表格被截斷、只有標題
    沒有詳細內容、檢索結果提示「文件被切分為多個片段」），使用此工具取得
    該文件的完整內容。

    參數：
        filename (str): 文件名稱，從 retrieve_documents 結果的來源欄位取得，
                       例如 "劉柏屏.md"、"護理之家.md"

    回傳：
        該文件的完整內容，包含所有片段合併後的文字。

    使用時機：
        1. 檢索結果顯示「共 N 個片段，目前顯示第 M 個」
        2. 內容明顯被截斷（如表格只有標題沒有資料列）
        3. 需要文件的完整上下文才能正確回答問題
    """
    logger.info("Tool[get_full_document] called with filename: {}", filename)

    try:
        # 從 Qdrant 取得該文件的所有 chunks
        chunks = qdrant_service.fetch_all_chunks_by_filename(filename)

        if not chunks:
            logger.warning("Tool[get_full_document] no chunks found for: {}", filename)
            return (
                f"{ToolOutputLabels.FULL_DOCUMENT_RESULT}\n"
                f"文件名稱: {filename}\n"
                f"狀態: 找不到此文件\n\n"
                f"⚠️ 知識庫中沒有找到名為「{filename}」的文件。\n"
                f"請確認文件名稱是否正確（應從 retrieve_documents 結果中取得）。"
            )

        # 合併所有 chunks 的文字內容
        # 優先使用原始 text（不含 context 前綴），避免重複的 metadata
        texts = []
        metadata = {}
        for chunk in chunks:
            payload = chunk.get("payload", {})
            # 使用原始 text，因為 contextualized_text 包含重複的 <context> 標籤
            text = payload.get("text", "")
            if text:
                texts.append(text)
            # 保留第一個 chunk 的 metadata 作為文件 metadata
            if not metadata:
                metadata = {
                    "source": payload.get("source", ""),
                    "entry_type": payload.get("entry_type", ""),
                    "module": payload.get("module", ""),
                    "total_chunks": payload.get("total_chunks", len(chunks)),
                }

        # 合併文字（chunks 已按 chunk_index 排序）
        full_content = "\n".join(texts)

        # 構建結果
        result_lines = [ToolOutputLabels.FULL_DOCUMENT_RESULT]
        result_lines.append(f"文件名稱: {filename}")
        result_lines.append(f"片段數量: {len(chunks)}")
        if metadata.get("entry_type"):
            result_lines.append(f"類型: {metadata['entry_type']}")
        if metadata.get("module"):
            result_lines.append(f"模組: {metadata['module']}")
        result_lines.append("")
        result_lines.append("--- 完整內容 ---")
        result_lines.append(full_content)

        logger.info(
            "Tool[get_full_document] returned {} chunks, {} chars for {}",
            len(chunks),
            len(full_content),
            filename,
        )

        return "\n".join(result_lines)

    except Exception as e:  # noqa: BLE001
        logger.error("Tool[get_full_document] error: {}", e)
        return (
            f"{ToolOutputLabels.FULL_DOCUMENT_ERROR}\n"
            f"取得文件「{filename}」時發生錯誤: {str(e)}\n\n"
            f"請稍後再試，或使用 retrieve_documents 重新檢索相關內容。"
        )


@tool
def get_form_download_links_tool(query: str) -> str:
    """
    表單下載連結查詢工具 - 根據查詢文字找出相關申請表或範本的下載連結。

    常見問題：
        - 「開放式課程掛置申請表在哪裡下載？」
        - 「SPOC 課程掛置的 Excel 申請表下載連結？」

    行為：
        1. 使用 form_download_service.search_form_downloads 進行語義搜尋
        2. 擷取表單名稱、檔案格式、下載路徑與完整 URL
        3. 回傳可直接給最終回答引用的文字摘要
    """
    logger.info("Tool[get_form_download_links] called with query: {}", query[:80])

    try:
        forms = form_download_service.search_form_downloads(
            query=query,
            top_k=5,
            score_threshold=0.3,
        )

        if not forms:
            logger.warning("Tool[get_form_download_links] found no forms")
            return (
                f"{ToolOutputLabels.FORM_DOWNLOAD_RESULT}\n"
                f"找到的表單數量: 0\n\n"
                f"{ToolOutputHints.NO_FORM_WARNING}"
            )

        lines: list[str] = []
        lines.append(ToolOutputLabels.FORM_DOWNLOAD_RESULT)
        lines.append(f"找到的表單數量: {len(forms)}")
        lines.append("")

        for idx, form in enumerate(forms, 1):
            form_name = form.get("form_name") or form.get("form_id") or "未命名表單"
            formats = form.get("formats") or []
            formats_str = "、".join(str(f) for f in formats) if formats else "（未標示格式）"
            download_urls = form.get("download_urls") or form.get("download_paths") or []
            notes = form.get("notes")

            lines.append(f"{idx}. 表單名稱：{form_name}")
            lines.append(f"   檔案格式：{formats_str}")
            for url in download_urls:
                lines.append(f"   下載連結：{url}")
            if notes:
                lines.append(f"   備註：{notes}")
            lines.append("")

        logger.info("Tool[get_form_download_links] found {} forms", len(forms))
        return "\n".join(lines)

    except Exception as e:  # noqa: BLE001
        logger.error("Tool[get_form_download_links] error: {}", e)
        return (
            f"{ToolOutputLabels.FORM_DOWNLOAD_ERROR}\n"
            f"查詢表單下載資訊時發生錯誤: {str(e)}\n\n"
            f"{ToolOutputHints.FORM_DOWNLOAD_ERROR_WARNING}"
        )


@tool
def list_by_metadata_tool(
    filter_field: str,
    filter_value: str,
    entry_type: str | None = None,
) -> str:
    """
    Metadata 過濾列表工具 - 列出符合條件的所有文件摘要。

    適用於聚合型查詢，例如：
        - 「心臟科有哪些醫師？」→ filter_field="department", filter_value="心臟"
        - 「骨科的門診排班？」→ filter_field="department", filter_value="骨科", entry_type="門診時刻表"
        - 「有哪些衛教單？」→ filter_field="entry_type", filter_value="衛教單"

    參數：
        filter_field (str): 要過濾的 metadata 欄位名稱，常用值：
            - "department": 按科別過濾（支援模糊匹配，如「心臟」會匹配「心臟血管科」）
            - "entry_type": 按文件類型過濾（如「門診時刻表」「衛教單」）
            - "doctor": 按醫師名稱過濾
        filter_value (str): 過濾條件的值（支援模糊匹配）
        entry_type (str, optional): 額外的文件類型過濾，用於限定結果範圍

    回傳：
        符合條件的文件清單摘要，包含數量統計和各文件的基本資訊。

    注意：
        - 此工具只返回文件的 metadata 摘要，不包含完整內容
        - 若需要某文件的詳細內容，請使用 get_full_document_tool
        - 科別名稱支援模糊匹配（如「心臟」會匹配「心臟血管科」）
    """
    logger.info(
        "Tool[list_by_metadata] called with field={}, value={}, entry_type={}",
        filter_field,
        filter_value,
        entry_type,
    )

    try:
        # 優先使用模糊匹配（支援簡稱如「心臟」匹配「心臟血管科」）
        files = qdrant_service.list_by_fuzzy_filter(
            field_name=filter_field,
            query_value=filter_value,
            entry_type=entry_type,
            limit=50,
        )

        # 如果模糊匹配沒結果，嘗試精確匹配（向後相容）
        if not files:
            filters: dict[str, str] = {filter_field: filter_value}
            if entry_type:
                filters["entry_type"] = entry_type
            files = qdrant_service.list_by_filter(filters=filters, limit=50)

        if not files:
            logger.warning(
                "Tool[list_by_metadata] no files found for {}={}",
                filter_field,
                filter_value,
            )
            entry_type_str = f', entry_type="{entry_type}"' if entry_type else ''
            return (
                f"{ToolOutputLabels.METADATA_FILTER_RESULT}\n"
                f"過濾條件: {filter_field}=\"{filter_value}\"{entry_type_str}\n"
                f"符合的文件數量: 0\n\n"
                f"⚠️ 知識庫中沒有找到符合此條件的文件。\n\n"
                f"{ToolOutputHints.NO_METADATA_RESULT_HINT}"
            )

        # 建構結果摘要
        result_lines = [ToolOutputLabels.METADATA_FILTER_RESULT]
        entry_type_suffix = f', entry_type="{entry_type}"' if entry_type else ''
        result_lines.append(f"過濾條件: {filter_field}=\"{filter_value}\"{entry_type_suffix}")
        result_lines.append(f"符合的文件數量: {len(files)}")
        result_lines.append("")

        # 按不同場景格式化輸出
        if filter_field == "department" or (filter_field == "entry_type" and filter_value == "門診時刻表"):
            # 科別查詢或門診查詢：突出醫師資訊
            doctors_info = []
            for f in files:
                doctor = f.get("doctor", "")
                if doctor:
                    schedule_count = f.get("schedule_count")
                    if schedule_count:
                        doctors_info.append(f"- {doctor}醫師（{schedule_count} 筆門診）")
                    else:
                        doctors_info.append(f"- {doctor}醫師")

            if doctors_info:
                dept = filter_value if filter_field == "department" else "（多科別）"
                result_lines.append(f"【{dept}】共 {len(doctors_info)} 位醫師：")
                result_lines.extend(doctors_info)
            else:
                # 沒有醫師資訊，列出文件標題
                result_lines.append("文件清單：")
                for f in files:
                    title = f.get("title", f.get("filename", "未知"))
                    result_lines.append(f"- {title}")
        else:
            # 其他查詢：列出文件標題
            result_lines.append("文件清單：")
            for f in files:
                title = f.get("title", f.get("filename", "未知"))
                entry_t = f.get("entry_type", "")
                if entry_t:
                    result_lines.append(f"- {title}（{entry_t}）")
                else:
                    result_lines.append(f"- {title}")

        result_lines.append("")
        result_lines.append(ToolOutputHints.GET_FULL_DOCUMENT_HINT)

        logger.info(
            "Tool[list_by_metadata] found {} files for {}={}",
            len(files),
            filter_field,
            filter_value,
        )
        return "\n".join(result_lines)

    except Exception as e:  # noqa: BLE001
        logger.error("Tool[list_by_metadata] error: {}", e)
        return (
            f"{ToolOutputLabels.METADATA_FILTER_ERROR}\n"
            f"查詢時發生錯誤: {str(e)}\n\n"
            f"請改用 retrieve_documents_tool 進行語義搜尋。"
        )


@tool
def export_form_file_tool(
    form_id: str,
    format: str = "xlsx",
    data: dict | None = None,
) -> str:
    """
    表單匯出工具 - 根據 form_template 欄位結構與使用者提供的資料，產出 xlsx / csv 檔案。

    使用說明（給 Agent / LLM）：
    1. form_id 應對應 RAG 中的 form_template.form_id，例如：
       - "open-course-suspension-application"
       - "spoc-course-suspension-application"
    2. format 僅支援 "xlsx" 或 "csv"（預設為 "xlsx"）。
    3. data 為欄位填寫內容的 dict，其中 key 需對應 form_template.fields[].key，
       例如：
       {
         "org_unit": "某某機關及服務部門",
         "applicant_name": "王小明",
         "course_name": "開放式課程 XXX",
         "reason": "說明掛置原因..."
       }
    4. 在呼叫本工具前，請先透過多輪對話確認所有「必填欄位」都有清楚的值，
       若資訊不足，請優先向使用者追問，不要急著呼叫本工具。

    回傳：
        一段可讀文字，內含可直接下載的檔案連結（/files/{filename}），
        例如：
        - 已為您產生「開放式課程掛置申請」xlsx 檔案，下載連結：...
    """
    logger.info(
        "Tool[export_form_file] called with form_id={}, format={}", form_id, format
    )

    try:
        result = form_export_service.export_form_data(
            form_id=form_id,
            fmt=format,
            data=data or {},
        )
        filename = result.get("filename")
        download_url = result.get("download_url")
        fmt = result.get("format")

        return (
            f"{ToolOutputLabels.FORM_EXPORT_RESULT}\n"
            f"已根據 form_id='{form_id}' 產生 {fmt} 檔案：{filename}\n"
            f"下載連結：{download_url}\n\n"
            "請提醒使用者：下載後仍可依實際情況調整內容，並依學院規定流程送審。"
        )

    except Exception as e:  # noqa: BLE001
        logger.error("Tool[export_form_file] error: {}", e)
        return (
            f"{ToolOutputLabels.FORM_EXPORT_ERROR}\n"
            f"產生表單檔案時發生錯誤: {str(e)}\n\n"
            f"{ToolOutputHints.FORM_EXPORT_ERROR_WARNING}"
        )


