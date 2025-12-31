"""
表單下載查詢服務模組。

此模組提供「專門查詢表單下載路徑」的服務，供 RAG / LangGraph 當作 Tool 使用。

設計重點：
- 僅關注與表單相關的文件，例如：
    - entry_type: form_catalog
    - entry_type: form_template
- 從向量檢索結果的 metadata 中萃取結構化的表單下載資訊，例如：
    - 表單名稱
    - 對應檔案格式
    - 下載路徑（/files/*.xlsx 等）
- 回傳結構化 list[dict]，方便上層節點組成 SystemMessage 或其他格式。
"""

from typing import Any

from loguru import logger

from chatbot_rag.core.config import settings
from chatbot_rag.services.retriever_service import retriever_service


class FormDownloadService:
    """
    表單下載查詢服務。

    提供根據自然語言查詢，找出最相關的表單下載資訊。
    """

    def search_form_downloads(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """
        根據查詢文字搜尋相關表單下載資訊。

        流程：
        1. 使用 retriever_service 進行語義檢索（不過濾 entry_type）
        2. 從每個檢索結果的 metadata 中萃取：
            - entry_type == "form_catalog"：
                - required_forms[].name / form_id / format / download_paths
            - entry_type == "form_template"：
                - file_templates[].format / path
        3. 將所有找到的表單下載資訊統一整理為 list[dict]

        回傳資料結構（示意）：
            {
                "entry_type": "form_catalog" | "form_template",
                "source_filename": "form-catalog-open-course-suspension.md",
                "module": "課程掛置與下架",
                "score": 0.93,
                "form_id": "open-course-suspension-application",
                "form_name": "開放式課程掛置申請表（開放式課程）",
                "formats": ["xlsx"],
                "download_paths": ["/files/1開放式課程掛置申請表範本1140417.xlsx"],
                "notes": "...",  # 如有
            }
        """
        logger.info(
            "Searching form downloads for query='{}', top_k={}, threshold={}",
            query[:80],
            top_k,
            score_threshold,
        )

        # 先用較低 threshold 做語義檢索，避免漏掉相關表單說明
        raw_docs = retriever_service.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        results: list[dict[str, Any]] = []

        for doc in raw_docs:
            metadata = doc.get("metadata") or {}
            entry_type = metadata.get("entry_type")
            if entry_type not in ("form_catalog", "form_template"):
                continue

            base_info: dict[str, Any] = {
                "entry_type": entry_type,
                "source_filename": metadata.get("filename", doc.get("filename")),
                "module": metadata.get("module"),
                "score": doc.get("score"),
            }

            # 1) form_catalog：通常描述一組情境與所需申請表
            if entry_type == "form_catalog":
                required_forms = metadata.get("required_forms") or []
                if isinstance(required_forms, list):
                    for form in required_forms:
                        if not isinstance(form, dict):
                            continue
                        form_id = form.get("form_id")
                        form_name = form.get("name") or form_id
                        formats = form.get("format") or []
                        if isinstance(formats, list):
                            formats_list = [str(f) for f in formats]
                        else:
                            formats_list = [str(formats)]

                        download_paths = form.get("download_paths") or []
                        if isinstance(download_paths, list):
                            paths_list = [str(p) for p in download_paths]
                        elif download_paths:
                            paths_list = [str(download_paths)]
                        else:
                            paths_list = []

                        notes = form.get("notes")

                        if not paths_list:
                            # 沒有實際下載路徑就略過，避免給出空資訊
                            continue

                        download_urls = [
                            self._build_download_url(path) for path in paths_list
                        ]

                        results.append(
                            {
                                **base_info,
                                "form_id": form_id,
                                "form_name": form_name,
                                "formats": formats_list,
                                "download_paths": paths_list,
                                "download_urls": download_urls,
                                "notes": notes,
                            }
                        )

            # 2) form_template：直接描述某一種表單範本與實體檔案
            if entry_type == "form_template":
                file_templates = metadata.get("file_templates") or []
                form_name = metadata.get("name")
                form_id = metadata.get("form_id")
                if isinstance(file_templates, list):
                    for tpl in file_templates:
                        if not isinstance(tpl, dict):
                            continue
                        fmt = tpl.get("format") or tpl.get("type") or ""
                        path = tpl.get("path") or ""
                        if not path:
                            continue

                        download_url = self._build_download_url(str(path))

                        results.append(
                            {
                                **base_info,
                                "form_id": form_id,
                                "form_name": form_name or form_id,
                                "formats": [str(fmt)] if fmt else [],
                                "download_paths": [str(path)],
                                "download_urls": [download_url],
                                "notes": None,
                            }
                        )

        logger.info("Form download search found {} results", len(results))
        return results

    @staticmethod
    def _build_download_url(path: str) -> str:
        """
        將資料中的下載路徑轉換為完整 URL。

        規則：
        - 若 path 已是 http:// 或 https:// 開頭，直接回傳
        - 若以 "/" 開頭，視為相對於 API 根路徑，例如 "/files/xxx.xlsx"
            -> 使用 settings.public_base_url 去組合
        - 其他情況，一律視為相對路徑，前面補上一個 "/"
        """
        path = path.strip()
        if path.startswith("http://") or path.startswith("https://"):
            return path

        base = settings.public_base_url.rstrip("/")

        if path.startswith("/"):
            return f"{base}{path}"

        return f"{base}/{path}"


# 全域實例，供其他模組直接導入使用
form_download_service = FormDownloadService()


