"""
表單匯出服務模組。

此模組負責：
- 根據 form_template 的結構化欄位定義與使用者提供的資料，產生對應的 xlsx / csv 檔案
- 檔案會儲存在專案根目錄下的 `files/` 資料夾，透過 `/files/{filename}` 供前端下載

目前的實作採用「通用表格」格式，而非直接修改官方 Excel 範本：
- 第一列為欄位說明欄位：field_key, label, required, value
- 後續各列對應每一個欄位

後續若有需要，可擴充為讀取原始範本並對應各欄位到特定儲存格。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import csv
import yaml
from loguru import logger
from openpyxl import Workbook

from chatbot_rag.core.config import settings
from chatbot_rag.services.retriever_service import retriever_service


@dataclass
class FormTemplateInfo:
    form_id: str
    name: str | None
    fields: List[Dict[str, Any]]


class FormExportService:
    """
    根據 form_template 欄位定義與資料，輸出 xlsx / csv 檔案。
    """

    def __init__(self) -> None:
        # /files 目錄，由 file_routes.py 也會使用相同路徑
        project_root = Path(__file__).resolve().parents[3]
        self.files_dir = project_root / "files"

    def _ensure_files_dir(self) -> None:
        self.files_dir.mkdir(parents=True, exist_ok=True)

    def _find_form_template(self, form_id: str) -> FormTemplateInfo:
        """
        找出對應的 form_template。

        優先順序：
        1. 透過向量檢索（若已完成向量化，可直接從 metadata 取得 fields）
        2. 若向量檢索未命中，改為直接掃描 rag_test_data/docs 目錄下的 markdown
           frontmatter（entry_type=form_template 且 form_id 匹配）
        """
        logger.info("Searching form_template for form_id='{}'", form_id)
        # 1) 先嘗試從向量檢索取得（若已向量化，這是最快方式）
        try:
            docs = retriever_service.retrieve(
                query=form_id,
                top_k=5,
                score_threshold=0.0,
            )

            for doc in docs:
                metadata = doc.get("metadata") or {}
                if metadata.get("entry_type") != "form_template":
                    continue
                if metadata.get("form_id") != form_id:
                    continue

                fields = metadata.get("fields") or []
                if not isinstance(fields, list):
                    fields = []

                return FormTemplateInfo(
                    form_id=form_id,
                    name=metadata.get("name"),
                    fields=fields,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Vector-based form_template lookup failed for form_id={} : {}",
                form_id,
                e,
            )

        # 2) 若向量檢索未命中，直接從 rag_test_data/docs 掃描 markdown frontmatter
        project_root = Path(__file__).resolve().parents[3]
        rag_dir = project_root / "rag_test_data" / "docs"

        if not rag_dir.exists():
            raise ValueError(
                f"找不到對應的 form_template（form_id={form_id}，且 docs 目錄不存在）"
            )

        for md_path in rag_dir.rglob("*.md"):
            try:
                text = md_path.read_text(encoding="utf-8")
            except Exception:
                continue

            # 簡單解析 YAML frontmatter：以 --- 開頭與下一個 --- 之間為 YAML
            if not text.lstrip().startswith("---"):
                continue

            parts = text.split("---", 2)
            if len(parts) < 3:
                continue

            frontmatter_str = parts[1]
            try:
                meta = yaml.safe_load(frontmatter_str) or {}
            except Exception:
                continue

            if not isinstance(meta, dict):
                continue

            if meta.get("entry_type") != "form_template":
                continue
            if meta.get("form_id") != form_id:
                continue

            fields = meta.get("fields") or []
            if not isinstance(fields, list):
                fields = []

            logger.info(
                "Found form_template via file scan: form_id={}, file={}",
                form_id,
                md_path,
            )
            return FormTemplateInfo(
                form_id=form_id,
                name=meta.get("name"),
                fields=fields,
            )

        raise ValueError(f"找不到對應的 form_template（form_id={form_id}）")

    def _build_safe_filename(self, form_id: str, ext: str) -> str:
        now = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in form_id)
        return f"{safe_id}-{now}.{ext}"

    def _export_to_csv(
        self,
        tmpl: FormTemplateInfo,
        data: Dict[str, Any],
    ) -> Path:
        self._ensure_files_dir()
        filename = self._build_safe_filename(tmpl.form_id, "csv")
        dst = self.files_dir / filename

        logger.info("Exporting form data to CSV: {}", dst)

        with dst.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["field_key", "label", "required", "value"])

            for field in tmpl.fields:
                key = str(field.get("key") or "")
                label = str(field.get("label") or "")
                required = bool(field.get("required", False))
                value = data.get(key, "")
                writer.writerow([key, label, "Y" if required else "", str(value)])

        return dst

    def _export_to_xlsx(
        self,
        tmpl: FormTemplateInfo,
        data: Dict[str, Any],
    ) -> Path:
        self._ensure_files_dir()
        filename = self._build_safe_filename(tmpl.form_id, "xlsx")
        dst = self.files_dir / filename

        logger.info("Exporting form data to XLSX: {}", dst)

        wb = Workbook()
        ws = wb.active
        ws.title = "表單填寫內容"

        ws.append(["field_key", "label", "required", "value"])

        for field in tmpl.fields:
            key = str(field.get("key") or "")
            label = str(field.get("label") or "")
            required = bool(field.get("required", False))
            value = data.get(key, "")
            ws.append([key, label, "Y" if required else "", str(value)])

        wb.save(dst)
        return dst

    def export_form_data(
        self,
        *,
        form_id: str,
        fmt: str = "xlsx",
        data: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        依照指定 form_id 與資料輸出 xlsx / csv 檔案。

        Args:
            form_id: 例如 "open-course-suspension-application"
            fmt: "xlsx" 或 "csv"
            data: 欄位填寫內容，key 需對應 form_template.fields[].key

        Returns:
            dict: {
              "filename": "<檔案名稱>",
              "download_url": "<可直接下載的完整 URL>",
              "format": "xlsx" | "csv"
            }
        """
        if not form_id:
            raise ValueError("form_id 不可為空")

        fmt_lower = fmt.lower()
        if fmt_lower not in {"xlsx", "csv"}:
            raise ValueError("format 僅支援 'xlsx' 或 'csv'")

        data = data or {}

        tmpl = self._find_form_template(form_id)

        # TODO: 可在此檢查必填欄位是否都有提供，並回傳警告

        if fmt_lower == "csv":
            dst = self._export_to_csv(tmpl, data)
        else:
            dst = self._export_to_xlsx(tmpl, data)

        base = settings.public_base_url.rstrip("/")
        download_url = f"{base}/files/{dst.name}"

        logger.info(
            "Form export completed. form_id={}, format={}, file={}",
            form_id,
            fmt_lower,
            dst,
        )

        return {
            "filename": dst.name,
            "download_url": download_url,
            "format": fmt_lower,
        }


form_export_service = FormExportService()



