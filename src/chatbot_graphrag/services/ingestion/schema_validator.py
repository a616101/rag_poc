"""
YAML Schema 驗證器

驗證精選 Markdown 文件中的 YAML frontmatter。

主要功能：
- 抽取 YAML frontmatter
- 驗證文件 schema
- 轉換為 DocumentMetadata
- 支援不同文件類型的必填/選填欄位
"""

import logging
import re
from typing import Any, Optional

import yaml
from pydantic import ValidationError

from chatbot_graphrag.core.constants import DocType
from chatbot_graphrag.models.pydantic.ingestion import (
    CuratedDocSchema,
    DocumentMetadata,
)

logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """帶有詳細訊息的 Schema 驗證錯誤。"""

    def __init__(self, message: str, errors: Optional[list[dict[str, Any]]] = None):
        super().__init__(message)
        self.errors = errors or []


class SchemaValidator:
    """
    驗證精選文件的 YAML frontmatter schema。

    根據 doc_type 支援不同的 schema：
    - procedure: steps, requirements, fees, forms
    - guide.location: building, floor, room, coordinates
    - guide.transport: transport_modes, parking
    - physician: name, specialties, schedule, contact
    - hospital_team: team_name, members, services
    """

    # 每種 doc_type 的必填欄位
    # 注意：大多數必填欄位改為選填，因為實際文檔使用不同的欄位結構:
    # - hospital_team 使用 name (dict) 而非 team_name，service_scope.services 而非 services
    # - guide.location 可能使用不同的位置欄位結構
    # - physician 的 specialties 可由 expertise 推導，在 CuratedDocSchema.normalize_fields 中處理
    TYPE_REQUIRED_FIELDS: dict[DocType, list[str]] = {
        DocType.PROCEDURE: [],  # steps 為選填，文檔可能有不同結構
        DocType.GUIDE_LOCATION: [],  # building 為選填，文檔可能使用 locations
        DocType.GUIDE_TRANSPORT: [],  # transport_modes 為選填
        DocType.GUIDE_GENERIC: [],
        DocType.PHYSICIAN: ["name"],  # name 為必填 (dict 格式)
        DocType.HOSPITAL_TEAM: ["name"],  # name 為必填 (dict 格式)，services 在 service_scope 內
        DocType.FAQ: [],
        DocType.GENERIC: [],
        DocType.EDUCATION_HANDOUT: [],
        DocType.EDUCATION_GENERAL: [],
        DocType.PROCESS: [],
        DocType.PROCESS_GENERIC: [],
    }

    # 每種 doc_type 的選填欄位
    TYPE_OPTIONAL_FIELDS: dict[DocType, list[str]] = {
        DocType.PROCEDURE: ["steps", "requirements", "fees", "forms", "timeline"],
        DocType.GUIDE_LOCATION: ["building", "floor", "room", "coordinates", "landmarks", "locations"],
        DocType.GUIDE_TRANSPORT: ["transport_modes", "parking", "accessibility", "routes"],
        DocType.GUIDE_GENERIC: ["summary", "audience", "retrieval", "source"],
        DocType.PHYSICIAN: [
            "schedule", "contact", "education", "experience",
            "expertise", "specialties", "certifications", "memberships",
            "languages", "org", "departments", "role", "retrieval", "source",
        ],
        DocType.HOSPITAL_TEAM: [
            "members", "locations", "hours", "contact", "parent_department",
            "subspecialties", "availability", "service_scope", "retrieval", "source",
        ],
        DocType.FAQ: ["questions", "category"],
        DocType.GENERIC: [],
        DocType.EDUCATION_HANDOUT: [
            "summary", "audience", "education", "retrieval", "source",
            "last_reviewed", "lang",
        ],
        DocType.EDUCATION_GENERAL: ["summary", "audience", "retrieval", "source"],
        DocType.PROCESS: ["steps", "requirements", "fees", "forms"],
        DocType.PROCESS_GENERIC: ["steps", "requirements", "summary"],
    }

    # YAML frontmatter 模式
    FRONTMATTER_PATTERN = re.compile(
        r"^---\s*\n(.*?)\n---\s*\n",
        re.DOTALL,
    )

    def __init__(self):
        """初始化 schema 驗證器。"""
        self._initialized = True

    def extract_frontmatter(self, content: str) -> tuple[Optional[dict[str, Any]], str]:
        """
        從 Markdown 內容中抽取 YAML frontmatter。

        Args:
            content: 帶有 YAML frontmatter 的原始 Markdown 內容

        Returns:
            (frontmatter_dict, body_content) 元組
        """
        match = self.FRONTMATTER_PATTERN.match(content)

        if not match:
            return None, content

        try:
            frontmatter = yaml.safe_load(match.group(1))
            body = content[match.end():]
            return frontmatter, body.strip()
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse YAML frontmatter: {e}")
            return None, content

    def validate_frontmatter(
        self,
        frontmatter: dict[str, Any],
        strict: bool = True,
    ) -> CuratedDocSchema:
        """
        根據 schema 驗證 YAML frontmatter。

        Args:
            frontmatter: 已解析的 YAML frontmatter 字典
            strict: 是否強制必填欄位

        Returns:
            驗證過的 CuratedDocSchema

        Raises:
            SchemaValidationError: 驗證失敗時
        """
        # 根據基礎 schema 驗證
        try:
            schema = CuratedDocSchema.model_validate(frontmatter)
        except ValidationError as e:
            errors = [
                {
                    "field": err["loc"][0] if err["loc"] else "unknown",
                    "message": err["msg"],
                    "type": err["type"],
                }
                for err in e.errors()
            ]
            raise SchemaValidationError(
                f"Frontmatter validation failed: {len(errors)} errors",
                errors=errors,
            )

        # 驗證 doc_type 特定的必填欄位
        if strict:
            doc_type = self._parse_doc_type(schema.doc_type)
            required = self.TYPE_REQUIRED_FIELDS.get(doc_type, [])
            missing = []

            for field in required:
                if getattr(schema, field, None) is None:
                    missing.append(field)

            if missing:
                raise SchemaValidationError(
                    f"Missing required fields for {doc_type.value}: {missing}",
                    errors=[
                        {"field": field, "message": "required field", "type": "missing"}
                        for field in missing
                    ],
                )

        return schema

    def validate_document(
        self,
        content: str,
        strict: bool = True,
    ) -> tuple[CuratedDocSchema, str]:
        """
        驗證帶有 YAML frontmatter 的完整 Markdown 文件。

        Args:
            content: 完整的 Markdown 內容
            strict: 是否強制必填欄位

        Returns:
            (validated_schema, body_content) 元組

        Raises:
            SchemaValidationError: 驗證失敗時
        """
        frontmatter, body = self.extract_frontmatter(content)

        if frontmatter is None:
            raise SchemaValidationError("No YAML frontmatter found in document")

        schema = self.validate_frontmatter(frontmatter, strict=strict)

        # 驗證文件主體非空
        if not body.strip():
            raise SchemaValidationError("Document body is empty")

        return schema, body

    def to_document_metadata(
        self,
        schema: CuratedDocSchema,
    ) -> DocumentMetadata:
        """
        將 CuratedDocSchema 轉換為 DocumentMetadata。

        Args:
            schema: 驗證過的 CuratedDocSchema

        Returns:
            DocumentMetadata 實例
        """
        doc_type = self._parse_doc_type(schema.doc_type)

        # 收集自訂欄位
        custom_fields: dict[str, Any] = {}
        optional_fields = self.TYPE_OPTIONAL_FIELDS.get(doc_type, [])

        for field in optional_fields:
            value = getattr(schema, field, None)
            if value is not None:
                custom_fields[field] = value

        # 將類型特定的必填欄位加入自訂
        required_fields = self.TYPE_REQUIRED_FIELDS.get(doc_type, [])
        for field in required_fields:
            value = getattr(schema, field, None)
            if value is not None:
                custom_fields[field] = value

        # 解析生效日期
        effective_date = None
        if schema.effective_date:
            try:
                from datetime import datetime

                effective_date = datetime.fromisoformat(schema.effective_date)
            except ValueError:
                pass

        return DocumentMetadata(
            doc_type=doc_type,
            title=schema.title,
            description=schema.description,
            language=schema.language,
            department=schema.department,
            tags=schema.tags,
            version=schema.version,
            effective_date=effective_date,
            acl_groups=schema.acl_groups,
            custom_fields=custom_fields,
        )

    def _parse_doc_type(self, doc_type_str: str) -> DocType:
        """將 doc_type 字串解析為 DocType 列舉。"""
        try:
            return DocType(doc_type_str)
        except ValueError:
            logger.warning(f"Unknown doc_type: {doc_type_str}, using GENERIC")
            return DocType.GENERIC

    def get_type_schema_info(self, doc_type: DocType) -> dict[str, Any]:
        """
        取得文件類型的 schema 資訊。

        Args:
            doc_type: 文件類型

        Returns:
            包含必填和選填欄位的字典
        """
        return {
            "doc_type": doc_type.value,
            "required_fields": self.TYPE_REQUIRED_FIELDS.get(doc_type, []),
            "optional_fields": self.TYPE_OPTIONAL_FIELDS.get(doc_type, []),
        }


# 單例實例
schema_validator = SchemaValidator()
