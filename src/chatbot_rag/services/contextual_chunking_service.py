"""
Contextual Chunking 服務模組

此模組實作 Anthropic 提出的 Contextual Retrieval 技術，
為每個 chunk 添加文檔脈絡，提升檢索精準度。

主要功能：
- Level 1: 使用 frontmatter + section headers 自動生成脈絡（不需 LLM）
- Level 2: 使用 LLM 生成語義脈絡描述（可選）
"""

import time
from typing import Optional

from langchain_openai import ChatOpenAI
from loguru import logger

from chatbot_rag.core.config import settings
from chatbot_rag.services.markdown_parser import Section, markdown_parser

# LLM 重試設定
LLM_MAX_RETRIES = 3  # 最大重試次數
LLM_RETRY_DELAY = 2.0  # 重試間隔（秒）
LLM_RETRY_BACKOFF = 2.0  # 退避倍數


class ContextualChunkingService:
    """
    Contextual Chunking 服務

    為每個 chunk 生成脈絡描述，讓 embedding 能捕捉更完整的語義。
    支援兩個層級的脈絡生成：
    - Level 1: 結構化脈絡（使用 metadata + section headers）
    - Level 2: 語義脈絡（使用 LLM 生成描述）
    """

    def __init__(self):
        """初始化服務，LLM client 採用延遲載入"""
        self._llm_client: Optional[ChatOpenAI] = None

    @property
    def llm_client(self) -> ChatOpenAI:
        """
        取得或建立 LLM client（延遲載入）。

        使用獨立的輕量設定，不帶 reasoning 模式。
        """
        if self._llm_client is None:
            model = settings.contextual_chunking_model or settings.chat_model
            logger.info(
                f"Initializing contextual chunking LLM client with model: {model}"
            )
            self._llm_client = ChatOpenAI(
                model=model,
                temperature=settings.contextual_chunking_temperature,
                max_tokens=settings.contextual_chunking_max_tokens,
                openai_api_key=settings.openai_api_key,
                openai_api_base=settings.openai_api_base,
                streaming=False,
            )
        return self._llm_client

    def generate_context_level1(
        self,
        doc_metadata: dict,
        section_path: str,
    ) -> str:
        """
        生成 Level 1 脈絡（結構化，不需 LLM）。

        使用 frontmatter metadata 和 section headers 組合脈絡。

        Args:
            doc_metadata: 文檔的 metadata（來自 frontmatter）
            section_path: 章節路徑（例如 "三、資源 > （一）專區"）

        Returns:
            str: 脈絡描述字串
        """
        doc_title = doc_metadata.get("doc_title") or doc_metadata.get("filename", "")
        entry_type = doc_metadata.get("entry_type", "")
        module = doc_metadata.get("module", "")

        # 組合脈絡
        parts = [f"文件：{doc_title}"]

        type_module_parts = []
        if entry_type:
            type_module_parts.append(f"類型：{entry_type}")
        if module:
            type_module_parts.append(f"模組：{module}")
        if type_module_parts:
            parts.append(" | ".join(type_module_parts))

        if section_path:
            parts.append(f"章節：{section_path}")

        return "\n".join(parts)

    def generate_context_level2(
        self,
        chunk_text: str,
        doc_metadata: dict,
        section_path: str,
    ) -> str:
        """
        生成 Level 2 脈絡（語義描述，需要 LLM）。

        使用 LLM 分析 chunk 內容並生成簡短的語義描述。
        包含重試機制以處理 LLM 服務暫時不可用的情況（如 LMStudio model crash）。

        Args:
            chunk_text: chunk 的文字內容
            doc_metadata: 文檔的 metadata
            section_path: 章節路徑

        Returns:
            str: LLM 生成的語義描述（1-2 句話）
        """
        doc_title = doc_metadata.get("doc_title") or doc_metadata.get("filename", "")

        prompt = f"""請用一句話（20-40字）描述以下文字片段的主要內容。
不要重複文件標題或章節名稱，專注於描述這段文字在說什麼。

文件：{doc_title}
章節：{section_path or "(無)"}

文字片段：
{chunk_text[:800]}

輸出格式：直接輸出描述，不要加引號、標點符號開頭或任何前綴。"""

        last_error = None
        delay = LLM_RETRY_DELAY

        for attempt in range(LLM_MAX_RETRIES):
            try:
                response = self.llm_client.invoke([{"role": "user", "content": prompt}])
                content = response.content
                if isinstance(content, str):
                    return content.strip()
                return str(content).strip()
            except Exception as e:
                last_error = e
                error_str = str(e)

                # 檢查是否為可重試的錯誤（如 400 model not loaded、連線錯誤等）
                is_retryable = (
                    "400" in error_str
                    or "No models loaded" in error_str
                    or "Connection" in error_str
                    or "timeout" in error_str.lower()
                )

                if is_retryable and attempt < LLM_MAX_RETRIES - 1:
                    logger.warning(
                        f"LLM context generation failed (attempt {attempt + 1}/{LLM_MAX_RETRIES}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                    delay *= LLM_RETRY_BACKOFF  # 指數退避
                else:
                    # 不可重試或已達最大重試次數
                    break

        logger.warning(
            f"LLM context generation failed after {LLM_MAX_RETRIES} attempts: {last_error}"
        )
        return ""

    def generate_context(
        self,
        chunk_text: str,
        doc_metadata: dict,
        section_path: str,
        use_llm: bool = True,
    ) -> str:
        """
        生成完整的脈絡描述。

        結合 Level 1（結構化）和 Level 2（語義）脈絡。

        Args:
            chunk_text: chunk 的文字內容
            doc_metadata: 文檔的 metadata
            section_path: 章節路徑
            use_llm: 是否使用 LLM 生成 Level 2 脈絡

        Returns:
            str: 完整的脈絡描述
        """
        # Level 1: 結構化脈絡
        context_parts = [self.generate_context_level1(doc_metadata, section_path)]

        # Level 2: 語義脈絡（可選）
        if use_llm and settings.contextual_chunking_use_llm:
            semantic_context = self.generate_context_level2(
                chunk_text, doc_metadata, section_path
            )
            if semantic_context:
                context_parts.append(f"內容摘要：{semantic_context}")

        return "\n".join(context_parts)

    def contextualize_chunk(
        self,
        chunk: dict,
        doc_content: str,
        doc_metadata: dict,
        sections: Optional[list[Section]] = None,
        use_llm: bool = True,
    ) -> dict:
        """
        為單個 chunk 添加脈絡。

        Args:
            chunk: 原始 chunk 字典，包含 text 和 metadata
            doc_content: 完整的文檔內容（用於定位章節）
            doc_metadata: 文檔的 metadata
            sections: 預先解析的章節列表（可選）
            use_llm: 是否使用 LLM

        Returns:
            dict: 更新後的 chunk，新增 contextualized_text 和 section_path
        """
        chunk_text = chunk.get("text", "")

        # 找出 chunk 所屬的章節
        section_path = markdown_parser.find_section_path_for_text(
            doc_content, chunk_text, sections
        )

        # 生成脈絡
        context = self.generate_context(
            chunk_text=chunk_text,
            doc_metadata=doc_metadata,
            section_path=section_path,
            use_llm=use_llm,
        )

        # 組合脈絡化文字
        contextualized_text = f"<context>\n{context}\n</context>\n{chunk_text}"

        # 更新 chunk
        chunk["contextualized_text"] = contextualized_text
        chunk["section_path"] = section_path

        return chunk

    def contextualize_chunks(
        self,
        chunks: list[dict],
        doc_content: str,
        doc_metadata: dict,
        use_llm: bool = True,
    ) -> list[dict]:
        """
        批量為 chunks 添加脈絡。

        Args:
            chunks: chunk 列表
            doc_content: 完整的文檔內容
            doc_metadata: 文檔的 metadata
            use_llm: 是否使用 LLM

        Returns:
            list[dict]: 更新後的 chunk 列表
        """
        if not chunks:
            return chunks

        # 預先解析章節結構（避免重複解析）
        sections = markdown_parser.parse_sections(doc_content)

        logger.info(
            f"Contextualizing {len(chunks)} chunks "
            f"(use_llm={use_llm}, sections={len(sections)})"
        )

        contextualized_chunks = []
        for i, chunk in enumerate(chunks):
            try:
                contextualized = self.contextualize_chunk(
                    chunk=chunk,
                    doc_content=doc_content,
                    doc_metadata=doc_metadata,
                    sections=sections,
                    use_llm=use_llm,
                )
                contextualized_chunks.append(contextualized)

                if (i + 1) % 10 == 0:
                    logger.debug(f"Contextualized {i + 1}/{len(chunks)} chunks")

            except Exception as e:
                logger.warning(f"Failed to contextualize chunk {i}: {e}")
                # 失敗時保留原始 chunk
                chunk["contextualized_text"] = chunk.get("text", "")
                chunk["section_path"] = ""
                contextualized_chunks.append(chunk)

        logger.info(f"Contextualization complete: {len(contextualized_chunks)} chunks")
        return contextualized_chunks


# 全域實例
contextual_chunking_service = ContextualChunkingService()
