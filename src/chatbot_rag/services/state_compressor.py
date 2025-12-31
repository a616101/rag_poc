"""
State 壓縮服務

減少大型對話的記憶體佔用，透過訊息壓縮和 context 精簡來優化資源使用。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from loguru import logger


@dataclass
class CompressionStats:
    """壓縮統計資訊"""

    original_messages: int = 0
    compressed_messages: int = 0
    original_chunks: int = 0
    compressed_chunks: int = 0
    original_chars: int = 0
    compressed_chars: int = 0

    @property
    def message_reduction(self) -> float:
        """訊息數量減少比例"""
        if self.original_messages == 0:
            return 0.0
        return 1 - (self.compressed_messages / self.original_messages)

    @property
    def char_reduction(self) -> float:
        """字元數減少比例"""
        if self.original_chars == 0:
            return 0.0
        return 1 - (self.compressed_chars / self.original_chars)


class StateCompressor:
    """
    State 壓縮器

    提供對話歷史壓縮和 retrieval chunks 精簡功能。
    """

    def __init__(
        self,
        keep_recent_messages: int = 4,
        max_chunks: int = 5,
        max_chunk_chars: int = 2000,
    ):
        """
        初始化 State 壓縮器

        Args:
            keep_recent_messages: 保留最近 N 則完整訊息
            max_chunks: 最大 chunks 數量
            max_chunk_chars: 每個 chunk 的最大字元數
        """
        self.keep_recent_messages = keep_recent_messages
        self.max_chunks = max_chunks
        self.max_chunk_chars = max_chunk_chars

    def compress_messages(
        self,
        messages: List[BaseMessage],
        keep_recent: Optional[int] = None,
    ) -> List[BaseMessage]:
        """
        壓縮對話訊息

        保留最近 N 則完整訊息，較早訊息轉為摘要。

        Args:
            messages: 原始訊息列表
            keep_recent: 保留最近 N 則（預設使用 self.keep_recent_messages）

        Returns:
            List[BaseMessage]: 壓縮後的訊息列表
        """
        if keep_recent is None:
            keep_recent = self.keep_recent_messages

        if not messages or len(messages) <= keep_recent:
            return messages

        # 分離最近訊息和較早訊息
        recent = messages[-keep_recent:]
        older = messages[:-keep_recent]

        if not older:
            return recent

        # 為較早訊息生成摘要
        summary_parts = []
        for msg in older:
            role = msg.__class__.__name__.replace("Message", "").lower()
            content = str(msg.content) if hasattr(msg, "content") else str(msg)
            # 截斷過長的內容
            if len(content) > 150:
                content = content[:150] + "..."
            summary_parts.append(f"[{role}]: {content}")

        summary_text = (
            "[Earlier conversation summary]\n" + "\n".join(summary_parts)
        )

        # 限制摘要長度
        if len(summary_text) > 1000:
            summary_text = summary_text[:1000] + "..."

        logger.debug(
            "[StateCompressor] Compressed messages: {} -> {} (summary: {} chars)",
            len(messages),
            len(recent) + 1,
            len(summary_text),
        )

        return [SystemMessage(content=summary_text)] + list(recent)

    def compress_raw_chunks(
        self,
        chunks: List[str],
        max_chunks: Optional[int] = None,
        max_chars_per_chunk: Optional[int] = None,
    ) -> List[str]:
        """
        壓縮 retrieval chunks

        保留最相關的 chunks，移除冗餘內容。

        Args:
            chunks: 原始 chunks 列表
            max_chunks: 最大 chunks 數量
            max_chars_per_chunk: 每個 chunk 的最大字元數

        Returns:
            List[str]: 壓縮後的 chunks 列表
        """
        if max_chunks is None:
            max_chunks = self.max_chunks
        if max_chars_per_chunk is None:
            max_chars_per_chunk = self.max_chunk_chars

        if not chunks:
            return []

        original_count = len(chunks)
        original_chars = sum(len(c) for c in chunks)

        # 限制 chunks 數量
        if len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]

        # 限制每個 chunk 的長度
        compressed = []
        for chunk in chunks:
            if len(chunk) > max_chars_per_chunk:
                # 智慧截斷：嘗試在段落邊界截斷
                truncated = self._smart_truncate(chunk, max_chars_per_chunk)
                compressed.append(truncated)
            else:
                compressed.append(chunk)

        compressed_chars = sum(len(c) for c in compressed)

        if original_count != len(compressed) or original_chars != compressed_chars:
            logger.debug(
                "[StateCompressor] Compressed chunks: {} -> {}, chars: {} -> {}",
                original_count,
                len(compressed),
                original_chars,
                compressed_chars,
            )

        return compressed

    def _smart_truncate(self, text: str, max_chars: int) -> str:
        """
        智慧截斷文字

        嘗試在段落或句子邊界截斷。

        Args:
            text: 原始文字
            max_chars: 最大字元數

        Returns:
            str: 截斷後的文字
        """
        if len(text) <= max_chars:
            return text

        # 嘗試在段落邊界截斷
        paragraphs = text.split("\n\n")
        result = []
        char_count = 0

        for para in paragraphs:
            if char_count + len(para) + 2 <= max_chars:
                result.append(para)
                char_count += len(para) + 2
            else:
                # 這個段落會超出限制
                remaining = max_chars - char_count - 5  # 預留 "..." 空間
                if remaining > 50:
                    # 嘗試在句子邊界截斷
                    truncated = self._truncate_at_sentence(para, remaining)
                    result.append(truncated + "...")
                break

        if not result:
            return text[:max_chars - 3] + "..."

        return "\n\n".join(result)

    def _truncate_at_sentence(self, text: str, max_chars: int) -> str:
        """
        在句子邊界截斷

        Args:
            text: 原始文字
            max_chars: 最大字元數

        Returns:
            str: 截斷後的文字
        """
        if len(text) <= max_chars:
            return text

        # 尋找句子結束符號
        sentence_ends = ["。", "！", "？", ".", "!", "?", "\n"]

        best_pos = 0
        for end_char in sentence_ends:
            pos = text.rfind(end_char, 0, max_chars)
            if pos > best_pos:
                best_pos = pos

        if best_pos > max_chars // 2:
            return text[:best_pos + 1]

        return text[:max_chars]

    def compress_state(
        self,
        state: Dict[str, Any],
        compress_messages: bool = True,
        compress_chunks: bool = True,
    ) -> tuple[Dict[str, Any], CompressionStats]:
        """
        壓縮整個 state

        Args:
            state: 原始 state 字典
            compress_messages: 是否壓縮訊息
            compress_chunks: 是否壓縮 chunks

        Returns:
            tuple: (壓縮後的 state, 壓縮統計)
        """
        stats = CompressionStats()
        new_state = dict(state)

        # 壓縮訊息
        if compress_messages and "messages" in state:
            messages = state["messages"]
            if isinstance(messages, list):
                stats.original_messages = len(messages)
                stats.original_chars += sum(
                    len(str(m.content) if hasattr(m, "content") else str(m))
                    for m in messages
                )

                compressed = self.compress_messages(messages)
                new_state["messages"] = compressed

                stats.compressed_messages = len(compressed)
                stats.compressed_chars += sum(
                    len(str(m.content) if hasattr(m, "content") else str(m))
                    for m in compressed
                )

        # 壓縮 retrieval chunks
        if compress_chunks and "retrieval" in state:
            retrieval = state.get("retrieval", {})
            if isinstance(retrieval, dict):
                raw_chunks = retrieval.get("raw_chunks", [])
                if isinstance(raw_chunks, list) and raw_chunks:
                    stats.original_chunks = len(raw_chunks)
                    stats.original_chars += sum(len(c) for c in raw_chunks)

                    compressed = self.compress_raw_chunks(raw_chunks)
                    new_retrieval = dict(retrieval)
                    new_retrieval["raw_chunks"] = compressed
                    new_state["retrieval"] = new_retrieval

                    stats.compressed_chunks = len(compressed)
                    stats.compressed_chars += sum(len(c) for c in compressed)

        return new_state, stats

    def should_compress(
        self,
        state: Dict[str, Any],
        message_threshold: int = 10,
        chunk_threshold: int = 8,
    ) -> bool:
        """
        判斷是否需要壓縮

        Args:
            state: 當前 state
            message_threshold: 訊息數量閾值
            chunk_threshold: chunks 數量閾值

        Returns:
            bool: 是否需要壓縮
        """
        messages = state.get("messages", [])
        if isinstance(messages, list) and len(messages) > message_threshold:
            return True

        retrieval = state.get("retrieval", {})
        if isinstance(retrieval, dict):
            raw_chunks = retrieval.get("raw_chunks", [])
            if isinstance(raw_chunks, list) and len(raw_chunks) > chunk_threshold:
                return True

        return False


# 建立全域 state compressor 實例
state_compressor = StateCompressor()
