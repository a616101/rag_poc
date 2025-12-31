"""
Hierarchical Context 管理服務

智慧 context 配額分配，避免硬截斷，透過分層摘要來優化 token 使用。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class ContextAllocation:
    """Context 配額設定"""

    system_prompt: float = 0.15  # 15%
    retrieved_docs: float = 0.45  # 45%
    conversation: float = 0.25  # 25%
    current_question: float = 0.10  # 10%
    buffer: float = 0.05  # 5%

    def validate(self) -> bool:
        """驗證配額總和是否為 1.0"""
        total = (
            self.system_prompt
            + self.retrieved_docs
            + self.conversation
            + self.current_question
            + self.buffer
        )
        return abs(total - 1.0) < 0.001


@dataclass
class ContextComponent:
    """Context 組件"""

    name: str
    content: str
    priority: int = 1  # 優先級，1 最高
    max_chars: Optional[int] = None
    current_level: int = 1  # 當前摘要層級


class ContextManager:
    """
    Hierarchical Context 管理器

    提供智慧 context 配額分配，當 context 過長時使用分層摘要而非硬截斷。
    """

    def __init__(
        self,
        max_chars: int = 12000,  # 約 4000 tokens
        allocation: Optional[ContextAllocation] = None,
    ):
        """
        初始化 Context Manager

        Args:
            max_chars: 最大字元數限制（約 3 字元 = 1 token）
            allocation: Context 配額設定
        """
        self.max_chars = max_chars
        self.allocation = allocation or ContextAllocation()

        if not self.allocation.validate():
            logger.warning(
                "[ContextManager] Allocation percentages don't sum to 1.0"
            )

    def get_char_budget(self, component: str) -> int:
        """
        取得指定組件的字元預算

        Args:
            component: 組件名稱

        Returns:
            int: 分配的字元數
        """
        allocation_map = {
            "system_prompt": self.allocation.system_prompt,
            "retrieved_docs": self.allocation.retrieved_docs,
            "conversation": self.allocation.conversation,
            "current_question": self.allocation.current_question,
            "buffer": self.allocation.buffer,
        }
        percentage = allocation_map.get(component, 0.1)
        return int(self.max_chars * percentage)

    def allocate(
        self,
        components: Dict[str, str],
        priorities: Optional[Dict[str, int]] = None,
    ) -> Dict[str, str]:
        """
        根據優先級和配額智慧分配 context

        Args:
            components: 組件名稱到內容的映射
            priorities: 組件優先級（數字越小優先級越高）

        Returns:
            Dict[str, str]: 處理後的組件內容
        """
        if priorities is None:
            priorities = {
                "current_question": 1,
                "system_prompt": 2,
                "retrieved_docs": 3,
                "conversation": 4,
            }

        result: Dict[str, str] = {}
        remaining_budget = self.max_chars

        # 按優先級排序
        sorted_components = sorted(
            components.items(),
            key=lambda x: priorities.get(x[0], 10),
        )

        for name, content in sorted_components:
            budget = self.get_char_budget(name)

            if len(content) <= budget:
                result[name] = content
                remaining_budget -= len(content)
            else:
                # 需要進行摘要或截斷
                summarized = self._summarize_content(
                    content,
                    target_chars=budget,
                    component_name=name,
                )
                result[name] = summarized
                remaining_budget -= len(summarized)

        return result

    def _summarize_content(
        self,
        content: str,
        target_chars: int,
        component_name: str,
    ) -> str:
        """
        對內容進行分層摘要

        Level 1: 完整內容
        Level 2: 段落摘要（保留關鍵句）
        Level 3: 關鍵句摘要

        Args:
            content: 原始內容
            target_chars: 目標字元數
            component_name: 組件名稱

        Returns:
            str: 摘要後的內容
        """
        if len(content) <= target_chars:
            return content

        # Level 2: 段落摘要 - 保留每段的前幾行
        paragraphs = content.split("\n\n")
        if len(paragraphs) > 1:
            summarized_paragraphs = []
            chars_per_para = target_chars // len(paragraphs)

            for para in paragraphs:
                if len(para) <= chars_per_para:
                    summarized_paragraphs.append(para)
                else:
                    # 保留段落開頭
                    lines = para.split("\n")
                    kept_lines = []
                    char_count = 0
                    for line in lines:
                        if char_count + len(line) <= chars_per_para:
                            kept_lines.append(line)
                            char_count += len(line) + 1
                        else:
                            break
                    if kept_lines:
                        summarized_paragraphs.append("\n".join(kept_lines) + "...")
                    else:
                        summarized_paragraphs.append(para[:chars_per_para] + "...")

            result = "\n\n".join(summarized_paragraphs)
            if len(result) <= target_chars:
                logger.debug(
                    f"[ContextManager] Level 2 summary for {component_name}: "
                    f"{len(content)} -> {len(result)} chars"
                )
                return result

        # Level 3: 關鍵句摘要 - 直接截斷並保留結構
        # 嘗試保留文檔分隔符
        if "---" in content:
            docs = content.split("---")
            chars_per_doc = target_chars // len(docs)
            truncated_docs = []
            for doc in docs:
                doc = doc.strip()
                if doc:
                    if len(doc) <= chars_per_doc:
                        truncated_docs.append(doc)
                    else:
                        truncated_docs.append(doc[:chars_per_doc] + "...")
            result = "\n\n---\n\n".join(truncated_docs)
        else:
            # 簡單截斷
            result = content[:target_chars] + "..."

        logger.debug(
            f"[ContextManager] Level 3 summary for {component_name}: "
            f"{len(content)} -> {len(result)} chars"
        )
        return result[:target_chars]

    def compress_retrieved_docs(
        self,
        raw_chunks: List[str],
        max_chars: Optional[int] = None,
    ) -> str:
        """
        壓縮檢索到的文件內容

        Args:
            raw_chunks: 原始文件片段列表
            max_chars: 最大字元數（預設使用 retrieved_docs 配額）

        Returns:
            str: 壓縮後的內容
        """
        if max_chars is None:
            max_chars = self.get_char_budget("retrieved_docs")

        if not raw_chunks:
            return ""

        # 計算每個 chunk 的平均配額
        total_length = sum(len(chunk) for chunk in raw_chunks)

        if total_length <= max_chars:
            return "\n\n---\n\n".join(raw_chunks)

        # 需要壓縮
        chars_per_chunk = max_chars // len(raw_chunks)
        compressed_chunks = []

        for chunk in raw_chunks:
            if len(chunk) <= chars_per_chunk:
                compressed_chunks.append(chunk)
            else:
                # 保留開頭部分
                compressed_chunks.append(chunk[:chars_per_chunk] + "...")

        result = "\n\n---\n\n".join(compressed_chunks)

        logger.debug(
            f"[ContextManager] Compressed {len(raw_chunks)} docs: "
            f"{total_length} -> {len(result)} chars"
        )

        return result[:max_chars]

    def compress_conversation_history(
        self,
        messages: List[Dict[str, Any]],
        max_chars: Optional[int] = None,
        keep_recent: int = 4,
    ) -> str:
        """
        壓縮對話歷史

        保留最近 N 則完整訊息，較早的訊息進行摘要。

        Args:
            messages: 訊息列表
            max_chars: 最大字元數
            keep_recent: 保留最近 N 則完整訊息

        Returns:
            str: 壓縮後的對話歷史
        """
        if max_chars is None:
            max_chars = self.get_char_budget("conversation")

        if not messages:
            return ""

        # 分離最近訊息和較早訊息
        recent = messages[-keep_recent:] if len(messages) > keep_recent else messages
        older = messages[:-keep_recent] if len(messages) > keep_recent else []

        # 格式化最近訊息
        recent_parts = []
        for msg in recent:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            recent_parts.append(f"[{role}]: {content}")
        recent_text = "\n".join(recent_parts)

        if not older:
            if len(recent_text) <= max_chars:
                return recent_text
            return recent_text[:max_chars] + "..."

        # 為較早訊息生成摘要
        older_summary_chars = max_chars - len(recent_text) - 50  # 留一些 buffer
        if older_summary_chars > 100:
            older_parts = []
            for msg in older:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:100]
                older_parts.append(f"{role}: {content}")
            older_summary = "[Earlier conversation summary]\n" + "\n".join(older_parts)
            older_summary = older_summary[:older_summary_chars]

            return older_summary + "\n\n" + recent_text
        else:
            return recent_text[:max_chars]


# 建立全域 context manager 實例
context_manager = ContextManager()
