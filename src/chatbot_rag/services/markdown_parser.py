"""
Markdown 結構解析模組

此模組提供 Markdown 文件的結構化解析功能，包括：
- 解析章節標題（## / ### headers）
- 追蹤每個文字片段所屬的章節
- 支援巢狀章節結構
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Section:
    """表示 Markdown 文件中的一個章節"""

    level: int  # 標題層級（1=h1, 2=h2, 3=h3...）
    title: str  # 章節標題文字
    start_pos: int  # 章節開始位置（字元索引）
    end_pos: int = -1  # 章節結束位置（-1 表示延伸到文件結尾）
    parent: Optional["Section"] = None  # 父章節（用於建立巢狀結構）
    children: list["Section"] = field(default_factory=list)


class MarkdownParser:
    """
    Markdown 結構解析器

    主要功能：
    1. 解析 Markdown 文件中的所有章節標題
    2. 建立章節的層級結構
    3. 根據文字位置找出其所屬章節路徑
    """

    # 匹配 Markdown 標題的正則表達式
    # 支援 # 到 ###### 的標題
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def parse_sections(self, content: str) -> list[Section]:
        """
        解析 Markdown 內容中的所有章節。

        Args:
            content: Markdown 文件內容

        Returns:
            list[Section]: 章節列表，按出現順序排列
        """
        sections: list[Section] = []

        # 找出所有標題
        for match in self.HEADER_PATTERN.finditer(content):
            level = len(match.group(1))  # # 的數量決定層級
            title = match.group(2).strip()
            start_pos = match.start()

            section = Section(
                level=level,
                title=title,
                start_pos=start_pos,
            )
            sections.append(section)

        # 設定每個章節的結束位置
        for i, section in enumerate(sections):
            if i + 1 < len(sections):
                section.end_pos = sections[i + 1].start_pos
            else:
                section.end_pos = len(content)

        # 建立父子關係
        self._build_hierarchy(sections)

        return sections

    def _build_hierarchy(self, sections: list[Section]) -> None:
        """
        建立章節的父子關係。

        Args:
            sections: 章節列表（會被原地修改）
        """
        if not sections:
            return

        # 使用堆疊追蹤當前的章節層級
        stack: list[Section] = []

        for section in sections:
            # 彈出所有層級 >= 當前章節的項目
            while stack and stack[-1].level >= section.level:
                stack.pop()

            # 如果堆疊不為空，最後一個就是父章節
            if stack:
                section.parent = stack[-1]
                stack[-1].children.append(section)

            stack.append(section)

    def find_section_for_position(
        self, sections: list[Section], position: int
    ) -> Optional[Section]:
        """
        根據字元位置找出其所屬的章節。

        Args:
            sections: 章節列表
            position: 文字的字元位置

        Returns:
            Optional[Section]: 所屬章節，如果找不到則返回 None
        """
        result: Optional[Section] = None

        for section in sections:
            if section.start_pos <= position < section.end_pos:
                result = section
                # 繼續尋找更具體的子章節
            elif section.start_pos > position:
                break

        return result

    def get_section_path(self, section: Optional[Section]) -> str:
        """
        取得章節的完整路徑（從根到當前章節）。

        Args:
            section: 章節物件

        Returns:
            str: 章節路徑，例如 "三、加盟機關可獲得之資源 > （一）專屬訓練專區及網址"
        """
        if section is None:
            return ""

        path_parts: list[str] = []
        current: Optional[Section] = section

        while current is not None:
            path_parts.append(current.title)
            current = current.parent

        # 反轉順序（從根到葉）
        path_parts.reverse()

        return " > ".join(path_parts)

    def find_section_path_for_text(
        self, content: str, text: str, sections: Optional[list[Section]] = None
    ) -> str:
        """
        找出指定文字片段所屬的章節路徑。

        Args:
            content: 完整的 Markdown 內容
            text: 要查找的文字片段
            sections: 預先解析的章節列表（可選，如未提供則會自動解析）

        Returns:
            str: 章節路徑
        """
        if sections is None:
            sections = self.parse_sections(content)

        # 找出文字在內容中的位置
        position = content.find(text)
        if position == -1:
            # 嘗試找部分匹配（取前 100 個字元）
            position = content.find(text[:100]) if len(text) > 100 else -1

        if position == -1:
            return ""

        section = self.find_section_for_position(sections, position)
        return self.get_section_path(section)


# 全域實例
markdown_parser = MarkdownParser()
