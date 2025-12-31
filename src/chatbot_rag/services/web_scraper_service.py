"""
網頁爬取服務模組

此模組使用 Crawl4AI 從指定 URL 遞迴爬取網頁內容並轉換為 RAG 文件格式。
Crawl4AI 支援 JavaScript 動態渲染，自動產出 LLM-ready 的 Markdown 格式。

主要功能：
- 使用 Crawl4AI 爬取網頁（支援 JS 渲染）
- 遞迴爬取同網域的頁面（可設定深度限制）
- 自動生成 YAML frontmatter 元資料
- 支援並發爬取提升效率
- 內容清理策略系統

使用方式：
    from chatbot_rag.services.web_scraper_service import web_scraper_service

    # 爬取單一 URL
    result = await web_scraper_service.scrape_url("https://example.com")

    # 遞迴爬取網站
    results = await web_scraper_service.crawl_website(
        "https://example.com",
        max_depth=3,
        max_pages=100
    )
"""

import asyncio
import hashlib
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse, urlunparse

from loguru import logger

from chatbot_rag.core.config import settings


# ============================================================================
# 清理策略系統 (Cleaning Strategy System)
# ============================================================================


@dataclass
class CleaningStrategy:
    """
    網頁內容清理策略

    定義如何清理特定網域或所有網域的網頁內容。
    包含正則表達式模式、精確匹配字串，以及各種過濾規則。

    屬性：
        name: 策略名稱（用於識別和日誌）
        noise_patterns: 正則表達式模式列表，匹配到的內容會被移除
        noise_exact: 精確匹配字串列表，完全相同的行會被跳過
        noise_prefixes: 前綴匹配列表，以這些字串開頭的行會被跳過
        noise_contains: 包含匹配列表，包含這些字串的行會被跳過
        min_line_length: 最小行長度（短於此長度的行會被跳過，除非是標題/列表/連結）
        skip_timestamp_lines: 是否跳過純時間戳行
        skip_copyright_lines: 是否跳過版權宣告行
    """
    name: str = "default"
    noise_patterns: list[str] = field(default_factory=list)
    noise_exact: list[str] = field(default_factory=list)
    noise_prefixes: list[str] = field(default_factory=list)
    noise_contains: list[str] = field(default_factory=list)
    min_line_length: int = 5
    skip_timestamp_lines: bool = True
    skip_copyright_lines: bool = True

    @classmethod
    def from_dict(cls, data: dict, name: str = "custom") -> "CleaningStrategy":
        """從字典建立 CleaningStrategy 實例"""
        return cls(
            name=data.get("name", name),
            noise_patterns=data.get("noise_patterns", []),
            noise_exact=data.get("noise_exact", []),
            noise_prefixes=data.get("noise_prefixes", []),
            noise_contains=data.get("noise_contains", []),
            min_line_length=data.get("min_line_length", 5),
            skip_timestamp_lines=data.get("skip_timestamp_lines", True),
            skip_copyright_lines=data.get("skip_copyright_lines", True),
        )

    def to_dict(self) -> dict:
        """將策略轉換為字典格式"""
        return {
            "name": self.name,
            "noise_patterns": self.noise_patterns,
            "noise_exact": self.noise_exact,
            "noise_prefixes": self.noise_prefixes,
            "noise_contains": self.noise_contains,
            "min_line_length": self.min_line_length,
            "skip_timestamp_lines": self.skip_timestamp_lines,
            "skip_copyright_lines": self.skip_copyright_lines,
        }

    def merge_with(self, other: "CleaningStrategy") -> "CleaningStrategy":
        """合併另一個策略（用於將網域策略與預設策略合併）"""
        return CleaningStrategy(
            name=self.name,
            noise_patterns=list(set(other.noise_patterns + self.noise_patterns)),
            noise_exact=list(set(other.noise_exact + self.noise_exact)),
            noise_prefixes=list(set(other.noise_prefixes + self.noise_prefixes)),
            noise_contains=list(set(other.noise_contains + self.noise_contains)),
            min_line_length=self.min_line_length,
            skip_timestamp_lines=self.skip_timestamp_lines,
            skip_copyright_lines=self.skip_copyright_lines,
        )


class CleaningStrategyRegistry:
    """
    清理策略註冊表

    管理預設策略和各網域特定的策略。支援動態註冊和查詢。
    """

    def __init__(self, auto_load_config: bool = True):
        """初始化註冊表，設定預設策略"""
        self._default_strategy = self._create_default_strategy()
        self._domain_strategies: dict[str, CleaningStrategy] = {}
        self._init_builtin_strategies()

        if auto_load_config:
            self._load_from_settings()

    def _create_default_strategy(self) -> CleaningStrategy:
        """建立預設清理策略（適用於所有網站）"""
        return CleaningStrategy(
            name="default",
            noise_patterns=[
                r'bread\w*',
                r'left\s*nav',
                r'right\s*nav',
                r'leftNav',
                r'rightNav',
                r'\bleft\b',
                r'\bright\b',
                r'\bnav\b',
                r'Copyright.*$',
                r'©.*$',
                r'All Rights Reserved.*$',
                r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}//\w+',
                r'Loading\.+',
                r'請稍等\.+',
            ],
            noise_exact=[
                '首頁',
                'Home',
                '更多',
                'more',
                '返回',
                'back',
                '跳至主要內容',
                'Loading',
            ],
            noise_prefixes=[],
            noise_contains=[],
            min_line_length=5,
            skip_timestamp_lines=True,
            skip_copyright_lines=True,
        )

    def _init_builtin_strategies(self):
        """初始化內建的網域策略"""
        # 屏東基督教醫院網站策略
        self.register_domain("www.ptch.org.tw", CleaningStrategy(
            name="ptch.org.tw",
            noise_patterns=[
                r'IOSAndroid',
                r'掛號APP',
                r'看診進度',
            ],
            noise_exact=[
                'IOSAndroid',
                '諮詢與預約',
                '看診進度',
                '掛號APP',
                '角色與努力',
                '其他綜合服務',
                '線上服務',
                'IOS',
                'Android',
            ],
            noise_prefixes=[
                'IOS[',
                'IOS [',
            ],
            noise_contains=[
                'google.com/store/apps',
                'apple.com/app',
            ],
        ))

    def register_domain(self, domain: str, strategy: CleaningStrategy):
        """註冊網域特定的清理策略"""
        domain = domain.lower().strip()
        if domain.startswith("http"):
            domain = urlparse(domain).netloc
        self._domain_strategies[domain] = strategy
        logger.debug(f"Registered cleaning strategy '{strategy.name}' for domain: {domain}")

    def get_strategy(self, url_or_domain: str, merge_with_default: bool = True) -> CleaningStrategy:
        """取得適用於指定 URL 或網域的清理策略"""
        if url_or_domain.startswith("http"):
            domain = urlparse(url_or_domain).netloc.lower()
        else:
            domain = url_or_domain.lower().strip()

        domain_strategy = self._domain_strategies.get(domain)

        if domain_strategy is None:
            if domain.startswith("www."):
                domain_strategy = self._domain_strategies.get(domain[4:])
            else:
                domain_strategy = self._domain_strategies.get(f"www.{domain}")

        if domain_strategy is None:
            return self._default_strategy

        if merge_with_default:
            return domain_strategy.merge_with(self._default_strategy)

        return domain_strategy

    @property
    def default_strategy(self) -> CleaningStrategy:
        """取得預設策略"""
        return self._default_strategy

    def list_domains(self) -> list[str]:
        """列出所有已註冊的網域"""
        return list(self._domain_strategies.keys())

    def get_all_strategies(self) -> dict[str, dict]:
        """取得所有策略（包含預設策略）"""
        result = {"_default": self._default_strategy.to_dict()}
        for domain, strategy in self._domain_strategies.items():
            result[domain] = strategy.to_dict()
        return result

    def register_domain_from_dict(self, domain: str, data: dict) -> CleaningStrategy:
        """從字典註冊網域策略"""
        strategy = CleaningStrategy.from_dict(data, name=domain)
        self.register_domain(domain, strategy)
        return strategy

    def unregister_domain(self, domain: str) -> bool:
        """移除網域策略"""
        domain = domain.lower().strip()
        if domain in self._domain_strategies:
            del self._domain_strategies[domain]
            logger.info(f"Unregistered cleaning strategy for domain: {domain}")
            return True
        return False

    def _load_from_settings(self):
        """從設定檔（環境變數）載入額外的清理策略"""
        import json

        strategies_json = settings.scraper_cleaning_strategies
        if not strategies_json or not strategies_json.strip():
            return

        try:
            strategies_data = json.loads(strategies_json)
            if not isinstance(strategies_data, dict):
                logger.warning("SCRAPER_CLEANING_STRATEGIES must be a JSON object")
                return

            for domain, strategy_data in strategies_data.items():
                if not isinstance(strategy_data, dict):
                    logger.warning(f"Strategy for domain '{domain}' must be a JSON object, skipping")
                    continue

                strategy = CleaningStrategy.from_dict(strategy_data, name=domain)
                self.register_domain(domain, strategy)
                logger.info(f"Loaded cleaning strategy for domain '{domain}' from settings")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse SCRAPER_CLEANING_STRATEGIES: {e}")


# 全域策略註冊表實例
cleaning_strategy_registry = CleaningStrategyRegistry()


class WebScraperService:
    """
    網頁爬取服務類別（使用 Crawl4AI）

    此類別使用 Crawl4AI 從網頁遞迴爬取內容並轉換為 RAG 系統可用的 Markdown 文件格式。
    Crawl4AI 支援 JavaScript 動態渲染，能處理 SPA 和動態載入的內容。
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        max_concurrent: int = 5,
    ):
        """
        初始化網頁爬取服務

        參數：
            output_dir: 爬取結果輸出目錄
            max_concurrent: 最大並發請求數，預設 5
        """
        project_root = Path(settings.default_docs_path).resolve().parent.parent

        self.output_dir = Path(output_dir or project_root / "rag_test_data/docs/網頁更新")
        self.max_concurrent = max_concurrent

        # 確保輸出目錄存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 爬取狀態
        self.visited_urls: set[str] = set()
        self.semaphore: Optional[asyncio.Semaphore] = None

    def normalize_url(self, url: str) -> str:
        """標準化 URL（移除 fragment、標準化路徑）"""
        parsed = urlparse(url)
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc.lower(),
            parsed.path.rstrip('/') or '/',
            parsed.params,
            parsed.query,
            ''
        ))
        return normalized

    def is_valid_url(self, url: str, base_domain: str) -> bool:
        """檢查 URL 是否有效且屬於同一網域"""
        try:
            parsed = urlparse(url)

            if parsed.scheme not in ['http', 'https']:
                return False

            if parsed.netloc.lower() != base_domain.lower():
                return False

            excluded_extensions = [
                '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                '.zip', '.rar', '.7z', '.tar', '.gz',
                '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico', '.webp',
                '.mp3', '.mp4', '.avi', '.mov', '.wmv',
                '.css', '.js', '.json', '.xml',
            ]
            path_lower = parsed.path.lower()
            for ext in excluded_extensions:
                if path_lower.endswith(ext):
                    return False

            return True
        except Exception:
            return False

    def extract_links_from_markdown(self, markdown: str, base_url: str) -> list[str]:
        """
        從 Markdown 內容中提取連結

        參數：
            markdown: Markdown 內容
            base_url: 基礎 URL（用於解析相對路徑）

        回傳：
            list[str]: 標準化後的連結列表
        """
        base_domain = urlparse(base_url).netloc
        links = []

        # 從 Markdown 連結提取: [text](url)
        md_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        for match in re.finditer(md_pattern, markdown):
            href = match.group(2).strip()

            if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:', 'data:')):
                continue

            try:
                if not href.startswith(('http://', 'https://')):
                    href = urljoin(base_url, href)

                normalized = self.normalize_url(href)

                if self.is_valid_url(normalized, base_domain):
                    links.append(normalized)
            except Exception:
                continue

        return list(set(links))

    def extract_links_from_html(self, html: str, base_url: str) -> list[str]:
        """
        從 HTML 內容中提取連結（用於遞迴爬取時發現新頁面）

        此方法專門用於遞迴爬取時的連結發現，從原始 HTML 中提取所有同網域連結，
        確保不會因為 LLM/fit 模式的內容過濾而遺漏頁面。

        參數：
            html: 原始 HTML 內容
            base_url: 基礎 URL（用於解析相對路徑）

        回傳：
            list[str]: 標準化後的連結列表
        """
        base_domain = urlparse(base_url).netloc
        links = []

        # 從 HTML href 屬性提取連結
        href_pattern = r'href=["\']([^"\']+)["\']'
        for match in re.finditer(href_pattern, html, re.IGNORECASE):
            href = match.group(1).strip()

            # 跳過無效連結
            if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:', 'data:')):
                continue

            try:
                # 解析相對路徑
                if not href.startswith(('http://', 'https://')):
                    href = urljoin(base_url, href)

                normalized = self.normalize_url(href)

                if self.is_valid_url(normalized, base_domain):
                    links.append(normalized)
            except Exception:
                continue

        return list(set(links))

    def _fix_broken_markdown(self, content: str) -> str:
        """
        輕量後備修復：處理 LLM 輸出中的少數格式問題

        主要依賴 LLM 提示詞產出正確格式，此方法只做最小化的後備處理。

        參數：
            content: LLM 輸出的 Markdown 內容

        回傳：
            str: 清理後的內容
        """
        # 1. 行尾的 -- 或 --- 轉為獨立分隔線（偶爾 LLM 會這樣輸出）
        content = re.sub(r'(\S)\s+-{2,3}\s*$', r'\1\n\n---', content, flags=re.MULTILINE)

        # 2. 修復被換行切斷的粗體（少見但可能發生）
        content = re.sub(r'\*\*([^*\n]+)\n+([^*\n]+)\*\*', r'**\1\2**', content)

        # 3. 修復被換行切斷的連結 URL
        content = re.sub(r'\[([^\]]+)\]\(([^)\s]+)\n+([^)\s]+)\)', r'[\1](\2\3)', content)

        return content

    def _fix_missing_newlines(self, content: str) -> str:
        """
        處理 LLM 輸出的換行問題

        優化後的 LLM 提示詞應產出正確格式，此方法只做輕量的後備處理。

        參數：
            content: LLM 輸出的 Markdown 內容

        回傳：
            str: 處理後的內容
        """
        # 輕量後備修復
        content = self._fix_broken_markdown(content)

        # 清理多餘的換行（超過 2 個連續換行變成 2 個）
        content = re.sub(r'\n{3,}', '\n\n', content)

        # 移除開頭的多餘換行
        content = content.lstrip('\n')

        return content

    def _clean_markdown(
        self,
        content: str,
        url: str = "",
        title: str = "",
        blacklist_exact: Optional[list[str]] = None,
        blacklist_patterns: Optional[list[str]] = None,
        group_images: bool = False,
    ) -> str:
        """
        輕量清理 Markdown 內容

        Crawl4AI 的 fit 模式和 LLM 模式已經能產出乾淨的 Markdown，
        這裡只做最基本的格式整理，避免過度清理導致內容遺失。

        重要：LLM 產出的內容已經是標準 Markdown 格式，不應該被過度處理。

        參數：
            content: 原始 Markdown 內容
            url: 來源 URL（保留參數相容性，目前未使用）
            title: 頁面標題（用於定位主要內容區塊）
            blacklist_exact: 精確匹配的黑名單字串列表（API 額外指定）
            blacklist_patterns: 正則表達式模式黑名單列表（API 額外指定）
            group_images: 是否根據圖片檔名分組（適用於頁籤式頁面）

        回傳：
            str: 清理後的內容
        """
        # 輕量後備修復（只處理真正的格式錯誤）
        content = self._fix_broken_markdown(content)

        # 基於 title 的內容擷取：移除主要內容區塊之前的導航雜訊
        if title:
            content = self._extract_main_content_by_title(content, title)

        # 套用內容黑名單過濾
        content = self._filter_blacklisted_content(content, blacklist_exact, blacklist_patterns)

        # 圖片分組（適用於頁籤式頁面如「樓層指引」）
        if group_images:
            content = self._group_images_by_section(content)

        # 移除行尾空白（但保留換行符和縮排）
        content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)

        # 最多保留兩個連續換行（一個空行）
        content = re.sub(r'\n{3,}', '\n\n', content)

        # 移除開頭和結尾的空白
        content = content.strip()

        return content

    def _group_images_by_section(self, content: str) -> str:
        """
        根據圖片檔名自動分組圖片到對應的標題下

        這個功能專門用於處理像「樓層指引」這類頁面，其中圖片檔名包含了
        分類資訊（如 路加B2.jpg、馬太大樓1F.jpg），可以自動將連續的圖片
        重新組織成帶標題的結構化內容。

        處理邏輯：
        1. 只處理「沒有標題」的連續圖片區塊
        2. 如果圖片前面已有 ### 標題，則不進行分組（保持原樣）
        3. 從連續的圖片區塊中提取圖片及其類別（從檔名）
        4. 將圖片按類別分組，並在每組前插入標題

        參數：
            content: 原始 Markdown 內容

        回傳：
            str: 重新組織後的 Markdown 內容
        """
        lines = content.split('\n')
        result_lines = []
        image_buffer = []  # 暫存連續的圖片行
        last_heading = None  # 記錄最後一個標題行

        # 定義圖片分類規則（從檔名提取類別）
        def extract_image_category(image_line: str) -> Optional[str]:
            """從圖片 URL 中提取類別名稱"""
            # 匹配 ![...](url)
            match = re.search(r'!\[[^\]]*\]\(([^)]+)\)', image_line)
            if not match:
                return None

            url = match.group(1)
            # URL 解碼
            from urllib.parse import unquote
            decoded_url = unquote(url)

            # 從檔名提取大樓名稱
            filename = decoded_url.split('/')[-1]

            # 常見的大樓名稱模式（用於樓層指引頁面）
            building_patterns = [
                (r'^各大樓分佈平面圖', '各大樓分佈平面圖'),
                (r'^路加', '路加大樓(門診)'),
                (r'^馬太大樓', '馬太大樓'),
                (r'^馬太', '馬太大樓'),
                (r'^馬可大樓', '馬可大樓'),
                (r'^馬可', '馬可大樓'),
                (r'^韓偉大樓', '韓偉大樓(急診)'),
                (r'^韓偉', '韓偉大樓(急診)'),
                (r'^約翰大樓', '約翰大樓'),
                (r'^約翰', '約翰大樓'),
                (r'^恩慈大樓', '恩慈大樓'),
                (r'^恩慈', '恩慈大樓'),
            ]

            for pattern, category in building_patterns:
                if re.match(pattern, filename, re.IGNORECASE):
                    return category

            return None

        def flush_image_buffer(buffer: list[str], has_heading_in_section: bool) -> list[str]:
            """
            處理圖片緩衝區

            參數：
                buffer: 圖片行列表
                has_heading_in_section: 這些圖片所在的區段是否有標題

            回傳：
                list[str]: 處理後的行
            """
            if not buffer:
                return []

            # 如果這個區段已有標題，不進行分組，直接返回原始圖片
            if has_heading_in_section:
                return buffer

            # 沒有標題，進行分組
            return self._process_image_buffer(buffer, extract_image_category)

        # 處理內容
        i = 0
        current_section_has_heading = False  # 追蹤當前區段是否有 ### 標題

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # 檢查是否是 ## 標題（新區段開始）
            is_h2 = stripped.startswith('## ') and not stripped.startswith('### ')

            # 檢查是否是 ### 標題
            is_h3 = stripped.startswith('###') and not stripped.startswith('####')

            # 檢查是否是圖片行
            is_image = re.match(r'^\s*!\[', line)

            if is_h2:
                # 新的 ## 區段開始，重置標題追蹤
                if image_buffer:
                    grouped_content = flush_image_buffer(image_buffer, current_section_has_heading)
                    result_lines.extend(grouped_content)
                    image_buffer = []
                current_section_has_heading = False
                result_lines.append(line)
            elif is_h3:
                # ### 標題，標記當前區段有標題
                if image_buffer:
                    grouped_content = flush_image_buffer(image_buffer, current_section_has_heading)
                    result_lines.extend(grouped_content)
                    image_buffer = []
                current_section_has_heading = True
                result_lines.append(line)
            elif is_image:
                image_buffer.append(line)
            else:
                # 其他行
                if image_buffer:
                    grouped_content = flush_image_buffer(image_buffer, current_section_has_heading)
                    result_lines.extend(grouped_content)
                    image_buffer = []
                result_lines.append(line)

            i += 1

        # 處理最後的圖片緩衝區
        if image_buffer:
            grouped_content = flush_image_buffer(image_buffer, current_section_has_heading)
            result_lines.extend(grouped_content)

        return '\n'.join(result_lines)

    def _process_image_buffer(
        self,
        image_lines: list[str],
        category_extractor: callable,
    ) -> list[str]:
        """
        處理連續的圖片行，按類別分組

        參數：
            image_lines: 圖片行列表
            category_extractor: 從圖片行提取類別的函數

        回傳：
            list[str]: 分組後的 Markdown 行
        """
        if not image_lines:
            return []

        # 按類別分組
        from collections import OrderedDict
        categories: OrderedDict[str, list[str]] = OrderedDict()
        uncategorized = []

        for line in image_lines:
            category = category_extractor(line)
            if category:
                if category not in categories:
                    categories[category] = []
                categories[category].append(line)
            else:
                uncategorized.append(line)

        # 如果沒有有效分類或只有一個分類，保持原樣
        if len(categories) <= 1 and not uncategorized:
            return image_lines if not categories else list(categories.values())[0]

        # 產生分組後的輸出
        result = []
        for category, images in categories.items():
            result.append(f'\n### {category}\n')
            result.extend(images)

        # 未分類的圖片放最後
        if uncategorized:
            result.append('\n### 其他圖片\n')
            result.extend(uncategorized)

        return result

    def _filter_blacklisted_content(
        self,
        content: str,
        extra_exact: Optional[list[str]] = None,
        extra_patterns: Optional[list[str]] = None,
    ) -> str:
        """
        過濾黑名單內容

        合併預設黑名單和 API 指定的額外黑名單，移除匹配的行。

        參數：
            content: 原始內容
            extra_exact: API 額外指定的精確匹配字串列表
            extra_patterns: API 額外指定的正則表達式列表

        回傳：
            str: 過濾後的內容
        """
        # 合併預設黑名單和額外黑名單
        exact_list = list(settings.scraper_content_blacklist_exact)
        if extra_exact:
            exact_list.extend(extra_exact)

        pattern_list = list(settings.scraper_content_blacklist_patterns)
        if extra_patterns:
            pattern_list.extend(extra_patterns)

        # 如果沒有任何黑名單，直接返回
        if not exact_list and not pattern_list:
            return content

        # 編譯正則表達式（忽略無效的模式）
        compiled_patterns = []
        for pattern in pattern_list:
            try:
                compiled_patterns.append(re.compile(pattern))
            except re.error as e:
                logger.warning(f"Invalid blacklist regex pattern '{pattern}': {e}")

        # 逐行過濾
        lines = content.split('\n')
        filtered_lines = []
        removed_count = 0

        for line in lines:
            stripped_line = line.strip()

            # 檢查精確匹配
            if stripped_line in exact_list:
                removed_count += 1
                logger.debug(f"Removed blacklisted line (exact): '{stripped_line}'")
                continue

            # 檢查正則匹配
            matched = False
            for pattern in compiled_patterns:
                if pattern.match(stripped_line):
                    matched = True
                    removed_count += 1
                    logger.debug(f"Removed blacklisted line (pattern): '{stripped_line}'")
                    break

            if not matched:
                filtered_lines.append(line)

        if removed_count > 0:
            logger.info(f"Content blacklist removed {removed_count} lines")

        return '\n'.join(filtered_lines)

    def _extract_main_content_by_title(self, content: str, title: str) -> str:
        """
        基於頁面標題定位並擷取主要內容區塊

        分析發現網頁結構通常是：
        1. 導航雜訊（麵包屑、側欄選單、線上服務連結等）
        2. 主要內容區塊（以 ## {title關鍵字} 開頭）
        3. 頁尾雜訊（版權、日期等）

        此方法從 title 中提取關鍵字，找到對應的 ## 標題，
        並移除該標題之前的導航雜訊。

        參數：
            content: 原始 Markdown 內容
            title: 頁面標題（如 "醫療費用訊息 - 屏基醫療財團法人屏東基督教醫院"）

        回傳：
            str: 擷取後的主要內容
        """
        if not title or not content:
            return content

        # 從 title 提取主題關鍵字
        title_keyword = self._extract_title_keyword(title)
        if not title_keyword:
            return content

        # 尋找 ## {title_keyword} 標題的位置
        # 使用正則表達式匹配 ##（可能有空格）後面接標題關鍵字
        pattern = rf'^##\s*{re.escape(title_keyword)}\s*$'
        match = re.search(pattern, content, re.MULTILINE)

        if match:
            main_start = match.start()

            # 檢查擷取的內容是否太短（可能誤判）
            main_content = content[main_start:]
            if len(main_content) < 100:
                logger.debug(f"Title-based extraction too short, keeping original content")
                return content

            # 計算移除了多少字元（用於日誌）
            removed_chars = main_start
            if removed_chars > 50:
                logger.debug(
                    f"Removed {removed_chars} chars of navigation noise before '## {title_keyword}'"
                )

            return main_content

        # 如果沒找到精確匹配，嘗試模糊匹配（標題可能有細微差異）
        # 使用標題關鍵字的前幾個字元進行匹配
        if len(title_keyword) >= 4:
            fuzzy_keyword = title_keyword[:min(8, len(title_keyword))]
            fuzzy_pattern = rf'^##\s*{re.escape(fuzzy_keyword)}'
            fuzzy_match = re.search(fuzzy_pattern, content, re.MULTILINE)

            if fuzzy_match:
                main_content = content[fuzzy_match.start():]
                if len(main_content) >= 100:
                    logger.debug(
                        f"Fuzzy title match: removed {fuzzy_match.start()} chars before '## {fuzzy_keyword}...'"
                    )
                    return main_content

        return content

    def _extract_title_keyword(self, title: str) -> Optional[str]:
        """
        從頁面標題中提取主題關鍵字

        處理多種標題格式：
        1. "醫療費用訊息 - 屏基醫療財團法人屏東基督教醫院" → "醫療費用訊息"
        2. "屏基醫療財團法人屏東基督教醫院" → None（首頁，無特定主題）
        3. "網站地圖 - 屏基醫療財團法人屏東基督教醫院" → "網站地圖"

        參數：
            title: 完整頁面標題

        回傳：
            Optional[str]: 主題關鍵字，若無法提取則返回 None
        """
        if not title:
            return None

        # 醫院名稱（用於識別和排除）
        hospital_name = "屏基醫療財團法人屏東基督教醫院"

        # 如果標題就是醫院名稱（首頁），不進行擷取
        if title.strip() == hospital_name:
            return None

        # 嘗試用 " - " 分隔
        if " - " in title:
            keyword = title.split(" - ")[0].strip()
            # 確保關鍵字不是醫院名稱本身
            if keyword and keyword != hospital_name and len(keyword) >= 2:
                return keyword

        # 嘗試用 "-" 分隔（無空格）
        if "-" in title:
            keyword = title.split("-")[0].strip()
            if keyword and keyword != hospital_name and len(keyword) >= 2:
                return keyword

        # 嘗試用 "|" 分隔（某些網站使用）
        if "|" in title:
            keyword = title.split("|")[0].strip()
            if keyword and keyword != hospital_name and len(keyword) >= 2:
                return keyword

        # 如果標題包含醫院名稱但有其他內容，嘗試移除醫院名稱
        if hospital_name in title:
            keyword = title.replace(hospital_name, "").strip(" -|")
            if keyword and len(keyword) >= 2:
                return keyword

        # 無法提取有效關鍵字
        return None

    def _generate_title_from_url(self, url: str) -> str:
        """從 URL 生成標題"""
        parsed = urlparse(url)
        path = parsed.path.strip('/')

        if path:
            last_part = path.split('/')[-1]
            if '.' in last_part:
                last_part = last_part.rsplit('.', 1)[0]
            title = last_part.replace('-', ' ').replace('_', ' ')
            return title.title() if title else parsed.netloc
        return parsed.netloc

    def generate_filename(self, url: str, title: str) -> str:
        """根據 URL 和標題生成檔案名稱"""
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split("/") if p]

        query_id = ""
        if parsed.query:
            for param in ['id', 'unit_id', 'page_id', 'article_id']:
                match = re.search(rf'{param}=(\w+)', parsed.query)
                if match:
                    query_id = f"_{param}{match.group(1)}"
                    break

        if path_parts:
            base_name = "_".join(path_parts[-2:]) if len(path_parts) >= 2 else path_parts[-1]
            base_name = base_name + query_id
        else:
            base_name = hashlib.md5(url.encode()).hexdigest()[:12]

        safe_name = re.sub(r'[\\/:*?"<>|#]', '_', base_name)
        safe_name = re.sub(r'_+', '_', safe_name)
        safe_name = safe_name.strip('_')[:100]

        return safe_name or "webpage"

    def save_as_markdown(self, content: str, metadata: dict, filename: str) -> Path:
        """將內容儲存為 Markdown 檔案"""
        frontmatter_lines = ["---"]
        for key, value in metadata.items():
            if isinstance(value, str) and (":" in value or "\n" in value or '"' in value):
                value = f'"{value}"'
            frontmatter_lines.append(f"{key}: {value}")
        frontmatter_lines.append("---")
        frontmatter_lines.append("")

        full_content = "\n".join(frontmatter_lines) + content

        file_path = self.output_dir / f"{filename}.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_content)

        logger.debug(f"Saved: {file_path}")
        return file_path

    async def scrape_url(
        self,
        url: str,
        extraction_mode: Optional[str] = None,
        llm_instruction: Optional[str] = None,
        content_blacklist_exact: Optional[list[str]] = None,
        content_blacklist_patterns: Optional[list[str]] = None,
        group_images: bool = False,
    ) -> dict:
        """
        使用 Crawl4AI 爬取單一 URL 並轉換為 Markdown 文件

        參數：
            url: 要爬取的網頁 URL
            extraction_mode: 提取模式 - raw/fit/llm（None = 使用全域設定）
            llm_instruction: 自訂 LLM 提取指令（僅 llm 模式使用）
            content_blacklist_exact: 額外的精確匹配黑名單（與預設黑名單合併）
            content_blacklist_patterns: 額外的正則表達式黑名單（與預設黑名單合併）
            group_images: 是否根據圖片檔名分組（適用於頁籤式頁面如「樓層指引」）

        回傳：
            dict: 爬取結果
        """
        from chatbot_rag.services.crawl4ai_client import crawl4ai_client

        mode = extraction_mode or settings.crawl4ai_extraction_mode
        logger.info(f"Scraping: {url} (extraction_mode: {mode})")

        try:
            result = await crawl4ai_client.crawl_url(
                url,
                extraction_mode=extraction_mode,
                llm_instruction=llm_instruction,
            )

            if not result.success:
                logger.warning(f"Crawl4AI failed for {url}: {result.error}")
                return {
                    "success": False,
                    "url": url,
                    "error": result.error,
                }

            markdown_content = result.markdown

            if not markdown_content or len(markdown_content) < 50:
                logger.warning(f"Crawl4AI returned insufficient content for {url}")
                return {
                    "success": False,
                    "url": url,
                    "error": "Content too short or empty",
                }

            # 應用輕量清理（只處理格式問題，保留換行）+ 黑名單過濾 + 圖片分組
            markdown_content = self._clean_markdown(
                markdown_content,
                url,
                blacklist_exact=content_blacklist_exact,
                blacklist_patterns=content_blacklist_patterns,
                group_images=group_images,
            )

            # 生成元資料
            parsed_url = urlparse(url)
            metadata = {
                "title": result.title or self._generate_title_from_url(url),
                "category": "網頁更新",
                "entry_type": "自動爬取",
                "source_url": url,
                "scraped_at": result.crawled_at,
                "domain": parsed_url.netloc,
            }

            # 儲存檔案
            filename = self.generate_filename(url, metadata.get("title", ""))
            file_path = self.save_as_markdown(markdown_content, metadata, filename)

            return {
                "success": True,
                "url": url,
                "file_path": str(file_path),
                "filename": filename,
                "title": metadata.get("title", ""),
                "content_length": len(markdown_content),
                "llm_extracted": result.llm_extracted,
            }

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return {
                "success": False,
                "url": url,
                "error": str(e),
            }

    async def crawl_website(
        self,
        start_url: str,
        max_depth: int = 3,
        max_pages: int = 100,
        url_pattern: Optional[str] = None,
        exclude_patterns: Optional[list[str]] = None,
        extraction_mode: Optional[str] = None,
        llm_instruction: Optional[str] = None,
        global_visited_urls: Optional[set[str]] = None,
        content_blacklist_exact: Optional[list[str]] = None,
        content_blacklist_patterns: Optional[list[str]] = None,
        group_images: bool = False,
    ) -> dict:
        """
        使用 Crawl4AI 遞迴爬取網站

        從起始 URL 開始，自動發現並爬取同網域的所有頁面。
        Crawl4AI 支援 JavaScript 渲染，能處理動態載入的內容。

        參數：
            start_url: 起始 URL
            max_depth: 最大爬取深度（預設 3）
            max_pages: 最大頁面數量（預設 100）
            url_pattern: URL 包含過濾正則表達式（只爬取符合的 URL）
            exclude_patterns: URL 排除模式列表，支援兩種匹配模式：
                - 包含匹配（預設）：字串存在於 URL 中即排除，如 "/user"
                - 精確匹配：以 "exact:" 開頭，URL 必須完全相符，如 "exact:https://example.com/"
            extraction_mode: 提取模式 - raw/fit/llm（None = 使用全域設定）
            llm_instruction: 自訂 LLM 提取指令（僅 llm 模式使用）
            global_visited_urls: 全域已訪問 URL 集合（用於多網址遞迴爬取時去重）
            content_blacklist_exact: 額外的精確匹配黑名單（與預設黑名單合併）
            content_blacklist_patterns: 額外的正則表達式黑名單（與預設黑名單合併）
            group_images: 是否根據圖片檔名分組（適用於頁籤式頁面如「樓層指引」）

        回傳：
            dict: 爬取結果摘要
        """
        # 儲存設定供 _crawl_single 使用
        self._current_extraction_mode = extraction_mode
        self._current_llm_instruction = llm_instruction
        self._current_blacklist_exact = content_blacklist_exact
        self._current_blacklist_patterns = content_blacklist_patterns
        self._current_group_images = group_images

        mode = extraction_mode or settings.crawl4ai_extraction_mode
        logger.info(f"Starting crawl from {start_url}, max_depth={max_depth}, max_pages={max_pages}, extraction_mode={mode}")

        # 初始化（如果有傳入全域集合，則使用它；否則建立新的）
        if global_visited_urls is not None:
            self.visited_urls = global_visited_urls
        else:
            self.visited_urls = set()
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

        base_domain = urlparse(start_url).netloc
        start_url = self.normalize_url(start_url)

        # BFS 爬取佇列: (url, depth)
        queue = deque([(start_url, 0)])
        results = []
        discovered_urls = {start_url}

        url_regex = re.compile(url_pattern) if url_pattern else None

        # 建立排除檢查函數
        def should_exclude(check_url: str) -> bool:
            """
            檢查 URL 是否應該被排除

            支援兩種匹配模式：
            1. 精確匹配：以 "exact:" 開頭，URL 必須完全相符
               例如："exact:https://www.ptch.org.tw/" 只排除首頁
            2. 包含匹配：預設模式，URL 包含該字串即排除
               例如："/user" 會排除所有包含 /user 的 URL
            """
            if not exclude_patterns:
                return False
            for pattern in exclude_patterns:
                if pattern.startswith("exact:"):
                    # 精確匹配模式
                    exact_url = pattern[6:]  # 移除 "exact:" 前綴
                    if check_url == exact_url:
                        logger.debug(f"Excluding URL (exact match '{exact_url}'): {check_url}")
                        return True
                else:
                    # 包含匹配模式（預設）
                    if pattern in check_url:
                        logger.debug(f"Excluding URL (contains '{pattern}'): {check_url}")
                        return True
            return False

        while queue and len(results) < max_pages:
            batch = []
            while queue and len(batch) < self.max_concurrent:
                url, depth = queue.popleft()

                if url in self.visited_urls:
                    continue

                if depth > max_depth:
                    continue

                # 檢查排除模式
                if should_exclude(url):
                    self.visited_urls.add(url)  # 標記為已訪問，避免重複檢查
                    continue

                if url_regex and not url_regex.search(url):
                    continue

                batch.append((url, depth))
                self.visited_urls.add(url)

            if not batch:
                continue

            tasks = [self._crawl_single(url, depth, base_domain) for url, depth in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for (url, depth), result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error crawling {url}: {result}")
                    results.append({
                        "success": False,
                        "url": url,
                        "error": str(result),
                    })
                    continue

                scrape_result, new_links = result

                if scrape_result["success"]:
                    results.append(scrape_result)
                    logger.info(f"[{len(results)}/{max_pages}] Scraped: {scrape_result.get('title', url)[:50]}")

                    if depth < max_depth:
                        for link in new_links:
                            # 在加入 queue 前先檢查排除模式
                            if link not in discovered_urls and link not in self.visited_urls:
                                if not should_exclude(link):
                                    discovered_urls.add(link)
                                    queue.append((link, depth + 1))
                else:
                    results.append(scrape_result)

                if len([r for r in results if r["success"]]) >= max_pages:
                    break

        success_count = sum(1 for r in results if r["success"])
        failed_count = len(results) - success_count

        logger.info(f"Crawl completed: {success_count} succeeded, {failed_count} failed, {len(discovered_urls)} discovered")

        return {
            "start_url": start_url,
            "total_discovered": len(discovered_urls),
            "total_crawled": len(results),
            "success": success_count,
            "failed": failed_count,
            "results": results,
        }

    async def _crawl_single(
        self,
        url: str,
        depth: int,
        base_domain: str,
    ) -> tuple[dict, list[str]]:
        """
        使用 Crawl4AI 爬取單一頁面並提取連結

        參數：
            url: 要爬取的 URL
            depth: 當前深度
            base_domain: 基礎網域

        回傳：
            tuple[dict, list[str]]: (爬取結果, 新發現的連結)
        """
        from chatbot_rag.services.crawl4ai_client import crawl4ai_client

        # 取得 crawl_website 設定的參數
        extraction_mode = getattr(self, "_current_extraction_mode", None)
        llm_instruction = getattr(self, "_current_llm_instruction", None)
        blacklist_exact = getattr(self, "_current_blacklist_exact", None)
        blacklist_patterns = getattr(self, "_current_blacklist_patterns", None)
        group_images = getattr(self, "_current_group_images", False)

        async with self.semaphore:
            try:
                result = await crawl4ai_client.crawl_url(
                    url,
                    extraction_mode=extraction_mode,
                    llm_instruction=llm_instruction,
                )

                if not result.success:
                    return {
                        "success": False,
                        "url": url,
                        "depth": depth,
                        "error": result.error,
                    }, []

                markdown_content = result.markdown

                if not markdown_content or len(markdown_content) < 50:
                    return {
                        "success": False,
                        "url": url,
                        "depth": depth,
                        "error": "Content too short",
                    }, []

                # 取得頁面標題（用於內容擷取）
                page_title = result.title or self._generate_title_from_url(url)

                # 應用清理策略（包含基於 title 的導航雜訊移除）+ 黑名單過濾 + 圖片分組
                markdown_content = self._clean_markdown(
                    markdown_content,
                    url,
                    page_title,
                    blacklist_exact=blacklist_exact,
                    blacklist_patterns=blacklist_patterns,
                    group_images=group_images,
                )

                # 從原始 HTML 提取連結（用於遞迴爬取）
                # 重要：使用 HTML 而非 Markdown，因為 LLM/fit 模式會過濾掉大量導航連結
                if result.html:
                    new_links = self.extract_links_from_html(result.html, url)
                else:
                    # 備用：如果沒有 HTML，則使用 Markdown
                    new_links = self.extract_links_from_markdown(result.markdown, url)

                # 生成元資料
                parsed_url = urlparse(url)
                metadata = {
                    "title": page_title,
                    "category": "網頁更新",
                    "entry_type": "自動爬取",
                    "source_url": url,
                    "scraped_at": result.crawled_at,
                    "domain": parsed_url.netloc,
                }

                # 儲存
                filename = self.generate_filename(url, metadata.get("title", ""))
                file_path = self.save_as_markdown(markdown_content, metadata, filename)

                return {
                    "success": True,
                    "url": url,
                    "depth": depth,
                    "file_path": str(file_path),
                    "filename": filename,
                    "title": metadata.get("title", ""),
                    "content_length": len(markdown_content),
                    "links_found": len(new_links),
                    "llm_extracted": result.llm_extracted,
                }, new_links

            except Exception as e:
                logger.error(f"Error crawling {url}: {e}")
                return {
                    "success": False,
                    "url": url,
                    "depth": depth,
                    "error": str(e),
                }, []


# 全域實例
web_scraper_service = WebScraperService()
