"""
Crawl4AI SDK 客戶端服務

此模組使用 Crawl4AI Python SDK 進行網頁爬取，
支援 JavaScript 動態渲染並返回 LLM-ready 的 Markdown 格式內容。

Crawl4AI SDK 特點：
- 直接使用 AsyncWebCrawler，無需額外的 Docker 服務
- 支援 JavaScript 動態渲染（透過 Playwright）
- 自動產出乾淨的 Markdown
- 支援並發爬取
- 內建多種內容過濾策略（fit、LLM 等）

使用方式：
    from chatbot_rag.services.crawl4ai_client import crawl4ai_client

    # 爬取單一頁面
    result = await crawl4ai_client.crawl_url("https://example.com")

    # 批次爬取多個頁面
    results = await crawl4ai_client.crawl_urls(["https://a.com", "https://b.com"])
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

from loguru import logger

from chatbot_rag.core.config import settings


@dataclass
class CrawlResult:
    """
    爬取結果資料類別

    屬性：
        url: 來源 URL
        success: 是否成功
        markdown: Markdown 格式內容
        html: 原始 HTML（可選）
        title: 頁面標題
        metadata: 額外的元資料
        error: 錯誤訊息（若失敗）
        crawled_at: 爬取時間
        llm_extracted: 是否使用了 LLM 提取
        llm_content: LLM 提取的內容（若有）
    """
    url: str
    success: bool
    markdown: str = ""
    html: str = ""
    title: str = ""
    metadata: dict = None
    error: str = ""
    crawled_at: str = ""
    llm_extracted: bool = False
    llm_content: str = ""

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.crawled_at:
            self.crawled_at = datetime.now().isoformat()


class Crawl4AIClient:
    """
    Crawl4AI SDK 客戶端

    使用 crawl4ai 的 AsyncWebCrawler 進行網頁爬取，
    支援多種內容提取模式：raw、fit、llm。
    """

    def __init__(
        self,
        headless: Optional[bool] = None,
        verbose: Optional[bool] = None,
        timeout: Optional[float] = None,
    ):
        """
        初始化 Crawl4AI 客戶端

        參數：
            headless: 是否使用無頭模式（預設從設定讀取）
            verbose: 是否啟用詳細日誌（預設從設定讀取）
            timeout: 請求逾時時間（秒）
        """
        self.headless = headless if headless is not None else settings.crawl4ai_headless
        self.verbose = verbose if verbose is not None else settings.crawl4ai_verbose
        self.timeout = timeout or settings.crawl4ai_timeout
        self._crawler = None
        self._lock = asyncio.Lock()

    async def _get_crawler(self):
        """
        取得或建立 AsyncWebCrawler 實例

        使用單例模式管理 crawler 實例，避免重複建立瀏覽器。
        若 Playwright 瀏覽器尚未安裝，會自動嘗試安裝。
        """
        if self._crawler is None:
            async with self._lock:
                if self._crawler is None:
                    from crawl4ai import AsyncWebCrawler, BrowserConfig

                    browser_config = BrowserConfig(
                        headless=self.headless,
                        verbose=self.verbose,
                    )

                    try:
                        self._crawler = AsyncWebCrawler(config=browser_config)
                        await self._crawler.start()
                        logger.info("Crawl4AI AsyncWebCrawler initialized")
                    except Exception as e:
                        error_str = str(e)
                        # 檢查是否是 Playwright 瀏覽器未安裝的錯誤
                        if "playwright install" in error_str.lower() or "executable doesn't exist" in error_str.lower():
                            logger.warning("Playwright browsers not installed, attempting to install...")
                            await self._install_playwright_browsers()
                            # 重新嘗試建立 crawler
                            self._crawler = AsyncWebCrawler(config=browser_config)
                            await self._crawler.start()
                            logger.info("Crawl4AI AsyncWebCrawler initialized after browser installation")
                        else:
                            raise

        return self._crawler

    async def _install_playwright_browsers(self):
        """
        安裝 Playwright 瀏覽器

        在 Docker 環境或首次使用時自動安裝所需的瀏覽器。
        """
        import subprocess
        import sys

        logger.info("Installing Playwright browsers...")

        try:
            # 使用 playwright install chromium 只安裝 chromium
            result = subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 分鐘超時
            )

            if result.returncode == 0:
                logger.info("Playwright chromium browser installed successfully")
            else:
                logger.error(f"Failed to install Playwright browsers: {result.stderr}")
                raise RuntimeError(f"Playwright installation failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Playwright installation timed out")
        except FileNotFoundError:
            # 如果 playwright 命令不存在，嘗試使用 crawl4ai 的安裝腳本
            logger.warning("playwright command not found, trying crawl4ai-setup...")
            try:
                result = subprocess.run(
                    ["crawl4ai-setup"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    logger.info("Crawl4AI setup completed successfully")
                else:
                    raise RuntimeError(f"crawl4ai-setup failed: {result.stderr}")
            except FileNotFoundError:
                raise RuntimeError(
                    "Neither 'playwright' nor 'crawl4ai-setup' command found. "
                    "Please run 'playwright install chromium' or 'crawl4ai-setup' manually."
                )

    async def close(self):
        """關閉 AsyncWebCrawler"""
        if self._crawler is not None:
            async with self._lock:
                if self._crawler is not None:
                    await self._crawler.close()
                    self._crawler = None
                    logger.info("Crawl4AI AsyncWebCrawler closed")

    async def health_check(self) -> bool:
        """
        檢查 Crawl4AI SDK 是否可用

        回傳：
            bool: SDK 是否可用
        """
        try:
            # 嘗試 import 並建立 crawler
            from crawl4ai import AsyncWebCrawler
            return True
        except Exception as e:
            logger.warning(f"Crawl4AI SDK health check failed: {e}")
            return False

    async def crawl_url(
        self,
        url: str,
        wait_for: Optional[str] = None,
        js_code: Optional[str] = None,
        css_selector: Optional[str] = None,
        excluded_tags: Optional[list[str]] = None,
        screenshot: bool = False,
        extra_params: Optional[dict] = None,
        extraction_mode: Optional[str] = None,
        llm_instruction: Optional[str] = None,
    ) -> CrawlResult:
        """
        爬取單一 URL

        參數：
            url: 要爬取的 URL
            wait_for: 等待特定元素出現（CSS selector）
            js_code: 執行的 JavaScript 代碼
            css_selector: 只提取特定區塊的內容
            excluded_tags: 要排除的 HTML 標籤
            screenshot: 是否擷取截圖
            extra_params: 額外的參數（目前未使用）
            extraction_mode: 提取模式 - raw/fit/llm（None 則使用全域設定）
            llm_instruction: LLM 提取指令（僅 llm 模式使用）

        回傳：
            CrawlResult: 爬取結果

        注意：
            LLM 模式採用兩階段處理：
            1. 先用 PruningContentFilter 取得乾淨的 markdown（避免 LLMContentFilter 的格式問題）
            2. 再用自訂 LLM 呼叫來整理 markdown 內容
        """
        from crawl4ai import CrawlerRunConfig, CacheMode
        from crawl4ai import DefaultMarkdownGenerator, PruningContentFilter

        # 決定提取模式
        mode = extraction_mode or settings.crawl4ai_extraction_mode
        if mode not in ("raw", "fit", "llm", "balanced"):
            logger.warning(f"Invalid extraction_mode '{mode}', falling back to 'fit'")
            mode = "fit"

        logger.debug(f"Crawling {url} with extraction_mode={mode}")

        try:
            crawler = await self._get_crawler()

            # 建立基本配置
            config_params = {
                "cache_mode": CacheMode.BYPASS,
                "page_timeout": int(self.timeout * 1000),  # 轉換為毫秒
                "verbose": self.verbose,
            }

            # 添加可選參數
            if wait_for:
                config_params["wait_for"] = wait_for
            if js_code:
                config_params["js_code"] = js_code
            if css_selector:
                config_params["css_selector"] = css_selector
            if excluded_tags:
                config_params["excluded_tags"] = excluded_tags
            if screenshot:
                config_params["screenshot"] = True

            # 根據模式設定 Markdown 生成器
            if mode == "fit":
                # fit 模式：使用標準門檻過濾
                config_params["markdown_generator"] = DefaultMarkdownGenerator(
                    content_filter=PruningContentFilter(
                        threshold=settings.crawl4ai_fit_threshold,
                        threshold_type="fixed",
                        min_word_threshold=0,
                    )
                )
            elif mode == "llm":
                # llm 模式：使用較低門檻，保留更多內容讓 LLM 判斷
                config_params["markdown_generator"] = DefaultMarkdownGenerator(
                    content_filter=PruningContentFilter(
                        threshold=settings.crawl4ai_llm_threshold,
                        threshold_type="fixed",
                        min_word_threshold=0,
                    )
                )
            elif mode == "balanced":
                # balanced 模式：更低門檻，保留更多內容（包括圖片、連結）
                config_params["markdown_generator"] = DefaultMarkdownGenerator(
                    content_filter=PruningContentFilter(
                        threshold=settings.crawl4ai_balanced_threshold,
                        threshold_type="fixed",
                        min_word_threshold=0,
                    )
                )
            # raw 模式：不設定 content_filter，取得完整原始 markdown

            run_config = CrawlerRunConfig(**config_params)

            # 執行爬取
            result = await crawler.arun(url=url, config=run_config)

            if not result.success:
                error_msg = result.error_message or "Crawl failed"
                logger.warning(f"Crawl4AI failed for {url}: {error_msg}")
                return CrawlResult(
                    url=url,
                    success=False,
                    error=error_msg,
                )

            # 提取 Markdown 內容
            markdown_content = ""
            llm_extracted = False

            if result.markdown:
                # result.markdown 可能是 StringCompatibleMarkdown 物件
                if mode == "raw":
                    # raw 模式使用完整的原始 markdown
                    markdown_content = str(result.markdown)
                elif mode == "balanced":
                    # balanced 模式：使用較低門檻的 fit_markdown，保留更多內容
                    markdown_content = (
                        result.markdown.fit_markdown
                        if hasattr(result.markdown, 'fit_markdown') and result.markdown.fit_markdown
                        else str(result.markdown)
                    )
                else:
                    # fit 和 llm 模式使用 fit_markdown（經過 PruningContentFilter 處理）
                    # llm 模式使用較低門檻，保留更多內容
                    markdown_content = (
                        result.markdown.fit_markdown
                        if hasattr(result.markdown, 'fit_markdown') and result.markdown.fit_markdown
                        else str(result.markdown)
                    )

            # LLM 模式：使用自訂 LLM 呼叫來整理 markdown
            if mode == "llm" and markdown_content:
                instruction = llm_instruction or settings.crawl4ai_llm_instruction
                processed_content = await self._process_with_llm(markdown_content, instruction, url)
                if processed_content:
                    markdown_content = processed_content
                    llm_extracted = True
                else:
                    logger.warning(f"LLM processing failed for {url}, using fit_markdown instead")

            # 記錄提取結果的詳細資訊
            logger.debug(
                f"Extraction result for {url}: "
                f"has_markdown={result.markdown is not None}, "
                f"has_fit_markdown={hasattr(result.markdown, 'fit_markdown') and bool(result.markdown.fit_markdown) if result.markdown else False}, "
                f"content_length={len(markdown_content)}"
            )

            if not markdown_content:
                return CrawlResult(
                    url=url,
                    success=False,
                    error="Empty markdown returned",
                )

            # 提取標題
            title = ""
            if result.metadata and isinstance(result.metadata, dict):
                title = result.metadata.get("title", "")

            # 從 URL 生成標題（若無）
            if not title:
                parsed = urlparse(url)
                title = parsed.path.split("/")[-1] or parsed.netloc

            logger.info(f"Crawl completed: {url} ({len(markdown_content)} chars, mode={mode})")

            return CrawlResult(
                url=url,
                success=True,
                markdown=markdown_content,
                html=result.html or "",
                title=title,
                metadata=result.metadata if isinstance(result.metadata, dict) else {},
                llm_extracted=llm_extracted,
            )

        except asyncio.TimeoutError:
            error_msg = f"Crawl4AI timeout for URL: {url}"
            logger.warning(error_msg)
            return CrawlResult(
                url=url,
                success=False,
                error=error_msg,
            )
        except Exception as e:
            error_msg = f"Crawl4AI error for {url}: {e}"
            logger.error(error_msg)
            return CrawlResult(
                url=url,
                success=False,
                error=error_msg,
            )

    async def _process_with_llm(
        self,
        markdown_content: str,
        instruction: str,
        url: str,
    ) -> Optional[str]:
        """
        使用 LLM 處理 markdown 內容

        這是 llm 模式的第二階段處理，接收已經過 PruningContentFilter
        處理的乾淨 markdown，然後用 LLM 進一步整理。

        參數：
            markdown_content: 經過 fit 處理的 markdown 內容
            instruction: LLM 處理指令
            url: 來源 URL（用於日誌）

        回傳：
            Optional[str]: 處理後的 markdown 內容，失敗時返回 None
        """
        import httpx

        try:
            # 建立 prompt - 避免在 prompt 中使用可能被誤解為內容的範例文字
            default_instruction = """從網頁原始內容中提取有價值的主要資訊，輸出乾淨的 Markdown。

移除：導航元素、頁首頁尾、側邊欄、社群按鈕、廣告、重複內容、無意義符號、裝飾性文字。

保留：標題、正文、列表資訊、表格、有效連結、聯絡方式、步驟說明。

格式：使用 Markdown 標題層級，列表項目各自一行，段落間空行分隔，連結用 [文字](網址) 格式。

只輸出整理後的內容，不加任何說明或評論。"""

            system_prompt = instruction if instruction else default_instruction
            user_prompt = f"{markdown_content}"

            # 呼叫 LLM API
            api_base = settings.openai_api_base.rstrip('/')
            api_key = settings.openai_api_key
            model = settings.crawl4ai_llm_provider or settings.chat_model

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": settings.contextual_chunking_temperature,
                        "max_tokens": 8000,
                    },
                )

                if response.status_code != 200:
                    logger.error(f"LLM API error: {response.status_code} - {response.text}")
                    return None

                result = response.json()
                processed_content = result["choices"][0]["message"]["content"]

                # 確保內容有換行符
                if processed_content and '\n' not in processed_content:
                    logger.warning("LLM output missing newlines, applying fix...")
                    processed_content = self._fix_missing_newlines(processed_content)

                logger.info(
                    f"LLM processing completed for {url}: "
                    f"input={len(markdown_content)} chars, output={len(processed_content)} chars"
                )
                return processed_content

        except Exception as e:
            logger.error(f"LLM processing error for {url}: {e}")
            return None

    def _enrich_with_media_links(self, fit_content: str, raw_content: str) -> str:
        """
        從 raw 內容中補充 fit 內容缺失的圖片和下載連結

        balanced 模式專用：fit_markdown 可能會過濾掉圖片、PDF 下載連結等，
        此方法從 raw markdown 中提取這些資源並補充到 fit 結果中。

        參數：
            fit_content: 經過 PruningContentFilter 處理的內容
            raw_content: 原始完整的 markdown

        回傳：
            str: 補充後的 markdown 內容
        """
        import re

        if not raw_content or not fit_content:
            return fit_content or raw_content

        # 收集 fit 內容中已有的連結
        existing_links = set(re.findall(r'\[([^\]]*)\]\(([^)]+)\)', fit_content))
        existing_urls = {url for _, url in existing_links}

        # 從 raw 內容提取圖片（![alt](url)）
        raw_images = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', raw_content)

        # 從 raw 內容提取下載連結（常見檔案類型）
        download_extensions = (
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.zip', '.rar', '.7z', '.csv', '.odt', '.ods', '.odp'
        )
        raw_links = re.findall(r'\[([^\]]*)\]\(([^)]+)\)', raw_content)
        download_links = [
            (text, url) for text, url in raw_links
            if any(url.lower().endswith(ext) for ext in download_extensions)
            and url not in existing_urls
        ]

        # 收集缺失的圖片
        missing_images = [
            (alt, url) for alt, url in raw_images
            if url not in existing_urls
        ]

        # 如果沒有缺失內容，直接返回
        if not missing_images and not download_links:
            return fit_content

        # 建立補充區塊
        additions = []

        if missing_images:
            additions.append("\n\n---\n\n**相關圖片：**\n")
            for alt, url in missing_images[:20]:  # 限制最多 20 張圖片
                display_alt = alt if alt else "圖片"
                additions.append(f"- ![{display_alt}]({url})\n")

        if download_links:
            additions.append("\n\n**相關下載：**\n")
            for text, url in download_links[:20]:  # 限制最多 20 個下載
                display_text = text if text else url.split('/')[-1]
                additions.append(f"- [{display_text}]({url})\n")

        if additions:
            logger.info(
                f"Balanced mode enriched: +{len(missing_images)} images, "
                f"+{len(download_links)} downloads"
            )
            return fit_content + ''.join(additions)

        return fit_content

    def _fix_missing_newlines(self, content: str) -> str:
        """
        修復缺少換行符的 Markdown 內容

        當 LLM 返回的內容沒有換行符時，嘗試在適當的位置插入換行。
        這通常發生在某些 LLM 輸出格式異常時。

        參數：
            content: 原始 Markdown 內容

        回傳：
            str: 修復後的內容
        """
        import re

        # 在標題前添加換行（## 或 ### 等）
        content = re.sub(r'(?<!^)(?<!\n)(#{1,6}\s)', r'\n\n\1', content)

        # 在分隔線前後添加換行
        content = re.sub(r'(?<!\n)(---+)(?!\n)', r'\n\n\1\n\n', content)

        # 在列表項目前添加換行（- 開頭）
        content = re.sub(r'(?<!\n)(?<!^)(- \*\*)', r'\n\1', content)
        content = re.sub(r'(?<!\n)(?<!^)(- [A-Z\u4e00-\u9fff])', r'\n\1', content)

        # 在粗體段落標題後添加換行（**標題** 後面緊跟內容）
        content = re.sub(r'(\*\*[^*]+\*\*)(?!\n)(?=\S)', r'\1\n', content)

        # 清理多餘的連續換行
        content = re.sub(r'\n{3,}', '\n\n', content)

        logger.info(f"Fixed missing newlines, new length: {len(content)}")
        return content

    async def crawl_urls(
        self,
        urls: list[str],
        max_concurrent: int = 5,
        extraction_mode: Optional[str] = None,
        llm_instruction: Optional[str] = None,
        **kwargs,
    ) -> list[CrawlResult]:
        """
        批次爬取多個 URL

        參數：
            urls: URL 列表
            max_concurrent: 最大並發數
            extraction_mode: 提取模式 - raw/fit/llm
            llm_instruction: LLM 提取指令（僅 llm 模式使用）
            **kwargs: 傳遞給 crawl_url 的參數

        回傳：
            list[CrawlResult]: 爬取結果列表
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def crawl_with_semaphore(url: str) -> CrawlResult:
            async with semaphore:
                return await self.crawl_url(
                    url,
                    extraction_mode=extraction_mode,
                    llm_instruction=llm_instruction,
                    **kwargs
                )

        tasks = [crawl_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 處理例外
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(CrawlResult(
                    url=urls[i],
                    success=False,
                    error=str(result),
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def crawl_website(
        self,
        start_url: str,
        max_pages: int = 100,
        max_depth: int = 3,
        same_domain_only: bool = True,
        extraction_mode: Optional[str] = None,
        llm_instruction: Optional[str] = None,
        **kwargs,
    ) -> list[CrawlResult]:
        """
        遞迴爬取網站

        參數：
            start_url: 起始 URL
            max_pages: 最大頁面數
            max_depth: 最大深度
            same_domain_only: 是否只爬同網域
            extraction_mode: 提取模式 - raw/fit/llm
            llm_instruction: LLM 提取指令（僅 llm 模式使用）
            **kwargs: 傳遞給 crawl_url 的參數

        回傳：
            list[CrawlResult]: 爬取結果列表
        """
        parsed_start = urlparse(start_url)
        base_domain = parsed_start.netloc

        visited: set[str] = set()
        results: list[CrawlResult] = []
        queue: list[tuple[str, int]] = [(start_url, 0)]  # (url, depth)

        while queue and len(results) < max_pages:
            url, depth = queue.pop(0)

            # 標準化 URL
            if url in visited:
                continue
            visited.add(url)

            # 爬取頁面
            result = await self.crawl_url(
                url,
                extraction_mode=extraction_mode,
                llm_instruction=llm_instruction,
                **kwargs
            )
            results.append(result)

            logger.info(
                f"Crawled [{len(results)}/{max_pages}] depth={depth}: "
                f"{url[:60]}... {'✓' if result.success else '✗'}"
            )

            # 如果成功且未達最大深度，解析連結
            if result.success and depth < max_depth:
                links = self._extract_links(result.markdown, result.html, base_domain)

                for link in links:
                    if link not in visited:
                        # 檢查網域
                        if same_domain_only:
                            parsed_link = urlparse(link)
                            if parsed_link.netloc != base_domain:
                                continue
                        queue.append((link, depth + 1))

        return results

    def _extract_links(
        self,
        markdown: str,
        html: str,
        base_domain: str,
    ) -> list[str]:
        """
        從內容中提取連結

        參數：
            markdown: Markdown 內容
            html: HTML 內容
            base_domain: 基礎網域

        回傳：
            list[str]: 連結列表
        """
        import re

        links = []

        # 從 Markdown 連結提取: [text](url)
        md_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        for match in re.finditer(md_pattern, markdown):
            url = match.group(2)
            if url.startswith(('http://', 'https://')):
                links.append(url)

        # 從 HTML 提取 (備用)
        if not links and html:
            href_pattern = r'href=["\']([^"\']+)["\']'
            for match in re.finditer(href_pattern, html):
                url = match.group(1)
                if url.startswith(('http://', 'https://')):
                    links.append(url)

        # 去重
        return list(set(links))


# 全域實例
crawl4ai_client = Crawl4AIClient()
