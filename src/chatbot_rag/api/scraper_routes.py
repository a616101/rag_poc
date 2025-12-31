"""
網頁爬取 API 路由。

提供統一的爬取端點，支援：
- 單一網址爬取
- 多個網址批次爬取
- 遞迴爬取（自動發現同網域連結）

所有端點都在 /api/v1/scraper 路徑下。
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from loguru import logger

from chatbot_rag.services.web_scraper_service import web_scraper_service


router = APIRouter(prefix="/api/v1/scraper", tags=["Scraper"])


# ==============================================================================
# Pydantic Models
# ==============================================================================


class ContentBlacklist(BaseModel):
    """
    內容黑名單設定

    用於移除爬取內容中的雜訊文字。
    API 指定的黑名單會與預設黑名單合併使用。
    """
    exact: Optional[list[str]] = None  # 精確匹配的行（完全相同才移除）
    patterns: Optional[list[str]] = None  # 正則表達式模式（匹配的行會被移除）


class ScrapeRequest(BaseModel):
    """
    爬取請求

    支援四種使用方式：
    1. 單一網址：只填 url
    2. 多個網址批次爬取：填 urls 陣列
    3. 單一網址遞迴爬取：填 url + recursive=True
    4. 多個網址遞迴爬取：填 urls + recursive=True（每個網址都會遞迴爬取）
    """
    url: Optional[str] = None  # 單一網址或遞迴爬取的起始網址
    urls: Optional[list[str]] = None  # 多個網址批次爬取（或多個遞迴起點）
    recursive: bool = False  # 是否遞迴爬取
    max_depth: int = 3  # 遞迴爬取最大深度
    max_pages: int = 100  # 遞迴爬取最大頁面數
    url_pattern: Optional[str] = None  # URL 包含過濾正則表達式（只爬取符合的 URL）
    exclude_patterns: Optional[list[str]] = None  # URL 排除模式：預設包含匹配，"exact:URL" 精確匹配
    extraction_mode: Optional[str] = None  # raw/fit/llm，None = 使用全域設定
    llm_instruction: Optional[str] = None  # 自訂 LLM 提取指令（僅 llm 模式）
    content_blacklist: Optional[ContentBlacklist] = None  # 額外的內容黑名單（與預設黑名單合併）
    group_images: bool = False  # 是否根據圖片檔名分組（適用於頁籤式頁面如「樓層指引」）

    @field_validator('url')
    @classmethod
    def validate_url_format(cls, v: Optional[str]) -> Optional[str]:
        """驗證 URL 必須以 http:// 或 https:// 開頭"""
        if v is None:
            return v
        v = v.strip()
        if not v.startswith(('http://', 'https://')):
            raise ValueError(f"URL 必須以 http:// 或 https:// 開頭，收到: '{v}'")
        return v

    @field_validator('urls')
    @classmethod
    def validate_urls_format(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        """驗證 URLs 列表中每個 URL 格式"""
        if v is None:
            return v
        validated = []
        for url in v:
            url = url.strip()
            if not url.startswith(('http://', 'https://')):
                raise ValueError(f"URL 必須以 http:// 或 https:// 開頭，收到: '{url}'")
            validated.append(url)
        return validated

    @field_validator('extraction_mode')
    @classmethod
    def validate_extraction_mode(cls, v: Optional[str]) -> Optional[str]:
        """驗證 extraction_mode 必須是 raw/fit/llm/balanced 之一"""
        if v is not None and v not in ('raw', 'fit', 'llm', 'balanced'):
            raise ValueError(f"extraction_mode 必須是 raw、fit、llm 或 balanced，收到: '{v}'")
        return v

    def model_post_init(self, __context) -> None:
        """驗證至少要有 url 或 urls"""
        if not self.url and not self.urls:
            raise ValueError("必須提供 url 或 urls 其中之一")
        if self.url and self.urls:
            raise ValueError("不能同時提供 url 和 urls，請擇一使用")
        # max_pages 在多網址遞迴模式下是每個起始網址的上限
        if self.urls and self.recursive and self.max_pages > 50:
            # 警告：多網址遞迴可能產生大量頁面
            pass


class ScrapeResultItem(BaseModel):
    """單一爬取結果"""
    success: bool
    url: str
    title: Optional[str] = None
    filename: Optional[str] = None
    file_path: Optional[str] = None
    content_length: Optional[int] = None
    depth: Optional[int] = None
    links_found: Optional[int] = None
    llm_extracted: bool = False
    error: Optional[str] = None


class ScrapeResponse(BaseModel):
    """爬取回應"""
    success: bool
    mode: str  # single, batch, recursive, batch_recursive
    total: int
    succeeded: int
    failed: int
    extraction_mode: str
    results: list[ScrapeResultItem]


# ==============================================================================
# 爬取 API
# ==============================================================================


@router.post("/scrape", response_model=ScrapeResponse)
async def scrape(request: ScrapeRequest):
    """
    統一爬取端點。

    支援三種使用方式：

    **1. 單一網址爬取**
    ```json
    {
        "url": "https://example.com/page"
    }
    ```

    **2. 多個網址批次爬取**
    ```json
    {
        "urls": [
            "https://example.com/page1",
            "https://example.com/page2"
        ]
    }
    ```

    **3. 單一網址遞迴爬取（自動發現同網域連結）**
    ```json
    {
        "url": "https://example.com",
        "recursive": true,
        "max_depth": 3,
        "max_pages": 100
    }
    ```

    **4. 多個網址遞迴爬取（每個網址都會遞迴爬取）**
    ```json
    {
        "urls": [
            "https://example.com/section1",
            "https://example.com/section2"
        ],
        "recursive": true,
        "max_depth": 2,
        "max_pages": 50
    }
    ```

    **通用參數：**
    - `extraction_mode`: 提取模式 - raw/fit/llm（預設使用全域設定）
    - `llm_instruction`: 自訂 LLM 提取指令（僅 llm 模式）
    - `url_pattern`: URL 過濾正則表達式（僅遞迴模式）

    Returns:
        爬取結果，包含所有爬取頁面的詳細資訊
    """
    extraction_mode = request.extraction_mode or "fit"

    # 判斷爬取模式
    if request.urls and request.recursive:
        # 多個網址遞迴爬取
        mode = "batch_recursive"
        logger.info(
            f"[Scraper] 多網址遞迴爬取 {len(request.urls)} 個起始網址, "
            f"max_depth={request.max_depth}, max_pages={request.max_pages}, "
            f"extraction_mode={extraction_mode}"
        )
        results = await _scrape_multiple_recursive(
            urls=request.urls,
            max_depth=request.max_depth,
            max_pages=request.max_pages,
            url_pattern=request.url_pattern,
            exclude_patterns=request.exclude_patterns,
            extraction_mode=request.extraction_mode,
            llm_instruction=request.llm_instruction,
            content_blacklist=request.content_blacklist,
            group_images=request.group_images,
        )
    elif request.urls:
        # 批次爬取多個網址（不遞迴）
        mode = "batch"
        logger.info(f"[Scraper] 批次爬取 {len(request.urls)} 個網址, extraction_mode={extraction_mode}")
        results = await _scrape_multiple_urls(
            urls=request.urls,
            extraction_mode=request.extraction_mode,
            llm_instruction=request.llm_instruction,
            content_blacklist=request.content_blacklist,
            group_images=request.group_images,
        )
    elif request.recursive:
        # 單一網址遞迴爬取
        mode = "recursive"
        logger.info(
            f"[Scraper] 遞迴爬取: {request.url}, "
            f"max_depth={request.max_depth}, max_pages={request.max_pages}, "
            f"extraction_mode={extraction_mode}"
        )
        results = await _scrape_recursive(
            url=request.url,
            max_depth=request.max_depth,
            max_pages=request.max_pages,
            url_pattern=request.url_pattern,
            exclude_patterns=request.exclude_patterns,
            extraction_mode=request.extraction_mode,
            llm_instruction=request.llm_instruction,
            content_blacklist=request.content_blacklist,
            group_images=request.group_images,
        )
    else:
        # 單一網址爬取
        mode = "single"
        logger.info(f"[Scraper] 爬取單一網址: {request.url}, extraction_mode={extraction_mode}")
        results = await _scrape_single_url(
            url=request.url,
            extraction_mode=request.extraction_mode,
            llm_instruction=request.llm_instruction,
            content_blacklist=request.content_blacklist,
            group_images=request.group_images,
        )

    # 統計結果
    succeeded = sum(1 for r in results if r.success)
    failed = len(results) - succeeded

    return ScrapeResponse(
        success=failed == 0,
        mode=mode,
        total=len(results),
        succeeded=succeeded,
        failed=failed,
        extraction_mode=extraction_mode,
        results=results,
    )


# ==============================================================================
# 內部爬取函數
# ==============================================================================


async def _scrape_single_url(
    url: str,
    extraction_mode: Optional[str],
    llm_instruction: Optional[str],
    content_blacklist: Optional[ContentBlacklist] = None,
    group_images: bool = False,
) -> list[ScrapeResultItem]:
    """爬取單一網址"""
    try:
        result = await web_scraper_service.scrape_url(
            url=url,
            extraction_mode=extraction_mode,
            llm_instruction=llm_instruction,
            content_blacklist_exact=content_blacklist.exact if content_blacklist else None,
            content_blacklist_patterns=content_blacklist.patterns if content_blacklist else None,
            group_images=group_images,
        )

        return [ScrapeResultItem(
            success=result["success"],
            url=url,
            title=result.get("title"),
            filename=result.get("filename"),
            file_path=result.get("file_path"),
            content_length=result.get("content_length"),
            llm_extracted=result.get("llm_extracted", False),
            error=result.get("error"),
        )]
    except Exception as e:
        logger.error(f"[Scraper] 爬取失敗 {url}: {e}")
        return [ScrapeResultItem(
            success=False,
            url=url,
            error=str(e),
        )]


async def _scrape_multiple_urls(
    urls: list[str],
    extraction_mode: Optional[str],
    llm_instruction: Optional[str],
    content_blacklist: Optional[ContentBlacklist] = None,
    group_images: bool = False,
) -> list[ScrapeResultItem]:
    """批次爬取多個網址（去重）"""
    results = []

    # 去重：標準化 URL 後過濾重複
    seen_urls: set[str] = set()
    unique_urls = []
    for url in urls:
        normalized = web_scraper_service.normalize_url(url)
        if normalized not in seen_urls:
            unique_urls.append(url)
            seen_urls.add(normalized)
        else:
            logger.info(f"[Scraper] 跳過重複網址: {url}")

    for url in unique_urls:
        try:
            result = await web_scraper_service.scrape_url(
                url=url,
                extraction_mode=extraction_mode,
                llm_instruction=llm_instruction,
                content_blacklist_exact=content_blacklist.exact if content_blacklist else None,
                content_blacklist_patterns=content_blacklist.patterns if content_blacklist else None,
                group_images=group_images,
            )

            results.append(ScrapeResultItem(
                success=result["success"],
                url=url,
                title=result.get("title"),
                filename=result.get("filename"),
                file_path=result.get("file_path"),
                content_length=result.get("content_length"),
                llm_extracted=result.get("llm_extracted", False),
                error=result.get("error"),
            ))
        except Exception as e:
            logger.error(f"[Scraper] 爬取失敗 {url}: {e}")
            results.append(ScrapeResultItem(
                success=False,
                url=url,
                error=str(e),
            ))

    return results


async def _scrape_multiple_recursive(
    urls: list[str],
    max_depth: int,
    max_pages: int,
    url_pattern: Optional[str],
    exclude_patterns: Optional[list[str]],
    extraction_mode: Optional[str],
    llm_instruction: Optional[str],
    content_blacklist: Optional[ContentBlacklist] = None,
    group_images: bool = False,
) -> list[ScrapeResultItem]:
    """多個網址遞迴爬取（每個網址都作為起始點進行遞迴，全域去重）"""
    all_results = []

    # 全域已訪問 URL 集合，跨所有起始網址共享
    # 注意：不要在這裡預先加入起始網址，讓 crawl_website 內部處理
    global_visited_urls: set[str] = set()

    # 先對起始網址去重（只去除輸入列表中的重複）
    seen_start_urls: set[str] = set()
    unique_start_urls = []
    for url in urls:
        normalized = web_scraper_service.normalize_url(url)
        if normalized not in seen_start_urls:
            unique_start_urls.append(url)
            seen_start_urls.add(normalized)
        else:
            logger.info(f"[Scraper] 跳過重複的起始網址: {url}")

    for url in unique_start_urls:
        # 檢查這個起始網址是否已經在之前的遞迴中被爬取過
        normalized = web_scraper_service.normalize_url(url)
        if normalized in global_visited_urls:
            logger.info(f"[Scraper] 跳過已爬取的起始網址: {url}")
            continue
        logger.info(f"[Scraper] 開始遞迴爬取起始網址: {url} (已訪問 {len(global_visited_urls)} 個網址)")
        try:
            crawl_result = await web_scraper_service.crawl_website(
                start_url=url,
                max_depth=max_depth,
                max_pages=max_pages,
                url_pattern=url_pattern,
                exclude_patterns=exclude_patterns,
                extraction_mode=extraction_mode,
                llm_instruction=llm_instruction,
                global_visited_urls=global_visited_urls,
                content_blacklist_exact=content_blacklist.exact if content_blacklist else None,
                content_blacklist_patterns=content_blacklist.patterns if content_blacklist else None,
                group_images=group_images,
            )

            for result in crawl_result.get("results", []):
                all_results.append(ScrapeResultItem(
                    success=result["success"],
                    url=result.get("url", ""),
                    title=result.get("title"),
                    filename=result.get("filename"),
                    file_path=result.get("file_path"),
                    content_length=result.get("content_length"),
                    depth=result.get("depth"),
                    links_found=result.get("links_found"),
                    llm_extracted=result.get("llm_extracted", False),
                    error=result.get("error"),
                ))

            logger.info(
                f"[Scraper] 起始網址 {url} 遞迴爬取完成: "
                f"{crawl_result.get('success', 0)} 成功, {crawl_result.get('failed', 0)} 失敗"
            )
        except Exception as e:
            logger.error(f"[Scraper] 遞迴爬取失敗 {url}: {e}")
            all_results.append(ScrapeResultItem(
                success=False,
                url=url,
                error=str(e),
            ))

    logger.info(f"[Scraper] 多網址遞迴爬取完成: 共訪問 {len(global_visited_urls)} 個不重複網址")
    return all_results


async def _scrape_recursive(
    url: str,
    max_depth: int,
    max_pages: int,
    url_pattern: Optional[str],
    exclude_patterns: Optional[list[str]],
    extraction_mode: Optional[str],
    llm_instruction: Optional[str],
    content_blacklist: Optional[ContentBlacklist] = None,
    group_images: bool = False,
) -> list[ScrapeResultItem]:
    """單一網址遞迴爬取網站"""
    try:
        crawl_result = await web_scraper_service.crawl_website(
            start_url=url,
            max_depth=max_depth,
            max_pages=max_pages,
            url_pattern=url_pattern,
            exclude_patterns=exclude_patterns,
            extraction_mode=extraction_mode,
            llm_instruction=llm_instruction,
            content_blacklist_exact=content_blacklist.exact if content_blacklist else None,
            content_blacklist_patterns=content_blacklist.patterns if content_blacklist else None,
            group_images=group_images,
        )

        results = []
        for result in crawl_result.get("results", []):
            results.append(ScrapeResultItem(
                success=result["success"],
                url=result.get("url", ""),
                title=result.get("title"),
                filename=result.get("filename"),
                file_path=result.get("file_path"),
                content_length=result.get("content_length"),
                depth=result.get("depth"),
                links_found=result.get("links_found"),
                llm_extracted=result.get("llm_extracted", False),
                error=result.get("error"),
            ))

        return results
    except Exception as e:
        logger.error(f"[Scraper] 遞迴爬取失敗: {e}")
        return [ScrapeResultItem(
            success=False,
            url=url,
            error=str(e),
        )]
