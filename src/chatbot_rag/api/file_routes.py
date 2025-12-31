"""
檔案下載 API 路由模組。

此模組提供將專案根目錄下 `files/` 目錄中的檔案，透過 HTTP 下載的功能。

設計目標：
- 讓 `files/` 目錄中的所有檔案，都可以透過 URL 下載，例如：
  - http://localhost:8000/files/1開放式課程掛置申請表範本1140417.xlsx
  - http://localhost:8000/files/課程掛置測驗題範本1060701.csv
- 提供列出可下載檔案的端點，方便前端動態產生下載連結清單。

安全考量：
- 僅允許存取專案根目錄下的 `files/` 目錄
- 防止路徑跳脫（path traversal），例如：`../../etc/passwd`
"""

from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from chatbot_rag.core.logging import get_logger

# 建立路由器實例，所有端點都會掛載在 /files 底下
router = APIRouter(prefix="/files", tags=["files"])

# ---------------------------------------------------------------------------
# 檔案根目錄設定
# ---------------------------------------------------------------------------
# 目前此檔案位置為：src/chatbot_rag/api/file_routes.py
# parents[0] = api
# parents[1] = chatbot_rag
# parents[2] = src
# parents[3] = 專案根目錄（包含 files/ 資料夾）
PROJECT_ROOT = Path(__file__).resolve().parents[3]
FILES_DIR = PROJECT_ROOT / "files"

logger = get_logger()


def _ensure_within_files_dir(target: Path) -> None:
    """
    確保目標路徑位於 FILES_DIR 目錄之內，避免路徑跳脫攻擊。

    Args:
        target: 要檢查的目標檔案完整路徑

    Raises:
        HTTPException: 若目標路徑不在 FILES_DIR 之內，則拋出 400 錯誤
    """

    try:
        # Python 3.9+ 可使用 is_relative_to 來檢查路徑是否位於特定目錄下
        if not target.resolve().is_relative_to(FILES_DIR.resolve()):
            raise HTTPException(status_code=400, detail="Invalid file path")
    except AttributeError:
        # 若執行環境較舊沒有 is_relative_to，改用手動判斷
        resolved_files = FILES_DIR.resolve()
        resolved_target = target.resolve()
        if str(resolved_target).startswith(str(resolved_files)):
            return
        raise HTTPException(status_code=400, detail="Invalid file path")


@router.get("/", response_model=List[str])
async def list_files() -> List[str]:
    """
    列出 `files/` 目錄下所有可下載的檔案名稱。

    Returns:
        List[str]: 檔案名稱列表（不包含路徑）
    """
    if not FILES_DIR.exists():
        logger.warning(f"`files` 目錄不存在：{FILES_DIR}")
        return []

    if not FILES_DIR.is_dir():
        logger.error(f"`files` 路徑不是資料夾：{FILES_DIR}")
        raise HTTPException(status_code=500, detail="Files directory is not a folder")

    files: List[str] = [
        p.name for p in FILES_DIR.iterdir() if p.is_file()
    ]

    logger.info(f"列出 files 目錄檔案，共 {len(files)} 筆")
    return files


@router.get("/{filename}")
async def download_file(filename: str) -> FileResponse:
    """
    下載指定檔案。

    路徑格式：
        GET /files/{filename}

    範例：
        - /files/1開放式課程掛置申請表範本1140417.xlsx
        - /files/課程掛置測驗題範本1060701.csv

    Args:
        filename: 檔案名稱（僅限 `files/` 目錄下的檔案）

    Returns:
        FileResponse: 將檔案以附件形式回傳，Browser 可直接下載。
    """
    if not FILES_DIR.exists():
        logger.error(f"`files` 目錄不存在：{FILES_DIR}")
        raise HTTPException(status_code=500, detail="Files directory not found")

    file_path = FILES_DIR / filename

    # 防止路徑跳脫攻擊
    _ensure_within_files_dir(file_path)

    if not file_path.is_file():
        logger.warning(f"要求下載不存在的檔案：{file_path}")
        raise HTTPException(status_code=404, detail="File not found")

    logger.info(f"提供檔案下載：{file_path}")

    # 使用 application/octet-stream，讓瀏覽器自動下載
    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=filename,
    )


