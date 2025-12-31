"""
Alembic 遷移 CLI 封裝

提供便捷的命令列介面來執行資料庫遷移操作。
"""
import subprocess
import sys
from pathlib import Path


def get_alembic_ini_path() -> Path:
    """
    取得 alembic.ini 路徑。

    Returns:
        alembic.ini 的絕對路徑
    """
    return Path(__file__).parent.parent / "alembic.ini"


def run_alembic(*args: str) -> int:
    """
    執行 alembic 命令。

    Args:
        *args: 傳遞給 alembic 的命令列參數

    Returns:
        命令的退出碼
    """
    alembic_ini = get_alembic_ini_path()
    cmd = ["alembic", "-c", str(alembic_ini), *args]
    return subprocess.call(cmd)


def upgrade(revision: str = "head") -> int:
    """
    升級資料庫到指定版本。

    Args:
        revision: 目標版本，預設為 "head"（最新）

    Returns:
        命令的退出碼
    """
    return run_alembic("upgrade", revision)


def downgrade(revision: str = "-1") -> int:
    """
    降級資料庫。

    Args:
        revision: 目標版本，預設為 "-1"（上一個版本）

    Returns:
        命令的退出碼
    """
    return run_alembic("downgrade", revision)


def revision(message: str, autogenerate: bool = True) -> int:
    """
    建立新的遷移腳本。

    Args:
        message: 遷移描述訊息
        autogenerate: 是否自動生成遷移內容

    Returns:
        命令的退出碼
    """
    args = ["revision", "-m", message]
    if autogenerate:
        args.append("--autogenerate")
    return run_alembic(*args)


def current() -> int:
    """
    顯示當前資料庫版本。

    Returns:
        命令的退出碼
    """
    return run_alembic("current")


def history() -> int:
    """
    顯示遷移歷史。

    Returns:
        命令的退出碼
    """
    return run_alembic("history")


def stamp(revision: str = "head") -> int:
    """
    標記資料庫版本（不執行遷移）。

    用於將現有資料庫標記為已完成某個遷移版本。

    Args:
        revision: 目標版本，預設為 "head"

    Returns:
        命令的退出碼
    """
    return run_alembic("stamp", revision)


def main() -> int:
    """
    CLI 入口點。

    直接傳遞命令列參數給 alembic。
    """
    args = sys.argv[1:] if len(sys.argv) > 1 else ["--help"]
    return run_alembic(*args)


if __name__ == "__main__":
    sys.exit(main())
