"""
文件 Repository - PostgreSQL CRUD 操作

提供文件資料存取的封裝層。
"""

from typing import Optional, Sequence
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from chatbot_graphrag.models.sqlalchemy import Doc


class DocumentRepository:
    """
    文件 Repository。

    封裝 Doc 模型的資料庫操作。

    Example:
        async with get_async_session() as session:
            repo = DocumentRepository(session)
            doc = await repo.get_by_id("doc_123")
    """

    def __init__(self, session: AsyncSession):
        """
        初始化 Repository。

        Args:
            session: SQLAlchemy AsyncSession
        """
        self.session = session

    async def get_by_id(self, doc_id: str) -> Optional[Doc]:
        """
        根據 ID 取得文件。

        Args:
            doc_id: 文件 ID

        Returns:
            Doc 實例或 None
        """
        result = await self.session.execute(
            select(Doc).where(Doc.id == doc_id, Doc.status != "deleted")
        )
        return result.scalar_one_or_none()

    async def get_by_content_hash(self, content_hash: str) -> Optional[Doc]:
        """
        根據 content_hash 取得文件。

        Args:
            content_hash: 內容 SHA-256 hash

        Returns:
            Doc 實例或 None
        """
        result = await self.session.execute(
            select(Doc).where(
                Doc.content_hash == content_hash,
                Doc.status != "deleted",
            )
        )
        return result.scalar_one_or_none()

    async def exists(self, doc_id: str) -> bool:
        """
        檢查文件是否存在。

        Args:
            doc_id: 文件 ID

        Returns:
            True 如果文件存在且未刪除
        """
        doc = await self.get_by_id(doc_id)
        return doc is not None

    async def get_content_hash(self, doc_id: str) -> Optional[str]:
        """
        取得文件的 content_hash。

        Args:
            doc_id: 文件 ID

        Returns:
            content_hash 或 None
        """
        result = await self.session.execute(
            select(Doc.content_hash).where(
                Doc.id == doc_id,
                Doc.status != "deleted",
            )
        )
        return result.scalar_one_or_none()

    async def create(self, doc: Doc) -> Doc:
        """
        建立新文件。

        Args:
            doc: Doc 實例

        Returns:
            建立的 Doc 實例
        """
        self.session.add(doc)
        await self.session.flush()
        return doc

    async def update_version(
        self,
        doc_id: str,
        new_hash: str,
    ) -> Optional[Doc]:
        """
        更新文件版本和 hash。

        Args:
            doc_id: 文件 ID
            new_hash: 新的 content_hash

        Returns:
            更新後的 Doc 實例或 None
        """
        await self.session.execute(
            update(Doc)
            .where(Doc.id == doc_id)
            .values(
                content_hash=new_hash,
                current_version=Doc.current_version + 1,
            )
        )
        return await self.get_by_id(doc_id)

    async def soft_delete(self, doc_id: str) -> bool:
        """
        軟刪除文件。

        Args:
            doc_id: 文件 ID

        Returns:
            True 如果刪除成功
        """
        result = await self.session.execute(
            update(Doc)
            .where(Doc.id == doc_id)
            .values(status="deleted")
        )
        return result.rowcount > 0

    async def list_by_type(
        self,
        doc_type: str,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Doc]:
        """
        根據類型列出文件。

        Args:
            doc_type: 文件類型
            limit: 最大返回數量
            offset: 起始偏移

        Returns:
            Doc 列表
        """
        result = await self.session.execute(
            select(Doc)
            .where(Doc.doc_type == doc_type, Doc.status != "deleted")
            .order_by(Doc.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def list_by_department(
        self,
        department: str,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Doc]:
        """
        根據部門列出文件。

        Args:
            department: 部門名稱
            limit: 最大返回數量
            offset: 起始偏移

        Returns:
            Doc 列表
        """
        result = await self.session.execute(
            select(Doc)
            .where(Doc.department == department, Doc.status != "deleted")
            .order_by(Doc.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()
