"""
ACL 節點

文件存取權限的存取控制列表檢查。

此節點驗證使用者是否有適當的權限存取請求的資源，
基於其 ACL 群組和租戶設定進行過濾。
"""

import logging
from typing import Any

from chatbot_graphrag.graph_workflow.types import FilterContext, GraphRAGState

logger = logging.getLogger(__name__)


async def acl_node(state: GraphRAGState) -> dict[str, Any]:
    """
    用於存取控制驗證的 ACL 節點。

    驗證使用者是否有適當的權限存取請求的資源，
    基於其 ACL 群組和租戶設定。

    Returns:
        更新後的狀態，包含 acl_denied 標誌
    """
    import time
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("acl", "START")

    start_time = time.time()
    acl_groups = state.get("acl_groups", [])
    tenant_id = state.get("tenant_id", "default")
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))

    logger.debug(f"ACL 檢查: tenant={tenant_id}, groups={acl_groups}")

    # 在生產系統中，這會：
    # 1. 驗證租戶存在且處於活躍狀態
    # 2. 驗證使用者的 ACL 群組有效
    # 3. 檢查任何租戶特定的限制
    # 4. 設置下游檢索的過濾器

    # 目前，對有效租戶允許所有存取
    if not tenant_id:
        logger.warning("ACL denied: no tenant_id")
        return {
            "acl_denied": True,
            "retrieval_path": retrieval_path + ["acl:denied:no_tenant"],
            "timing": {**timing, "acl_ms": (time.time() - start_time) * 1000},
        }

    # 如果未提供則使用預設 ACL 群組
    if not acl_groups:
        acl_groups = ["public"]
        logger.debug("使用預設 ACL 群組: public")

    # 為下游檢索節點建立過濾上下文（第 1 階段）
    filter_context = FilterContext(
        tenant_id=tenant_id,
        acl_groups=acl_groups,
        department=None,  # Can be extracted from user profile
        doc_types=None,  # Can be set based on query intent
    )

    logger.debug(f"ACL passed: tenant={tenant_id}, groups={acl_groups}")
    emit_status("acl", "DONE")
    return {
        "acl_denied": False,
        "acl_groups": acl_groups,
        "tenant_id": tenant_id,
        "filter_context": filter_context,
        "retrieval_path": retrieval_path + ["acl:passed"],
        "timing": {**timing, "acl_ms": (time.time() - start_time) * 1000},
    }
