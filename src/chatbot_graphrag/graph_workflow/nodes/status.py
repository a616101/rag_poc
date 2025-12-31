"""
狀態事件輔助函數

為 LangGraph 工作流程節點提供統一的狀態事件發送。
用於通知前端關於處理階段的資訊。
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def emit_status(node: str, stage: str) -> None:
    """
    向串流寫入器發送狀態事件。

    即使不在串流上下文中，此函數也可以安全呼叫。
    狀態事件由前端用於顯示處理進度。

    Args:
        node: 節點名稱（例如 "guard"、"retrieval"、"rerank"）
        stage: 節點內的階段（例如 "START"、"PROCESSING"、"DONE"）

    範例：
        emit_status("guard", "START")
        # ... 執行工作 ...
        emit_status("guard", "DONE")
    """
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
        if writer:
            writer({
                "channel": "status",
                "node": node,
                "stage": stage,
            })
    except Exception as e:
        # 如果不在串流上下文中則靜默忽略
        logger.debug(f"Status emit skipped (not in stream context): {e}")
