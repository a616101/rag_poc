"""
Langfuse trace 統一管理模組。

此模組提供統一的 trace context 管理，參考 chatbot_rag 的實作方式。
使用此模組可確保：
1. 整個 workflow 共用同一個 trace_id
2. 流程結束後可取得 trace_id 回傳前端
3. 支援 user feedback / custom scores API
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


@dataclass
class TraceContext:
    """Trace 執行結果，包含 trace_id 供後續 feedback 使用。"""

    trace_id: str
    handler: Any  # langfuse.langchain.CallbackHandler


def _is_langfuse_available() -> bool:
    """Check if Langfuse is properly configured."""
    try:
        from chatbot_graphrag.core.config import settings
        return bool(
            settings.langfuse_enabled
            and settings.langfuse_public_key
            and settings.langfuse_secret_key
        )
    except Exception:
        return False


@contextmanager
def create_trace_context(
    *,
    name: str = "graphrag-workflow",
    trace_id_seed: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
    input_data: Optional[dict] = None,
) -> Generator[Optional[TraceContext], None, None]:
    """
    建立統一的 Langfuse trace context。

    使用方式：
        with create_trace_context(
            trace_id_seed="request_12345",
            user_id="user-123",
            input_data={"question": q}
        ) as ctx:
            if ctx:
                result = graph.stream(state, config={"callbacks": [ctx.handler]})
                # ctx.trace_id 可回傳給前端

    參數：
        name: trace 名稱，預設為 "graphrag-workflow"
        trace_id_seed: 用於產生 deterministic trace_id 的種子（如 request_id）
        user_id: 用戶 ID，用於 Langfuse 用戶分析
        session_id: 會話 ID，用於聚合同一會話的 trace
        tags: 標籤列表，用於 Langfuse 過濾
        metadata: 額外元數據
        input_data: trace 輸入資料

    Yields：
        TraceContext: 包含 trace_id 和 handler 的上下文物件，若 Langfuse 不可用則為 None
    """
    if not _is_langfuse_available():
        logger.debug("Langfuse not configured, skipping trace context")
        yield None
        return

    try:
        from langfuse import get_client, propagate_attributes
        from langfuse.langchain import CallbackHandler

        langfuse = get_client()

        # 產生 trace_id
        if trace_id_seed:
            trace_id = langfuse.create_trace_id(seed=trace_id_seed)
        else:
            trace_id = langfuse.create_trace_id()

        # 建立 handler
        handler = CallbackHandler()

        with langfuse.start_as_current_observation(
            as_type="span",
            name=name,
            metadata=metadata,
            trace_context={"trace_id": trace_id},
        ):
            # 設定 trace 層級的 input 和 tags
            trace_updates: dict = {}
            if input_data is not None:
                trace_updates["input"] = input_data
            if tags:
                trace_updates["tags"] = tags
            if trace_updates:
                langfuse.update_current_trace(**trace_updates)

            with propagate_attributes(
                user_id=user_id,
                session_id=session_id,
            ):
                yield TraceContext(
                    trace_id=trace_id,
                    handler=handler,
                )

    except ImportError:
        logger.warning("Langfuse not installed, skipping trace context")
        yield None
    except Exception as e:
        logger.warning(f"Failed to create trace context: {e}")
        yield None


def update_trace_with_result(
    *,
    output: Optional[dict] = None,
    metadata: Optional[dict] = None,
    tags: Optional[list[str]] = None,
    scores: Optional[dict[str, float]] = None,
) -> None:
    """
    更新當前 trace 的結果資訊。

    應在 workflow 結束時呼叫，用於記錄 output、metadata 和 scores。

    參數：
        output: trace 輸出資料
        metadata: 額外元數據
        tags: 標籤列表
        scores: 評分字典 {name: value}
    """
    if not _is_langfuse_available():
        return

    try:
        from langfuse import get_client

        langfuse = get_client()

        # 更新 trace
        trace_updates: dict = {}
        if output is not None:
            trace_updates["output"] = output
        if metadata is not None:
            trace_updates["metadata"] = metadata
        if tags:
            trace_updates["tags"] = tags

        if trace_updates:
            langfuse.update_current_trace(**trace_updates)

        # 記錄 scores
        if scores:
            for name, value in scores.items():
                if value is not None:
                    langfuse.score_current_trace(
                        name=name,
                        value=float(value),
                    )

    except Exception as e:
        logger.warning(f"Failed to update trace: {e}")
