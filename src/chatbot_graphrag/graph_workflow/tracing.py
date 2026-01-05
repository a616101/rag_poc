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


def get_callbacks_from_config(config: dict | None = None) -> list:
    """
    從 LangGraph config 提取 Langfuse callbacks。

    Args:
        config: LangGraph 傳入的 config 字典

    Returns:
        callbacks 列表，若無則返回空列表
    """
    if not config:
        return []
    return config.get("callbacks", [])


@contextmanager
def traced_span(
    name: str,
    *,
    input_data: dict | None = None,
) -> Generator[Optional[Any], None, None]:
    """
    建立 Langfuse span 的 context manager。

    用於追蹤非節點的操作（如 Ragas 評估、外部 API 呼叫等）。

    使用方式：
        with traced_span("ragas_evaluation", input_data={"question": q}):
            result = await evaluator.evaluate(sample)

    Args:
        name: span 名稱
        input_data: 輸入資料

    Yields:
        observation 物件（可用於更新 output），若 Langfuse 不可用則為 None
    """
    if not _is_langfuse_available():
        yield None
        return

    try:
        from langfuse import get_client

        langfuse = get_client()

        # 使用 as observation 捕獲觀察物件，讓呼叫者可以更新 output
        with langfuse.start_as_current_observation(
            as_type="span",
            name=name,
            input=input_data,
        ) as observation:
            yield observation

    except ImportError:
        logger.debug("Langfuse not available for span tracing")
        yield None
    except Exception as e:
        logger.warning(f"Traced span {name} error: {e}")
        yield None


def _truncate_for_trace(val: Any, max_str_len: int = 500, max_list_len: int = 5, max_depth: int = 2) -> Any:
    """
    截斷大型資料結構以避免 Langfuse 卡住。

    Args:
        val: 要截斷的值
        max_str_len: 字串最大長度
        max_list_len: 列表最大項目數
        max_depth: 最大遞迴深度（防止深層嵌套）

    Returns:
        截斷後的值
    """
    if max_depth <= 0:
        if isinstance(val, (dict, list)):
            return f"[truncated: {type(val).__name__}]"
        return val

    if val is None:
        return None

    if isinstance(val, str):
        if len(val) > max_str_len:
            return val[:max_str_len] + "..."
        return val

    if isinstance(val, (int, float, bool)):
        return val

    if isinstance(val, list):
        if len(val) > max_list_len:
            # 只保留前幾項，並添加摘要
            truncated = [
                _truncate_for_trace(item, max_str_len, max_list_len, max_depth - 1)
                for item in val[:max_list_len]
            ]
            return {"_truncated": True, "_total": len(val), "_sample": truncated}
        return [
            _truncate_for_trace(item, max_str_len, max_list_len, max_depth - 1)
            for item in val
        ]

    if isinstance(val, dict):
        # 對於字典，只保留摘要資訊
        result = {}
        for k, v in val.items():
            if isinstance(v, list) and len(v) > max_list_len:
                # 大型列表只記錄數量
                result[k] = f"[{len(v)} items]"
            elif isinstance(v, dict) and len(v) > 10:
                # 大型字典只記錄 key 數量
                result[k] = f"{{dict with {len(v)} keys}}"
            else:
                result[k] = _truncate_for_trace(v, max_str_len, max_list_len, max_depth - 1)
        return result

    # 其他類型轉為字串並截斷
    str_val = str(val)
    if len(str_val) > max_str_len:
        return str_val[:max_str_len] + "..."
    return str_val


def traced_node(
    node_name: str,
    *,
    input_keys: list[str] | None = None,
    output_keys: list[str] | None = None,
):
    """
    為節點添加 Langfuse span 追蹤的裝飾器。

    使用方式：
        @traced_node("guard", input_keys=["question"], output_keys=["guard_blocked"])
        async def guard_node(state: GraphRAGState, config: dict | None = None):
            ...

    Args:
        node_name: span 名稱（顯示在 Langfuse 中）
        input_keys: 要記錄的 state input keys
        output_keys: 要記錄的 result output keys

    Returns:
        裝飾後的異步函數
    """
    from functools import wraps

    def decorator(func):
        @wraps(func)
        async def wrapper(state, config: dict | None = None):
            # 若 Langfuse 不可用，直接執行原函數
            if not _is_langfuse_available():
                return await func(state, config)

            try:
                from langfuse import get_client

                langfuse = get_client()

                # 準備 input 資料（使用截斷函數）
                _input_keys = input_keys or ["question"]
                input_data = {}
                for k in _input_keys:
                    val = state.get(k)
                    if val is not None:
                        input_data[k] = _truncate_for_trace(val)

                # 開啟 span 並執行節點
                # 使用 as observation 捕獲觀察物件以便更新 output
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name=node_name,
                    input=input_data,
                ) as observation:
                    result = await func(state, config)

                    # 記錄 output 資料（使用截斷函數）
                    if result and output_keys:
                        output_data = {}
                        for k in output_keys:
                            if k in result:
                                output_data[k] = _truncate_for_trace(result[k])
                        if output_data:
                            # 使用 observation.update() 而非 langfuse.update_current_observation()
                            observation.update(output=output_data)

                    return result

            except ImportError:
                logger.debug("Langfuse not available for tracing")
                return await func(state, config)
            except Exception as e:
                logger.warning(f"Traced node {node_name} error: {e}")
                # 即使追蹤失敗，仍執行原函數
                return await func(state, config)

        return wrapper

    return decorator
