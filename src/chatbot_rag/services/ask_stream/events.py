"""
SSE 事件發送工具函數。

提供：
- emit_node_event: 發送節點狀態事件
- emit_llm_meta_event: 發送 LLM usage/duration 事件
- truncate_tool_output_for_sse: 截斷過長的工具輸出
"""

from typing import Any, Callable, Optional


def truncate_tool_output_for_sse(output: str, max_length: int = 8000) -> str:
    """
    為 SSE 事件截斷工具輸出，保留開頭和結尾以便前端顯示。
    """
    if len(output) <= max_length:
        return output

    head_len = int(max_length * 0.6)
    tail_len = int(max_length * 0.3)
    head = output[:head_len]
    tail = output[-tail_len:]
    omitted_len = len(output) - head_len - tail_len

    return f"{head}\n\n... [省略 {omitted_len} 字元] ...\n\n{tail}"


def emit_node_event(
    writer: Callable[[dict], None],
    *,
    node: str,
    phase: str,
    payload: dict,
) -> None:
    """
    發送節點事件。

    參數：
        writer: SSE writer 函數
        node: 節點名稱
        phase: 階段名稱
        payload: 事件負載（包含 stage、channel 等）
    """
    base_event: dict[str, Any] = {
        "source": "ask_stream",
        "node": node,
        "phase": phase,
    }
    event = {**base_event, **payload}
    writer(event)


def emit_llm_meta_event(
    writer: Callable[[dict], None],
    *,
    node: str,
    phase: str,
    component: str,
    usage: Optional[dict[str, Any]],
) -> None:
    """將單一 LLM 呼叫的 usage/duration 轉換為 meta 事件。"""
    if not usage:
        return

    meta: dict[str, Any] = {"component": component}
    if usage:
        meta["usage"] = usage

    emit_node_event(
        writer,
        node=node,
        phase=phase,
        payload={
            "channel": "meta",
            "meta": meta,
        },
    )
