"""
GraphRAG Ask Service - 統一的問答服務層

此模組提供統一的問答服務，整合 Langfuse tracing：
- run_ask: 非串流問答（用於 /ask 端點）
- run_ask_stream: 串流問答（用於 /ask/stream 端點）

遵循 chatbot_rag 的 tracing 模式，確保整個 workflow 都被正確追蹤。
"""

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any, Optional

from chatbot_graphrag.graph_workflow import (
    build_graphrag_workflow,
    create_initial_state,
)
from chatbot_graphrag.graph_workflow.tracing import (
    create_trace_context,
    update_trace_with_result,
)

logger = logging.getLogger(__name__)


async def run_ask(
    question: str,
    *,
    acl_groups: Optional[list[str]] = None,
    tenant_id: str = "default",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    執行非串流問答。

    Args:
        question: 使用者問題
        acl_groups: 存取控制群組
        tenant_id: 租戶 ID
        user_id: 用戶 ID（用於 Langfuse）
        session_id: 會話 ID（用於 Langfuse）
        conversation_id: 對話 ID

    Returns:
        包含 answer, sources, confidence, trace_id 等的結果字典
    """
    acl_groups = acl_groups or ["public"]
    request_id = uuid.uuid4().hex[:8]

    with create_trace_context(
        name="graphrag-ask",
        trace_id_seed=request_id,
        user_id=user_id,
        session_id=session_id or conversation_id,
        tags=["graphrag", "non-stream"],
        metadata={"tenant_id": tenant_id},
        input_data={"question": question, "tenant_id": tenant_id},
    ) as ctx:
        workflow = build_graphrag_workflow()
        initial_state = create_initial_state(
            question=question,
            acl_groups=acl_groups,
            tenant_id=tenant_id,
        )

        trace_id = ctx.trace_id if ctx else request_id
        initial_state["trace_id"] = trace_id

        # 執行工作流程
        config: dict[str, Any] = {"configurable": {"thread_id": trace_id}}
        if ctx and ctx.handler:
            config["callbacks"] = [ctx.handler]

        result = await workflow.ainvoke(initial_state, config)

        # 更新 trace
        if ctx:
            update_trace_with_result(
                output={
                    "answer": result.get("final_answer", ""),
                    "answer_length": len(result.get("final_answer", "")),
                },
                metadata={
                    "retrieval_path": result.get("retrieval_path", []),
                    "timing": result.get("timing", {}),
                },
                scores={
                    "confidence": result.get("confidence", 0.0),
                    "groundedness": result.get("groundedness_score", 0.0),
                },
            )

    # 格式化來源
    sources = []
    evidence_table = result.get("evidence_table", [])
    for item in evidence_table[:5]:
        source_info = {
            "chunk_id": getattr(item, "chunk_id", ""),
            "content": getattr(item, "content", "")[:200] if hasattr(item, "content") else "",
            "source_doc": getattr(item, "source_doc", ""),
            "relevance_score": getattr(item, "relevance_score", 0.0),
        }
        sources.append(source_info)

    return {
        "answer": result.get("final_answer", ""),
        "sources": sources,
        "confidence": result.get("confidence", 0.0),
        "trace_id": trace_id,
        "thread_id": result.get("thread_id", ""),
        "status": "completed",
    }


async def run_ask_stream(
    question: str,
    *,
    acl_groups: Optional[list[str]] = None,
    tenant_id: str = "default",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    include_sources: bool = True,
) -> AsyncIterator[dict[str, Any]]:
    """
    執行串流問答。

    與 chatbot_rag 相同的模式：整個串流操作都在 trace context 內執行。

    Args:
        question: 使用者問題
        acl_groups: 存取控制群組
        tenant_id: 租戶 ID
        user_id: 用戶 ID（用於 Langfuse）
        session_id: 會話 ID（用於 Langfuse）
        include_sources: 是否包含來源

    Yields:
        SSE 事件字典：
        - {"type": "response.start", "trace_id": "..."}
        - {"type": "response.chunk", "content": "..."}
        - {"type": "response.sources", "sources": [...]}
        - {"type": "response.done", "confidence": 0.8}
    """
    import asyncio

    acl_groups = acl_groups or ["public"]
    request_id = uuid.uuid4().hex[:8]

    with create_trace_context(
        name="graphrag-stream",
        trace_id_seed=request_id,
        user_id=user_id,
        session_id=session_id,
        tags=["graphrag", "stream"],
        metadata={"tenant_id": tenant_id},
        input_data={"question": question, "tenant_id": tenant_id},
    ) as ctx:
        trace_id = ctx.trace_id if ctx else request_id

        # 發送開始事件
        yield {"type": "response.start", "trace_id": trace_id}

        try:
            workflow = build_graphrag_workflow()
            initial_state = create_initial_state(
                question=question,
                acl_groups=acl_groups,
                tenant_id=tenant_id,
            )
            initial_state["trace_id"] = trace_id

            # 執行工作流程
            config: dict[str, Any] = {"configurable": {"thread_id": trace_id}}
            if ctx and ctx.handler:
                config["callbacks"] = [ctx.handler]

            result = await workflow.ainvoke(initial_state, config)

            # 串流回答片段
            answer = result.get("final_answer", "")
            if answer:
                chunk_size = 50
                for i in range(0, len(answer), chunk_size):
                    chunk = answer[i : i + chunk_size]
                    yield {"type": "response.chunk", "content": chunk}
                    await asyncio.sleep(0.02)

            # 發送來源
            if include_sources:
                evidence_table = result.get("evidence_table", [])
                sources = []
                for item in evidence_table[:5]:
                    content = getattr(item, "content", "")
                    sources.append({
                        "chunk_id": getattr(item, "chunk_id", ""),
                        "content": content[:200] + "..." if len(content) > 200 else content,
                        "source_doc": getattr(item, "source_doc", ""),
                        "relevance_score": getattr(item, "relevance_score", 0.0),
                    })
                if sources:
                    yield {"type": "response.sources", "sources": sources}

            # 更新 trace
            if ctx:
                update_trace_with_result(
                    output={
                        "answer": answer,
                        "answer_length": len(answer),
                    },
                    metadata={
                        "retrieval_path": result.get("retrieval_path", []),
                        "timing": result.get("timing", {}),
                    },
                    scores={
                        "confidence": result.get("confidence", 0.0),
                        "groundedness": result.get("groundedness_score", 0.0),
                    },
                )

            # 發送完成事件
            yield {
                "type": "response.done",
                "confidence": result.get("confidence", 0.0),
                "trace_id": trace_id,
            }

        except Exception as e:
            logger.error(f"串流問答錯誤: {e}")
            yield {"type": "response.error", "error": str(e)}


class AskService:
    """問答服務封裝類別。"""

    async def ask(
        self,
        question: str,
        **kwargs,
    ) -> dict[str, Any]:
        """執行非串流問答。"""
        return await run_ask(question, **kwargs)

    async def ask_stream(
        self,
        question: str,
        **kwargs,
    ) -> AsyncIterator[dict[str, Any]]:
        """執行串流問答。"""
        async for event in run_ask_stream(question, **kwargs):
            yield event


# 單例服務實例
ask_service = AskService()
