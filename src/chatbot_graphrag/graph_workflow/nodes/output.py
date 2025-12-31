"""
輸出節點

最終答案生成和遙測。
實現 OWASP LLM02（輸出處理）緩解措施。

主要節點：
- final_answer_node: 使用 LLM 生成最終答案
- telemetry_node: 記錄追蹤資訊到 Langfuse
"""

import html
import logging
import re
from typing import Any

from chatbot_graphrag.graph_workflow.types import GraphRAGState

logger = logging.getLogger(__name__)

# 最大輸出長度以防止資源耗盡
MAX_OUTPUT_LENGTH = 10000

# 要從輸出中剝離的模式（內部標記、系統提示）
OUTPUT_STRIP_PATTERNS = [
    r"<\|system\|>.*?<\|/system\|>",
    r"<system>.*?</system>",
    r"\[INTERNAL\].*?\[/INTERNAL\]",
    r"###\s*System:.*?(?=###|$)",
    r"DEBUG:.*?(?:\n|$)",
]

COMPILED_OUTPUT_STRIP = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in OUTPUT_STRIP_PATTERNS]


def sanitize_output(text: str, escape_html: bool = True) -> str:
    """
    清理 LLM 輸出以安全顯示（OWASP LLM02）。

    Args:
        text: 原始 LLM 輸出
        escape_html: 是否跳脫 HTML 實體

    Returns:
        安全用於前端顯示的清理過的文字
    """
    if not text:
        return text

    # 1. 剝離內部標記和洩漏的系統提示
    for pattern in COMPILED_OUTPUT_STRIP:
        text = pattern.sub("", text)

    # 2. 如果啟用則跳脫 HTML 實體（防止 XSS）
    if escape_html:
        text = html.escape(text)

    # 3. 限制輸出長度
    if len(text) > MAX_OUTPUT_LENGTH:
        text = text[:MAX_OUTPUT_LENGTH] + "\n...(回答已截斷)"
        logger.warning(f"Output truncated from {len(text)} to {MAX_OUTPUT_LENGTH} chars")

    # 4. 清理過多的空白
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


async def final_answer_node(state: GraphRAGState, config: dict | None = None) -> dict[str, Any]:
    """
    最終答案生成節點。

    使用 LLM 與上下文生成最終答案。

    Args:
        state: 當前圖譜狀態
        config: 包含 Langfuse 追蹤回調的可選 LangGraph 配置

    Returns:
        更新後的狀態，包含 final_answer 和引用
    """
    import time

    # 從配置中提取 Langfuse 追蹤的回調
    callbacks = config.get("callbacks", []) if config else []

    start_time = time.time()
    # 優先使用 resolved_question（來自帶有追蹤上下文的查詢分解器）
    # 而非 normalized_question 和原始問題
    question = (
        state.get("resolved_question")
        or state.get("normalized_question")
        or state.get("question", "")
    )
    user_language = state.get("user_language", "zh-TW")
    context_text = state.get("context_text", "")
    evidence_table = state.get("evidence_table", [])
    groundedness_score = state.get("groundedness_score", 0.0)
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))

    # 處理被阻擋/拒絕的情況
    if state.get("guard_blocked"):
        reason = state.get("guard_reason", "安全檢查未通過")
        answer = f"""很抱歉，我無法處理您的問題 🙏

**原因：** {reason}

---

如果您認為這是誤判，或有其他問題需要協助，歡迎重新提問或聯繫我們的客服人員。我們會盡力幫助您！"""
        return {
            "final_answer": answer,
            "confidence": 0.0,
            "citations": [],
            "retrieval_path": retrieval_path + ["final_answer:blocked"],
            "timing": {**timing, "final_answer_ms": (time.time() - start_time) * 1000},
        }

    if state.get("acl_denied"):
        answer = """很抱歉，您目前沒有權限訪問相關資訊 🔒

---

如需取得相關資料的存取權限，請聯繫您的管理員或客服人員。

有其他問題歡迎隨時詢問，我會盡力協助您！"""
        return {
            "final_answer": answer,
            "confidence": 0.0,
            "citations": [],
            "retrieval_path": retrieval_path + ["final_answer:acl_denied"],
            "timing": {**timing, "final_answer_ms": (time.time() - start_time) * 1000},
        }

    # 處理 HITL 逾時（第 3 階段）
    hitl_timeout_at = state.get("hitl_timeout_at")
    if hitl_timeout_at and state.get("hitl_required") and not state.get("hitl_resolved"):
        import time as time_module

        if time_module.time() > hitl_timeout_at:
            from chatbot_graphrag.graph_workflow.nodes.quality import get_hitl_fallback_response

            answer = get_hitl_fallback_response(state)
            logger.warning("HITL timeout - using fallback response")
            return {
                "final_answer": answer,
                "confidence": 0.0,
                "citations": [],
                "hitl_timed_out": True,
                "retrieval_path": retrieval_path + ["final_answer:hitl_timeout"],
                "timing": {**timing, "final_answer_ms": (time.time() - start_time) * 1000},
            }

    # 處理 HITL 拒絕（第 3 階段）
    if state.get("hitl_approved") is False:
        if user_language.startswith("zh"):
            answer = """您的問題已經由我們的審核人員仔細處理 📋

---

很抱歉，目前無法提供完整的回答。

### 建議您可以：
- 聯繫我們的客服人員取得專人協助
- 親自到醫院服務台洽詢

感謝您的耐心等待，我們會持續改進服務品質！"""
        else:
            answer = """Your question was carefully reviewed by our team 📋

---

Unfortunately, we couldn't provide a complete answer at this time.

### We suggest you:
- Contact our support team for personalized assistance
- Visit the hospital service desk in person

Thank you for your patience. We're continuously improving our services!"""

        return {
            "final_answer": answer,
            "confidence": 0.0,
            "citations": [],
            "hitl_rejected": True,
            "retrieval_path": retrieval_path + ["final_answer:hitl_rejected"],
            "timing": {**timing, "final_answer_ms": (time.time() - start_time) * 1000},
        }

    # 檢查我們是否已經有答案（來自 direct_answer 或快取）
    existing_answer = state.get("final_answer", "")
    if existing_answer:
        # 通過 SSE 事件將快取的答案串流到前端
        # 這確保前端即使對於快取的回應也能接收內容
        import asyncio
        logger.info(f"Found existing answer ({len(existing_answer)} chars), attempting to stream...")

        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()

            if writer is None:
                logger.warning("get_stream_writer() returned None - streaming not available")
            else:
                logger.debug(f"Got stream writer: {type(writer)}")

                # 發送開始狀態
                writer({
                    "node": "final_answer",
                    "channel": "status",
                    "stage": "GENERATING",
                })

                # 分塊串流快取的答案以實現平滑顯示
                # 使用較小的塊和延遲來模擬自然打字效果
                chunk_size = 20  # Smaller chunks for smoother streaming
                delay_per_chunk = 0.02  # 20ms delay between chunks (simulates ~50 chars/sec)
                chunks_sent = 0
                for i in range(0, len(existing_answer), chunk_size):
                    chunk = existing_answer[i:i + chunk_size]
                    writer({
                        "node": "final_answer",
                        "channel": "answer",
                        "delta": chunk,
                    })
                    chunks_sent += 1
                    # 添加延遲以實現自然串流效果
                    await asyncio.sleep(delay_per_chunk)

                logger.info(f"Streamed {chunks_sent} chunks for cached answer")

                # 發送完成狀態
                writer({
                    "node": "final_answer",
                    "channel": "status",
                    "stage": "DONE",
                })

                # 如果可用則從 evidence_table 建構並發送來源
                sources_data = []
                if evidence_table:
                    for idx, evidence in enumerate(evidence_table):
                        content = getattr(evidence, 'content', '') or ''
                        sources_data.append({
                            "index": idx + 1,
                            "chunk_id": getattr(evidence, 'chunk_id', str(idx)),
                            "content": content[:200] + "..." if len(content) > 200 else content,
                            "source_doc": getattr(evidence, 'source_doc', ''),
                            "relevance_score": round(getattr(evidence, 'relevance_score', 0.0), 3),
                        })

                if sources_data:
                    writer({
                        "node": "final_answer",
                        "channel": "sources",
                        "sources": sources_data,
                    })

                logger.info(f"Streamed cached answer: {len(existing_answer)} chars")

        except Exception as e:
            logger.error(f"Error streaming cached answer: {e}", exc_info=True)

        return {
            "retrieval_path": retrieval_path + ["final_answer:existing"],
            "timing": {**timing, "final_answer_ms": (time.time() - start_time) * 1000},
        }

    # 記錄正在使用哪個問題
    original_q = state.get("question", "")
    resolved_q = state.get("resolved_question", "")
    logger.info(f"Generating final answer - original: '{original_q[:30]}...', resolved: '{resolved_q[:50] if resolved_q else 'N/A'}'")

    try:
        # 為 LLM 建構提示
        if context_text:
            system_prompt = """# 你是誰
你是屏東基督教醫院的「服務小天使」，一個親切、專業且充滿關懷的醫療資訊助理。
你的使命是用溫暖的語氣，幫助民眾解答醫療相關的疑問。

# 回答風格
- **親切溫暖**：像朋友一樣關心對方，適時表達同理心
- **專業可靠**：根據提供的參考資料回答，不編造資訊
- **清晰易懂**：用淺顯易懂的語言說明，避免過於專業的術語

# 回答格式要求（Markdown）
- 使用 **粗體** 強調重點資訊
- 使用條列式（- 或 1.）列出步驟或多項內容
- 使用標題（## 或 ###）區分不同段落
- 適當使用分隔線（---）區隔不同主題
- 引用來源時使用 [數字] 格式

# 重要規則
1. 只根據提供的參考資料回答，不要編造資訊
2. 如果資料不足，請誠實說明並建議諮詢專業人員
3. 回答要完整但簡潔
4. 結尾加上溫馨的祝福語或關心的話語"""

            user_prompt = f"""參考資料：
{context_text}

問題：{question}

請根據以上參考資料，以親切溫暖的方式回答問題，並使用 Markdown 格式美化回答："""
        else:
            # 無上下文 - 生成降級回應
            if user_language.startswith("zh"):
                answer = """很抱歉，我目前沒有找到與您問題相關的資訊 😔

---

### 您可以嘗試以下方式：
- 重新描述您的問題，提供更多細節
- 使用不同的關鍵字搜尋
- 聯繫我們的客服人員獲取專人協助

如有任何疑問，歡迎隨時詢問，我會盡力幫助您！💪"""
            else:
                answer = """I'm sorry, I couldn't find relevant information for your question 😔

---

### You can try the following:
- Rephrase your question with more details
- Use different keywords
- Contact our support team for personalized assistance

Feel free to ask anytime, and I'll do my best to help! 💪"""

            return {
                "final_answer": answer,
                "confidence": 0.1,
                "citations": [],
                "retrieval_path": retrieval_path + ["final_answer:no_context"],
                "timing": {**timing, "final_answer_ms": (time.time() - start_time) * 1000},
            }

        # 使用帶有並發控制的真實 LLM 串流
        from langgraph.config import get_stream_writer

        from chatbot_graphrag.core.concurrency import llm_concurrency
        from chatbot_graphrag.services.llm import llm_factory

        # 取得用於自訂串流的串流寫入器
        writer = get_stream_writer()

        # 根據狀態配置選擇 LLM 後端
        agent_backend = state.get("agent_backend", "responses")
        concurrency_backend = "responses" if agent_backend == "responses" else "chat"

        if agent_backend == "responses":
            streaming_llm = llm_factory.create_responses_llm(streaming=True)
        else:
            streaming_llm = llm_factory.create_chat_completion_llm(streaming=True)

        # 發送開始狀態事件
        writer({
            "node": "final_answer",
            "channel": "status",
            "stage": "GENERATING",
        })

        # 準備訊息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 串流回應
        answer_tokens: list[str] = []
        reasoning_tokens: list[str] = []
        final_usage = None

        # 建構帶有 Langfuse 追蹤回調的配置
        stream_config = {"callbacks": callbacks} if callbacks else {}

        # 使用並發控制進行串流 LLM 呼叫
        # 信號量在整個串流持續期間持有
        async with llm_concurrency.acquire(concurrency_backend):
            async for chunk in streaming_llm.astream(messages, config=stream_config):
                add_kwargs = getattr(chunk, "additional_kwargs", {}) or {}
                channel = add_kwargs.get("channel")
                delta = add_kwargs.get("delta") or ""

                # 主要答案內容（Responses API）
                if channel == "output_text" and delta:
                    answer_tokens.append(delta)
                    writer({
                        "node": "final_answer",
                        "channel": "answer",
                        "delta": delta,
                    })
                # 推理內容（僅 Responses API）
                elif channel == "reasoning" and delta:
                    reasoning_tokens.append(delta)
                    writer({
                        "node": "final_answer",
                        "channel": "reasoning",
                        "delta": delta,
                    })
                # 元資訊（用量）
                elif channel == "meta":
                    responses_meta = add_kwargs.get("responses_meta", {})
                    final_usage = responses_meta.get("usage")

                # 處理 Chat Completions API 內容
                content = getattr(chunk, "content", None)
                if isinstance(content, str) and content and channel != "output_text":
                    answer_tokens.append(content)
                    writer({
                        "node": "final_answer",
                        "channel": "answer",
                        "delta": content,
                    })

        answer = "".join(answer_tokens)

        # 發送元事件
        writer({
            "node": "final_answer",
            "channel": "meta",
            "meta": {
                "usage": final_usage,
                "reasoning_text": "".join(reasoning_tokens),
            },
        })

        # Send completion status
        writer({
            "node": "final_answer",
            "channel": "status",
            "stage": "DONE",
        })

        # 使用證據表或 chunk 建構並發送來源事件
        sources_data = []

        # 首先嘗試 evidence_table（結構化證據 - EvidenceItem 資料類別）
        logger.info(f"Building sources: evidence_table has {len(evidence_table)} items")
        if evidence_table:
            for idx, evidence in enumerate(evidence_table):
                # EvidenceItem 是一個資料類別，包含：chunk_id, content, relevance_score, source_doc
                content = getattr(evidence, 'content', '') or ''
                sources_data.append({
                    "index": idx + 1,
                    "chunk_id": getattr(evidence, 'chunk_id', str(idx)),
                    "content": content[:200] + "..." if len(content) > 200 else content,
                    "source_doc": getattr(evidence, 'source_doc', ''),
                    "relevance_score": round(getattr(evidence, 'relevance_score', 0.0), 3),
                })
        else:
            # 回退到 expanded_chunks 或 reranked_chunks
            expanded_chunks = state.get("expanded_chunks", [])
            reranked_chunks = state.get("reranked_chunks", [])
            logger.info(f"Fallback: expanded_chunks={len(expanded_chunks)}, reranked_chunks={len(reranked_chunks)}")

            chunks = expanded_chunks or reranked_chunks
            for idx, chunk in enumerate(chunks[:5]):  # Limit to top 5
                chunk_id = chunk.chunk_id if hasattr(chunk, 'chunk_id') else str(idx)
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                doc_id = chunk.doc_id if hasattr(chunk, 'doc_id') else ""
                score = chunk.score if hasattr(chunk, 'score') else 0.0

                sources_data.append({
                    "index": idx + 1,
                    "chunk_id": chunk_id,
                    "content": content[:200] + "..." if len(content) > 200 else content,
                    "source_doc": doc_id,
                    "relevance_score": round(score, 3) if isinstance(score, float) else 0.0,
                })

        logger.info(f"Sources data: {len(sources_data)} items to send")

        if sources_data:
            writer({
                "node": "final_answer",
                "channel": "sources",
                "sources": sources_data,
            })

        # 在清理之前從答案中抽取引用
        citations = re.findall(r"\[(\d+)\]", answer)
        unique_citations = list(dict.fromkeys(citations))

        # 清理輸出以安全顯示（第 2 階段：OWASP LLM02）
        # 注意：escape_html=False 以保留 Markdown 格式
        # 如需要，HTML 跳脫應在 API/前端層完成
        answer = sanitize_output(answer, escape_html=False)

        # 根據落地性和證據計數計算信心分數
        evidence_count = len(evidence_table)
        confidence = min(1.0, (groundedness_score * 0.6) + (min(evidence_count, 5) / 5 * 0.4))

        logger.info(f"Generated answer: {len(answer)} chars, confidence={confidence:.2f}")

        return {
            "final_answer": answer,
            "confidence": confidence,
            "citations": unique_citations,
            "retrieval_path": retrieval_path + ["final_answer:generated"],
            "timing": {**timing, "final_answer_ms": (time.time() - start_time) * 1000},
        }

    except Exception as e:
        logger.error(f"Final answer generation error: {e}")

        # 降級回應
        if user_language.startswith("zh"):
            answer = """很抱歉，生成回答時發生了一些問題 😓

---

### 請您稍後再試，或者：
- 重新整理頁面後再次提問
- 嘗試用不同的方式描述您的問題
- 聯繫客服人員獲取協助

造成不便，敬請見諒！我們會盡快修復問題。"""
        else:
            answer = """Sorry, we encountered an issue while generating the response 😓

---

### Please try again later, or:
- Refresh the page and ask again
- Try rephrasing your question
- Contact support for assistance

We apologize for the inconvenience and will fix this soon!"""

        return {
            "final_answer": answer,
            "confidence": 0.0,
            "citations": [],
            "error": str(e),
            "retrieval_path": retrieval_path + ["final_answer:error"],
            "timing": {**timing, "final_answer_ms": (time.time() - start_time) * 1000},
        }


async def telemetry_node(state: GraphRAGState) -> dict[str, Any]:
    """
    遙測節點。

    將追蹤資訊記錄到 Langfuse 以供可觀測性。
    第 4 階段：啟用 Langfuse 整合與 Ragas 分數。
    使用來自 chatbot_rag 的統一追蹤模組模式。

    Returns:
        更新後的狀態，包含 trace_id
    """
    import time
    import uuid

    from chatbot_graphrag.core.config import settings
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("telemetry", "START")

    start_time = time.time()
    question = state.get("question", "")
    final_answer = state.get("final_answer", "")
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))
    confidence = state.get("confidence", 0.0)
    groundedness_score = state.get("groundedness_score", 0.0)

    # 如果不存在則生成追蹤 ID
    trace_id = state.get("trace_id") or str(uuid.uuid4())

    logger.debug(f"Recording telemetry: trace_id={trace_id}")

    # 抽取版本欄位（第 0 階段）
    index_version = state.get("index_version", "")
    pipeline_version = state.get("pipeline_version", "")
    prompt_version = state.get("prompt_version", "")
    config_hash = state.get("config_hash", "")

    # 抽取 Ragas 指標（第 4 階段）
    ragas_metrics = state.get("ragas_metrics", {})
    ragas_sampled = state.get("ragas_sampled", False)

    try:
        telemetry_data = {
            "trace_id": trace_id,
            "question": question[:100],
            "answer_length": len(final_answer),
            "retrieval_path": retrieval_path,
            "confidence": confidence,
            "groundedness_score": groundedness_score,
            "timing": timing,
            "total_ms": sum(timing.values()) if timing else 0,
            # 用於可重現性的版本欄位（第 0 階段）
            "index_version": index_version,
            "pipeline_version": pipeline_version,
            "prompt_version": prompt_version,
            "config_hash": config_hash,
            # Ragas 指標（第 4 階段）
            "ragas_sampled": ragas_sampled,
            "ragas_metrics": ragas_metrics,
        }

        logger.info(f"Telemetry: {telemetry_data}")

        # 注意：Langfuse 追蹤更新現在在 API 路由中處理
        # 在工作流程完成後以確保正確的上下文傳播。
        # telemetry_data 在這裡準備用於日誌記錄並在狀態中返回
        # 供 API 路由使用 update_trace_with_result()。

        emit_status("telemetry", "DONE")
        return {
            "trace_id": trace_id,
            "retrieval_path": retrieval_path + ["telemetry"],
            "timing": {**timing, "telemetry_ms": (time.time() - start_time) * 1000},
        }

    except Exception as e:
        logger.warning(f"Telemetry error: {e}")
        return {
            "trace_id": trace_id,
            "retrieval_path": retrieval_path + ["telemetry:error"],
            "timing": {**timing, "telemetry_ms": (time.time() - start_time) * 1000},
        }
