import { Injectable, inject } from '@angular/core';
import { SSEService } from './sse.service';
import { MessageService } from './message.service';
import { QuestionPayload, StreamEventRaw, MetaSummary } from '../models';

interface StreamTimer {
  requestStart: number;
  firstEvent?: number;
}

/**
 * Service for handling API calls and SSE streaming.
 */
@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private readonly sseService = inject(SSEService);
  private readonly messageService = inject(MessageService);
  private readonly streamTimers = new Map<number, StreamTimer>();

  /**
   * Send a question and stream the answer.
   * @param apiPath - API endpoint path
   * @param payload - Question payload
   * @param assistantMessageId - ID of the assistant message to update
   * @param callbacks - Optional callbacks for additional handling
   */
  async streamAnswer(
    apiPath: string,
    payload: QuestionPayload,
    assistantMessageId: number,
    callbacks?: {
      onThinkingChange?: (thinking: boolean) => void;
      onScrollNeeded?: () => void;
      onComplete?: (messageId: number) => void;
    }
  ): Promise<void> {
    this.streamTimers.set(assistantMessageId, {
      requestStart: performance.now()
    });

    let durationFinalized = false;
    try {
      const response = await fetch(apiPath, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok || !response.body) {
        throw new Error(
          `Network error: ${response.status} ${response.statusText}`
        );
      }

      for await (const event of this.sseService.parseStream(response.body)) {
        this.handleStreamEvent(event, assistantMessageId, callbacks);
      }

      // Finalize stream
      this.messageService.updateAssistantDetails(assistantMessageId, (details) => {
        details.isStreaming = false;
        details.expanded = false;
      });
      this.finalizeStreamDuration(assistantMessageId);
      durationFinalized = true;

      callbacks?.onComplete?.(assistantMessageId);
    } finally {
      if (!durationFinalized) {
        this.finalizeStreamDuration(assistantMessageId);
      }
    }
  }

  private handleStreamEvent(
    event: StreamEventRaw,
    messageId: number,
    callbacks?: {
      onThinkingChange?: (thinking: boolean) => void;
      onScrollNeeded?: () => void;
    }
  ): void {
    const timer = this.streamTimers.get(messageId);
    if (timer && timer.firstEvent === undefined) {
      timer.firstEvent = performance.now();
    }

    const channel = event.channel;

    switch (channel) {
      case 'status':
        this.handleStatusEvent(event, messageId, callbacks);
        break;

      case 'rewrite_llm':
        this.messageService.updateAssistantDetails(messageId, (details) => {
          details.planning += event.delta ?? '';
          details.isStreaming = true;
          details.expanded = true;
        });
        break;

      case 'retrieval':
        this.messageService.updateAssistantDetails(messageId, (details) => {
          const count = (event['documents_count'] as number | undefined) ?? 0;
          const query = event.search_query ?? '';
          details.retrieval = `已完成檢索，找到 ${count} 筆相關文檔。${
            query ? `\n搜尋查詢：${query}` : ''
          }`;
          details.isStreaming = true;
          details.expanded = true;
        });
        break;

      case 'reasoning_summary':
      case 'reasoning':
        this.messageService.updateAssistantDetails(messageId, (details) => {
          details.reasoning += event.delta ?? '';
          details.isStreaming = true;
          details.expanded = true;
        });
        callbacks?.onThinkingChange?.(true);
        break;

      case 'answer':
        if (event.delta && typeof event.delta === 'string') {
          this.messageService.appendToMessage(messageId, event.delta);
        }
        this.messageService.updateAssistantDetails(messageId, (details) => {
          details.isStreaming = true;
        });
        callbacks?.onThinkingChange?.(false);
        break;

      case 'meta':
        this.handleMetaEvent(event, messageId);
        callbacks?.onThinkingChange?.(false);
        break;

      case 'meta_summary':
        this.handleMetaSummaryEvent(event, messageId);
        break;

      default:
        break;
    }

    callbacks?.onScrollNeeded?.();
  }

  private handleMetaEvent(event: StreamEventRaw, messageId: number): void {
    this.messageService.updateAssistantDetails(messageId, (details) => {
      const meta = (event.meta ?? {}) as {
        usage?: Record<string, unknown>;
        usage_metadata?: Record<string, unknown>;
        token_usage?: Record<string, unknown>;
        conversation_summary?: unknown;
        [key: string]: unknown;
      };

      const rawUsage =
        (meta.usage as Record<string, unknown>) ??
        (meta.usage_metadata as Record<string, unknown>) ??
        (meta.token_usage as Record<string, unknown>);
      const usage = this.sseService.extractUsageStats(rawUsage);

      details.meta = meta;

      if (
        usage.totalTokens !== undefined ||
        usage.inputTokens !== undefined ||
        usage.outputTokens !== undefined
      ) {
        details.tokenUsage = this.sseService.mergeUsageStats(
          details.tokenUsage,
          usage
        );
      }

      this.messageService.updateConversationSummary(meta.conversation_summary);
    });
  }

  private handleMetaSummaryEvent(
    event: StreamEventRaw,
    messageId: number
  ): void {
    this.messageService.updateAssistantDetails(messageId, (details) => {
      const summary = (event['summary'] ?? {}) as MetaSummary;

      const totalUsage = this.sseService.extractUsageStats(
        summary.total_usage as Record<string, unknown> | undefined
      );

      if (
        totalUsage.totalTokens !== undefined ||
        totalUsage.inputTokens !== undefined ||
        totalUsage.outputTokens !== undefined
      ) {
        details.tokenUsage = {
          totalTokens: totalUsage.totalTokens ?? details.tokenUsage?.totalTokens,
          inputTokens: totalUsage.inputTokens ?? details.tokenUsage?.inputTokens,
          outputTokens: totalUsage.outputTokens ?? details.tokenUsage?.outputTokens
        };
      }

      const traceId = (event['trace_id'] as string) ?? summary.trace_id;
      if (traceId) {
        details.traceId = traceId;
      }

      details.isStreaming = false;
    });

    const summaryPayload = event['summary'] as MetaSummary | undefined;
    if (summaryPayload?.conversation_summary) {
      this.messageService.updateConversationSummary(
        summaryPayload.conversation_summary
      );
    }
  }

  private handleStatusEvent(
    event: StreamEventRaw,
    messageId: number,
    callbacks?: { onThinkingChange?: (thinking: boolean) => void }
  ): void {
    const node = event.node;
    const stage = event.stage;
    const nodeStage = (event.node_stage as string | undefined) ?? stage;

    this.messageService.updateAssistantDetails(messageId, (details) => {
      details.isStreaming = true;

      let message: string | null = null;
      const ensureExpanded = () => {
        if (!details.expanded) {
          details.expanded = true;
        }
      };

      // Handle GraphRAG nodes
      message = this.getGraphRAGStatusMessage(event, node, nodeStage, details, ensureExpanded, callbacks);

      // Handle v1 legacy events if no message yet
      if (!message) {
        message = this.getLegacyStatusMessage(event, node, stage, nodeStage, details, callbacks);
      }

      if (message) {
        details.statusMessages = [...details.statusMessages, message];
      }
    });
  }

  private getGraphRAGStatusMessage(
    event: StreamEventRaw,
    node: string | undefined,
    nodeStage: string | undefined,
    details: { expanded: boolean; isStreaming: boolean; retrieval: string },
    ensureExpanded: () => void,
    callbacks?: { onThinkingChange?: (thinking: boolean) => void }
  ): string | null {
    if (node === 'language_normalizer') {
      ensureExpanded();
      if (nodeStage === 'language_normalizer_start') {
        return '🌐 語言標準化：開始偵測使用者偏好語言...';
      } else if (nodeStage === 'language_normalizer_done') {
        const lang = (event.user_language as string | undefined) ?? '未知語言';
        return `🌐 語言標準化完成，統一採用「${lang}」。`;
      }
    } else if (node === 'planner') {
      ensureExpanded();
      if (nodeStage === 'planner_start') {
        return '🧭 任務規劃：LLM 正在判斷意圖與工具需求...';
      } else if (nodeStage === 'planner_done') {
        const intent = (event.intent as string | undefined) ?? 'unknown';
        const taskType = (event['task_type'] as string | undefined) ?? intent;
        const shouldRetrieve = Boolean(event['should_retrieve'] ?? event['need_retrieval']);
        const retrieveHint = shouldRetrieve ? '需要檢索' : '不需檢索';
        return `🧭 任務規劃完成：任務類型「${taskType}」，意圖「${intent}」，${retrieveHint}。`;
      } else if (nodeStage === 'planner_error') {
        const err = (event.error as string | undefined) ?? '未知錯誤';
        details.isStreaming = false;
        return `🧭 任務規劃失敗：${err}`;
      }
    } else if (node === 'followup_transform') {
      ensureExpanded();
      if (nodeStage === 'followup_start') {
        return '📝 追問流程：確認是否僅需重寫上一輪回答...';
      } else if (nodeStage === 'followup_done') {
        const fallback = Boolean(event.fallback_to_retrieval);
        return fallback
          ? '📝 未找到上一輪回答，改回一般檢索流程。'
          : '📝 已確認為追問任務，將直接處理上一輪回答。';
      }
    } else if (node === 'query_builder') {
      ensureExpanded();
      if (nodeStage === 'query_builder_start') {
        const loop = (event.loop as number | undefined) ?? 1;
        return `🔎 Query Builder 第 ${loop} 輪開始，準備整理檢索條件...`;
      } else if (nodeStage === 'query_builder_done') {
        const loop = (event.loop as number | undefined) ?? 1;
        const query = (event.query as string | undefined) ?? '';
        details.retrieval = `檢索查詢：${query || '（空）'}\n（第 ${loop} 輪）`;
        return `🔎 Query Builder 完成（第 ${loop} 輪），檢索查詢：「${query || '（空）'}」。`;
      }
    } else if (node === 'tool_executor') {
      ensureExpanded();
      if (nodeStage === 'tool_executor_start') {
        return '🛠️ 工具執行器：準備呼叫規劃中的工具...';
      } else if (nodeStage === 'tool_executor_call') {
        const toolName = (event.tool_name as string | undefined) ?? '(未知工具)';
        const args = event.tool_args !== undefined ? JSON.stringify(event.tool_args).slice(0, 200) : '';
        return `🛠️ 呼叫工具：${toolName}${args ? `，參數：${args}` : ''}`;
      } else if (nodeStage === 'tool_executor_result') {
        const toolName = (event.tool_name as string | undefined) ?? '(未知工具)';
        const output = (event.tool_output as string | undefined) ?? '';
        return `🛠️ 工具結果：${toolName}${output ? ` → ${output.slice(0, 200)}...` : ''}`;
      } else if (nodeStage === 'tool_executor_done') {
        const usedTools = (event.used_tools as string[] | undefined) ?? [];
        const documents = (event.documents_count as number | undefined) ?? 0;
        return `🛠️ 工具執行完成，使用 ${usedTools.length > 0 ? usedTools.join(', ') : '無'}，取得 ${documents} 份內容。`;
      }
    } else if (node === 'retrieval_checker') {
      ensureExpanded();
      if (nodeStage === 'retrieval_checker_start') {
        return '📚 檢閱檢索結果，評估是否需要重試...';
      } else if (nodeStage === 'retrieval_checker_retry') {
        const loop = (event.loop as number | undefined) ?? 1;
        return `📚 未找到足夠資料，準備第 ${loop + 1} 輪檢索。`;
      } else if (nodeStage === 'retrieval_checker_no_hits') {
        return '📚 檢索仍無結果，將以 fallback 策略回應。';
      } else if (nodeStage === 'retrieval_checker_done') {
        const count = (event.documents_count as number | undefined) ?? 0;
        if (count > 0) {
          details.retrieval = `找到 ${count} 份相關內容，準備生成回答。`;
        }
        return `📚 已選出 ${count} 份相關內容，準備交給 LLM。`;
      }
    } else if (node === 'response_synth') {
      ensureExpanded();
      if (nodeStage === 'response_generating') {
        const intent = (event.intent as string | undefined) ?? '';
        const tools = (event.used_tools as string[] | undefined) ?? [];
        const loop = (event.loops as number | undefined) ?? event.loop ?? 1;
        return `✍️ 正在生成回答（意圖：${intent || '一般問題'}，迴圈：${loop}，使用工具：${tools.length ? tools.join(', ') : '無'}）。`;
      } else if (nodeStage === 'response_reasoning') {
        callbacks?.onThinkingChange?.(true);
        return '🧠 LLM 正在輸出 reasoning 內容...';
      } else if (nodeStage === 'response_done') {
        details.isStreaming = false;
        callbacks?.onThinkingChange?.(false);
        return '✅ 回答已完成。';
      }
    } else if (node === 'telemetry') {
      if (nodeStage === 'telemetry_summary') {
        details.isStreaming = false;
        return '📊 已上傳本輪對話的遙測統計。';
      }
    }

    return null;
  }

  private getLegacyStatusMessage(
    event: StreamEventRaw,
    node: string | undefined,
    stage: string | undefined,
    nodeStage: string | undefined,
    details: { expanded: boolean; isStreaming: boolean },
    callbacks?: { onThinkingChange?: (thinking: boolean) => void }
  ): string | null {
    const phase = event.phase;

    // v1 unified_agent events
    if (node === 'unified_agent') {
      if (stage === 'unified_agent_start') {
        details.expanded = true;
        return '🚀 Unified Agent 開始處理...';
      } else if (stage === 'unified_agent_analyzing') {
        details.expanded = true;
        return '🔍 正在分析問題意圖與決定工具...';
      } else if (stage === 'unified_agent_tool_call') {
        const toolName = event.tool_name ?? '(未知工具)';
        const args = event.tool_args ? JSON.stringify(event.tool_args).slice(0, 200) : '';
        details.expanded = true;
        return `🔧 呼叫工具：${toolName}${args ? `（${args}）` : ''}`;
      } else if (stage === 'unified_agent_tool_result') {
        const toolName = event.tool_name ?? '(未知工具)';
        const outputRaw = event.tool_output;
        details.expanded = true;
        return `📋 工具結果：${toolName}${outputRaw ? ` → ${outputRaw.slice(0, 200)}...` : ''}`;
      } else if (stage === 'unified_agent_generating') {
        const intent = event.intent ?? '';
        const isOutOfScope = event.is_out_of_scope ?? false;
        const usedTools = event.used_tools ?? [];
        let toolsInfo = '';
        if (usedTools.length > 0) {
          toolsInfo = `，使用工具：${usedTools.join(', ')}`;
        }
        details.expanded = true;
        if (isOutOfScope) {
          return `✍️ 準備回應（超出服務範圍）${toolsInfo}`;
        }
        return `✍️ 準備生成回答（意圖：${intent || '一般問題'}）${toolsInfo}`;
      } else if (stage === 'unified_agent_done') {
        const loops = event.loops ?? 1;
        const usedTools = event.used_tools ?? [];
        const intent = event.intent ?? '';
        details.isStreaming = false;
        return `✅ Unified Agent 完成（迴圈：${loops}，意圖：${intent}，工具：${usedTools.length > 0 ? usedTools.join(', ') : '無'}）`;
      } else if (stage === 'unified_agent_error') {
        const err = (event.error as string | undefined) ?? '未知錯誤';
        details.isStreaming = false;
        return `❌ Unified Agent 錯誤：${err}`;
      } else if (stage === 'reasoning_start') {
        callbacks?.onThinkingChange?.(true);
        details.expanded = true;
        return '🧠 開始進行深度 reasoning...';
      } else if (stage === 'reasoning_end') {
        callbacks?.onThinkingChange?.(false);
        return '🧠 Reasoning 階段結束。';
      } else if (stage === 'answer_start') {
        return '💬 開始串流最終回答...';
      } else if (stage === 'answer_end') {
        details.expanded = false;
        return '💬 回答完成。';
      }
    }

    // v1 guard events
    if (node === 'guard' && (phase === 'planning' || phase === 'guard')) {
      if (stage === 'guard_start') {
        return '🛡️ Guard 節點：檢查請求安全性...';
      } else if (stage === 'guard_end') {
        const blocked = event.blocked ?? false;
        if (blocked) {
          return '🛡️ Guard 節點：請求已被攔截。';
        }
        return '🛡️ Guard 節點：通過安全檢查。';
      }
    }

    // v1 model events
    if (
      (node === 'model' || node === 'rag_model' || node === 'fallback_model') &&
      phase === 'generation'
    ) {
      if (stage === 'reasoning_start') {
        callbacks?.onThinkingChange?.(true);
        details.expanded = true;
        return '模型開始進行深度 reasoning。';
      } else if (stage === 'reasoning_end') {
        callbacks?.onThinkingChange?.(false);
        details.expanded = false;
        return 'reasoning 階段結束。';
      } else if (stage === 'answer_start') {
        return '開始串流最終回答內容。';
      } else if (stage === 'answer_end') {
        details.isStreaming = false;
        details.expanded = false;
        return '最終回答已完成。';
      }
    }

    return null;
  }

  private finalizeStreamDuration(messageId: number): void {
    const timer = this.streamTimers.get(messageId);
    if (!timer) return;

    const endAt = performance.now();
    const startAt = timer.firstEvent ?? timer.requestStart;
    const durationMs = Math.max(0, endAt - startAt);

    this.messageService.updateAssistantDetails(messageId, (details) => {
      details.durationMs = durationMs;
    });

    this.streamTimers.delete(messageId);
  }
}
