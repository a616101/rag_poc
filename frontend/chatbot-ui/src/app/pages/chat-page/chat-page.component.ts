import { CommonModule } from '@angular/common';
import { Component, computed, inject, signal, ViewChild, ElementRef, HostListener } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';
import { MarkdownModule } from 'ngx-markdown';

type ChatRole = 'user' | 'assistant';

interface AssistantDetails {
  /** 規劃 / Query 重寫過程（rewrite_llm channel 的 delta 累積） */
  planning: string;
  /** 檢索階段資訊（retrieval channel / status） */
  retrieval: string;
  /** LLM reasoning 詳細內容（reasoning channel 的 delta 累積） */
  reasoning: string;
  /** 階段狀態文字（status channel） */
  statusMessages: string[];
  /** LLM meta 資訊（包含 token 與完整 reasoning 摘要） */
  meta?: unknown;
  /** Token 使用統計（累加所有 LLM node 的 meta.usage） */
  tokenUsage?: {
    totalTokens?: number;
    inputTokens?: number;
    outputTokens?: number;
  };
  /** 前端以 SSE 起訖量測的耗時（毫秒） */
  durationMs?: number;
  /** 是否展開細節面板 */
  expanded: boolean;
  /** 是否仍在串流中（任一階段） */
  isStreaming: boolean;
  /** Langfuse trace ID（用於回饋 API） */
  traceId?: string;
  /** 用戶已提交的評分 */
  userFeedback?: 'up' | 'down' | null;
}

interface ChatMessage {
  id: number;
  role: ChatRole;
  content: string;
  /** 只有助手訊息會帶有詳細推理資訊 */
  details?: AssistantDetails;
}

interface LlmConfigPayload {
  model?: string;
  reasoning_effort?: 'low' | 'medium' | 'high';
  reasoning_summary?: 'auto' | 'concise' | 'detailed';
}

interface QuestionPayload {
  question: string;
  conversation_history?: { role: ChatRole; content: string }[] | null;
  top_k?: number;
  llm_config?: LlmConfigPayload;
  enable_conversation_summary?: boolean;
  conversation_summary?: string;
}

/**
 * /api/v1/rag/ask/stream 系列端點事件格式：
 *
 * 兩個端點皆採 Unified Agent LangGraph：
 * - /ask/stream：Chat backend，偏向最終回答
 * - /ask/stream_chat：Responses backend，包含 reasoning summary
 *
 * 事件共通欄位：
 * {
 *   "source": "ask_stream" | "unified_agent",
 *   "node": "guard" | "planner" | "tool_executor" | ...,
 *   "phase": "planning" | "agent" | "generation" | "summary",
 *   "channel": "status" | "rewrite_llm" | "answer" | "reasoning" | "meta" | "meta_summary",
 *   "stage": "unified_agent_*" | "language_normalizer_*" | ...,
 *   "delta": "逐字輸出",
 *   "used_tools": [...],
 *   "meta": {...}
 * }
 */
interface StreamEventRaw {
  source?: string;
  node?: string;
  phase?: string;
  channel?: string;
  stage?: string;
  node_stage?: string;
  delta?: string;
  // rewrite 節點相關欄位
  is_out_of_scope?: boolean;
  search_query?: string;
  intent?: string;
  // guard 節點相關欄位
  blocked?: boolean;
  meta?: unknown;
  error?: unknown;
  // Agent 工具相關欄位
  tool_name?: string;
  tool_args?: unknown;
  tool_output?: string;
  //  Unified Agent 相關欄位
  used_tools?: string[];
  loops?: number;
  loop?: number;
  user_language?: string;
  query?: string;
  documents_count?: number;
  fallback_to_retrieval?: boolean;
  // 其他欄位保留彈性
  [key: string]: unknown;
}

@Component({
  selector: 'app-chat-page',
  standalone: true,
  imports: [CommonModule, FormsModule, MarkdownModule],
  templateUrl: './chat-page.component.html'
})
export class ChatPageComponent {
  private readonly route = inject(ActivatedRoute);
  private readonly streamTimers = new Map<
    number,
    { requestStart: number; firstEvent?: number }
  >();

  @ViewChild('messagesContainer')
  private messagesContainer?: ElementRef<HTMLDivElement>;

  readonly pageTitle: string =
    this.route.snapshot.data['title'] ?? 'RAG 串流聊天';

  /**
   * 後端 API 路徑：
   * - /api/v1/rag/ask/stream
   * - /api/v1/rag/ask/stream_chat
   * 由路由 data.apiPath 傳入，並在畫面上顯示。
   */
  readonly apiPath: string =
    this.route.snapshot.data['apiPath'] ?? '/api/v1/rag/ask/stream';

  readonly messages = signal<ChatMessage[]>([]);

  private readonly inputSignal = signal('');

  get inputValue(): string {
    return this.inputSignal();
  }

  set inputValue(value: string) {
    this.inputSignal.set(value);
  }

  readonly isLoading = signal(false);
  readonly isThinking = signal(false);
  readonly errorMessage = signal<string | null>(null);

  private nextId = 1;
  private conversationSummary = '';
  conversationSummaryEnabled = false;

  readonly modelOptions = [
    'openai/gpt-oss-20b',
    'gpt-oss-20b',
    'llama3.1-ffm-8b-32k-chat',
    'llama3.3-ffm-70b-32k-chat'
  ];

  readonly reasoningEffortOptions: Array<'low' | 'medium' | 'high'> = [
    'low',
    'medium',
    'high'
  ];

  selectedModel: string = '';
  selectedReasoningEffort: 'low' | 'medium' | 'high' = 'low';
  topK = 3;

  readonly canSend = computed(
    () => !this.isLoading() && this.inputSignal().trim().length > 0
  );

  // 工具列狀態
  copiedMessageId: number | null = null;
  readonly feedbackLoading = signal(false);
  showFeedbackPopup = false;
  pendingFeedbackMessageId: number | null = null;
  feedbackComment = '';

  // Lightbox 圖片放大狀態
  lightboxImageSrc: string | null = null;

  onConversationSummaryToggle(enabled: boolean): void {
    this.conversationSummaryEnabled = enabled;
    if (!enabled) {
      this.conversationSummary = '';
    }
  }

  // ========== Lightbox 圖片放大功能 ==========

  /**
   * 監聽 markdown 區塊內的圖片點擊事件
   */
  @HostListener('click', ['$event'])
  onDocumentClick(event: MouseEvent): void {
    const target = event.target as HTMLElement;
    // 檢查是否點擊了 markdown-body 內的圖片
    if (
      target.tagName === 'IMG' &&
      target.closest('.markdown-body')
    ) {
      event.preventDefault();
      event.stopPropagation();
      const imgSrc = (target as HTMLImageElement).src;
      if (imgSrc) {
        this.openLightbox(imgSrc);
      }
    }
  }

  /**
   * 開啟 lightbox 顯示放大圖片
   */
  openLightbox(src: string): void {
    this.lightboxImageSrc = src;
  }

  /**
   * 關閉 lightbox
   */
  closeLightbox(): void {
    this.lightboxImageSrc = null;
  }

  /**
   * 按下 ESC 鍵關閉 lightbox
   */
  @HostListener('document:keydown.escape')
  onEscapeKey(): void {
    if (this.lightboxImageSrc) {
      this.closeLightbox();
    }
  }

  onTextareaKeydown(event: KeyboardEvent): void {
    const isEnter = event.key === 'Enter';
    const isShift = event.shiftKey;
    const isComposing =
      (event as any).isComposing === true ||
      ((event.target as any)?.isComposing === true) ||
      // 某些輸入法在組字時會回報 keyCode 229
      (event as any).keyCode === 229;

    if (!isEnter) {
      return;
    }

    // 組字中：交給輸入法處理，不截斷事件
    if (isComposing) {
      return;
    }

    // Shift+Enter → 換行
    if (isShift) {
      return;
    }

    // 單純 Enter → 送出提問
    event.preventDefault();
    void this.onSubmit();
  }

  async onSubmit(event?: SubmitEvent): Promise<void> {
    event?.preventDefault();

    const question = this.inputSignal().trim();
    if (!question || this.isLoading()) {
      return;
    }

    this.errorMessage.set(null);
    this.isLoading.set(true);
    this.isThinking.set(false);

    const history = this.buildConversationHistory();

    // 先清空輸入框
    this.inputSignal.set('');

    // 將當前問題與空的 assistant 回應加入畫面
    this.addMessage('user', question);
    const assistantMessageId = this.addMessage('assistant', '');

    let safeTopK = Number(this.topK) || 3;
    if (safeTopK < 1) safeTopK = 1;
    if (safeTopK > 10) safeTopK = 10;

    const payload: QuestionPayload = {
      question,
      conversation_history: history,
      top_k: safeTopK,
      llm_config: {
        model: this.selectedModel || undefined,
        reasoning_effort: this.selectedReasoningEffort,
        reasoning_summary: 'auto'
      },
      enable_conversation_summary: this.conversationSummaryEnabled,
      conversation_summary: this.conversationSummaryEnabled
        ? this.conversationSummary.trim() || undefined
        : undefined
    };

    try {
      await this.streamAnswer(payload, assistantMessageId);
    } catch (error) {
      console.error('stream error', error);
      this.errorMessage.set('請求失敗，請稍後再試。');
    } finally {
      this.isLoading.set(false);
      this.isThinking.set(false);
    }
  }

  private addMessage(role: ChatRole, content: string): number {
    const id = this.nextId++;
    this.messages.update((list) => [
      ...list,
      {
        id,
        role,
        content,
        details:
          role === 'assistant'
            ? {
                planning: '',
                retrieval: '',
                reasoning: '',
                statusMessages: [],
                meta: undefined,
                expanded: false,
                isStreaming: false
              }
            : undefined
      }
    ]);
    this.scrollToBottomIfNearBottom();
    return id;
  }

  private appendToMessage(id: number, chunk: string): void {
    if (!chunk) return;
    this.messages.update((list) =>
      list.map((m) =>
        m.id === id ? { ...m, content: m.content + chunk } : m
      )
    );
    this.scrollToBottomIfNearBottom();
  }

  private buildConversationHistory():
    | { role: ChatRole; content: string }[]
    | null {
    const msgs = this.messages();
    if (!msgs.length) {
      return null;
    }
    return msgs.map((m) => ({ role: m.role, content: m.content }));
  }

  private async streamAnswer(
    payload: QuestionPayload,
    assistantMessageId: number
  ): Promise<void> {
    this.streamTimers.set(assistantMessageId, {
      requestStart: performance.now()
    });

    let durationFinalized = false;
    try {
      const response = await fetch(this.apiPath, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok || !response.body) {
        throw new Error(
          `Network error: ${response.status} ${response.statusText}`
        );
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const rawLine of lines) {
          const line = rawLine.trim();
          if (!line || !line.startsWith('data:')) continue;

          const jsonStr = line.slice(5).trim();
          if (!jsonStr) continue;

          let event: StreamEventRaw;
          try {
            event = JSON.parse(jsonStr);
          } catch (e) {
            console.warn('Failed to parse SSE event', e, jsonStr);
            continue;
          }

          this.handleStreamEvent(event, assistantMessageId);
        }
      }

      // 收尾處理殘餘緩衝
      const remaining = buffer.trim();
      if (remaining && remaining.startsWith('data:')) {
        const jsonStr = remaining.slice(5).trim();
        if (jsonStr) {
          try {
            const event: StreamEventRaw = JSON.parse(jsonStr);
            this.handleStreamEvent(event, assistantMessageId);
          } catch {
            // ignore
          }
        }
      }

      // 串流結束時，將當前助手訊息標記為非串流中
      this.updateAssistantDetails(assistantMessageId, (details) => {
        details.isStreaming = false;
        details.expanded = false;
      });
      this.scrollToMessageTop(assistantMessageId);
      this.finalizeStreamDuration(assistantMessageId);
      durationFinalized = true;
    } finally {
      if (!durationFinalized) {
        this.finalizeStreamDuration(assistantMessageId);
      }
    }
  }

  private handleStreamEvent(
    event: StreamEventRaw,
    assistantMessageId: number
  ): void {
    const timer = this.streamTimers.get(assistantMessageId);
    if (timer && timer.firstEvent === undefined) {
      timer.firstEvent = performance.now();
    }

    const channel = event.channel;

    switch (channel) {
      case 'status':
        this.handleStatusEvent(event, assistantMessageId);
        break;

      case 'rewrite_llm':
        this.updateAssistantDetails(assistantMessageId, (details) => {
          details.planning += event.delta ?? '';
          details.isStreaming = true;
          details.expanded = true; // 正在串流規劃時自動展開
        });
        break;

      case 'retrieval':
        // 注意：此 channel 已棄用，檢索現在由 Agent 工具處理
        // 保留此處理以維持向後兼容
        this.updateAssistantDetails(assistantMessageId, (details) => {
          const count = (event['documents_count'] as number | undefined) ?? 0;
          const query = event.search_query ?? '';
          details.retrieval = `已完成檢索，找到 ${count} 筆相關文檔。${
            query ? `\n搜尋查詢：${query}` : ''
          }`;
          details.isStreaming = true;
          details.expanded = true;
        });
        break;

      case 'reasoning_summary': case 'reasoning':
        // 將 reasoning delta 累積到詳細推理區塊
        this.updateAssistantDetails(assistantMessageId, (details) => {
          details.reasoning += event.delta ?? '';
          details.isStreaming = true;
          details.expanded = true;
        });
        this.isThinking.set(true);
        break;

      case 'answer':
        // 最終回答內容：直接串流到助手訊息的氣泡內容
        if (event.delta && typeof event.delta === 'string') {
          this.appendToMessage(assistantMessageId, event.delta);
        }
        this.updateAssistantDetails(assistantMessageId, (details) => {
          details.isStreaming = true;
        });
        this.isThinking.set(false);
        break;

      case 'meta': {
        // 累加所有 LLM node 的 tokens 和 duration（而非覆蓋）
        this.updateAssistantDetails(assistantMessageId, (details) => {
          const meta = (event.meta ?? {}) as {
            usage?: {
              total_tokens?: number;
              input_tokens?: number;
              output_tokens?: number;
            };
            usage_metadata?: Record<string, unknown>;
            token_usage?: Record<string, unknown>;
            [key: string]: unknown;
          };

          const rawUsage =
            (meta.usage as Record<string, unknown>) ??
            (meta.usage_metadata as Record<string, unknown>) ??
            (meta.token_usage as Record<string, unknown>);
          const usage = this.extractUsageStats(rawUsage);

          // 保存最後一個 meta（用於顯示其他詳細資訊）
          details.meta = meta;

          if (
            usage.totalTokens !== undefined ||
            usage.inputTokens !== undefined ||
            usage.outputTokens !== undefined
          ) {
            const prevTokens = details.tokenUsage ?? {};
            details.tokenUsage = {
              totalTokens:
                (prevTokens.totalTokens ?? 0) + (usage.totalTokens ?? 0),
              inputTokens:
                (prevTokens.inputTokens ?? 0) + (usage.inputTokens ?? 0),
              outputTokens:
                (prevTokens.outputTokens ?? 0) + (usage.outputTokens ?? 0)
            };
          }

          this.updateConversationSummary(meta['conversation_summary']);
          // 只有最後一個 node 結束時才設為 false（由 answer_end 控制）
          // details.isStreaming = false;
        });
        this.isThinking.set(false);
        break;
      }

      case 'meta_summary': {
        // 後端發送的完整統計摘要（包含整個 graph 的累計 tokens 和 duration）
        this.updateAssistantDetails(assistantMessageId, (details) => {
          const summary = (event['summary'] ?? {}) as {
            total_usage?: {
              total_tokens?: number;
              input_tokens?: number;
              output_tokens?: number;
            };
            trace_id?: string;
            [key: string]: unknown;
          };

          const totalUsage = this.extractUsageStats(
            summary.total_usage as Record<string, unknown> | undefined
          );

          if (
            totalUsage.totalTokens !== undefined ||
            totalUsage.inputTokens !== undefined ||
            totalUsage.outputTokens !== undefined
          ) {
            details.tokenUsage = {
              totalTokens:
                totalUsage.totalTokens ?? details.tokenUsage?.totalTokens,
              inputTokens:
                totalUsage.inputTokens ?? details.tokenUsage?.inputTokens,
              outputTokens:
                totalUsage.outputTokens ?? details.tokenUsage?.outputTokens
            };
          }

          // 提取 trace_id 供回饋 API 使用
          const traceId =
            (event['trace_id'] as string) ?? summary.trace_id;
          if (traceId) {
            details.traceId = traceId;
          }

          details.isStreaming = false;
        });
        const summaryPayload = event['summary'] as
          | { conversation_summary?: unknown }
          | undefined;
        if (summaryPayload) {
          this.updateConversationSummary(summaryPayload.conversation_summary);
        }
        break;
      }

      default:
        // 其他 channel 暫時忽略或之後再視需求擴充
        break;
    }
    this.scrollToBottomIfNearBottom();
  }

  private handleStatusEvent(
    event: StreamEventRaw,
    assistantMessageId: number
  ): void {
    const node = event.node;
    const phase = event.phase;
    const stage = event.stage;
    const nodeStage = (event.node_stage as string | undefined) ?? stage;

    this.updateAssistantDetails(assistantMessageId, (details) => {
      details.isStreaming = true;

      let message: string | null = null;
      const ensureExpanded = () => {
        if (!details.expanded) {
          details.expanded = true;
        }
      };

      // ========== Unified Agent 節點（語言標準化、規劃、檢索等） ==========
      if (node === 'language_normalizer') {
        ensureExpanded();
        if (nodeStage === 'language_normalizer_start') {
          message = '🌐 語言標準化：開始偵測使用者偏好語言...';
        } else if (nodeStage === 'language_normalizer_done') {
          const lang = (event.user_language as string | undefined) ?? '未知語言';
          message = `🌐 語言標準化完成，統一採用「${lang}」。`;
        }
      } else if (node === 'planner') {
        ensureExpanded();
        if (nodeStage === 'planner_start') {
          message = '🧭 任務規劃：LLM 正在判斷意圖與工具需求...';
        } else if (nodeStage === 'planner_done') {
          const intent = (event.intent as string | undefined) ?? 'unknown';
          const taskType = (event['task_type'] as string | undefined) ?? intent;
          const shouldRetrieve = Boolean(
            event['should_retrieve'] ?? event['need_retrieval']
          );
          const retrieveHint = shouldRetrieve ? '需要檢索' : '不需檢索';
          message = `🧭 任務規劃完成：任務類型「${taskType}」，意圖「${intent}」，${retrieveHint}。`;
        } else if (nodeStage === 'planner_error') {
          const err = (event.error as string | undefined) ?? '未知錯誤';
          message = `🧭 任務規劃失敗：${err}`;
          details.isStreaming = false;
        }
      } else if (node === 'followup_transform') {
        ensureExpanded();
        if (nodeStage === 'followup_start') {
          message = '📝 追問流程：確認是否僅需重寫上一輪回答...';
        } else if (nodeStage === 'followup_done') {
          const fallback = Boolean(event.fallback_to_retrieval);
          message = fallback
            ? '📝 未找到上一輪回答，改回一般檢索流程。'
            : '📝 已確認為追問任務，將直接處理上一輪回答。';
        }
      } else if (node === 'query_builder') {
        ensureExpanded();
        if (nodeStage === 'query_builder_start') {
          const loop = (event.loop as number | undefined) ?? 1;
          message = `🔎 Query Builder 第 ${loop} 輪開始，準備整理檢索條件...`;
        } else if (nodeStage === 'query_builder_done') {
          const loop = (event.loop as number | undefined) ?? 1;
          const query = (event.query as string | undefined) ?? '';
          message = `🔎 Query Builder 完成（第 ${loop} 輪），檢索查詢：「${query || '（空）'}」。`;
          details.retrieval = `檢索查詢：${query || '（空）'}\n（第 ${loop} 輪）`;
        }
      } else if (node === 'tool_executor') {
        ensureExpanded();
        if (nodeStage === 'tool_executor_start') {
          message = '🛠️ 工具執行器：準備呼叫規劃中的工具...';
        } else if (nodeStage === 'tool_executor_call') {
          const toolName = (event.tool_name as string | undefined) ?? '(未知工具)';
          const args =
            event.tool_args !== undefined
              ? JSON.stringify(event.tool_args).slice(0, 200)
              : '';
          message = `🛠️ 呼叫工具：${toolName}${args ? `，參數：${args}` : ''}`;
        } else if (nodeStage === 'tool_executor_result') {
          const toolName = (event.tool_name as string | undefined) ?? '(未知工具)';
          const output = (event.tool_output as string | undefined) ?? '';
          message = `🛠️ 工具結果：${toolName}${
            output ? ` → ${output.slice(0, 200)}...` : ''
          }`;
        } else if (nodeStage === 'tool_executor_done') {
          const usedTools = (event.used_tools as string[] | undefined) ?? [];
          const documents = (event.documents_count as number | undefined) ?? 0;
          message = `🛠️ 工具執行完成，使用 ${
            usedTools.length > 0 ? usedTools.join(', ') : '無'
          }，取得 ${documents} 份內容。`;
        }
      } else if (node === 'retrieval_checker') {
        ensureExpanded();
        if (nodeStage === 'retrieval_checker_start') {
          message = '📚 檢閱檢索結果，評估是否需要重試...';
        } else if (nodeStage === 'retrieval_checker_retry') {
          const loop = (event.loop as number | undefined) ?? 1;
          message = `📚 未找到足夠資料，準備第 ${loop + 1} 輪檢索。`;
        } else if (nodeStage === 'retrieval_checker_no_hits') {
          message = '📚 檢索仍無結果，將以 fallback 策略回應。';
        } else if (nodeStage === 'retrieval_checker_done') {
          const count = (event.documents_count as number | undefined) ?? 0;
          message = `📚 已選出 ${count} 份相關內容，準備交給 LLM。`;
          if (count > 0) {
            details.retrieval = `找到 ${count} 份相關內容，準備生成回答。`;
          }
        }
      } else if (node === 'response_synth') {
        ensureExpanded();
        if (nodeStage === 'response_generating') {
          const intent = (event.intent as string | undefined) ?? '';
          const tools = (event.used_tools as string[] | undefined) ?? [];
          const loop = (event.loops as number | undefined) ?? event.loop ?? 1;
          message = `✍️ 正在生成回答（意圖：${
            intent || '一般問題'
          }，迴圈：${loop}，使用工具：${
            tools.length ? tools.join(', ') : '無'
          }）。`;
        } else if (nodeStage === 'response_reasoning') {
          message = '🧠 LLM 正在輸出 reasoning 內容...';
          this.isThinking.set(true);
        } else if (nodeStage === 'response_done') {
          message = '✅ 回答已完成。';
          details.isStreaming = false;
          this.isThinking.set(false);
        }
      } else if (node === 'telemetry') {
        if (nodeStage === 'telemetry_summary') {
          message = '📊 已上傳本輪對話的遙測統計。';
          details.isStreaming = false;
        }
      }

      // ========== v1 舊架構事件 ==========
      if (!message && node === 'rewrite' && phase === 'planning') {
        if (stage === 'rewrite_start') {
          message = '開始進行 Query 重寫與規劃（planning）...';
        } else if (stage === 'rewrite_done') {
          const isOutOfScope = event.is_out_of_scope ?? false;
          const searchQuery = event.search_query ?? '';
          const intent = event.intent ?? '';
          // 顯示重寫結果：是否超出範圍、搜尋查詢、意圖
          const parts: string[] = ['完成 Query 重寫'];
          if (isOutOfScope) {
            parts.push('（判定為超出服務範圍）');
          } else if (searchQuery) {
            parts.push(`，搜尋查詢：「${searchQuery}」`);
          }
          if (intent) {
            parts.push(`，意圖：${intent}`);
          }
          message = parts.join('');
        }
      } else if (
        !message &&
        node === 'guard' &&
        (phase === 'planning' || phase === 'guard')
      ) {
        // guard 節點的狀態事件
        if (stage === 'guard_start') {
          message = '🛡️ Guard 節點：檢查請求安全性...';
        } else if (stage === 'guard_end') {
          const blocked = event.blocked ?? false;
          if (blocked) {
            message = '🛡️ Guard 節點：請求已被攔截。';
          } else {
            message = '🛡️ Guard 節點：通過安全檢查。';
          }
        }
      } else if (!message && node === 'agent' && phase === 'planning') {
        // v1 舊版 Agent 事件
        if (stage === 'agent_planning_start') {
          message = 'Agent 開始規劃，準備決定是否調用工具（檢索文件 / 表單下載）。';
          details.expanded = true;
        } else if (stage === 'agent_planning_done') {
          message = 'Agent 規劃完成，已整理出一份整合工具結果的摘要。';
        } else if (stage === 'agent_planning_error') {
          const err = (event.error as string | undefined) ?? '未知錯誤';
          message = `Agent 規劃過程發生錯誤：${err}`;
          details.isStreaming = false;
        } else if (stage === 'agent_tool_call') {
          const toolName = (event.tool_name as string | undefined) ?? '(未知工具)';
          const args =
            event.tool_args !== undefined
              ? JSON.stringify(event.tool_args).slice(0, 200)
              : '';
          message = `Agent 工具呼叫：${toolName}${
            args ? `，參數：${args}` : ''
          }`;
          details.expanded = true;
        } else if (stage === 'agent_tool_result') {
          const toolName = (event.tool_name as string | undefined) ?? '(未知工具)';
          const outputRaw = event.tool_output as string | undefined;
          message = `Agent 工具結果：${toolName}${
            outputRaw ? ` → ${outputRaw.slice(0, 300)}` : ''
          }`;
          details.expanded = true;
        }
      }
      // ==========  Unified Agent 事件 ==========
      else if (!message && node === 'unified_agent') {
        if (stage === 'unified_agent_start') {
          message = '🚀 Unified Agent 開始處理...';
          details.expanded = true;
        } else if (stage === 'unified_agent_analyzing') {
          message = '🔍 正在分析問題意圖與決定工具...';
          details.expanded = true;
        } else if (stage === 'unified_agent_tool_call') {
          const toolName = event.tool_name ?? '(未知工具)';
          const args = event.tool_args
            ? JSON.stringify(event.tool_args).slice(0, 200)
            : '';
          message = `🔧 呼叫工具：${toolName}${args ? `（${args}）` : ''}`;
          details.expanded = true;
        } else if (stage === 'unified_agent_tool_result') {
          const toolName = event.tool_name ?? '(未知工具)';
          const outputRaw = event.tool_output;
          message = `📋 工具結果：${toolName}${
            outputRaw ? ` → ${outputRaw.slice(0, 200)}...` : ''
          }`;
          details.expanded = true;
        } else if (stage === 'unified_agent_generating') {
          const intent = event.intent ?? '';
          const isOutOfScope = event.is_out_of_scope ?? false;
          const usedTools = event.used_tools ?? [];
          
          let toolsInfo = '';
          if (usedTools.length > 0) {
            toolsInfo = `，使用工具：${usedTools.join(', ')}`;
          }
          
          if (isOutOfScope) {
            message = `✍️ 準備回應（超出服務範圍）${toolsInfo}`;
          } else {
            message = `✍️ 準備生成回答（意圖：${intent || '一般問題'}）${toolsInfo}`;
          }
          details.expanded = true;
        } else if (stage === 'unified_agent_done') {
          const loops = event.loops ?? 1;
          const usedTools = event.used_tools ?? [];
          const intent = event.intent ?? '';
          message = `✅ Unified Agent 完成（迴圈：${loops}，意圖：${intent}，工具：${usedTools.length > 0 ? usedTools.join(', ') : '無'}）`;
          details.isStreaming = false;
        } else if (stage === 'unified_agent_error') {
          const err = (event.error as string | undefined) ?? '未知錯誤';
          message = `❌ Unified Agent 錯誤：${err}`;
          details.isStreaming = false;
        } else if (stage === 'reasoning_start') {
          message = '🧠 開始進行深度 reasoning...';
          this.isThinking.set(true);
          details.expanded = true;
        } else if (stage === 'reasoning_end') {
          message = '🧠 Reasoning 階段結束。';
          this.isThinking.set(false);
        } else if (stage === 'answer_start') {
          message = '💬 開始串流最終回答...';
        } else if (stage === 'answer_end') {
          message = '💬 回答完成。';
          details.expanded = false;
        }
      }
      // ========== v1 舊版 model 節點事件 ==========
      else if (
        !message &&
        // 支援新的節點名稱：rag_model 和 fallback_model
        (node === 'model' || node === 'rag_model' || node === 'fallback_model') &&
        phase === 'generation'
      ) {
        if (stage === 'reasoning_start') {
          message = '模型開始進行深度 reasoning。';
          this.isThinking.set(true);
          details.expanded = true;
        } else if (stage === 'reasoning_end') {
          message = 'reasoning 階段結束。';
          this.isThinking.set(false);
          // reasoning 結束時，自動收合思考細節面板
          details.expanded = false;
        } else if (stage === 'answer_start') {
          message = '開始串流最終回答內容。';
        } else if (stage === 'answer_end') {
          message = '最終回答已完成。';
          details.isStreaming = false;
          details.expanded = false;
        }
      }

      if (message) {
        details.statusMessages = [...details.statusMessages, message];
      }
    });
  }

  toggleDetails(messageId: number): void {
    this.updateAssistantDetails(messageId, (details) => {
      details.expanded = !details.expanded;
    });
  }

  private updateAssistantDetails(
    assistantMessageId: number,
    updater: (details: AssistantDetails) => void
  ): void {
    this.messages.update((list) =>
      list.map((m) => {
        if (m.id !== assistantMessageId || m.role !== 'assistant') {
          return m;
        }

        const baseDetails: AssistantDetails = m.details ?? {
          planning: '',
          retrieval: '',
          reasoning: '',
          statusMessages: [],
          meta: undefined,
          tokenUsage: undefined,
          durationMs: undefined,
          expanded: false,
          isStreaming: false
        };

        const copy = { ...baseDetails };
        updater(copy);
        return { ...m, details: copy };
      })
    );
  }

  private finalizeStreamDuration(assistantMessageId: number): void {
    const timer = this.streamTimers.get(assistantMessageId);
    if (!timer) return;

    const endAt = performance.now();
    const startAt = timer.firstEvent ?? timer.requestStart;
    const durationMs = Math.max(0, endAt - startAt);

    this.updateAssistantDetails(assistantMessageId, (details) => {
      details.durationMs = durationMs;
    });

    this.streamTimers.delete(assistantMessageId);
  }

  private updateConversationSummary(summary: unknown): void {
    if (!this.conversationSummaryEnabled) {
      return;
    }
    if (typeof summary === 'string') {
      this.conversationSummary = summary;
      return;
    }
    if (summary === null) {
      this.conversationSummary = '';
    }
  }

  private extractUsageStats(
    usageRaw: Record<string, unknown> | undefined
  ): {
    totalTokens?: number;
    inputTokens?: number;
    outputTokens?: number;
  } {
    const pick = (...keys: string[]) => {
      if (!usageRaw) return undefined;
      for (const key of keys) {
        const value = this.toNumber(usageRaw[key]);
        if (value !== undefined) {
          return value;
        }
      }
      return undefined;
    };

    return {
      totalTokens: pick('total_tokens', 'totalTokens'),
      inputTokens: pick(
        'input_tokens',
        'prompt_tokens',
        'inputTokens',
        'promptTokens'
      ),
      outputTokens: pick(
        'output_tokens',
        'completion_tokens',
        'outputTokens',
        'completionTokens'
      )
    };
  }

  private toNumber(value: unknown): number | undefined {
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
    if (typeof value === 'string') {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
    return undefined;
  }

  // ========== Markdown 預處理 ==========

  /**
   * 修正 CJK（中日韓）字元的 markdown 粗體/斜體語法
   * marked.js 遵循 CommonMark 規範，對中文字元緊鄰 ** 或 * 的情況解析有問題
   * 此函數在 ** 和中文字元之間插入零寬空格以修正此問題
   */
  normalizeMarkdown(content: string): string {
    if (!content) return content;

    // 在 ** 後緊接中文字元時，插入零寬空格
    // 在中文字元後緊接 ** 時，插入零寬空格
    // 使用 Unicode 範圍匹配 CJK 字元
    const cjkRange = '\\u4e00-\\u9fff\\u3400-\\u4dbf\\uf900-\\ufaff\\u3000-\\u303f\\uff00-\\uffef';

    // 處理粗體 **text**
    let result = content
      // **後接CJK：**中 → ** 中（加零寬空格）
      .replace(new RegExp(`(\\*\\*)([${cjkRange}])`, 'g'), '$1\u200B$2')
      // CJK後接**：中** → 中 **（加零寬空格）
      .replace(new RegExp(`([${cjkRange}])(\\*\\*)`, 'g'), '$1\u200B$2');

    // 處理斜體 *text*（單星號，但要避免影響粗體）
    // 這裡只處理單獨的 * 而非 **
    result = result
      .replace(new RegExp(`(?<!\\*)(\\*)(?!\\*)([${cjkRange}])`, 'g'), '$1\u200B$2')
      .replace(new RegExp(`([${cjkRange}])(?<!\\*)(\\*)(?!\\*)`, 'g'), '$1\u200B$2');

    return result;
  }

  // ========== 工具列功能方法 ==========

  /**
   * 複製回答內容到剪貼簿
   */
  async onCopyAnswer(messageId: number): Promise<void> {
    const message = this.messages().find((m) => m.id === messageId);
    if (!message?.content) return;

    try {
      await navigator.clipboard.writeText(message.content);
      this.copiedMessageId = messageId;

      // 2 秒後重置圖示
      setTimeout(() => {
        if (this.copiedMessageId === messageId) {
          this.copiedMessageId = null;
        }
      }, 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }

  /**
   * 重新生成回答
   * 移除當前助手訊息，重新送出對話歷史
   */
  async onRegenerate(messageId: number): Promise<void> {
    if (this.isLoading()) return;

    const msgs = this.messages();
    const targetIndex = msgs.findIndex((m) => m.id === messageId);
    if (targetIndex === -1) return;

    // 找到這個助手訊息對應的使用者問題（前一則訊息）
    const userMessage = msgs[targetIndex - 1];
    if (!userMessage || userMessage.role !== 'user') return;

    // 移除當前助手訊息
    this.messages.update((list) => list.filter((m) => m.id !== messageId));

    // 取得該問題之前的對話歷史（不包含被移除的助手訊息）
    const historyBeforeQuestion = msgs.slice(0, targetIndex - 1);

    // 重新發送請求
    this.errorMessage.set(null);
    this.isLoading.set(true);
    this.isThinking.set(false);

    const newAssistantMessageId = this.addMessage('assistant', '');

    let safeTopK = Number(this.topK) || 3;
    if (safeTopK < 1) safeTopK = 1;
    if (safeTopK > 10) safeTopK = 10;

    const payload: QuestionPayload = {
      question: userMessage.content,
      conversation_history:
        historyBeforeQuestion.length > 0
          ? historyBeforeQuestion.map((m) => ({ role: m.role, content: m.content }))
          : null,
      top_k: safeTopK,
      llm_config: {
        model: this.selectedModel || undefined,
        reasoning_effort: this.selectedReasoningEffort,
        reasoning_summary: 'auto'
      },
      enable_conversation_summary: this.conversationSummaryEnabled,
      conversation_summary: this.conversationSummaryEnabled
        ? this.conversationSummary.trim() || undefined
        : undefined
    };

    try {
      await this.streamAnswer(payload, newAssistantMessageId);
    } catch (error) {
      console.error('regenerate error', error);
      this.errorMessage.set('重新生成失敗，請稍後再試。');
    } finally {
      this.isLoading.set(false);
      this.isThinking.set(false);
    }
  }

  /**
   * 處理回饋按鈕點擊
   */
  onFeedback(messageId: number, score: 'up' | 'down'): void {
    const message = this.messages().find((m) => m.id === messageId);
    if (!message?.details?.traceId) return;

    if (score === 'down') {
      // 倒讚：開啟 popup 讓用戶填寫原因
      this.pendingFeedbackMessageId = messageId;
      this.feedbackComment = '';
      this.showFeedbackPopup = true;
      return;
    }

    // 讚：直接提交
    void this.submitFeedback(messageId, score, null);
  }

  /**
   * 提交回饋到後端
   */
  async submitFeedback(
    messageId: number,
    score: 'up' | 'down',
    comment: string | null
  ): Promise<void> {
    const message = this.messages().find((m) => m.id === messageId);
    if (!message?.details?.traceId) return;

    this.feedbackLoading.set(true);
    try {
      const response = await fetch('/api/v1/rag/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          trace_id: message.details.traceId,
          score,
          comment
        })
      });

      if (response.ok) {
        this.updateAssistantDetails(messageId, (details) => {
          details.userFeedback = score;
        });
      }
    } catch (err) {
      console.error('Failed to submit feedback:', err);
    } finally {
      this.feedbackLoading.set(false);
      this.showFeedbackPopup = false;
      this.pendingFeedbackMessageId = null;
      this.feedbackComment = '';
    }
  }

  /**
   * 確認倒讚並提交
   */
  onConfirmDownvote(): void {
    if (this.pendingFeedbackMessageId === null) return;
    void this.submitFeedback(
      this.pendingFeedbackMessageId,
      'down',
      this.feedbackComment.trim() || null
    );
  }

  /**
   * 取消倒讚彈窗
   */
  onCancelFeedbackPopup(): void {
    this.showFeedbackPopup = false;
    this.pendingFeedbackMessageId = null;
    this.feedbackComment = '';
  }

  private scrollToBottomIfNearBottom(): void {
    if (!this.messagesContainer) return;
    const el = this.messagesContainer.nativeElement;
    // 等待本輪變更套用到 DOM 後再捲動
    setTimeout(() => {
      try {
        const distanceToBottom =
          el.scrollHeight - (el.scrollTop + el.clientHeight);
        const threshold = 40; // 距離底部 40px 內才自動捲動
        if (distanceToBottom <= threshold) {
          el.scrollTop = el.scrollHeight;
        }
      } catch {
        // ignore
      }
    }, 0);
  }

  private scrollToMessageTop(messageId: number): void {
    // 將視窗捲動到指定訊息區塊的頂端位置
    setTimeout(() => {
      const el = document.getElementById(`message-${messageId}`);
      if (!el) return;
      try {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      } catch {
        // ignore
      }
    }, 0);
  }
}


