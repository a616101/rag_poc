import { CommonModule } from '@angular/common';
import {
  Component,
  computed,
  inject,
  ViewChild,
  ElementRef,
  HostListener
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';
import { MarkdownModule } from 'ngx-markdown';
import { MessageService, ApiService, FeedbackService } from '../../services';
import { QuestionPayload } from '../../models';

@Component({
  selector: 'app-chat-page',
  standalone: true,
  imports: [CommonModule, FormsModule, MarkdownModule],
  templateUrl: './chat-page.component.html'
})
export class ChatPageComponent {
  private readonly route = inject(ActivatedRoute);
  readonly messageService = inject(MessageService);
  private readonly apiService = inject(ApiService);
  readonly feedbackService = inject(FeedbackService);

  @ViewChild('messagesContainer')
  private messagesContainer?: ElementRef<HTMLDivElement>;

  readonly pageTitle: string =
    this.route.snapshot.data['title'] ?? 'RAG 串流聊天';

  readonly apiPath: string =
    this.route.snapshot.data['apiPath'] ?? '/api/v1/rag/ask/stream';

  inputValue = '';

  get inputSignal(): string {
    return this.inputValue;
  }

  set inputSignal(value: string) {
    this.inputValue = value;
  }

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

  selectedModel = '';
  selectedReasoningEffort: 'low' | 'medium' | 'high' = 'low';
  topK = 3;
  conversationSummaryEnabled = false;

  readonly canSend = computed(
    () => !this.messageService.isLoading() && this.inputValue.trim().length > 0
  );

  // Toolbar state
  copiedMessageId: number | null = null;
  showFeedbackPopup = false;
  pendingFeedbackMessageId: number | null = null;
  feedbackComment = '';

  // Lightbox state
  lightboxImageSrc: string | null = null;

  onConversationSummaryToggle(enabled: boolean): void {
    this.conversationSummaryEnabled = enabled;
    this.messageService.setSummaryEnabled(enabled);
  }

  // ========== Lightbox ==========

  @HostListener('click', ['$event'])
  onDocumentClick(event: MouseEvent): void {
    const target = event.target as HTMLElement;
    if (target.tagName === 'IMG' && target.closest('.markdown-body')) {
      event.preventDefault();
      event.stopPropagation();
      const imgSrc = (target as HTMLImageElement).src;
      if (imgSrc) {
        this.openLightbox(imgSrc);
      }
    }
  }

  openLightbox(src: string): void {
    this.lightboxImageSrc = src;
  }

  closeLightbox(): void {
    this.lightboxImageSrc = null;
  }

  @HostListener('document:keydown.escape')
  onEscapeKey(): void {
    if (this.lightboxImageSrc) {
      this.closeLightbox();
    }
  }

  // ========== Input handling ==========

  onTextareaKeydown(event: KeyboardEvent): void {
    const isEnter = event.key === 'Enter';
    const isShift = event.shiftKey;
    const isComposing =
      (event as any).isComposing === true ||
      (event.target as any)?.isComposing === true ||
      (event as any).keyCode === 229;

    if (!isEnter || isComposing) return;
    if (isShift) return; // Shift+Enter = newline

    event.preventDefault();
    void this.onSubmit();
  }

  async onSubmit(event?: SubmitEvent): Promise<void> {
    event?.preventDefault();

    const question = this.inputValue.trim();
    if (!question || this.messageService.isLoading()) return;

    this.messageService.errorMessage.set(null);
    this.messageService.isLoading.set(true);
    this.messageService.isThinking.set(false);

    const history = this.messageService.buildConversationHistory();
    this.inputValue = '';

    this.messageService.addMessage('user', question);
    const assistantMessageId = this.messageService.addMessage('assistant', '');

    let safeTopK = Number(this.topK) || 3;
    safeTopK = Math.max(1, Math.min(10, safeTopK));

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
        ? this.messageService.getConversationSummary().trim() || undefined
        : undefined
    };

    try {
      await this.apiService.streamAnswer(this.apiPath, payload, assistantMessageId, {
        onThinkingChange: (thinking) => this.messageService.isThinking.set(thinking),
        onScrollNeeded: () => this.scrollToBottomIfNearBottom(),
        onComplete: (id) => this.scrollToMessageTop(id)
      });
    } catch (error) {
      console.error('stream error', error);
      this.messageService.errorMessage.set('請求失敗，請稍後再試。');
    } finally {
      this.messageService.isLoading.set(false);
      this.messageService.isThinking.set(false);
    }
  }

  toggleDetails(messageId: number): void {
    this.messageService.updateAssistantDetails(messageId, (details) => {
      details.expanded = !details.expanded;
    });
  }

  // ========== Markdown preprocessing ==========

  normalizeMarkdown(content: string): string {
    if (!content) return content;

    const cjkRange =
      '\\u4e00-\\u9fff\\u3400-\\u4dbf\\uf900-\\ufaff\\u3000-\\u303f\\uff00-\\uffef';

    let result = content
      .replace(new RegExp(`(\\*\\*)([${cjkRange}])`, 'g'), '$1\u200B$2')
      .replace(new RegExp(`([${cjkRange}])(\\*\\*)`, 'g'), '$1\u200B$2');

    result = result
      .replace(
        new RegExp(`(?<!\\*)(\\*)(?!\\*)([${cjkRange}])`, 'g'),
        '$1\u200B$2'
      )
      .replace(
        new RegExp(`([${cjkRange}])(?<!\\*)(\\*)(?!\\*)`, 'g'),
        '$1\u200B$2'
      );

    return result;
  }

  // ========== Toolbar actions ==========

  async onCopyAnswer(messageId: number): Promise<void> {
    const message = this.messageService.findMessage(messageId);
    if (!message?.content) return;

    try {
      await navigator.clipboard.writeText(message.content);
      this.copiedMessageId = messageId;

      setTimeout(() => {
        if (this.copiedMessageId === messageId) {
          this.copiedMessageId = null;
        }
      }, 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }

  async onRegenerate(messageId: number): Promise<void> {
    if (this.messageService.isLoading()) return;

    const msgs = this.messageService.messages();
    const targetIndex = msgs.findIndex((m) => m.id === messageId);
    if (targetIndex === -1) return;

    const userMessage = msgs[targetIndex - 1];
    if (!userMessage || userMessage.role !== 'user') return;

    this.messageService.removeMessage(messageId);
    const historyBeforeQuestion = msgs.slice(0, targetIndex - 1);

    this.messageService.errorMessage.set(null);
    this.messageService.isLoading.set(true);
    this.messageService.isThinking.set(false);

    const newAssistantMessageId = this.messageService.addMessage('assistant', '');

    let safeTopK = Number(this.topK) || 3;
    safeTopK = Math.max(1, Math.min(10, safeTopK));

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
        ? this.messageService.getConversationSummary().trim() || undefined
        : undefined
    };

    try {
      await this.apiService.streamAnswer(
        this.apiPath,
        payload,
        newAssistantMessageId,
        {
          onThinkingChange: (thinking) =>
            this.messageService.isThinking.set(thinking),
          onScrollNeeded: () => this.scrollToBottomIfNearBottom(),
          onComplete: (id) => this.scrollToMessageTop(id)
        }
      );
    } catch (error) {
      console.error('regenerate error', error);
      this.messageService.errorMessage.set('重新生成失敗，請稍後再試。');
    } finally {
      this.messageService.isLoading.set(false);
      this.messageService.isThinking.set(false);
    }
  }

  // ========== Feedback ==========

  onFeedback(messageId: number, score: 'up' | 'down'): void {
    const message = this.messageService.findMessage(messageId);
    if (!message?.details?.traceId) return;

    if (score === 'down') {
      this.pendingFeedbackMessageId = messageId;
      this.feedbackComment = '';
      this.showFeedbackPopup = true;
      return;
    }

    void this.submitFeedback(messageId, score, null);
  }

  async submitFeedback(
    messageId: number,
    score: 'up' | 'down',
    comment: string | null
  ): Promise<void> {
    const message = this.messageService.findMessage(messageId);
    if (!message?.details?.traceId) return;

    const basePath = this.apiPath.replace(/\/ask\/stream(_chat)?$/, '');
    const result = await this.feedbackService.submitFeedback(basePath, {
      trace_id: message.details.traceId,
      score,
      comment
    });

    if (result.success) {
      this.messageService.updateAssistantDetails(messageId, (details) => {
        details.userFeedback = score;
      });
    }

    this.showFeedbackPopup = false;
    this.pendingFeedbackMessageId = null;
    this.feedbackComment = '';
  }

  onConfirmDownvote(): void {
    if (this.pendingFeedbackMessageId === null) return;
    void this.submitFeedback(
      this.pendingFeedbackMessageId,
      'down',
      this.feedbackComment.trim() || null
    );
  }

  onCancelFeedbackPopup(): void {
    this.showFeedbackPopup = false;
    this.pendingFeedbackMessageId = null;
    this.feedbackComment = '';
  }

  // ========== Scrolling ==========

  private scrollToBottomIfNearBottom(): void {
    if (!this.messagesContainer) return;
    const el = this.messagesContainer.nativeElement;
    setTimeout(() => {
      try {
        const distanceToBottom =
          el.scrollHeight - (el.scrollTop + el.clientHeight);
        const threshold = 40;
        if (distanceToBottom <= threshold) {
          el.scrollTop = el.scrollHeight;
        }
      } catch {
        // ignore
      }
    }, 0);
  }

  private scrollToMessageTop(messageId: number): void {
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
