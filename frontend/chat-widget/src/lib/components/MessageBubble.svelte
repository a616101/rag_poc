<script>
  import { createEventDispatcher } from 'svelte';
  import { renderMarkdown } from '../services/markdown.js';
  import TypingIndicator from './TypingIndicator.svelte';

  /**
   * @type {{
   *   id: string,
   *   role: 'user' | 'assistant',
   *   content: string,
   *   isStreaming: boolean,
   *   statusText?: string,
   *   sources?: Array<{index: number, chunk_id: string, content: string, source_doc: string, relevance_score: number}>,
   *   traceId?: string,
   *   userFeedback?: 'up' | 'down' | null
   * }}
   */
  export let message;

  /** @type {boolean} 回饋是否正在載入 */
  export let feedbackLoading = false;

  const dispatch = createEventDispatcher();

  $: isUser = message.role === 'user';
  $: isAssistant = message.role === 'assistant';
  // 顯示狀態提示：串流中、沒有內容、且有狀態文字
  $: showStatus = isAssistant && message.isStreaming && !message.content && message.statusText;
  // 顯示打字動畫：串流中、沒有內容、沒有狀態文字
  $: showTyping = isAssistant && message.isStreaming && !message.content && !message.statusText;
  $: htmlContent = isAssistant && message.content ? renderMarkdown(message.content) : '';
  // 參考來源：只在串流結束後且有 sources 時顯示
  $: hasSources = isAssistant && !message.isStreaming && message.sources && message.sources.length > 0;
  // 回饋按鈕：只在助手訊息、串流結束、有內容、有 traceId 時顯示
  $: showFeedbackButtons = isAssistant && !message.isStreaming && message.content && message.traceId;

  // 展開/收合狀態
  let sourcesExpanded = false;

  /**
   * 處理回饋按鈕點擊
   * @param {'up' | 'down'} score
   */
  function handleFeedback(score) {
    dispatch('feedback', { messageId: message.id, score });
  }
</script>

<div class="flex flex-col {isUser ? 'items-end' : 'items-start'}">
  <div
    class="max-w-[85%] rounded-2xl px-4 py-2.5 shadow-sm
           {isUser ? 'bg-primary text-white rounded-br-md' : 'bg-gray-100 text-gray-800 rounded-bl-md'}"
  >
    {#if isUser}
      <!-- 使用者訊息：純文字 -->
      <p class="text-sm whitespace-pre-wrap leading-relaxed">{message.content}</p>
    {:else if showStatus}
      <!-- 簡潔的單行狀態顯示 -->
      <div class="status-container">
        <div class="status-spinner"></div>
        <span class="status-text">{message.statusText}</span>
      </div>
    {:else if showTyping}
      <!-- 助手輸入中（無狀態時的 fallback） -->
      <TypingIndicator />
    {:else}
      <!-- 助手訊息：Markdown 渲染 -->
      <div class="markdown-body">
        {@html htmlContent}
      </div>
      {#if message.isStreaming}
        <span class="inline-block w-1.5 h-4 bg-primary ml-0.5 animate-pulse rounded-sm"></span>
      {/if}
    {/if}
  </div>

  <!-- 參考來源區塊 -->
  {#if hasSources}
    <div class="sources-container mt-2 max-w-[85%]">
      <button
        class="sources-toggle"
        on:click={() => sourcesExpanded = !sourcesExpanded}
      >
        <svg class="sources-icon" viewBox="0 0 20 20" fill="currentColor">
          <path fill-rule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clip-rule="evenodd" />
        </svg>
        <span>參考來源 ({message.sources.length})</span>
        <svg class="chevron-icon {sourcesExpanded ? 'rotate-180' : ''}" viewBox="0 0 20 20" fill="currentColor">
          <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
        </svg>
      </button>

      {#if sourcesExpanded}
        <div class="sources-list">
          {#each message.sources as source, i}
            <div class="source-item">
              <div class="source-header">
                <span class="source-index">[{source.index}]</span>
                <span class="source-doc">{source.source_doc || '未知來源'}</span>
                <span class="source-score">{(source.relevance_score * 100).toFixed(0)}%</span>
              </div>
              <p class="source-content">{source.content}</p>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  {/if}

  <!-- 回饋按鈕區塊 -->
  {#if showFeedbackButtons}
    <div class="feedback-buttons mt-2 max-w-[85%]">
      <div class="flex items-center gap-1">
        <!-- 讚按鈕 -->
        <button
          type="button"
          class="feedback-btn"
          class:feedback-btn-active-up={message.userFeedback === 'up'}
          on:click={() => handleFeedback('up')}
          disabled={feedbackLoading}
          title="這個回答很有幫助"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5"/>
          </svg>
        </button>

        <!-- 倒讚按鈕 -->
        <button
          type="button"
          class="feedback-btn"
          class:feedback-btn-active-down={message.userFeedback === 'down'}
          on:click={() => handleFeedback('down')}
          disabled={feedbackLoading}
          title="這個回答需要改進"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018a2 2 0 01.485.06l3.76.94m-7 10v5a2 2 0 002 2h.096c.5 0 .905-.405.905-.904 0-.715.211-1.413.608-2.008L17 13V4m-7 10h2m4-10h2a2 2 0 012 2v6a2 2 0 01-2 2h-2.5"/>
          </svg>
        </button>
      </div>
    </div>
  {/if}
</div>

<style>
  /* ========== 簡潔狀態顯示 ========== */
  .status-container {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 0;
  }

  .status-spinner {
    width: 16px;
    height: 16px;
    border: 2px solid rgba(99, 102, 241, 0.2);
    border-top-color: var(--widget-primary, #6366f1);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    flex-shrink: 0;
  }

  .status-text {
    font-size: 0.875rem;
    color: #6b7280;
    animation: fadeInOut 1.5s ease-in-out infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  @keyframes fadeInOut {
    0%, 100% {
      opacity: 0.6;
    }
    50% {
      opacity: 1;
    }
  }

  /* ========== 參考來源區塊 ========== */
  .sources-container {
    width: 100%;
  }

  .sources-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.813rem;
    color: #64748b;
    transition: all 0.2s ease;
    width: 100%;
  }

  .sources-toggle:hover {
    background: #f1f5f9;
    border-color: #cbd5e1;
    color: #475569;
  }

  .sources-icon {
    width: 14px;
    height: 14px;
    flex-shrink: 0;
  }

  .chevron-icon {
    width: 16px;
    height: 16px;
    margin-left: auto;
    transition: transform 0.2s ease;
  }

  .chevron-icon.rotate-180 {
    transform: rotate(180deg);
  }

  .sources-list {
    margin-top: 8px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .source-item {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 0.813rem;
  }

  .source-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
  }

  .source-index {
    font-weight: 600;
    color: var(--widget-primary, #6366f1);
    font-size: 0.75rem;
  }

  .source-doc {
    color: #334155;
    font-weight: 500;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .source-score {
    font-size: 0.688rem;
    padding: 2px 6px;
    background: #ecfdf5;
    color: #059669;
    border-radius: 4px;
    font-weight: 500;
  }

  .source-content {
    color: #64748b;
    font-size: 0.75rem;
    line-height: 1.5;
    margin: 0;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  /* ========== 回饋按鈕 ========== */
  .feedback-buttons {
    display: flex;
    align-items: center;
  }

  .feedback-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 6px;
    color: #94a3b8;
    background: transparent;
    border: none;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .feedback-btn:hover:not(:disabled) {
    color: #64748b;
    background: #f1f5f9;
  }

  .feedback-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .feedback-btn-active-up {
    color: #ffffff;
    background: #10b981;
  }

  .feedback-btn-active-up:hover:not(:disabled) {
    background: #059669;
    color: #ffffff;
  }

  .feedback-btn-active-down {
    color: #ffffff;
    background: #ef4444;
  }

  .feedback-btn-active-down:hover:not(:disabled) {
    background: #dc2626;
    color: #ffffff;
  }
</style>
