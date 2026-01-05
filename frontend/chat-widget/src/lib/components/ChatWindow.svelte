<script>
  import { onMount, onDestroy } from 'svelte';
  import Header from './Header.svelte';
  import MessageList from './MessageList.svelte';
  import InputArea from './InputArea.svelte';
  import FeedbackPopup from './FeedbackPopup.svelte';
  import {
    getConversationHistory,
    updateActivity,
    checkAndHandleSessionExpiry,
    getMessageTraceId,
    setMessageFeedback
  } from '../stores/messages.js';
  import { config } from '../stores/config.js';
  import { sendQuestion } from '../services/api.js';
  import { submitFeedback } from '../services/feedback.js';
  import { initMessenger } from '../utils/messenger.js';

  export let onMinimize = () => {};

  // 回饋相關狀態
  let feedbackLoading = false;
  let showFeedbackPopup = false;
  let pendingFeedbackMessageId = null;

  // 從 config 取得展開狀態
  $: isExpanded = $config.isExpanded;

  let sessionCheckInterval;

  // 初始化 postMessage 通訊
  onMount(() => {
    initMessenger();

    // 每分鐘檢查會話是否過期
    sessionCheckInterval = setInterval(() => {
      checkAndHandleSessionExpiry();
    }, 60 * 1000);

    // 監聽用戶活動
    window.addEventListener('focus', handleUserActivity);
  });

  onDestroy(() => {
    if (sessionCheckInterval) {
      clearInterval(sessionCheckInterval);
    }
    window.removeEventListener('focus', handleUserActivity);
  });

  function handleUserActivity() {
    updateActivity();
  }

  /**
   * 發送訊息
   * @param {string} text
   */
  function handleSend(text) {
    // 更新活動時間
    updateActivity();

    // 檢查會話過期（發送前再檢查一次）
    checkAndHandleSessionExpiry();

    // 使用滑動窗口取得對話歷史
    const history = getConversationHistory();

    sendQuestion(text, history);
  }

  /**
   * 處理快捷提問選擇
   * @param {string} text
   */
  function handleQuickReply(text) {
    handleSend(text);
  }

  /**
   * 處理來自 MessageBubble 的回饋事件
   * @param {{ detail: { messageId: string, score: 'up' | 'down' } }} event
   */
  async function handleFeedback(event) {
    const { messageId, score } = event.detail;

    // 如果是倒讚，顯示彈窗讓用戶填寫原因
    if (score === 'down') {
      pendingFeedbackMessageId = messageId;
      showFeedbackPopup = true;
      return;
    }

    // 讚直接送出
    await doSubmitFeedback(messageId, 'up', null);
  }

  /**
   * 確認送出倒讚回饋（從 popup）
   * @param {{ detail: { comment: string | null } }} event
   */
  async function handleConfirmDownvote(event) {
    const { comment } = event.detail;
    if (pendingFeedbackMessageId) {
      await doSubmitFeedback(pendingFeedbackMessageId, 'down', comment);
    }
    showFeedbackPopup = false;
    pendingFeedbackMessageId = null;
  }

  /**
   * 取消回饋彈窗
   */
  function handleCancelFeedback() {
    showFeedbackPopup = false;
    pendingFeedbackMessageId = null;
  }

  /**
   * 實際送出回饋到後端
   * @param {string} messageId
   * @param {'up' | 'down'} score
   * @param {string | null} comment
   */
  async function doSubmitFeedback(messageId, score, comment) {
    const traceId = getMessageTraceId(messageId);
    if (!traceId) {
      console.error('No trace ID found for message:', messageId);
      return;
    }

    feedbackLoading = true;
    try {
      const result = await submitFeedback(traceId, score, comment);
      if (result.success) {
        setMessageFeedback(messageId, score);
      } else {
        console.error('Feedback submission failed:', result.message);
      }
    } catch (error) {
      console.error('Feedback submission error:', error);
    } finally {
      feedbackLoading = false;
    }
  }
</script>

<div
  class="flex flex-col h-full bg-white overflow-hidden
         {isExpanded ? 'expanded-mode' : 'rounded-2xl shadow-2xl'}"
>
  <Header {onMinimize} />

  <div class="flex-1 flex flex-col overflow-hidden {isExpanded ? 'expanded-content' : ''}">
    <MessageList
      onQuickReply={handleQuickReply}
      {feedbackLoading}
      on:feedback={handleFeedback}
    />

    <InputArea onSend={handleSend} />
  </div>
</div>

<!-- 倒讚回饋彈窗 -->
<FeedbackPopup
  isOpen={showFeedbackPopup}
  isLoading={feedbackLoading}
  on:confirm={handleConfirmDownvote}
  on:cancel={handleCancelFeedback}
/>

<style>
  /* 全螢幕模式樣式 */
  .expanded-mode {
    background: #f5f5f5;
  }

  .expanded-content {
    max-width: 800px;
    width: 100%;
    height: 100%;
    margin: 0 auto;
    background: white;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
  }
</style>
