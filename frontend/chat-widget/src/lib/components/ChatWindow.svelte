<script>
  import { onMount, onDestroy } from 'svelte';
  import Header from './Header.svelte';
  import MessageList from './MessageList.svelte';
  import InputArea from './InputArea.svelte';
  import {
    getConversationHistory,
    updateActivity,
    checkAndHandleSessionExpiry
  } from '../stores/messages.js';
  import { config } from '../stores/config.js';
  import { sendQuestion } from '../services/api.js';
  import { initMessenger } from '../utils/messenger.js';

  export let onMinimize = () => {};

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
</script>

<div
  class="flex flex-col h-full bg-white overflow-hidden
         {isExpanded ? 'expanded-mode' : 'rounded-2xl shadow-2xl'}"
>
  <Header {onMinimize} />

  <div class="flex-1 flex flex-col overflow-hidden {isExpanded ? 'expanded-content' : ''}">
    <MessageList onQuickReply={handleQuickReply} />

    <InputArea onSend={handleSend} />
  </div>
</div>

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
