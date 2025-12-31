<script>
  import { config } from '../stores/config.js';
  import { messages, clearMessages, isLoading } from '../stores/messages.js';
  import { requestClose, requestToggleExpand } from '../utils/messenger.js';

  export let onMinimize = () => {};

  // 從 config store 取得展開狀態
  $: isExpanded = $config.isExpanded;

  function handleClose() {
    requestClose();
    onMinimize();
  }

  function handleClearChat() {
    if ($isLoading) return; // 載入中不允許清空
    clearMessages();
  }

  function handleToggleExpand() {
    requestToggleExpand();
  }

  $: hasMessages = $messages.length > 0;

  // 計算 header 樣式
  $: headerStyle = `background-color: ${$config.headerBgColor || $config.primaryColor}; color: ${$config.headerTextColor || '#ffffff'};`;
</script>

<header
  class="{isExpanded ? '' : 'rounded-t-2xl'}"
  style={headerStyle}
>
  <div class="flex items-center justify-between px-4 py-3 {isExpanded ? 'max-w-[800px] mx-auto' : ''}">
  <div class="flex items-center gap-3">
    <!-- Logo / Avatar -->
    {#if $config.headerIcon}
      <div class="w-10 h-10 rounded-full overflow-hidden bg-white flex items-center justify-center">
        <img src={$config.headerIcon} alt="" class="w-full h-full object-cover" />
      </div>
    {:else}
      <div class="w-9 h-9 rounded-full bg-white/20 flex items-center justify-center">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>
        </svg>
      </div>
    {/if}
    <div>
      <h1 class="font-semibold text-sm leading-tight">{$config.headerTitle}</h1>
      <p class="text-xs opacity-80">{$config.headerSubtitle || '在線服務中'}</p>
    </div>
  </div>

  <div class="flex items-center gap-1">
    <!-- 清空對話按鈕 -->
    {#if hasMessages}
      <button
        type="button"
        class="w-8 h-8 rounded-full hover:bg-white/20 flex items-center justify-center transition-colors
               {$isLoading ? 'opacity-50 cursor-not-allowed' : ''}"
        on:click={handleClearChat}
        disabled={$isLoading}
        aria-label="清空對話"
        title="開始新對話"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
        </svg>
      </button>
    {/if}

    <!-- 放大/縮小按鈕 -->
    <button
      type="button"
      class="w-8 h-8 rounded-full hover:bg-white/20 flex items-center justify-center transition-colors"
      on:click={handleToggleExpand}
      aria-label={isExpanded ? '縮小視窗' : '放大視窗'}
      title={isExpanded ? '縮小視窗' : '放大視窗'}
    >
      {#if isExpanded}
        <!-- 縮小圖示 -->
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M9 9V4.5M9 9H4.5M9 9L3.75 3.75M9 15v4.5M9 15H4.5M9 15l-5.25 5.25M15 9h4.5M15 9V4.5M15 9l5.25-5.25M15 15h4.5M15 15v4.5m0-4.5l5.25 5.25"/>
        </svg>
      {:else}
        <!-- 放大圖示 -->
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15"/>
        </svg>
      {/if}
    </button>

    <!-- 關閉按鈕 -->
    <button
      type="button"
      class="w-8 h-8 rounded-full hover:bg-white/20 flex items-center justify-center transition-colors"
      on:click={handleClose}
      aria-label="最小化"
    >
      <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
      </svg>
    </button>
  </div>
  </div>
</header>
