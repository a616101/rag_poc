<script>
  import { config } from '../stores/config.js';
  import { isLoading } from '../stores/messages.js';

  export let onSend = (/** @type {string} */ text) => {};

  // 取得展開狀態
  $: isExpanded = $config.isExpanded;

  let inputValue = '';
  let textareaEl;

  $: canSend = !$isLoading && inputValue.trim().length > 0;

  function handleSubmit() {
    const text = inputValue.trim();
    if (!text || $isLoading) return;

    onSend(text);
    inputValue = '';

    // 重設 textarea 高度
    if (textareaEl) {
      textareaEl.style.height = 'auto';
    }
  }

  function handleKeydown(event) {
    // 處理 IME 組字中的情況
    if (event.isComposing || event.keyCode === 229) {
      return;
    }

    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSubmit();
    }
  }

  function handleInput() {
    // 自動調整高度
    if (textareaEl) {
      textareaEl.style.height = 'auto';
      textareaEl.style.height = Math.min(textareaEl.scrollHeight, 120) + 'px';
    }
  }
</script>

<div class="border-t border-gray-200 px-4 py-3 bg-white {isExpanded ? '' : 'rounded-b-2xl'}">
  <form on:submit|preventDefault={handleSubmit} class="flex items-end gap-2">
    <div class="flex-1">
      <textarea
        bind:this={textareaEl}
        bind:value={inputValue}
        on:keydown={handleKeydown}
        on:input={handleInput}
        placeholder={$config.inputPlaceholder}
        disabled={$isLoading}
        rows="1"
        class="w-full resize-none rounded-xl border border-gray-300 bg-gray-50
               px-4 py-2.5 text-sm text-gray-800
               placeholder:text-gray-400
               focus:border-primary focus:bg-white focus:outline-none focus:ring-1 focus:ring-primary
               disabled:opacity-60 disabled:cursor-not-allowed
               transition-colors"
      ></textarea>
    </div>

    <button
      type="submit"
      disabled={!canSend}
      class="flex-shrink-0 w-10 h-10 rounded-full bg-primary text-white
             flex items-center justify-center
             hover:bg-primary-hover
             disabled:opacity-50 disabled:cursor-not-allowed
             transition-colors"
      aria-label="發送"
    >
      {#if $isLoading}
        <svg class="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
      {:else}
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>
        </svg>
      {/if}
    </button>
  </form>
</div>
