<script>
  /**
   * FeedbackPopup - 倒讚回饋彈窗
   *
   * 當用戶點擊倒讚時顯示，讓用戶可以填寫回饋原因。
   */
  import { createEventDispatcher } from 'svelte';

  /** @type {boolean} 是否顯示彈窗 */
  export let isOpen = false;

  /** @type {boolean} 是否正在載入 */
  export let isLoading = false;

  let comment = '';
  const dispatch = createEventDispatcher();

  /**
   * 確認送出
   */
  function handleConfirm() {
    dispatch('confirm', { comment: comment.trim() || null });
  }

  /**
   * 取消
   */
  function handleCancel() {
    comment = '';
    dispatch('cancel');
  }

  /**
   * 點擊背景關閉
   * @param {MouseEvent} e
   */
  function handleBackdropClick(e) {
    if (e.target === e.currentTarget) {
      handleCancel();
    }
  }

  /**
   * 按 Escape 關閉
   * @param {KeyboardEvent} e
   */
  function handleKeydown(e) {
    if (e.key === 'Escape' && isOpen) {
      handleCancel();
    }
  }
</script>

<svelte:window on:keydown={handleKeydown} />

{#if isOpen}
  <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-noninteractive-element-interactions -->
  <div
    class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
    on:click={handleBackdropClick}
    role="dialog"
    aria-modal="true"
    aria-labelledby="feedback-title"
  >
    <div class="w-full max-w-md mx-4 rounded-xl border border-slate-300 bg-white p-5 shadow-2xl">
      <h3
        id="feedback-title"
        class="text-base font-semibold text-slate-800 mb-3"
      >
        請告訴我們哪裡需要改進
      </h3>
      <p class="text-sm text-slate-500 mb-4">
        您的意見將幫助我們提升回答品質。（選填）
      </p>
      <textarea
        class="block w-full resize-none rounded-lg border border-slate-300 bg-slate-50 px-3 py-2 text-sm text-slate-800 outline-none transition placeholder:text-slate-400 focus:border-primary focus:ring-1 focus:ring-primary"
        rows="4"
        bind:value={comment}
        placeholder="例如：回答不夠完整、資訊有誤、格式難以閱讀..."
        disabled={isLoading}
      ></textarea>
      <div class="mt-4 flex justify-end gap-2">
        <button
          type="button"
          class="inline-flex items-center justify-center rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-50 transition disabled:opacity-50 disabled:cursor-not-allowed"
          on:click={handleCancel}
          disabled={isLoading}
        >
          取消
        </button>
        <button
          type="button"
          class="inline-flex items-center justify-center rounded-lg bg-red-500 px-4 py-2 text-sm font-medium text-white hover:bg-red-600 transition disabled:opacity-50 disabled:cursor-not-allowed"
          on:click={handleConfirm}
          disabled={isLoading}
        >
          {#if isLoading}
            <span class="mr-2 h-3 w-3 animate-spin rounded-full border-2 border-white border-t-transparent"></span>
            送出中...
          {:else}
            確認送出
          {/if}
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .focus\:border-primary:focus {
    border-color: var(--widget-primary, #E84967);
  }
  .focus\:ring-primary:focus {
    --tw-ring-color: var(--widget-primary, #E84967);
  }
</style>
