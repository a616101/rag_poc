<script>
  import { onMount, afterUpdate } from 'svelte';
  import { messages, showQuickReplies } from '../stores/messages.js';
  import { config } from '../stores/config.js';
  import MessageBubble from './MessageBubble.svelte';
  import QuickReplies from './QuickReplies.svelte';
  import ImageLightbox from './ImageLightbox.svelte';

  export let onQuickReply = (/** @type {string} */ text) => {};

  let container;
  let shouldAutoScroll = true;

  // Lightbox 狀態
  let lightboxOpen = false;
  let lightboxSrc = '';
  let lightboxAlt = '';

  // 追蹤是否接近底部
  function checkScrollPosition() {
    if (!container) return;
    const { scrollTop, scrollHeight, clientHeight } = container;
    const distanceToBottom = scrollHeight - scrollTop - clientHeight;
    shouldAutoScroll = distanceToBottom < 50;
  }

  // 捲動到底部
  function scrollToBottom() {
    if (!container || !shouldAutoScroll) return;
    container.scrollTop = container.scrollHeight;
  }

  // 處理圖片點擊
  function handleImageClick(event) {
    const target = event.target;
    if (target.tagName === 'IMG' && target.dataset.zoomable === 'true') {
      lightboxSrc = target.src;
      lightboxAlt = target.alt || 'Image';
      lightboxOpen = true;
    }
  }

  function closeLightbox() {
    lightboxOpen = false;
    lightboxSrc = '';
    lightboxAlt = '';
  }

  afterUpdate(() => {
    scrollToBottom();
  });

  onMount(() => {
    scrollToBottom();
  });
</script>

<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
<div
  bind:this={container}
  class="flex-1 overflow-y-auto px-4 py-4 space-y-3 custom-scrollbar"
  on:scroll={checkScrollPosition}
  on:click={handleImageClick}
>
  {#if $messages.length === 0}
    <!-- 歡迎訊息 -->
    <div class="flex justify-start">
      <div class="max-w-[85%] rounded-2xl rounded-bl-md bg-gray-100 text-gray-800 px-4 py-2.5 shadow-sm">
        <p class="text-sm leading-relaxed">{$config.welcomeMessage}</p>
      </div>
    </div>

    <!-- 快捷提問 -->
    {#if $showQuickReplies}
      <QuickReplies onSelect={onQuickReply} />
    {/if}
  {:else}
    {#each $messages as message (message.id)}
      <MessageBubble {message} />
    {/each}
  {/if}
</div>

<!-- 圖片放大 Lightbox -->
<ImageLightbox
  src={lightboxSrc}
  alt={lightboxAlt}
  isOpen={lightboxOpen}
  on:close={closeLightbox}
/>
