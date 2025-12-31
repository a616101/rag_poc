<script>
  import { config } from '../stores/config.js';

  export let onSelect = (/** @type {string} */ text) => {};

  $: quickReplies = $config.quickReplies || [];
  $: quickRepliesStyle = $config.quickRepliesStyle || 'outline';

  // 統一處理 quickReplies 格式 - 支援 string[] 或 {label, value}[]
  function getLabel(reply) {
    return typeof reply === 'string' ? reply : reply.label;
  }

  function getValue(reply) {
    return typeof reply === 'string' ? reply : reply.value;
  }

  // 根據樣式取得按鈕 class
  function getButtonClass(style) {
    const base = 'inline-flex items-center px-3 py-1.5 text-sm transition-colors duration-150';

    switch (style) {
      case 'solid':
        return `${base} rounded-lg bg-primary text-white hover:bg-primary-hover`;
      case 'pill':
        return `${base} rounded-full bg-primary text-white hover:bg-primary-hover`;
      case 'outline':
      default:
        return `${base} rounded-lg border border-primary/30 text-primary bg-primary/5 hover:bg-primary/10 hover:border-primary/50`;
    }
  }
</script>

{#if quickReplies.length > 0}
  <div class="px-4 py-3">
    <p class="text-xs text-gray-500 mb-2">快速提問：</p>
    <div class="flex flex-wrap gap-2">
      {#each quickReplies as reply}
        <button
          type="button"
          class={getButtonClass(quickRepliesStyle)}
          on:click={() => onSelect(getValue(reply))}
        >
          {getLabel(reply)}
        </button>
      {/each}
    </div>
  </div>
{/if}
