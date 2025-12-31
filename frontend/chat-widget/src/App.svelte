<script>
  import { onMount } from 'svelte';
  import ChatWindow from './lib/components/ChatWindow.svelte';
  import { updateConfig } from './lib/stores/config.js';

  let isMobile = false;

  // 偵測是否為手機
  function checkMobile() {
    isMobile = window.innerWidth < 640;
  }

  onMount(() => {
    checkMobile();
    window.addEventListener('resize', checkMobile);

    // 開發模式：從 URL 參數讀取配置
    if (import.meta.env.DEV) {
      const params = new URLSearchParams(window.location.search);
      const configFromUrl = {};

      if (params.get('apiEndpoint')) {
        configFromUrl.apiEndpoint = params.get('apiEndpoint');
      }
      if (params.get('primaryColor')) {
        configFromUrl.primaryColor = params.get('primaryColor');
      }
      if (params.get('headerTitle')) {
        configFromUrl.headerTitle = params.get('headerTitle');
      }
      if (params.get('welcomeMessage')) {
        configFromUrl.welcomeMessage = params.get('welcomeMessage');
      }
      if (params.get('quickReplies')) {
        try {
          configFromUrl.quickReplies = JSON.parse(params.get('quickReplies'));
        } catch {
          console.warn('Failed to parse quickReplies');
        }
      }

      if (Object.keys(configFromUrl).length > 0) {
        updateConfig(configFromUrl);
      }
    }

    return () => {
      window.removeEventListener('resize', checkMobile);
    };
  });

  function handleMinimize() {
    // 在獨立模式下，最小化沒有作用
    // 實際嵌入時由 Widget Loader 處理
    console.log('Minimize requested');
  }
</script>

<div class="w-full h-full" class:mobile={isMobile}>
  <ChatWindow onMinimize={handleMinimize} />
</div>

<style>
  /* 手機模式下確保全螢幕 */
  .mobile {
    position: fixed;
    inset: 0;
    z-index: 9999;
  }
</style>
