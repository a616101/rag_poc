/**
 * Chat Widget Loader
 * 輕量級 loader，用於在客戶網站上嵌入聊天機器人
 *
 * 使用方式：
 * <script
 *   src="https://your-domain.com/widget.js"
 *   data-api-endpoint="https://your-api.com/api/v1/rag/ask/stream"
 *   data-primary-color="#E84967"
 *   data-header-title="屏基AI小天使"
 *   data-header-subtitle="請問有什麼可以協助您？"
 *   data-welcome-message="您好，我是屏基AI小天使，請問有什麼可以協助您？"
 *   data-quick-replies='[{"label":"門診時間","value":"門診時間"},{"label":"看診進度","value":"看診進度"}]'
 * ></script>
 */

(function () {
  'use strict';

  // 防止重複載入
  if (window.__CHAT_WIDGET_LOADED__) return;
  window.__CHAT_WIDGET_LOADED__ = true;

  // ========== 工具函式 ==========

  const scriptTag = document.currentScript;

  /**
   * 取得 widget 的基礎路徑（絕對 URL）
   * 使用 document.baseURI 作為相對路徑的基準
   */
  function getBasePath() {
    if (scriptTag?.src) {
      try {
        // 使用 document.baseURI 解析相對路徑
        const url = new URL(scriptTag.src, document.baseURI);
        return url.href.replace(/\/widget\.js(\?.*)?$/, '');
      } catch {
        return '';
      }
    }
    return '';
  }

  // 提前計算 basePath，供後續使用
  const basePath = getBasePath();

  function getWidgetUrl() {
    // 預設使用與 loader 同源的 widget
    if (basePath) {
      return basePath + '/index.html';
    }
    return './index.html';
  }

  /**
   * 解析資源 URL - 將相對路徑轉換為絕對 URL
   * 支援：
   * - 絕對 URL (https://...)
   * - Data URL (data:image/...)
   * - 根路徑 (/path) - 相對於 widget 所在目錄
   * - 相對路徑 (./path, path) - 相對於 widget 所在目錄
   */
  function resolveAssetUrl(url) {
    if (!url) return '';

    // 已經是絕對 URL 或 data URL，直接返回
    if (url.startsWith('http://') || url.startsWith('https://') || url.startsWith('data:')) {
      return url;
    }

    // 如果有 basePath，解析相對路徑
    if (basePath) {
      // 根路徑 /path - 相對於 widget 所在目錄（不是網站根目錄）
      if (url.startsWith('/')) {
        return basePath + url;
      }
      // 相對路徑 ./path 或 path
      const cleanUrl = url.startsWith('./') ? url.slice(2) : url;
      return basePath + '/' + cleanUrl;
    }

    // 沒有 basePath 時，嘗試使用 document.baseURI 解析
    try {
      return new URL(url, document.baseURI).href;
    } catch {
      return url;
    }
  }

  function parseJSON(str, fallback) {
    if (!str) return fallback;
    try {
      return JSON.parse(str);
    } catch {
      return fallback;
    }
  }

  // ========== 配置讀取 ==========

  const config = {
    // API 配置
    apiEndpoint: scriptTag?.dataset.apiEndpoint || '/api/v1/rag/ask/stream_chat',
    widgetUrl: scriptTag?.dataset.widgetUrl || getWidgetUrl(),

    // 顏色配置
    primaryColor: scriptTag?.dataset.primaryColor || '#E84967',
    headerBgColor: scriptTag?.dataset.headerBgColor || '', // 空值則使用 primaryColor
    headerTextColor: scriptTag?.dataset.headerTextColor || '#ffffff',
    bubbleBgColor: scriptTag?.dataset.bubbleBgColor || '', // 空值則使用 primaryColor

    // 位置與大小
    position: scriptTag?.dataset.position || 'bottom-right',
    bubbleSize: scriptTag?.dataset.bubbleSize || 'medium',

    // 文案
    headerTitle: scriptTag?.dataset.headerTitle || '屏基AI小天使',
    headerSubtitle: scriptTag?.dataset.headerSubtitle || '請問有什麼可以協助您？',
    welcomeMessage: scriptTag?.dataset.welcomeMessage || '您好，我是屏基AI小天使，請問有什麼可以協助您？',
    inputPlaceholder: scriptTag?.dataset.inputPlaceholder || '輸入訊息...',

    // 圖示 (支援圖片 URL)
    headerIcon: scriptTag?.dataset.headerIcon || '',
    bubbleIcon: scriptTag?.dataset.bubbleIcon || '',

    // 快捷按鈕 - 支援 [{label, value}] 或 ["string"] 格式
    quickReplies: parseJSON(scriptTag?.dataset.quickReplies, []),
    quickRepliesStyle: scriptTag?.dataset.quickRepliesStyle || 'outline', // outline | solid | pill

    // 行為
    autoOpen: scriptTag?.dataset.autoOpen === 'true',
    autoOpenDelay: parseInt(scriptTag?.dataset.autoOpenDelay, 10) || 3000
  };

  // ========== 狀態 ==========

  let isOpen = false;
  let isExpanded = false;
  let container = null;
  let bubble = null;
  let iframe = null;
  let unreadCount = 0;

  // ========== 樣式 ==========

  const styles = `
    #chat-widget-container {
      position: fixed;
      bottom: 20px;
      ${config.position === 'bottom-left' ? 'left: 20px;' : 'right: 20px;'}
      z-index: 2147483647;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }

    #chat-widget-bubble {
      width: ${config.bubbleSize === 'small' ? '48px' : config.bubbleSize === 'large' ? '64px' : '56px'};
      height: ${config.bubbleSize === 'small' ? '48px' : config.bubbleSize === 'large' ? '64px' : '56px'};
      border-radius: 50%;
      background-color: ${config.bubbleBgColor || config.primaryColor};
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
      overflow: hidden;
    }

    #chat-widget-bubble:hover {
      transform: scale(1.08);
      box-shadow: 0 6px 24px rgba(0, 0, 0, 0.25);
    }

    #chat-widget-bubble svg {
      width: ${config.bubbleSize === 'small' ? '24px' : config.bubbleSize === 'large' ? '32px' : '28px'};
      height: ${config.bubbleSize === 'small' ? '24px' : config.bubbleSize === 'large' ? '32px' : '28px'};
      fill: white;
    }

    #chat-widget-bubble img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }

    #chat-widget-bubble-badge {
      position: absolute;
      top: -4px;
      right: -4px;
      min-width: 20px;
      height: 20px;
      padding: 0 6px;
      border-radius: 10px;
      background-color: #ef4444;
      color: white;
      font-size: 12px;
      font-weight: 600;
      display: none;
      align-items: center;
      justify-content: center;
    }

    #chat-widget-bubble-badge.show {
      display: flex;
    }

    #chat-widget-frame-container {
      position: absolute;
      bottom: 70px;
      ${config.position === 'bottom-left' ? 'left: 0;' : 'right: 0;'}
      width: 380px;
      height: 600px;
      max-height: calc(100vh - 100px);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
      opacity: 0;
      transform: translateY(20px) scale(0.95);
      pointer-events: none;
      transition: opacity 0.2s ease, transform 0.2s ease;
    }

    #chat-widget-frame-container.open {
      opacity: 1;
      transform: translateY(0) scale(1);
      pointer-events: auto;
    }

    #chat-widget-frame-container.expanded {
      position: fixed;
      inset: 0;
      width: 100%;
      height: 100%;
      max-height: 100%;
      border-radius: 0;
      bottom: 0;
    }

    #chat-widget-iframe {
      width: 100%;
      height: 100%;
      border: none;
      background: white;
    }

    /* 手機響應式 */
    @media (max-width: 480px) {
      #chat-widget-frame-container {
        position: fixed;
        inset: 0;
        width: 100%;
        height: 100%;
        max-height: 100%;
        border-radius: 0;
        bottom: 0;
      }

      #chat-widget-bubble {
        bottom: 16px;
        ${config.position === 'bottom-left' ? 'left: 16px;' : 'right: 16px;'}
      }
    }
  `;

  // ========== DOM 建立 ==========

  function createWidget() {
    // 注入樣式
    const styleEl = document.createElement('style');
    styleEl.textContent = styles;
    document.head.appendChild(styleEl);

    // 建立容器
    container = document.createElement('div');
    container.id = 'chat-widget-container';

    // 建立浮動按鈕
    bubble = document.createElement('div');
    bubble.id = 'chat-widget-bubble';

    // 判斷是否有自訂圖示 - 解析相對路徑
    const bubbleIconUrl = resolveAssetUrl(config.bubbleIcon);
    const bubbleContent = bubbleIconUrl
      ? `<img src="${bubbleIconUrl}" alt="Chat" />`
      : `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H5.17L4 17.17V4h16v12z"/>
          <path d="M7 9h2v2H7zm4 0h2v2h-2zm4 0h2v2h-2z"/>
        </svg>`;

    bubble.innerHTML = `
      ${bubbleContent}
      <span id="chat-widget-bubble-badge">0</span>
    `;
    bubble.addEventListener('click', toggleChat);

    // 建立 iframe 容器
    const frameContainer = document.createElement('div');
    frameContainer.id = 'chat-widget-frame-container';

    // 建立 iframe
    iframe = document.createElement('iframe');
    iframe.id = 'chat-widget-iframe';
    iframe.src = config.widgetUrl;
    iframe.title = config.headerTitle;
    iframe.allow = 'microphone';

    frameContainer.appendChild(iframe);
    container.appendChild(bubble);
    container.appendChild(frameContainer);
    document.body.appendChild(container);

    // 監聽 iframe 訊息
    window.addEventListener('message', handleMessage);

    // 自動開啟
    if (config.autoOpen) {
      setTimeout(openChat, config.autoOpenDelay);
    }
  }

  // ========== 事件處理 ==========

  function handleMessage(event) {
    const data = event.data;
    if (!data || typeof data !== 'object') return;
    if (data.source !== 'chat-widget-iframe') return;

    switch (data.type) {
      case 'READY':
        // iframe 準備就緒，發送配置
        sendConfigToIframe();
        break;

      case 'CLOSE':
        closeChat();
        break;

      case 'NEW_MESSAGE':
        if (!isOpen && data.payload?.unreadCount > 0) {
          unreadCount = data.payload.unreadCount;
          updateBadge();
        }
        break;

      case 'RESIZE':
        // 可選：動態調整大小
        break;

      case 'TOGGLE_EXPAND':
        toggleExpand();
        break;

      default:
        break;
    }
  }

  function sendConfigToIframe() {
    if (!iframe?.contentWindow) return;

    iframe.contentWindow.postMessage({
      source: 'chat-widget-loader',
      type: 'INIT_CONFIG',
      payload: {
        apiEndpoint: config.apiEndpoint,
        primaryColor: config.primaryColor,
        headerBgColor: config.headerBgColor || config.primaryColor,
        headerTextColor: config.headerTextColor,
        headerTitle: config.headerTitle,
        headerSubtitle: config.headerSubtitle,
        headerIcon: resolveAssetUrl(config.headerIcon),
        welcomeMessage: config.welcomeMessage,
        inputPlaceholder: config.inputPlaceholder,
        quickReplies: config.quickReplies,
        quickRepliesStyle: config.quickRepliesStyle
      }
    }, '*');
  }

  function toggleChat() {
    if (isOpen) {
      closeChat();
    } else {
      openChat();
    }
  }

  function openChat() {
    isOpen = true;
    unreadCount = 0;
    updateBadge();

    const frameContainer = document.getElementById('chat-widget-frame-container');
    if (frameContainer) {
      frameContainer.classList.add('open');
    }

    // 更換圖示為關閉
    updateBubbleIcon(true);
  }

  function closeChat() {
    isOpen = false;

    const frameContainer = document.getElementById('chat-widget-frame-container');
    if (frameContainer) {
      frameContainer.classList.remove('open');
    }

    // 更換圖示為聊天
    updateBubbleIcon(false);
  }

  function updateBubbleIcon(isClose) {
    if (!bubble) return;

    const svg = bubble.querySelector('svg');
    if (!svg) return;

    if (isClose) {
      svg.innerHTML = `
        <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
      `;
    } else {
      svg.innerHTML = `
        <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H5.17L4 17.17V4h16v12z"/>
        <path d="M7 9h2v2H7zm4 0h2v2h-2zm4 0h2v2h-2z"/>
      `;
    }
  }

  function updateBadge() {
    const badge = document.getElementById('chat-widget-bubble-badge');
    if (!badge) return;

    if (unreadCount > 0) {
      badge.textContent = unreadCount > 99 ? '99+' : unreadCount;
      badge.classList.add('show');
    } else {
      badge.classList.remove('show');
    }
  }

  function toggleExpand() {
    isExpanded = !isExpanded;

    const frameContainer = document.getElementById('chat-widget-frame-container');
    if (frameContainer) {
      if (isExpanded) {
        frameContainer.classList.add('expanded');
      } else {
        frameContainer.classList.remove('expanded');
      }
    }

    // 通知 iframe 展開狀態已更新
    sendExpandedStateToIframe();
  }

  function sendExpandedStateToIframe() {
    if (!iframe?.contentWindow) return;

    iframe.contentWindow.postMessage({
      source: 'chat-widget-loader',
      type: 'EXPANDED_STATE',
      payload: { isExpanded }
    }, '*');
  }

  // ========== 公開 API ==========

  window.ChatWidget = {
    open: openChat,
    close: closeChat,
    toggle: toggleChat,
    updateConfig: function (newConfig) {
      Object.assign(config, newConfig);
      sendConfigToIframe();
    }
  };

  // ========== 初始化 ==========

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createWidget);
  } else {
    createWidget();
  }
})();
