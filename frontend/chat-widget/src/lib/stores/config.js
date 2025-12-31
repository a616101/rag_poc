import { writable } from 'svelte/store';

/**
 * Widget 配置 store
 */
export const config = writable({
  // API 端點 (GraphRAG OpenAI 相容格式)
  apiEndpoint: '/api/v1/rag/ask/stream_chat',

  // 系統提示 (可選，會加入 messages 陣列)
  systemPrompt: '',

  // 顏色配置
  primaryColor: '#E84967',
  headerBgColor: '#E84967',
  headerTextColor: '#ffffff',

  // Header 配置
  headerTitle: '屏基AI小天使',
  headerSubtitle: '請問有什麼可以協助您？',
  headerIcon: '', // 圖片 URL

  // 文案
  welcomeMessage: '您好，我是屏基AI小天使，請問有什麼可以協助您？',
  inputPlaceholder: '輸入訊息...',

  // 快捷按鈕 - 支援 [{label, value}] 或 ["string"] 格式
  quickReplies: [],
  quickRepliesStyle: 'outline', // outline | solid | pill

  // 行為
  autoOpen: false,
  autoOpenDelay: 3000,

  // 展開狀態
  isExpanded: false
});

/**
 * 更新配置並套用 CSS 變數
 */
export function updateConfig(newConfig) {
  config.update(current => {
    const merged = { ...current, ...newConfig };

    // 更新 CSS 變數
    if (newConfig.primaryColor) {
      document.documentElement.style.setProperty('--widget-primary', newConfig.primaryColor);
      document.documentElement.style.setProperty('--widget-primary-hover', adjustColor(newConfig.primaryColor, -15));
      document.documentElement.style.setProperty('--widget-primary-light', adjustColor(newConfig.primaryColor, 40, 0.15));
    }

    if (newConfig.headerBgColor) {
      document.documentElement.style.setProperty('--widget-header-bg', newConfig.headerBgColor);
    }

    if (newConfig.headerTextColor) {
      document.documentElement.style.setProperty('--widget-header-text', newConfig.headerTextColor);
    }

    return merged;
  });
}

/**
 * 調整顏色明暗
 */
function adjustColor(hex, percent, alpha = 1) {
  const num = parseInt(hex.replace('#', ''), 16);
  const r = Math.min(255, Math.max(0, (num >> 16) + percent));
  const g = Math.min(255, Math.max(0, ((num >> 8) & 0x00FF) + percent));
  const b = Math.min(255, Math.max(0, (num & 0x0000FF) + percent));

  if (alpha < 1) {
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  return `#${(1 << 24 | r << 16 | g << 8 | b).toString(16).slice(1)}`;
}

/**
 * 更新展開狀態
 * @param {boolean} expanded
 */
export function updateExpanded(expanded) {
  config.update(current => ({
    ...current,
    isExpanded: expanded
  }));
}
