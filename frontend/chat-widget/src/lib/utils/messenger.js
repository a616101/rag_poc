import { updateConfig, updateExpanded } from '../stores/config.js';

/**
 * postMessage 通訊處理
 * 用於 Widget Loader 與 iframe 之間的通訊
 */

/**
 * 初始化 postMessage 監聽
 */
export function initMessenger() {
  window.addEventListener('message', handleMessage);

  // 通知父視窗 iframe 已準備就緒
  notifyReady();
}

/**
 * 處理來自父視窗的訊息
 * @param {MessageEvent} event
 */
function handleMessage(event) {
  // 安全檢查：確認來源（生產環境應該更嚴格）
  const data = event.data;
  if (!data || typeof data !== 'object') return;
  if (data.source !== 'chat-widget-loader') return;

  switch (data.type) {
    case 'INIT_CONFIG':
      // 接收初始配置
      if (data.payload) {
        updateConfig(data.payload);
      }
      break;

    case 'UPDATE_CONFIG':
      // 動態更新配置
      if (data.payload) {
        updateConfig(data.payload);
      }
      break;

    case 'EXPANDED_STATE':
      // 接收展開狀態
      if (data.payload && typeof data.payload.isExpanded === 'boolean') {
        updateExpanded(data.payload.isExpanded);
      }
      break;

    default:
      break;
  }
}

/**
 * 通知父視窗 iframe 已準備就緒
 */
function notifyReady() {
  sendToParent({
    type: 'READY'
  });
}

/**
 * 通知父視窗有新訊息（用於未讀計數）
 * @param {number} unreadCount
 */
export function notifyNewMessage(unreadCount) {
  sendToParent({
    type: 'NEW_MESSAGE',
    payload: { unreadCount }
  });
}

/**
 * 請求父視窗關閉聊天窗
 */
export function requestClose() {
  sendToParent({
    type: 'CLOSE'
  });
}

/**
 * 請求父視窗調整 iframe 大小
 * @param {Object} size
 * @param {number} size.width
 * @param {number} size.height
 */
export function requestResize(size) {
  sendToParent({
    type: 'RESIZE',
    payload: size
  });
}

/**
 * 請求父視窗切換放大/縮小模式
 */
export function requestToggleExpand() {
  sendToParent({
    type: 'TOGGLE_EXPAND'
  });
}

/**
 * 發送訊息到父視窗
 * @param {Object} message
 */
function sendToParent(message) {
  if (window.parent === window) return;

  window.parent.postMessage(
    {
      source: 'chat-widget-iframe',
      ...message
    },
    '*' // 生產環境應指定具體 origin
  );
}
