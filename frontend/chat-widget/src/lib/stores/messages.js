import { writable, derived, get } from 'svelte/store';

/**
 * 參考來源
 * @typedef {Object} Source
 * @property {number} index - 來源索引
 * @property {string} chunk_id - 區塊 ID
 * @property {string} content - 內容摘要
 * @property {string} source_doc - 來源文件名稱
 * @property {number} relevance_score - 相關性分數
 */

/**
 * 訊息類型
 * @typedef {'user' | 'assistant'} MessageRole
 *
 * @typedef {Object} Message
 * @property {string} id
 * @property {MessageRole} role
 * @property {string} content
 * @property {boolean} isStreaming
 * @property {string} [statusText] - 目前處理階段的提示文字
 * @property {Source[]} [sources] - 參考來源列表
 * @property {number} timestamp
 */

// ========== 會話設定 ==========

/** 滑動窗口大小：送給後端的最大對話輪數 */
const MAX_HISTORY_ROUNDS = 10;

/** 會話過期時間（毫秒）：30 分鐘 */
const SESSION_TIMEOUT_MS = 30 * 60 * 1000;

/** 最後活動時間 */
let lastActivityTime = Date.now();

// ========== Stores ==========

/** @type {import('svelte/store').Writable<Message[]>} */
export const messages = writable([]);

/** 是否正在載入中 */
export const isLoading = writable(false);

/** 錯誤訊息 */
export const errorMessage = writable(null);

/** 是否顯示快捷提問（僅在沒有訊息時顯示） */
export const showQuickReplies = derived(messages, $messages => $messages.length === 0);

/** 對話輪數 */
export const conversationRounds = derived(messages, $messages => {
  return Math.floor($messages.filter(m => m.role === 'user').length);
});

let messageIdCounter = 0;

/**
 * 產生唯一訊息 ID
 */
function generateId() {
  return `msg-${Date.now()}-${++messageIdCounter}`;
}

/**
 * 新增使用者訊息
 * @param {string} content
 * @returns {string} 訊息 ID
 */
export function addUserMessage(content) {
  const id = generateId();
  messages.update(msgs => [
    ...msgs,
    {
      id,
      role: 'user',
      content,
      isStreaming: false,
      timestamp: Date.now()
    }
  ]);
  return id;
}

/**
 * 新增助手訊息（串流中）
 * @param {string} [initialStatus] - 初始狀態文字
 * @returns {string} 訊息 ID
 */
export function addAssistantMessage(initialStatus = '正在處理...') {
  const id = generateId();
  messages.update(msgs => [
    ...msgs,
    {
      id,
      role: 'assistant',
      content: '',
      isStreaming: true,
      statusText: initialStatus,
      timestamp: Date.now()
    }
  ]);
  return id;
}

/**
 * 更新訊息的狀態文字
 * @param {string} id
 * @param {string} statusText
 */
export function updateMessageStatus(id, statusText) {
  messages.update(msgs =>
    msgs.map(msg => {
      if (msg.id !== id) return msg;

      // 如果是空字串（開始顯示內容），直接清除狀態
      if (!statusText) {
        return { ...msg, statusText: '' };
      }

      // 檢查是否與當前狀態相同（避免重複更新）
      if (msg.statusText === statusText) {
        return msg;
      }

      return { ...msg, statusText };
    })
  );
}

/**
 * 追加內容到指定訊息
 * @param {string} id
 * @param {string} chunk
 */
export function appendToMessage(id, chunk) {
  if (!chunk) return;
  messages.update(msgs =>
    msgs.map(msg =>
      msg.id === id ? { ...msg, content: msg.content + chunk } : msg
    )
  );
}

/**
 * 標記訊息串流結束
 * @param {string} id
 */
export function finishMessage(id) {
  messages.update(msgs =>
    msgs.map(msg =>
      msg.id === id ? { ...msg, isStreaming: false } : msg
    )
  );
}

/**
 * 設定訊息的參考來源
 * @param {string} id
 * @param {Source[]} sources
 */
export function setMessageSources(id, sources) {
  if (!sources || sources.length === 0) return;
  messages.update(msgs =>
    msgs.map(msg =>
      msg.id === id ? { ...msg, sources } : msg
    )
  );
}

/**
 * 清空所有訊息
 */
export function clearMessages() {
  messages.set([]);
  errorMessage.set(null);
  lastActivityTime = Date.now();
}

// ========== 會話管理 ==========

/**
 * 更新最後活動時間
 */
export function updateActivity() {
  lastActivityTime = Date.now();
}

/**
 * 檢查會話是否過期
 * @returns {boolean}
 */
export function isSessionExpired() {
  return Date.now() - lastActivityTime > SESSION_TIMEOUT_MS;
}

/**
 * 檢查並處理會話過期（如果過期則清空訊息）
 * @returns {boolean} 是否已過期並被清空
 */
export function checkAndHandleSessionExpiry() {
  if (isSessionExpired() && get(messages).length > 0) {
    clearMessages();
    return true;
  }
  return false;
}

/**
 * 取得要送給後端的對話歷史（滑動窗口）
 * @returns {Array<{role: string, content: string}> | null}
 */
export function getConversationHistory() {
  const allMessages = get(messages);
  if (allMessages.length === 0) return null;

  // 計算要保留的訊息數量（最近 N 輪 = N 個 user + N 個 assistant = 2N 條）
  const maxMessages = MAX_HISTORY_ROUNDS * 2;

  // 取最近的訊息
  const recentMessages = allMessages.slice(-maxMessages);

  // 轉換為後端格式
  return recentMessages
    .filter(m => m.content) // 排除空內容
    .map(m => ({
      role: m.role,
      content: m.content
    }));
}
