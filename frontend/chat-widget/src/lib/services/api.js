import { get } from 'svelte/store';
import { config } from '../stores/config.js';
import {
  isLoading,
  errorMessage,
  addUserMessage,
  addAssistantMessage,
  appendToMessage,
  finishMessage,
  updateMessageStatus,
  setMessageSources
} from '../stores/messages.js';

/**
 * OpenAI Chat Completion Chunk 結構
 * @typedef {Object} OpenAIChunk
 * @property {string} id - Completion ID
 * @property {string} object - "chat.completion.chunk"
 * @property {Array<{delta: {role?: string, content?: string}, finish_reason?: string}>} choices
 */

/**
 * Status 事件結構
 * @typedef {Object} StatusEvent
 * @property {string} type - "status"
 * @property {string} node - 節點名稱
 * @property {string} stage - 階段
 */

/**
 * 發送問題並處理 SSE 串流回應 (OpenAI 相容格式)
 * @param {string} question - 使用者問題
 * @param {Array<{role: string, content: string}>} [conversationHistory] - 對話歷史
 */
export async function sendQuestion(question, conversationHistory = null) {
  const $config = get(config);

  // 重置狀態節流計時器
  resetStatusThrottle();

  // 設定載入狀態
  isLoading.set(true);
  errorMessage.set(null);

  // 新增使用者訊息到畫面
  addUserMessage(question);

  // 新增空的助手訊息（串流中）
  const assistantMessageId = addAssistantMessage();

  try {
    // 建構 OpenAI 格式的 messages 陣列
    const messages = [];

    // 加入系統提示（如果有設定）
    if ($config.systemPrompt) {
      messages.push({
        role: 'system',
        content: $config.systemPrompt
      });
    }

    // 加入對話歷史
    if (conversationHistory && conversationHistory.length > 0) {
      messages.push(...conversationHistory);
    }

    // 加入當前問題
    messages.push({
      role: 'user',
      content: question
    });

    // 建構 OpenAI 相容的請求 payload
    const payload = {
      messages,
      stream: true,
      include_sources: true
    };

    const response = await fetch($config.apiEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`請求失敗: ${response.status} ${response.statusText}`);
    }

    if (!response.body) {
      throw new Error('回應沒有 body');
    }

    // 處理 SSE 串流
    await processSSEStream(response.body, assistantMessageId);

  } catch (error) {
    console.error('API Error:', error);
    errorMessage.set(error.message || '發生錯誤，請稍後再試');
    finishMessage(assistantMessageId);
  } finally {
    isLoading.set(false);
  }
}

/**
 * 處理 SSE 串流 (OpenAI 相容格式)
 * @param {ReadableStream} body
 * @param {string} messageId
 */
async function processSSEStream(body, messageId) {
  const reader = body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // 按行分割處理
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line || !line.startsWith('data:')) continue;

        const jsonStr = line.slice(5).trim();
        if (!jsonStr) continue;

        // 處理 OpenAI 串流結束標記
        if (jsonStr === '[DONE]') {
          continue;
        }

        try {
          const event = JSON.parse(jsonStr);
          handleStreamEvent(event, messageId);
        } catch (e) {
          console.warn('Failed to parse SSE event:', e, jsonStr);
        }
      }
    }

    // 處理殘餘緩衝
    if (buffer.trim() && buffer.startsWith('data:')) {
      const jsonStr = buffer.slice(5).trim();
      if (jsonStr && jsonStr !== '[DONE]') {
        try {
          const event = JSON.parse(jsonStr);
          handleStreamEvent(event, messageId);
        } catch {
          // ignore
        }
      }
    }

  } finally {
    finishMessage(messageId);
  }
}

/**
 * 節點階段對應的友善提示文字
 * 確保主要階段都有獨特的狀態顯示，讓用戶感知進度
 */
const STATUS_MESSAGES = {
  // ========== 階段一：問題分析（快速，合併） ==========
  guard_START: '正在分析問題...',
  acl_START: '正在分析問題...',
  normalize_START: '正在分析問題...',
  cache_lookup_START: '正在分析問題...',

  // ========== 階段二：理解意圖（較慢，獨立顯示） ==========
  intent_router_START: '正在理解您的問題...',

  // ========== 階段三：搜尋資料（較慢，獨立顯示） ==========
  hybrid_seed_START: '正在搜尋相關資料...',
  community_reports_START: '正在搜尋相關資料...',
  followups_START: '正在搜尋相關資料...',

  // ========== 階段四：篩選結果（較慢，獨立顯示） ==========
  rrf_merge_START: '正在整合搜尋結果...',
  rerank_START: '正在篩選最佳結果...',

  // ========== 階段五：知識圖譜分析（較慢，獨立顯示） ==========
  graph_seed_extract_START: '正在分析知識圖譜...',
  graph_traverse_START: '正在探索相關知識...',
  subgraph_to_queries_START: '正在擴展搜尋範圍...',
  hop_hybrid_START: '正在深度搜尋...',

  // ========== 階段六：整理資料 ==========
  chunk_expander_START: '正在整理參考資料...',
  context_packer_START: '正在整理參考資料...',
  evidence_table_START: '正在建立證據...',

  // ========== 階段七：品質驗證（較慢，獨立顯示） ==========
  groundedness_START: '正在驗證答案品質...',
  targeted_retry_START: '正在優化搜尋...',

  // ========== 階段八：生成回答 ==========
  direct_answer_START: '正在生成回答...',
  final_answer_GENERATING: '正在生成回答...',
  GENERATING: '正在生成回答...',

  // ========== 其他（不顯示給用戶） ==========
  interrupt_hitl_START: '需要人工審核...',
  cache_store_START: null,
  telemetry_START: null
};

/**
 * 根據 status 事件取得狀態文字
 * 只處理 START 和 GENERATING 狀態，忽略 DONE 狀態以簡化用戶體驗
 * @param {StatusEvent} event
 * @returns {string | null}
 */
function getStatusText(event) {
  const node = event.node || '';
  const stage = event.stage || '';

  // 忽略 DONE 狀態（用戶不需要看到每個步驟完成）
  if (stage === 'DONE') {
    return null;
  }

  // 1. 先嘗試 node_stage 組合鍵
  const compositeKey = `${node}_${stage}`;
  if (compositeKey && STATUS_MESSAGES.hasOwnProperty(compositeKey)) {
    return STATUS_MESSAGES[compositeKey]; // 可能是 null（表示不顯示）
  }

  // 2. 嘗試 stage 單獨鍵（如 GENERATING）
  if (stage && STATUS_MESSAGES.hasOwnProperty(stage)) {
    return STATUS_MESSAGES[stage];
  }

  // 3. 未知節點的預設處理
  return null;
}

// ========== 狀態顯示與內容緩衝機制 ==========
const MIN_STATUS_DISPLAY_TIME = 200; // 每個狀態最小顯示時間（毫秒）
const MAX_CONTENT_DELAY = 300; // 內容最大延遲時間（毫秒）

let statusQueue = [];
let isProcessingQueue = false;
let currentStatusTimer = null;
let lastShownStatus = '';

// 內容緩衝
let contentBuffer = [];
let contentBufferMessageId = null;
let contentStartTime = null;
let contentFlushTimer = null;

/**
 * 將狀態更新加入佇列
 * @param {string} messageId
 * @param {string} statusText
 */
function throttledStatusUpdate(messageId, statusText) {
  // 如果內容已經開始顯示，忽略新的狀態
  if (contentStartTime && contentBuffer.length === 0) {
    return;
  }

  // 如果與上次顯示或佇列中最後一個狀態相同，不重複加入
  if (statusText === lastShownStatus) {
    return;
  }
  const lastInQueue = statusQueue[statusQueue.length - 1];
  if (lastInQueue && lastInQueue.statusText === statusText) {
    return;
  }

  statusQueue.push({ messageId, statusText });
  processStatusQueue();
}

/**
 * 處理狀態佇列
 */
function processStatusQueue() {
  if (isProcessingQueue || statusQueue.length === 0) {
    // 如果佇列空了且有緩衝內容，顯示內容
    if (statusQueue.length === 0 && contentBuffer.length > 0) {
      flushContentBuffer();
    }
    return;
  }

  isProcessingQueue = true;
  const { messageId, statusText } = statusQueue.shift();

  updateMessageStatus(messageId, statusText);
  lastShownStatus = statusText;

  currentStatusTimer = setTimeout(() => {
    isProcessingQueue = false;
    processStatusQueue();
  }, MIN_STATUS_DISPLAY_TIME);
}

/**
 * 處理內容 chunk（可能緩衝或直接顯示）
 * @param {string} messageId
 * @param {string} content
 */
function handleContentChunk(messageId, content) {
  // 如果正在模擬串流，將新內容加入模擬佇列
  if (isSimulatingStream) {
    simulatedStreamQueue.push({
      messageId: messageId,
      content: content
    });
    return;
  }

  // 如果還有狀態在佇列中且未超時，緩衝內容
  const hasQueuedStatus = isProcessingQueue || statusQueue.length > 0;
  const withinTimeLimit = !contentStartTime || (Date.now() - contentStartTime < MAX_CONTENT_DELAY);

  if (hasQueuedStatus && withinTimeLimit) {
    // 第一個 content chunk
    if (contentStartTime === null) {
      contentStartTime = Date.now();
      contentBufferMessageId = messageId;

      // 設定最大延遲計時器
      contentFlushTimer = setTimeout(() => {
        flushContentBuffer();
      }, MAX_CONTENT_DELAY);
    }

    contentBuffer.push(content);
    return;
  }

  // 沒有等待的狀態或已超時，直接顯示
  flushContentBuffer();
  updateMessageStatus(messageId, '');
  appendToMessage(messageId, content);
}

// 串流模擬狀態
let isSimulatingStream = false;
let simulatedStreamQueue = [];
let simulatedStreamTimer = null;
const SIMULATED_CHUNK_SIZE = 5; // 每次顯示的字元數
const SIMULATED_CHUNK_DELAY = 20; // 每個 chunk 之間的延遲（毫秒）

/**
 * 刷新內容緩衝，以模擬串流方式顯示
 */
function flushContentBuffer() {
  // 清除計時器
  if (contentFlushTimer) {
    clearTimeout(contentFlushTimer);
    contentFlushTimer = null;
  }
  if (currentStatusTimer) {
    clearTimeout(currentStatusTimer);
    currentStatusTimer = null;
  }

  // 清空狀態佇列
  statusQueue = [];
  isProcessingQueue = false;

  // 如果有緩衝內容，以模擬串流方式顯示
  if (contentBuffer.length > 0 && contentBufferMessageId) {
    updateMessageStatus(contentBufferMessageId, '');
    const allContent = contentBuffer.join('');

    // 將內容加入模擬串流佇列
    simulatedStreamQueue.push({
      messageId: contentBufferMessageId,
      content: allContent
    });

    // 開始模擬串流（如果尚未開始）
    if (!isSimulatingStream) {
      processSimulatedStream();
    }
  }

  // 重置緩衝狀態（但保留 contentStartTime 表示內容已開始）
  contentBuffer = [];
  contentBufferMessageId = null;
}

/**
 * 處理模擬串流佇列
 */
function processSimulatedStream() {
  if (simulatedStreamQueue.length === 0) {
    isSimulatingStream = false;
    return;
  }

  isSimulatingStream = true;
  const item = simulatedStreamQueue[0];

  if (item.content.length === 0) {
    // 這個項目處理完了，移除並處理下一個
    simulatedStreamQueue.shift();
    processSimulatedStream();
    return;
  }

  // 取出一小塊內容
  const chunk = item.content.slice(0, SIMULATED_CHUNK_SIZE);
  item.content = item.content.slice(SIMULATED_CHUNK_SIZE);

  // 顯示這個 chunk
  appendToMessage(item.messageId, chunk);

  // 安排下一個 chunk
  simulatedStreamTimer = setTimeout(() => {
    processSimulatedStream();
  }, SIMULATED_CHUNK_DELAY);
}

/**
 * 重置所有狀態（在新對話開始時調用）
 */
function resetStatusThrottle() {
  statusQueue = [];
  isProcessingQueue = false;
  lastShownStatus = '';
  contentBuffer = [];
  contentBufferMessageId = null;
  contentStartTime = null;

  // 重置模擬串流狀態
  isSimulatingStream = false;
  simulatedStreamQueue = [];

  if (currentStatusTimer) {
    clearTimeout(currentStatusTimer);
    currentStatusTimer = null;
  }
  if (contentFlushTimer) {
    clearTimeout(contentFlushTimer);
    contentFlushTimer = null;
  }
  if (simulatedStreamTimer) {
    clearTimeout(simulatedStreamTimer);
    simulatedStreamTimer = null;
  }
}

/**
 * 處理單一 SSE 事件
 * @param {OpenAIChunk | StatusEvent | {type: 'sources', sources: Array}} event
 * @param {string} messageId
 */
function handleStreamEvent(event, messageId) {
  // 處理 status 事件（自定義格式）
  if (event.type === 'status') {
    const statusText = getStatusText(event);
    if (statusText) {
      throttledStatusUpdate(messageId, statusText);
    }
    return;
  }

  // 處理 sources 事件（參考來源）
  if (event.type === 'sources') {
    setMessageSources(messageId, event.sources);
    return;
  }

  // 處理 OpenAI Chat Completion Chunk
  if (event.choices && event.choices.length > 0) {
    const choice = event.choices[0];
    const delta = choice.delta;

    // 處理內容
    if (delta && delta.content) {
      // 使用內容緩衝機制，讓狀態有足夠時間顯示
      handleContentChunk(messageId, delta.content);
    }

    // 處理錯誤
    if (choice.finish_reason === 'error' && delta && delta.content) {
      errorMessage.set(delta.content);
    }
  }
}
