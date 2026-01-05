/**
 * Feedback Service - 用戶回饋 API
 *
 * 提供提交用戶回饋到後端的功能。
 */

import { get } from 'svelte/store';
import { config } from '../stores/config.js';

/**
 * 提交用戶回饋到後端
 * @param {string} traceId - Langfuse trace ID
 * @param {'up' | 'down'} score - 評分
 * @param {string | null} [comment] - 評論（倒讚時選填）
 * @returns {Promise<{success: boolean, message: string, score_id?: string}>}
 */
export async function submitFeedback(traceId, score, comment = null) {
  const $config = get(config);

  // 從 apiEndpoint 推斷 feedback endpoint
  // 例如: /api/v1/rag/ask/stream_chat -> /api/v1/rag/feedback
  const baseUrl = $config.apiEndpoint.replace(/\/ask\/stream(_chat)?$/, '');
  const feedbackUrl = `${baseUrl}/feedback`;

  try {
    const response = await fetch(feedbackUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        trace_id: traceId,
        score,
        comment
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();
    return {
      success: data.success,
      message: data.message,
      score_id: data.score_id
    };
  } catch (error) {
    console.error('Feedback submission failed:', error);
    return {
      success: false,
      message: error.message || 'Failed to submit feedback'
    };
  }
}
