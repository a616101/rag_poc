/**
 * API request models for the chat application.
 */

import { ChatRole } from './message';

export interface LlmConfigPayload {
  model?: string;
  reasoning_effort?: 'low' | 'medium' | 'high';
  reasoning_summary?: 'auto' | 'concise' | 'detailed';
}

export interface QuestionPayload {
  question: string;
  conversation_history?: { role: ChatRole; content: string }[] | null;
  top_k?: number;
  llm_config?: LlmConfigPayload;
  enable_conversation_summary?: boolean;
  conversation_summary?: string;
}

export interface FeedbackPayload {
  trace_id: string;
  score: 'up' | 'down';
  comment?: string | null;
}
