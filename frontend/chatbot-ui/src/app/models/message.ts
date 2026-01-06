/**
 * Message models for the chat application.
 */

export type ChatRole = 'user' | 'assistant';

export interface TokenUsage {
  totalTokens?: number;
  inputTokens?: number;
  outputTokens?: number;
}

export interface AssistantDetails {
  /** Query rewriting process (accumulated delta from rewrite_llm channel) */
  planning: string;
  /** Retrieval phase information (retrieval channel / status) */
  retrieval: string;
  /** LLM reasoning content (accumulated delta from reasoning channel) */
  reasoning: string;
  /** Stage status messages (status channel) */
  statusMessages: string[];
  /** LLM meta information */
  meta?: unknown;
  /** Token usage statistics */
  tokenUsage?: TokenUsage;
  /** Duration measured by frontend SSE start/end (milliseconds) */
  durationMs?: number;
  /** Whether details panel is expanded */
  expanded: boolean;
  /** Whether still streaming (any phase) */
  isStreaming: boolean;
  /** Langfuse trace ID (for feedback API) */
  traceId?: string;
  /** User submitted feedback score */
  userFeedback?: 'up' | 'down' | null;
}

export interface ChatMessage {
  id: number;
  role: ChatRole;
  content: string;
  /** Only assistant messages have detailed reasoning info */
  details?: AssistantDetails;
}

export function createEmptyAssistantDetails(): AssistantDetails {
  return {
    planning: '',
    retrieval: '',
    reasoning: '',
    statusMessages: [],
    meta: undefined,
    tokenUsage: undefined,
    durationMs: undefined,
    expanded: false,
    isStreaming: false
  };
}
