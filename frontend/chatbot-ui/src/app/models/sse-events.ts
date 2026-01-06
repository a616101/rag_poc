/**
 * SSE event type definitions for GraphRAG streaming responses.
 *
 * Both endpoints use Unified Agent LangGraph:
 * - /ask/stream: Chat backend, focused on final answer
 * - /ask/stream_chat: Responses backend, includes reasoning summary
 *
 * Common event fields:
 * {
 *   "source": "ask_stream" | "unified_agent",
 *   "node": "guard" | "planner" | "tool_executor" | ...,
 *   "phase": "planning" | "agent" | "generation" | "summary",
 *   "channel": "status" | "rewrite_llm" | "answer" | "reasoning" | "meta" | "meta_summary",
 *   "stage": "unified_agent_*" | "language_normalizer_*" | ...,
 *   "delta": "streaming output",
 *   "used_tools": [...],
 *   "meta": {...}
 * }
 */

export interface StreamEventRaw {
  source?: string;
  node?: string;
  phase?: string;
  channel?: string;
  stage?: string;
  node_stage?: string;
  delta?: string;
  // Rewrite node fields
  is_out_of_scope?: boolean;
  search_query?: string;
  intent?: string;
  // Guard node fields
  blocked?: boolean;
  meta?: unknown;
  error?: unknown;
  // Agent tool fields
  tool_name?: string;
  tool_args?: unknown;
  tool_output?: string;
  // Unified Agent fields
  used_tools?: string[];
  loops?: number;
  loop?: number;
  user_language?: string;
  query?: string;
  documents_count?: number;
  fallback_to_retrieval?: boolean;
  // Allow other fields
  [key: string]: unknown;
}

export type SSEChannel =
  | 'status'
  | 'rewrite_llm'
  | 'retrieval'
  | 'reasoning'
  | 'reasoning_summary'
  | 'answer'
  | 'meta'
  | 'meta_summary';

export interface UsageMetadata {
  total_tokens?: number;
  input_tokens?: number;
  output_tokens?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
}

export interface MetaSummary {
  total_usage?: UsageMetadata;
  trace_id?: string;
  conversation_summary?: string;
  [key: string]: unknown;
}
