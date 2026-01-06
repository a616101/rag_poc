import { Injectable } from '@angular/core';
import { StreamEventRaw, UsageMetadata } from '../models';
import { TokenUsage } from '../models';

export interface ParsedSSEEvent {
  event: StreamEventRaw;
  rawLine: string;
}

/**
 * Service for parsing and handling SSE (Server-Sent Events) streams.
 */
@Injectable({
  providedIn: 'root'
})
export class SSEService {
  /**
   * Parse SSE stream from a ReadableStream.
   * Yields parsed events as they arrive.
   */
  async *parseStream(
    body: ReadableStream<Uint8Array>
  ): AsyncGenerator<StreamEventRaw, void, unknown> {
    const reader = body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const rawLine of lines) {
          const event = this.parseLine(rawLine);
          if (event) {
            yield event;
          }
        }
      }

      // Process remaining buffer
      if (buffer.trim()) {
        const event = this.parseLine(buffer);
        if (event) {
          yield event;
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * Parse a single SSE line.
   */
  private parseLine(rawLine: string): StreamEventRaw | null {
    const line = rawLine.trim();
    if (!line || !line.startsWith('data:')) return null;

    const jsonStr = line.slice(5).trim();
    if (!jsonStr || jsonStr === '[DONE]') return null;

    try {
      return JSON.parse(jsonStr) as StreamEventRaw;
    } catch (e) {
      console.warn('Failed to parse SSE event', e, jsonStr);
      return null;
    }
  }

  /**
   * Extract usage statistics from various possible formats.
   */
  extractUsageStats(usageRaw: Record<string, unknown> | undefined): TokenUsage {
    const pick = (...keys: string[]): number | undefined => {
      if (!usageRaw) return undefined;
      for (const key of keys) {
        const value = this.toNumber(usageRaw[key]);
        if (value !== undefined) {
          return value;
        }
      }
      return undefined;
    };

    return {
      totalTokens: pick('total_tokens', 'totalTokens'),
      inputTokens: pick(
        'input_tokens',
        'prompt_tokens',
        'inputTokens',
        'promptTokens'
      ),
      outputTokens: pick(
        'output_tokens',
        'completion_tokens',
        'outputTokens',
        'completionTokens'
      )
    };
  }

  /**
   * Merge usage statistics (add new to previous).
   */
  mergeUsageStats(prev: TokenUsage | undefined, add: TokenUsage): TokenUsage {
    const prevTokens = prev ?? {};
    return {
      totalTokens: (prevTokens.totalTokens ?? 0) + (add.totalTokens ?? 0),
      inputTokens: (prevTokens.inputTokens ?? 0) + (add.inputTokens ?? 0),
      outputTokens: (prevTokens.outputTokens ?? 0) + (add.outputTokens ?? 0)
    };
  }

  private toNumber(value: unknown): number | undefined {
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
    if (typeof value === 'string') {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
    return undefined;
  }
}
