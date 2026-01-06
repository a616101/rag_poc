import { Injectable, signal } from '@angular/core';
import {
  ChatMessage,
  ChatRole,
  AssistantDetails,
  createEmptyAssistantDetails
} from '../models';

/**
 * Service for managing chat message state.
 */
@Injectable({
  providedIn: 'root'
})
export class MessageService {
  readonly messages = signal<ChatMessage[]>([]);
  readonly isLoading = signal(false);
  readonly isThinking = signal(false);
  readonly errorMessage = signal<string | null>(null);

  private nextId = 1;
  private conversationSummary = '';
  private summaryEnabled = false;

  /**
   * Add a new message to the list.
   * @returns The message ID
   */
  addMessage(role: ChatRole, content: string): number {
    const id = this.nextId++;
    this.messages.update((list) => [
      ...list,
      {
        id,
        role,
        content,
        details: role === 'assistant' ? createEmptyAssistantDetails() : undefined
      }
    ]);
    return id;
  }

  /**
   * Append content to an existing message.
   */
  appendToMessage(id: number, chunk: string): void {
    if (!chunk) return;
    this.messages.update((list) =>
      list.map((m) =>
        m.id === id ? { ...m, content: m.content + chunk } : m
      )
    );
  }

  /**
   * Update assistant message details.
   */
  updateAssistantDetails(
    messageId: number,
    updater: (details: AssistantDetails) => void
  ): void {
    this.messages.update((list) =>
      list.map((m) => {
        if (m.id !== messageId || m.role !== 'assistant') {
          return m;
        }

        const baseDetails: AssistantDetails = m.details ?? createEmptyAssistantDetails();
        const copy = { ...baseDetails };
        updater(copy);
        return { ...m, details: copy };
      })
    );
  }

  /**
   * Remove a message by ID.
   */
  removeMessage(id: number): void {
    this.messages.update((list) => list.filter((m) => m.id !== id));
  }

  /**
   * Find a message by ID.
   */
  findMessage(id: number): ChatMessage | undefined {
    return this.messages().find((m) => m.id === id);
  }

  /**
   * Get messages before a given index (for regeneration).
   */
  getMessagesBefore(messageId: number): ChatMessage[] {
    const msgs = this.messages();
    const targetIndex = msgs.findIndex((m) => m.id === messageId);
    if (targetIndex === -1) return [];
    return msgs.slice(0, targetIndex);
  }

  /**
   * Build conversation history for API request.
   */
  buildConversationHistory(): { role: ChatRole; content: string }[] | null {
    const msgs = this.messages();
    if (!msgs.length) {
      return null;
    }
    return msgs.map((m) => ({ role: m.role, content: m.content }));
  }

  /**
   * Clear all messages and reset state.
   */
  clearMessages(): void {
    this.messages.set([]);
    this.errorMessage.set(null);
    this.nextId = 1;
    this.conversationSummary = '';
  }

  /**
   * Set conversation summary mode.
   */
  setSummaryEnabled(enabled: boolean): void {
    this.summaryEnabled = enabled;
    if (!enabled) {
      this.conversationSummary = '';
    }
  }

  /**
   * Update conversation summary.
   */
  updateConversationSummary(summary: unknown): void {
    if (!this.summaryEnabled) return;
    if (typeof summary === 'string') {
      this.conversationSummary = summary;
    } else if (summary === null) {
      this.conversationSummary = '';
    }
  }

  /**
   * Get conversation summary.
   */
  getConversationSummary(): string {
    return this.conversationSummary;
  }

  /**
   * Check if summary mode is enabled.
   */
  isSummaryEnabled(): boolean {
    return this.summaryEnabled;
  }
}
