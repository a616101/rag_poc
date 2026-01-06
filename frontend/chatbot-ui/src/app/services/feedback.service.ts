import { Injectable, signal } from '@angular/core';
import { FeedbackPayload } from '../models';

/**
 * Service for handling user feedback submission.
 */
@Injectable({
  providedIn: 'root'
})
export class FeedbackService {
  readonly isLoading = signal(false);

  /**
   * Submit feedback to the backend.
   * @param apiBasePath - Base API path (e.g., '/api/v1/rag')
   * @param payload - Feedback payload
   * @returns Promise with success status
   */
  async submitFeedback(
    apiBasePath: string,
    payload: FeedbackPayload
  ): Promise<{ success: boolean; message?: string; score_id?: string }> {
    // Derive feedback endpoint from base path
    const feedbackUrl = `${apiBasePath}/feedback`;

    this.isLoading.set(true);
    try {
      const response = await fetch(feedbackUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();
      return {
        success: data.success ?? true,
        message: data.message,
        score_id: data.score_id
      };
    } catch (error) {
      console.error('Feedback submission failed:', error);
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Failed to submit feedback'
      };
    } finally {
      this.isLoading.set(false);
    }
  }
}
