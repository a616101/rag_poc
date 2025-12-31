"""
用戶回饋服務模組。

提供用戶評分功能，將讚/倒讚評分送至 Langfuse。
"""

import hashlib
from typing import Optional

from langfuse import get_client
from loguru import logger


class FeedbackService:
    """用戶回饋服務，整合 Langfuse 評分 API。"""

    SCORE_NAME = "user_feedback"

    def _generate_score_id(self, trace_id: str) -> str:
        """
        產生確定性的 score_id，用於 idempotency。

        同一個 trace_id + score_name 組合會產生相同的 score_id，
        確保重複評分時會更新而非建立新紀錄。
        """
        key = f"{trace_id}:{self.SCORE_NAME}"
        return hashlib.sha256(key.encode()).hexdigest()[:32]

    def submit_feedback(
        self,
        trace_id: str,
        score: str,  # "up" or "down"
        comment: Optional[str] = None,
    ) -> tuple[bool, str, Optional[str]]:
        """
        提交用戶回饋到 Langfuse。

        Args:
            trace_id: Langfuse trace ID
            score: 評分類型 ("up" 或 "down")
            comment: 評論內容（選填，通常倒讚時提供）

        Returns:
            tuple[bool, str, Optional[str]]: (成功與否, 訊息, score_id)

        Note:
            - score="up" → value=1
            - score="down" → value=0
            - 使用 BOOLEAN data_type
            - 使用 score_id 作為 idempotency key，重複評分會更新既有紀錄
        """
        langfuse = get_client()
        value = 1 if score == "up" else 0
        score_id = self._generate_score_id(trace_id)

        try:
            langfuse.create_score(
                name=self.SCORE_NAME,
                value=value,
                trace_id=trace_id,
                data_type="BOOLEAN",
                comment=comment,
                score_id=score_id,
            )
            logger.info(
                f"[FEEDBACK] Submitted: trace_id={trace_id}, score={score}, "
                f"value={value}, score_id={score_id}"
            )
            return True, "Feedback submitted", score_id

        except Exception as e:
            logger.error(f"[FEEDBACK] Failed to submit: {e}")
            return False, str(e), None


feedback_service = FeedbackService()
