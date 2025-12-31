"""
用戶回饋相關的 Pydantic Models。
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    """用戶回饋請求"""

    trace_id: str = Field(..., description="Langfuse trace ID")
    score: Literal["up", "down"] = Field(..., description="評分：up=讚, down=倒讚")
    comment: Optional[str] = Field(None, description="倒讚時的原因說明")


class FeedbackResponse(BaseModel):
    """用戶回饋回應"""

    success: bool
    message: str
    score_id: Optional[str] = None
