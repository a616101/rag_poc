"""
Trace 報表相關的 Pydantic Models。

本模組定義了 Langfuse Trace 報表 API 的請求和響應數據模型。
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """
    背景任務狀態枚舉。

    定義報表生成任務的狀態：
    - PENDING: 等待處理
    - PROCESSING: 處理中
    - COMPLETED: 已完成
    - FAILED: 失敗
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TraceReportRequest(BaseModel):
    """
    Trace 報表請求模型。

    用於請求生成 Langfuse Trace 的 Excel 報表。

    Attributes:
        start_date: 開始日期（必填）
        end_date: 結束日期（必填）
        user_id: 用戶 ID 過濾（選填）
        session_id: 會話 ID 過濾（選填）
        trace_name: Trace 名稱過濾（選填）
        tags: 標籤過濾（選填）
        score_names: 要包含的評分類型（選填，None 表示全部）
        limit: 最大 trace 數量
    """

    start_date: datetime = Field(..., description="開始日期")
    end_date: datetime = Field(..., description="結束日期")

    user_id: Optional[str] = Field(None, description="用戶 ID 過濾")
    session_id: Optional[str] = Field(None, description="會話 ID 過濾")
    trace_name: Optional[str] = Field(None, description="Trace 名稱過濾")
    tags: Optional[list[str]] = Field(None, description="標籤過濾")

    score_names: Optional[list[str]] = Field(
        None,
        description="要包含的評分類型，如 ['user_feedback', 'accuracy']，None 表示全部",
    )

    limit: int = Field(
        default=500,
        le=2000,
        ge=1,
        description="最大 trace 數量",
    )


class TraceReportTaskResponse(BaseModel):
    """
    背景任務響應模型。

    當 trace 數量超過同步處理閾值時返回此響應。

    Attributes:
        status: 任務狀態
        task_id: 任務 ID
        message: 狀態訊息
        estimated_traces: 預估的 trace 數量
        download_url: 下載連結
    """

    status: str = Field(..., description="任務狀態")
    task_id: str = Field(..., description="任務 ID")
    message: str = Field(..., description="狀態訊息")
    estimated_traces: int = Field(..., description="預估的 trace 數量")
    download_url: str = Field(..., description="下載連結")


class TaskInfo(BaseModel):
    """
    任務資訊模型。

    用於追蹤背景任務的狀態和結果。

    Attributes:
        task_id: 任務 ID
        status: 任務狀態
        estimated_traces: 預估的 trace 數量
        created_at: 建立時間
        completed_at: 完成時間（選填）
        filename: 生成的檔案名稱（選填）
        result: 生成的 Excel bytes（選填）
        error: 錯誤訊息（選填）
    """

    task_id: str = Field(..., description="任務 ID")
    status: TaskStatus = Field(..., description="任務狀態")
    estimated_traces: int = Field(..., description="預估的 trace 數量")
    created_at: datetime = Field(..., description="建立時間")
    completed_at: Optional[datetime] = Field(None, description="完成時間")
    filename: Optional[str] = Field(None, description="生成的檔案名稱")
    result: Optional[bytes] = Field(None, description="生成的 Excel bytes", exclude=True)
    error: Optional[str] = Field(None, description="錯誤訊息")


class TaskStatusResponse(BaseModel):
    """
    任務狀態查詢響應模型。

    用於查詢背景任務的當前狀態。

    Attributes:
        task_id: 任務 ID
        status: 任務狀態
        message: 狀態訊息
        created_at: 建立時間
        completed_at: 完成時間（選填）
    """

    task_id: str = Field(..., description="任務 ID")
    status: str = Field(..., description="任務狀態")
    message: str = Field(..., description="狀態訊息")
    created_at: datetime = Field(..., description="建立時間")
    completed_at: Optional[datetime] = Field(None, description="完成時間")
