"""
Trace 報表 API 路由模組。

本模組提供 Langfuse Trace 報表生成功能端點，包括：
- Excel 報表生成
- 背景任務管理
- 報表下載
"""

import io

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from chatbot_rag.models.report import (
    TaskInfo,
    TaskStatus,
    TaskStatusResponse,
    TraceReportRequest,
    TraceReportTaskResponse,
)
from chatbot_rag.services.trace_report_service import trace_report_service


# 建立報表路由器，所有端點前綴為 /api/v1/reports
router = APIRouter(prefix="/api/v1/reports", tags=["Reports"])


@router.post("/traces/excel")
async def generate_trace_report(request: TraceReportRequest):
    """
    生成 Langfuse Trace Excel 報表。

    根據時間範圍和過濾條件，從 Langfuse 獲取 Trace 資料並生成 Excel 報表。
    包含三個工作表：Trace Summary、Observations Detail、Scores。

    ## 處理模式

    - **同步模式**（≤200 traces）：直接返回 Excel 檔案
    - **背景模式**（>200 traces）：返回任務 ID，稍後下載

    ## 報表內容

    1. **Trace Summary**：Trace 總覽，包含 ID、時間、Token 用量、成本、評分等
    2. **Observations Detail**：過程詳情，包含每個步驟的類型、模型、時長等
    3. **Scores**：評分詳情，包含評分名稱、值、評語等

    Args:
        request (TraceReportRequest): 報表請求，包含：
            - start_date (datetime): 開始日期（必填）
            - end_date (datetime): 結束日期（必填）
            - user_id (str, optional): 用戶 ID 過濾
            - session_id (str, optional): 會話 ID 過濾
            - trace_name (str, optional): Trace 名稱過濾
            - tags (list[str], optional): 標籤過濾
            - score_names (list[str], optional): 要包含的評分類型
            - limit (int): 最大 trace 數量（預設 500，最大 2000）

    Returns:
        StreamingResponse: Excel 檔案（同步模式）
        TraceReportTaskResponse: 任務資訊（背景模式）

    Raises:
        HTTPException:
            - 500: 如果報表生成失敗

    Example:
        **同步下載（小量資料）**:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/reports/traces/excel" \\
             -H "Content-Type: application/json" \\
             -d '{
               "start_date": "2024-01-01T00:00:00Z",
               "end_date": "2024-01-31T23:59:59Z",
               "limit": 100
             }' \\
             --output trace_report.xlsx
        ```

        **背景任務（大量資料）**:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/reports/traces/excel" \\
             -H "Content-Type: application/json" \\
             -d '{
               "start_date": "2024-01-01T00:00:00Z",
               "end_date": "2024-12-31T23:59:59Z",
               "limit": 1000
             }'
        ```

        背景任務 Response:
        ```json
        {
          "status": "processing",
          "task_id": "abc123...",
          "message": "報表生成中，請稍後下載",
          "estimated_traces": 500,
          "download_url": "/api/v1/reports/traces/download/abc123..."
        }
        ```

        **使用過濾條件**:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/reports/traces/excel" \\
             -H "Content-Type: application/json" \\
             -d '{
               "start_date": "2024-01-01T00:00:00Z",
               "end_date": "2024-01-31T23:59:59Z",
               "user_id": "user_123",
               "tags": ["production"],
               "score_names": ["user_feedback", "accuracy"]
             }'
        ```

    Note:
        - 時間範圍不限制，但大範圍查詢可能需要較長時間
        - 超過 200 筆 traces 會自動切換為背景任務模式
        - score_names 為 None 時會包含所有評分類型
    """
    try:
        logger.info(
            f"[REPORT] Request: {request.start_date} ~ {request.end_date}, "
            f"limit={request.limit}"
        )

        result = await trace_report_service.generate_report(request)

        if isinstance(result, bytes):
            # 同步返回 Excel
            filename = (
                f"trace_report_{request.start_date.date()}_"
                f"{request.end_date.date()}.xlsx"
            )
            logger.info(f"[REPORT] Returning Excel file: {filename}")

            return StreamingResponse(
                io.BytesIO(result),
                media_type=(
                    "application/vnd.openxmlformats-officedocument"
                    ".spreadsheetml.sheet"
                ),
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )
        else:
            # 返回背景任務資訊
            task_info: TaskInfo = result
            logger.info(f"[REPORT] Created background task: {task_info.task_id}")

            return TraceReportTaskResponse(
                status="processing",
                task_id=task_info.task_id,
                message="報表生成中，請稍後下載",
                estimated_traces=task_info.estimated_traces,
                download_url=f"/api/v1/reports/traces/download/{task_info.task_id}",
            )

    except Exception as e:
        logger.error(f"[REPORT] Failed to generate report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/traces/download/{task_id}")
async def download_trace_report(task_id: str):
    """
    下載已生成的報表或查詢任務狀態。

    根據任務 ID 查詢背景任務的狀態，如果任務已完成則返回 Excel 檔案。

    Args:
        task_id (str): 背景任務 ID

    Returns:
        StreamingResponse: Excel 檔案（任務已完成）
        TaskStatusResponse: 任務狀態（任務進行中）

    Raises:
        HTTPException:
            - 404: 任務不存在
            - 500: 報表生成失敗

    Example:
        **查詢任務狀態**:
        ```bash
        curl -X GET "http://localhost:8000/api/v1/reports/traces/download/abc123"
        ```

        進行中 Response:
        ```json
        {
          "task_id": "abc123",
          "status": "processing",
          "message": "報表生成中",
          "created_at": "2024-01-15T10:30:00Z",
          "completed_at": null
        }
        ```

        **下載完成的報表**:
        ```bash
        curl -X GET "http://localhost:8000/api/v1/reports/traces/download/abc123" \\
             --output trace_report.xlsx
        ```

    Note:
        - 任務完成後會直接返回 Excel 檔案
        - 任務失敗時會返回 HTTP 500 錯誤
        - 任務資訊儲存在記憶體中，服務重啟後會遺失
    """
    task_info = trace_report_service.get_task(task_id)

    if task_info is None:
        raise HTTPException(status_code=404, detail="任務不存在")

    if task_info.status == TaskStatus.COMPLETED:
        # 返回 Excel 檔案
        logger.info(f"[REPORT] Downloading completed task: {task_id}")

        return StreamingResponse(
            io.BytesIO(task_info.result),
            media_type=(
                "application/vnd.openxmlformats-officedocument"
                ".spreadsheetml.sheet"
            ),
            headers={
                "Content-Disposition": f"attachment; filename={task_info.filename}"
            },
        )

    elif task_info.status == TaskStatus.FAILED:
        raise HTTPException(
            status_code=500,
            detail=f"報表生成失敗：{task_info.error}",
        )

    else:
        # 返回任務狀態
        return TaskStatusResponse(
            task_id=task_info.task_id,
            status=task_info.status.value,
            message="報表生成中",
            created_at=task_info.created_at,
            completed_at=task_info.completed_at,
        )
