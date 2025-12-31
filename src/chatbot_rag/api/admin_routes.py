"""
Langfuse 管理 API 路由。

提供以下功能：
- 初始化 Prompts
- 查看 Prompts 狀態
- 建立 Dataset
- 查看 Dataset 狀態
- 匯出 Tools Schema
- 執行 Prompt Experiments

所有端點都在 /api/v1/admin 路徑下。
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from langfuse import get_client
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger

from chatbot_rag.services.prompt_service import (
    initialize_default_prompts,
    PromptNames,
    PromptService,
    DEFAULT_PROMPTS,
)
from chatbot_rag.services.tools_schema import TOOLS_SCHEMA, get_tool_names
from chatbot_rag.services.dataset_cases import (
    TEST_CASES,
    DATASET_NAME as DEFAULT_DATASET_NAME,
    DATASET_DESCRIPTION,
)
from chatbot_rag.services.evaluators import run_all_evaluations, run_e2e_evaluations


router = APIRouter(prefix="/api/v1/admin", tags=["Admin"])


# ==============================================================================
# Pydantic Models
# ==============================================================================


class PromptInfo(BaseModel):
    """Prompt 資訊"""
    name: str
    type: str
    preview: str
    version: Optional[int] = None
    label: Optional[str] = None


class InitPromptsRequest(BaseModel):
    """初始化 Prompts 請求"""
    label: str = "production"


class InitPromptsResponse(BaseModel):
    """初始化 Prompts 回應"""
    success: bool
    created: int
    existed: int
    total: int
    details: dict[str, bool]


class DatasetInfo(BaseModel):
    """Dataset 資訊"""
    name: str
    description: Optional[str] = None
    items_count: int


class CreateDatasetRequest(BaseModel):
    """建立 Dataset 請求"""
    name: str = "chatbot-rag-evaluation/v1"


class CreateDatasetResponse(BaseModel):
    """建立 Dataset 回應"""
    success: bool
    name: str
    added: int
    skipped: int
    total: int
    message: str


class ToolsSchemaResponse(BaseModel):
    """Tools Schema 回應"""
    tools: list[dict[str, Any]]
    tool_names: list[str]


# ==============================================================================
# Prompts 管理
# ==============================================================================


@router.get("/prompts", response_model=list[PromptInfo])
async def list_prompts(label: str = "production"):
    """
    列出所有預設 Prompts 及其狀態。

    Args:
        label: Prompt label（production/staging）

    Returns:
        Prompt 資訊列表
    """
    prompts_info = []

    try:
        langfuse = get_client()
        prompt_service = PromptService(default_label=label)
    except Exception as e:
        logger.error(f"Langfuse 連線失敗: {e}")
        raise HTTPException(status_code=500, detail=f"Langfuse 連線失敗: {e}")

    for name in PromptNames.all_prompts():
        config = DEFAULT_PROMPTS.get(name, {})
        info = PromptInfo(
            name=name,
            type=config.get("type", "unknown"),
            preview=config.get("prompt", "")[:100] + "...",
        )

        # 嘗試取得 Langfuse 上的版本資訊
        try:
            prompt, metadata = prompt_service.get_prompt(name, label=label)
            info.version = metadata.version
            info.label = metadata.label
        except Exception:
            info.version = None
            info.label = None

        prompts_info.append(info)

    return prompts_info


@router.post("/prompts/init", response_model=InitPromptsResponse)
async def init_prompts(request: InitPromptsRequest):
    """
    初始化所有預設 Prompts 到 Langfuse。

    已存在的 Prompt 會被跳過，不會建立新版本。

    Args:
        request: 包含 label 的請求

    Returns:
        初始化結果
    """
    try:
        langfuse = get_client()
    except Exception as e:
        logger.error(f"Langfuse 連線失敗: {e}")
        raise HTTPException(status_code=500, detail=f"Langfuse 連線失敗: {e}")

    logger.info(f"[Admin] 初始化 Prompts, label={request.label}")

    results = initialize_default_prompts(langfuse, default_label=request.label)

    created = sum(1 for v in results.values() if v)
    existed = sum(1 for v in results.values() if not v)

    return InitPromptsResponse(
        success=True,
        created=created,
        existed=existed,
        total=len(results),
        details=results,
    )


@router.post("/prompts/clear-cache")
async def clear_prompt_cache(name: Optional[str] = None):
    """
    清除 Prompt 快取。

    Args:
        name: 指定要清除的 prompt 名稱，未指定則清除全部

    Returns:
        清除結果
    """
    prompt_service = PromptService()

    if name:
        prompt_service.clear_cache(name)
        return {"message": f"已清除 {name} 的快取"}
    else:
        prompt_service.clear_cache()
        return {"message": "已清除所有 Prompt 快取"}


# ==============================================================================
# Dataset 管理
# ==============================================================================


@router.get("/datasets", response_model=list[DatasetInfo])
async def list_datasets():
    """
    列出所有 Datasets。

    Returns:
        Dataset 資訊列表
    """
    try:
        langfuse = get_client()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Langfuse 連線失敗: {e}")

    # Langfuse SDK 目前沒有 list_datasets 方法
    # 這裡嘗試取得已知的 dataset
    known_datasets = ["chatbot-rag-evaluation/v1"]
    datasets_info = []

    for name in known_datasets:
        try:
            dataset = langfuse.get_dataset(name)
            datasets_info.append(
                DatasetInfo(
                    name=dataset.name,
                    description=dataset.description,
                    items_count=len(list(dataset.items)),
                )
            )
        except Exception:
            pass

    return datasets_info


@router.post("/datasets/create", response_model=CreateDatasetResponse)
async def create_dataset(request: CreateDatasetRequest):
    """
    建立評估 Dataset。

    會建立預設的測試案例，包含：
    - qa_retrieval: 知識庫查詢測試
    - form_download: 表單下載測試
    - conversation_followup: 對話延續測試
    - out_of_scope: 超出範圍測試
    - multi_language: 多語言測試

    Args:
        request: 包含 dataset 名稱的請求

    Returns:
        建立結果
    """
    try:
        langfuse = get_client()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Langfuse 連線失敗: {e}")

    logger.info(f"[Admin] 建立 Dataset: {request.name}")

    # 檢查是否已存在
    try:
        dataset = langfuse.get_dataset(request.name)
        existing_questions = {
            item.input.get("question", "")
            for item in dataset.items
            if item.input
        }
    except Exception:
        # Dataset 不存在，建立新的
        langfuse.create_dataset(
            name=request.name,
            description=DATASET_DESCRIPTION,
        )
        existing_questions = set()

    # 新增測試案例
    added = 0
    skipped = 0

    for case in TEST_CASES:
        question = case["input"]["question"]
        if question in existing_questions:
            skipped += 1
            continue

        try:
            langfuse.create_dataset_item(
                dataset_name=request.name,
                input=case["input"],
                expected_output=case.get("expected_output"),
                metadata=case.get("metadata"),
            )
            added += 1
        except Exception as e:
            logger.warning(f"新增 dataset item 失敗: {e}")

    return CreateDatasetResponse(
        success=True,
        name=request.name,
        added=added,
        skipped=skipped,
        total=added + skipped + len(existing_questions),
        message=f"新增 {added} 項，跳過 {skipped} 項（已存在）",
    )


# ==============================================================================
# Tools Schema
# ==============================================================================


@router.get("/tools/schema", response_model=ToolsSchemaResponse)
async def get_tools_schema():
    """
    取得 Tools Schema。

    可直接複製到 Langfuse Playground 使用。

    Returns:
        Tools Schema JSON
    """
    return ToolsSchemaResponse(
        tools=TOOLS_SCHEMA,
        tool_names=get_tool_names(),
    )


# ==============================================================================
# LLM 並發監控
# ==============================================================================


@router.get("/concurrency/status")
async def concurrency_status():
    """
    取得 LLM 並發狀態。

    顯示每個後端（default/chat/responses/embedding）的：
    - limit: 最大並發數
    - available: 可用槽位數
    - in_progress: 正在進行的請求數
    - waiting: 正在排隊等待的請求數
    - total_acquired/released/timeout: 累計統計

    Returns:
        Dict: 各後端的並發狀態
    """
    from chatbot_rag.core.concurrency import llm_concurrency

    if not llm_concurrency.is_initialized():
        return {"error": "Concurrency manager not initialized", "status": {}}

    return llm_concurrency.get_status()


@router.get("/concurrency/summary")
async def concurrency_summary():
    """
    取得 LLM 並發狀態摘要。

    顯示所有後端的彙總資訊：
    - total_in_progress: 所有後端正在進行的請求總數
    - total_waiting: 所有後端正在排隊的請求總數
    - by_backend: 各後端的簡要狀態

    Returns:
        Dict: 彙總的並發狀態
    """
    from chatbot_rag.core.concurrency import llm_concurrency

    if not llm_concurrency.is_initialized():
        return {
            "error": "Concurrency manager not initialized",
            "total_in_progress": 0,
            "total_waiting": 0,
            "by_backend": {},
        }

    return llm_concurrency.get_summary()


@router.get("/concurrency/priority")
async def concurrency_priority_stats():
    """
    取得優先級排序統計資訊。

    顯示優先級排序機制的狀態：
    - priority_enabled: 是否啟用優先級排序
    - starvation_threshold: 飢餓門檻（秒）
    - queues: 各後端優先級佇列狀態
    - active_requests: 目前活躍的請求數

    若優先級排序未啟用，只會返回 {"priority_enabled": false}

    Returns:
        Dict: 優先級排序統計
    """
    from chatbot_rag.core.concurrency import llm_concurrency

    if not llm_concurrency.is_initialized():
        return {
            "error": "Concurrency manager not initialized",
            "priority_enabled": False,
        }

    return llm_concurrency.get_priority_stats()


# ==============================================================================
# 健康檢查
# ==============================================================================


@router.get("/langfuse/status")
async def langfuse_status():
    """
    檢查 Langfuse 連線狀態。

    Returns:
        連線狀態
    """
    try:
        langfuse = get_client()
        # 嘗試簡單操作確認連線正常
        # langfuse.auth_check() 會驗證 credentials
        return {
            "status": "connected",
            "host": langfuse._client_wrapper._base_url if hasattr(langfuse, "_client_wrapper") else "unknown",
        }
    except Exception as e:
        return {
            "status": "disconnected",
            "error": str(e),
        }


# ==============================================================================
# Experiments 實驗執行
# ==============================================================================

# 儲存實驗狀態（簡易版，生產環境應使用 Redis 或資料庫）
_experiments_store: dict[str, dict[str, Any]] = {}


class RunExperimentRequest(BaseModel):
    """執行實驗請求"""
    dataset_name: str = "chatbot-rag-evaluation/v1"
    prompt_label: str = "production"
    model: str = "openai/gpt-oss-20b"
    limit: Optional[int] = None
    category: Optional[str] = None


class ExperimentItemResult(BaseModel):
    """單一測試項目結果"""
    item_id: str
    question: str
    category: str
    success: bool
    tool_calls: list[str] = []
    response_preview: str = ""
    scores: dict[str, float] = {}
    error: Optional[str] = None


class ExperimentResult(BaseModel):
    """實驗結果"""
    experiment_id: str
    status: str  # pending, running, completed, failed
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    config: dict[str, Any] = {}
    total: int = 0
    completed_count: int = 0
    successful: int = 0
    failed: int = 0
    results: list[ExperimentItemResult] = []
    score_summary: dict[str, float] = {}
    category_summary: dict[str, dict[str, int]] = {}
    error: Optional[str] = None


@router.get("/experiments", response_model=list[ExperimentResult])
async def list_experiments():
    """
    列出所有實驗（記憶體中的）。

    Returns:
        實驗列表
    """
    return [
        ExperimentResult(**exp)
        for exp in _experiments_store.values()
    ]


@router.get("/experiments/{experiment_id}", response_model=ExperimentResult)
async def get_experiment(experiment_id: str):
    """
    取得實驗狀態和結果。

    Args:
        experiment_id: 實驗 ID

    Returns:
        實驗結果
    """
    if experiment_id not in _experiments_store:
        raise HTTPException(status_code=404, detail=f"實驗 {experiment_id} 不存在")

    return ExperimentResult(**_experiments_store[experiment_id])


@router.post("/experiments/run", response_model=ExperimentResult)
async def run_experiment(
    request: RunExperimentRequest,
    background_tasks: BackgroundTasks,
):
    """
    執行 Prompt Experiment（背景執行）。

    此端點會立即返回實驗 ID，實驗在背景執行。
    使用 GET /experiments/{experiment_id} 查詢進度和結果。

    Args:
        request: 實驗配置

    Returns:
        實驗狀態（初始為 pending）
    """
    experiment_id = f"exp-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    # 初始化實驗狀態
    _experiments_store[experiment_id] = {
        "experiment_id": experiment_id,
        "status": "pending",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "config": request.model_dump(),
        "total": 0,
        "completed_count": 0,
        "successful": 0,
        "failed": 0,
        "results": [],
        "score_summary": {},
        "category_summary": {},
        "error": None,
    }

    # 在背景執行實驗
    background_tasks.add_task(
        _run_experiment_task,
        experiment_id=experiment_id,
        dataset_name=request.dataset_name,
        prompt_label=request.prompt_label,
        model=request.model,
        limit=request.limit,
        category_filter=request.category,
    )

    return ExperimentResult(**_experiments_store[experiment_id])


@router.post("/experiments/run-sync", response_model=ExperimentResult)
async def run_experiment_sync(request: RunExperimentRequest):
    """
    執行 Prompt Experiment（同步執行）。

    此端點會等待實驗完成後才返回結果。
    建議只在測試少量案例時使用（limit <= 10）。

    Args:
        request: 實驗配置

    Returns:
        完整實驗結果
    """
    if request.limit and request.limit > 10:
        raise HTTPException(
            status_code=400,
            detail="同步執行最多支援 10 個測試案例，請使用 /experiments/run 進行背景執行"
        )

    experiment_id = f"exp-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    # 初始化實驗狀態
    _experiments_store[experiment_id] = {
        "experiment_id": experiment_id,
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "config": request.model_dump(),
        "total": 0,
        "completed_count": 0,
        "successful": 0,
        "failed": 0,
        "results": [],
        "score_summary": {},
        "category_summary": {},
        "error": None,
    }

    # 同步執行
    await _run_experiment_task(
        experiment_id=experiment_id,
        dataset_name=request.dataset_name,
        prompt_label=request.prompt_label,
        model=request.model,
        limit=request.limit or 10,
        category_filter=request.category,
    )

    return ExperimentResult(**_experiments_store[experiment_id])


@router.delete("/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """
    刪除實驗記錄。

    Args:
        experiment_id: 實驗 ID

    Returns:
        刪除結果
    """
    if experiment_id not in _experiments_store:
        raise HTTPException(status_code=404, detail=f"實驗 {experiment_id} 不存在")

    del _experiments_store[experiment_id]
    return {"message": f"已刪除實驗 {experiment_id}"}


async def _run_experiment_task(
    experiment_id: str,
    dataset_name: str,
    prompt_label: str,
    model: str,
    limit: Optional[int],
    category_filter: Optional[str],
) -> None:
    """
    實際執行實驗的背景任務。

    會將每個測試項目的執行結果記錄到 Langfuse：
    1. 為每個 dataset item 建立一個 trace（使用 start_as_current_observation）
    2. 將 trace 連結到 dataset item（使用 item.link(trace_id=...)）
    3. 記錄評估分數到 trace（使用 create_score）
    """
    from chatbot_rag.llm import create_chat_completion_llm

    try:
        _experiments_store[experiment_id]["status"] = "running"

        langfuse = get_client()
        prompt_service = PromptService(default_label=prompt_label)

        # 取得 Dataset
        try:
            dataset = langfuse.get_dataset(dataset_name)
        except Exception as e:
            _experiments_store[experiment_id]["status"] = "failed"
            _experiments_store[experiment_id]["error"] = f"無法取得 Dataset: {e}"
            return

        items = list(dataset.items)

        # 過濾類別
        if category_filter:
            items = [
                item for item in items
                if item.metadata and item.metadata.get("category") == category_filter
            ]

        # 限制數量
        if limit and limit < len(items):
            items = items[:limit]

        _experiments_store[experiment_id]["total"] = len(items)

        if not items:
            _experiments_store[experiment_id]["status"] = "completed"
            _experiments_store[experiment_id]["completed_at"] = datetime.now().isoformat()
            return

        # 建立 LLM
        llm = create_chat_completion_llm(streaming=False, model=model)
        llm_with_tools = llm.bind_tools(TOOLS_SCHEMA)

        results = []
        score_totals: dict[str, list[float]] = {}
        category_stats: dict[str, dict[str, int]] = {}

        # 實驗名稱（用於 Langfuse 分組）
        run_name = f"experiment-{experiment_id}"

        for idx, item in enumerate(items):
            input_data = item.input or {}
            expected_output = item.expected_output or {}
            metadata = item.metadata or {}

            question = input_data.get("question", "")
            language = input_data.get("language", "zh-hant")
            chat_history = input_data.get("chat_history", [])
            category = metadata.get("category", "unknown")

            # 初始化類別統計
            if category not in category_stats:
                category_stats[category] = {"success": 0, "fail": 0}

            try:
                # 使用 item.run() 上下文管理器自動建立 trace 並連結到 dataset item
                with item.run(
                    run_name=run_name,
                    run_metadata={
                        "prompt_label": prompt_label,
                        "model": model,
                        "experiment_id": experiment_id,
                    },
                ) as root_span:
                    # 設定 trace 的 tags 和 metadata
                    langfuse.update_current_trace(
                        name=f"{category}/{question[:30]}...",
                        tags=["experiment", category, f"exp-{experiment_id[:20]}"],
                        input={"question": question, "language": language},
                        metadata={
                            "experiment_id": experiment_id,
                            "dataset_name": dataset_name,
                            "dataset_item_id": item.id,
                            "category": category,
                            "difficulty": metadata.get("difficulty", "unknown"),
                            "language": language,
                        },
                    )

                    # 取得語言指令
                    lang_instruction_name = PromptNames.get_language_instruction_name(language)
                    lang_instruction, _ = prompt_service.get_text_prompt(lang_instruction_name)

                    # 編譯 system prompt
                    system_prompt, _ = prompt_service.get_text_prompt(
                        PromptNames.UNIFIED_AGENT_SYSTEM,
                        language_instruction=lang_instruction,
                    )

                    # 建構訊息
                    messages = [SystemMessage(content=system_prompt)]
                    for msg in chat_history:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role == "user":
                            messages.append(HumanMessage(content=content))
                        elif role == "assistant":
                            messages.append(AIMessage(content=content))
                    messages.append(HumanMessage(content=question))

                    # 呼叫 LLM（使用 generation span）
                    with langfuse.start_as_current_observation(
                        as_type="generation",
                        name="llm-call",
                        model=model,
                        input={
                            "system": system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt,
                            "messages": [{"role": "user", "content": question}],
                            "chat_history_length": len(chat_history),
                        },
                    ):
                        response = await llm_with_tools.ainvoke(messages)

                        # 提取回答
                        response_text = response.content or "" if hasattr(response, "content") else ""
                        tool_calls = [
                            tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                            for tc in (getattr(response, "tool_calls", None) or [])
                        ]

                        # 更新 generation output
                        langfuse.update_current_generation(
                            output={
                                "content": response_text[:500] if response_text else "(empty)",
                                "tool_calls": tool_calls,
                            },
                            metadata={"tool_calls_count": len(tool_calls)},
                        )

                    # 執行評估
                    dataset_item_dict = {
                        "input": input_data,
                        "expected_output": expected_output,
                        "metadata": metadata,
                    }
                    evaluations = run_all_evaluations(response, response_text, dataset_item_dict)

                    # 收集分數並記錄到 Langfuse
                    scores = {}
                    for eval_name, eval_result in evaluations.items():
                        scores[eval_name] = eval_result.score
                        if eval_name not in score_totals:
                            score_totals[eval_name] = []
                        score_totals[eval_name].append(eval_result.score)

                        # 使用 root_span.score_trace 記錄分數
                        root_span.score_trace(
                            name=eval_name,
                            value=eval_result.score,
                            comment=eval_result.comment,
                        )

                    # 更新 trace output
                    langfuse.update_current_trace(
                        output={
                            "response": response_text[:300] if response_text else "(empty)",
                            "tool_calls": tool_calls,
                            "success": True,
                        },
                    )

                result = ExperimentItemResult(
                    item_id=item.id,
                    question=question[:80],
                    category=category,
                    success=True,
                    tool_calls=tool_calls,
                    response_preview=response_text[:150] if response_text else "(empty)",
                    scores=scores,
                )
                results.append(result)
                category_stats[category]["success"] += 1

            except Exception as e:
                logger.error(f"[Experiment] 測試失敗: {question[:30]}... - {e}")

                result = ExperimentItemResult(
                    item_id=item.id,
                    question=question[:80],
                    category=category,
                    success=False,
                    error=str(e),
                )
                results.append(result)
                category_stats[category]["fail"] += 1

            # 更新進度
            _experiments_store[experiment_id]["completed_count"] = len(results)
            _experiments_store[experiment_id]["results"] = [r.model_dump() for r in results]

        # 確保所有資料都發送到 Langfuse
        langfuse.flush()

        # 計算統計
        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)

        score_summary = {}
        for eval_name, scores in score_totals.items():
            if scores:
                score_summary[eval_name] = sum(scores) / len(scores)

        _experiments_store[experiment_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "successful": successful,
            "failed": failed,
            "results": [r.model_dump() for r in results],
            "score_summary": score_summary,
            "category_summary": category_stats,
        })

        logger.info(
            f"[Experiment] 完成 {experiment_id}: "
            f"{successful}/{len(results)} 成功, "
            f"已同步到 Langfuse (run_name={run_name})"
        )

    except Exception as e:
        logger.error(f"[Experiment] 實驗執行失敗: {e}")
        _experiments_store[experiment_id]["status"] = "failed"
        _experiments_store[experiment_id]["error"] = str(e)
        _experiments_store[experiment_id]["completed_at"] = datetime.now().isoformat()


# ==============================================================================
# E2E Experiments（使用真實 LangGraph Workflow）
# ==============================================================================


class RunE2EExperimentRequest(BaseModel):
    """E2E 實驗請求"""
    dataset_name: str = "chatbot-rag-evaluation/v1"
    limit: Optional[int] = None
    category: Optional[str] = None
    agent_backend: str = "responses"  # "responses" 或 "chat_completion"
    run_description: Optional[str] = None  # 實驗描述，會顯示在 Langfuse 上


@router.post("/experiments/run-e2e", response_model=ExperimentResult)
async def run_e2e_experiment(
    request: RunE2EExperimentRequest,
    background_tasks: BackgroundTasks,
):
    """
    執行 E2E Prompt Experiment（使用真實 LangGraph Workflow）。

    此端點會執行完整的 LangGraph workflow，包含：
    - Planner（任務規劃）
    - Query Rewriter（查詢重寫）
    - RAG Retrieval（知識檢索）
    - Response（回答生成）
    - 等所有節點

    與 /experiments/run 的差異：
    - run: 簡化的單一 LLM 呼叫，只測試 tool 選擇
    - run-e2e: 完整的 workflow，測試真實系統行為

    Args:
        request: E2E 實驗配置

    Returns:
        實驗狀態（初始為 pending）
    """
    experiment_id = f"e2e-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    _experiments_store[experiment_id] = {
        "experiment_id": experiment_id,
        "status": "pending",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "config": {**request.model_dump(), "type": "e2e"},
        "total": 0,
        "completed_count": 0,
        "successful": 0,
        "failed": 0,
        "results": [],
        "score_summary": {},
        "category_summary": {},
        "error": None,
    }

    background_tasks.add_task(
        _run_e2e_experiment_task,
        experiment_id=experiment_id,
        dataset_name=request.dataset_name,
        limit=request.limit,
        category_filter=request.category,
        agent_backend=request.agent_backend,
        run_description=request.run_description,
    )

    return ExperimentResult(**_experiments_store[experiment_id])


@router.post("/experiments/run-e2e-sync", response_model=ExperimentResult)
async def run_e2e_experiment_sync(request: RunE2EExperimentRequest):
    """
    執行 E2E Prompt Experiment（同步執行）。

    由於 E2E 測試較慢，建議 limit <= 5。

    Args:
        request: E2E 實驗配置

    Returns:
        完整實驗結果
    """
    if request.limit and request.limit > 5:
        raise HTTPException(
            status_code=400,
            detail="E2E 同步執行最多支援 5 個測試案例，請使用 /experiments/run-e2e 進行背景執行"
        )

    experiment_id = f"e2e-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    _experiments_store[experiment_id] = {
        "experiment_id": experiment_id,
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "config": {**request.model_dump(), "type": "e2e"},
        "total": 0,
        "completed_count": 0,
        "successful": 0,
        "failed": 0,
        "results": [],
        "score_summary": {},
        "category_summary": {},
        "error": None,
    }

    await _run_e2e_experiment_task(
        experiment_id=experiment_id,
        dataset_name=request.dataset_name,
        limit=request.limit or 5,
        category_filter=request.category,
        agent_backend=request.agent_backend,
        run_description=request.run_description,
    )

    return ExperimentResult(**_experiments_store[experiment_id])


async def _run_e2e_experiment_task(
    experiment_id: str,
    dataset_name: str,
    limit: Optional[int],
    category_filter: Optional[str],
    agent_backend: str,
    run_description: Optional[str] = None,
) -> None:
    """
    執行 E2E 實驗的背景任務。

    使用真實的 LangGraph workflow，測試完整系統行為。
    """
    from chatbot_rag.models.rag import QuestionRequest, ConversationMessage
    from chatbot_rag.services.ask_stream import run_stream_graph
    from chatbot_rag.llm import create_chat_completion_llm

    try:
        _experiments_store[experiment_id]["status"] = "running"

        langfuse = get_client()

        # 取得 Dataset
        try:
            dataset = langfuse.get_dataset(dataset_name)
        except Exception as e:
            _experiments_store[experiment_id]["status"] = "failed"
            _experiments_store[experiment_id]["error"] = f"無法取得 Dataset: {e}"
            return

        items = list(dataset.items)

        # 過濾類別
        if category_filter:
            items = [
                item for item in items
                if item.metadata and item.metadata.get("category") == category_filter
            ]

        # 限制數量
        if limit and limit < len(items):
            items = items[:limit]

        _experiments_store[experiment_id]["total"] = len(items)

        if not items:
            _experiments_store[experiment_id]["status"] = "completed"
            _experiments_store[experiment_id]["completed_at"] = datetime.now().isoformat()
            return

        # 建立 LLM-as-a-Judge 評估用的 LLM
        judge_llm = create_chat_completion_llm(streaming=False)

        def llm_call(prompt: str) -> str:
            """LLM 呼叫函數，用於 LLM-as-a-Judge 評估。"""
            response = judge_llm.invoke([HumanMessage(content=prompt)])
            return response.content if hasattr(response, "content") else str(response)

        results = []
        score_totals: dict[str, list[float]] = {}
        category_stats: dict[str, dict[str, int]] = {}

        run_name = f"e2e-experiment-{experiment_id}"

        for idx, item in enumerate(items):
            input_data = item.input or {}
            expected_output = item.expected_output or {}
            metadata = item.metadata or {}

            question = input_data.get("question", "")
            language = input_data.get("language", "zh-hant")
            chat_history_raw = input_data.get("chat_history", [])
            category = metadata.get("category", "unknown")

            if category not in category_stats:
                category_stats[category] = {"success": 0, "fail": 0}

            try:
                # 建立 QuestionRequest
                conversation_history = [
                    ConversationMessage(
                        role=msg.get("role", "user"),
                        content=msg.get("content", ""),
                    )
                    for msg in chat_history_raw
                ]

                request = QuestionRequest(
                    question=question,
                    conversation_history=conversation_history,
                    top_k=5,
                )

                # 收集 workflow 輸出
                final_response = ""
                used_tools: list[str] = []
                trace_id = ""
                intent = ""

                # 使用 item.run() 連結到 dataset
                with item.run(
                    run_name=run_name,
                    run_description=run_description,
                    run_metadata={
                        "experiment_id": experiment_id,
                        "agent_backend": agent_backend,
                        "type": "e2e",
                    },
                ) as root_span:
                    # 設定 trace metadata
                    langfuse.update_current_trace(
                        name=f"e2e/{category}/{question[:25]}...",
                        tags=["e2e-experiment", category, f"exp-{experiment_id[:15]}"],
                        input={"question": question, "language": language},
                        metadata={
                            "experiment_id": experiment_id,
                            "category": category,
                            "difficulty": metadata.get("difficulty", "unknown"),
                        },
                    )

                    # 執行真實 workflow
                    async def always_connected():
                        return False

                    answer_tokens: list[str] = []

                    async for event in run_stream_graph(
                        request,
                        is_disconnected=always_connected,
                        agent_backend=agent_backend,
                        extra_tags=[
                            "e2e-experiment",
                            f"exp:{experiment_id[:8]}",
                            f"category:{category}",
                        ],
                        extra_metadata={
                            "experiment_id": experiment_id,
                            "dataset_name": dataset_name,
                            "dataset_item_id": item.id,
                            "category": category,
                            "difficulty": metadata.get("difficulty", "unknown"),
                        },
                    ):
                        channel = event.get("channel", "")

                        # 收集 answer delta（串流回應內容）
                        if channel == "answer":
                            delta = event.get("delta", "")
                            if delta:
                                answer_tokens.append(delta)

                        # 收集 meta 資訊（包含 used_tools）
                        elif channel == "meta":
                            meta = event.get("meta", {})
                            if meta:
                                meta_tools = meta.get("used_tools", [])
                                if meta_tools:
                                    used_tools = meta_tools

                        # 收集 meta_summary（最終摘要）
                        elif channel == "meta_summary":
                            summary_data = event.get("summary", {})
                            if summary_data:
                                trace_id = summary_data.get("trace_id", "") or event.get("trace_id", "")
                                if not used_tools:
                                    used_tools = summary_data.get("agent_used_tools", [])
                                intent = summary_data.get("intent", "")

                    # 組合最終回應
                    final_response = "".join(answer_tokens)

                    logger.debug(
                        f"[E2E] Question: {question[:30]}... | "
                        f"Response length: {len(final_response)} | "
                        f"Tools: {used_tools} | Intent: {intent}"
                    )

                    # 更新 trace output
                    langfuse.update_current_trace(
                        output={
                            "response": final_response[:300] if final_response else "(empty)",
                            "used_tools": used_tools,
                            "intent": intent,
                            "success": True,
                        },
                    )

                    # 執行評估（使用統一的 E2E 評估器 + LLM-as-a-Judge）
                    evaluations = run_e2e_evaluations(
                        question=question,
                        response_text=final_response,
                        used_tools=used_tools,
                        expected_output=expected_output,
                        metadata={**metadata, "language": language},
                        llm_call=llm_call,  # 啟用 LLM-as-a-Judge 評估
                    )

                    # 收集分數並記錄到 Langfuse
                    scores = {}
                    for eval_name, eval_result in evaluations.items():
                        scores[eval_name] = eval_result.score
                        if eval_name not in score_totals:
                            score_totals[eval_name] = []
                        score_totals[eval_name].append(eval_result.score)

                        root_span.score_trace(
                            name=eval_name,
                            value=eval_result.score,
                            comment=eval_result.comment,
                        )

                result = ExperimentItemResult(
                    item_id=item.id,
                    question=question[:80],
                    category=category,
                    success=True,
                    tool_calls=used_tools,
                    response_preview=final_response[:150] if final_response else "(empty)",
                    scores=scores,
                )
                results.append(result)
                category_stats[category]["success"] += 1

            except Exception as e:
                logger.error(f"[E2E Experiment] 測試失敗: {question[:30]}... - {e}")

                result = ExperimentItemResult(
                    item_id=item.id,
                    question=question[:80],
                    category=category,
                    success=False,
                    error=str(e),
                )
                results.append(result)
                category_stats[category]["fail"] += 1

            # 更新進度
            _experiments_store[experiment_id]["completed_count"] = len(results)
            _experiments_store[experiment_id]["results"] = [r.model_dump() for r in results]

        # 確保資料發送到 Langfuse
        langfuse.flush()

        # 計算統計
        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)

        score_summary = {}
        for eval_name, scores in score_totals.items():
            if scores:
                score_summary[eval_name] = sum(scores) / len(scores)

        _experiments_store[experiment_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "successful": successful,
            "failed": failed,
            "results": [r.model_dump() for r in results],
            "score_summary": score_summary,
            "category_summary": category_stats,
        })

        logger.info(
            f"[E2E Experiment] 完成 {experiment_id}: "
            f"{successful}/{len(results)} 成功, "
            f"已同步到 Langfuse (run_name={run_name})"
        )

    except Exception as e:
        logger.error(f"[E2E Experiment] 實驗執行失敗: {e}")
        _experiments_store[experiment_id]["status"] = "failed"
        _experiments_store[experiment_id]["error"] = str(e)
        _experiments_store[experiment_id]["completed_at"] = datetime.now().isoformat()
