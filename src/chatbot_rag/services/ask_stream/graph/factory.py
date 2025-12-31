"""
Unified Agent Graph 工廠與建構邏輯。

LangGraph 14 節點工作流程：
```
START → guard → language_normalizer → cache_lookup
                                        ├─[hit]─→ cache_response → telemetry → END
                                        └─[miss]─→ intent_analyzer
                                                    ├─[direct]─→ response_synth ─┐
                                                    ├─[followup]→ followup_transform → response_synth
                                                    └─[retrieval]→ query_builder → tool_executor → reranker
                                                                                                      ↓
                                                                  result_evaluator ← chunk_expander ←┘
                                                                       ├─[retry]─→ query_builder
                                                                       └─[done]─→ response_synth → cache_store → telemetry → END
```

特性：
- Graph 編譯快取：避免每次請求重新編譯
- 快取鍵基於 base_llm_params、agent_backend、domain_id
- 節點建構函數透過閉包捕獲配置參數
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, cast

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from chatbot_rag.core.domain import DomainConfig, get_current_domain
from chatbot_rag.llm import State
from chatbot_rag.services.prompt_service import PromptService


# 模組層級的 Graph 快取字典
_graph_cache: Dict[str, CompiledStateGraph] = {}


def _make_cache_key(
    base_llm_params: dict,
    agent_backend: str,
    domain_id: str = "",
) -> str:
    """
    根據 LLM 參數、agent 後端和領域生成快取鍵。

    Args:
        base_llm_params: LLM 參數字典
        agent_backend: Agent 後端類型
        domain_id: 領域 ID

    Returns:
        str: 快取鍵（MD5 hash）
    """
    params_str = json.dumps(base_llm_params, sort_keys=True, default=str)
    key_source = f"{params_str}:{agent_backend}:{domain_id}"
    return hashlib.md5(key_source.encode("utf-8")).hexdigest()


def clear_graph_cache() -> None:
    """
    清除 Graph 快取。

    供測試或設定重載時使用。
    """
    global _graph_cache
    _graph_cache.clear()
    logger.info("[GRAPH_FACTORY] Graph cache cleared")


def get_graph_cache_stats() -> Dict[str, Any]:
    """
    取得 Graph 快取統計資訊。

    Returns:
        Dict: 包含快取數量等統計資訊
    """
    return {
        "cache_size": len(_graph_cache),
        "cache_keys": list(_graph_cache.keys()),
    }


from .nodes import (
    build_guard_node,
    build_cache_lookup_node,
    build_cache_response_node,
    build_cache_store_node,
    build_language_normalizer_node,
    build_followup_transform_node,
    build_query_builder_node,
    build_tool_executor_node,
    build_reranker_node,
    build_chunk_expander_node,
    build_response_node,
    build_telemetry_node,
    build_intent_analyzer_node,
    build_result_evaluator_node,
)
from .routing import (
    route_after_guard,
    route_after_cache_lookup,
    route_after_intent_analyzer,
    route_after_result_evaluator,
)


@dataclass(frozen=True)
class UnifiedAgentGraphFactory:
    """集中管理 Unified Agent Graph 節點註冊與邊定義。"""

    base_llm_params: dict
    agent_backend: str = "chat"
    prompt_service: Optional[PromptService] = field(default=None, hash=False)
    domain_config: Optional[DomainConfig] = field(default=None, hash=False)

    def compile(self):
        """編譯並回傳 LangGraph。"""
        builder = StateGraph(State)
        for name, node_builder in self._node_factories():
            builder.add_node(name, cast(Any, node_builder()))
        self._register_edges(builder)
        return builder.compile()

    def _node_factories(
        self,
    ) -> List[tuple[str, Callable[[], Callable[[State], State]]]]:
        """回傳所有節點名稱與其建構函數。"""
        return [
            ("guard", build_guard_node),
            ("cache_lookup", build_cache_lookup_node),
            ("cache_response", build_cache_response_node),
            (
                "language_normalizer",
                lambda: build_language_normalizer_node(
                    prompt_service=self.prompt_service,
                ),
            ),
            ("followup_transform", build_followup_transform_node),
            (
                "query_builder",
                lambda: build_query_builder_node(
                    self.base_llm_params,
                    agent_backend=self.agent_backend,
                    prompt_service=self.prompt_service,
                ),
            ),
            (
                "tool_executor",
                lambda: build_tool_executor_node(agent_backend=self.agent_backend),
            ),
            ("reranker", build_reranker_node),
            ("chunk_expander", build_chunk_expander_node),
            (
                "response_synth",
                lambda: build_response_node(
                    self.base_llm_params,
                    agent_backend=self.agent_backend,
                    prompt_service=self.prompt_service,
                ),
            ),
            ("cache_store", build_cache_store_node),
            (
                "telemetry",
                lambda: build_telemetry_node(prompt_service=self.prompt_service),
            ),
            (
                "intent_analyzer",
                lambda: build_intent_analyzer_node(
                    self.base_llm_params,
                    agent_backend=self.agent_backend,
                    prompt_service=self.prompt_service,
                    domain_config=self.domain_config,
                ),
            ),
            (
                "result_evaluator",
                lambda: build_result_evaluator_node(
                    self.base_llm_params,
                    agent_backend=self.agent_backend,
                    prompt_service=self.prompt_service,
                    domain_config=self.domain_config,
                ),
            ),
        ]

    def _register_edges(self, builder: StateGraph) -> None:
        """註冊所有邊與條件路由。"""
        builder.add_edge(START, "guard")

        # Guard → language_normalizer 或 end
        builder.add_conditional_edges(
            "guard",
            route_after_guard,
            {
                "language_normalizer": "language_normalizer",
                "end": END,
            },
        )

        # Language normalizer → cache_lookup（正規化後再查快取）
        builder.add_edge("language_normalizer", "cache_lookup")

        # Cache lookup → cache_response（命中）或 intent_analyzer（未命中）
        builder.add_conditional_edges(
            "cache_lookup",
            route_after_cache_lookup,
            {
                "cache_response": "cache_response",
                "intent_analyzer": "intent_analyzer",
            },
        )

        # Cache response → telemetry（快取命中路徑）
        builder.add_edge("cache_response", "telemetry")

        # Intent analyzer → followup_transform / response_synth / query_builder
        builder.add_conditional_edges(
            "intent_analyzer",
            route_after_intent_analyzer,
            {
                "followup_transform": "followup_transform",
                "response_synth": "response_synth",
                "query_builder": "query_builder",
            },
        )
        builder.add_edge("followup_transform", "response_synth")

        builder.add_edge("query_builder", "tool_executor")
        builder.add_edge("tool_executor", "reranker")
        builder.add_edge("reranker", "chunk_expander")
        builder.add_edge("chunk_expander", "result_evaluator")

        # Result evaluator → query_builder（重試）/ response_synth
        builder.add_conditional_edges(
            "result_evaluator",
            route_after_result_evaluator,
            {
                "query_builder": "query_builder",
                "response_synth": "response_synth",
            },
        )

        # Response → cache_store → telemetry
        builder.add_edge("response_synth", "cache_store")
        builder.add_edge("cache_store", "telemetry")
        builder.add_edge("telemetry", END)


def build_ask_graph(
    base_llm_params: dict,
    *,
    agent_backend: str = "chat",
    prompt_service: Optional[PromptService] = None,
    domain_config: Optional[DomainConfig] = None,
) -> CompiledStateGraph:
    """
    建立 Unified Agent 架構的 LangGraph。

    使用模組層級快取，避免每次請求都重新編譯 Graph。
    快取鍵基於 base_llm_params、agent_backend 和 domain_id。

    Args:
        base_llm_params: LLM 參數字典
        agent_backend: Agent 後端類型 (chat/responses)
        prompt_service: Prompt 服務（不影響快取鍵，因為 prompt 是動態載入的）
        domain_config: 領域設定（不影響快取鍵，僅影響節點行為）

    Returns:
        CompiledStateGraph: 編譯後的 LangGraph
    """
    global _graph_cache

    # 如果未提供 domain_config，使用當前領域
    if domain_config is None:
        domain_config = get_current_domain()

    domain_id = domain_config.domain_id if domain_config else ""
    cache_key = _make_cache_key(
        base_llm_params,
        agent_backend,
        domain_id,
    )

    if cache_key in _graph_cache:
        logger.debug(
            "[GRAPH_FACTORY] Cache hit for key=%s (backend=%s, domain=%s)",
            cache_key[:8],
            agent_backend,
            domain_id,
        )
        return _graph_cache[cache_key]

    logger.info(
        "[GRAPH_FACTORY] Cache miss, compiling graph for key=%s (backend=%s, domain=%s)",
        cache_key[:8],
        agent_backend,
        domain_id,
    )

    factory = UnifiedAgentGraphFactory(
        base_llm_params=base_llm_params,
        agent_backend=agent_backend,
        prompt_service=prompt_service,
        domain_config=domain_config,
    )
    compiled = factory.compile()

    _graph_cache[cache_key] = compiled

    logger.info(
        "[GRAPH_FACTORY] Graph compiled and cached (cache_size={})",
        len(_graph_cache),
    )

    return compiled
