# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chatbot RAG is a production-ready RAG-based chatbot API built with FastAPI, LangGraph, and Qdrant. It implements an **Agentic RAG** architecture where an intelligent agent plans, decides, and executes retrieval operations dynamically rather than following a fixed pipeline. The system supports multi-domain configurations and includes an embeddable chat widget.

## Common Commands

```bash
# Development server (with auto-reload)
uv run chatbot-dev

# Production server (configurable workers)
uv run chatbot-start

# High-performance production (auto-calculates workers: CPU*2+1)
uv run chatbot-prod

# Run all tests
uv run pytest

# Run specific test categories
uv run pytest -m security -v          # Security tests
uv run pytest -m injection -v         # Prompt injection tests
uv run pytest tests/test_api.py -v    # API tests only

# Format and lint
uv run ruff format .
uv run ruff check .
uv run ruff check --fix .

# Docker development
docker compose up app-dev
docker compose --profile test run --rm test
docker compose up chat-widget-dev     # Chat widget development
```

## Architecture

### Core Flow: LangGraph Computational Graph

The system uses a 14-node LangGraph workflow for intelligent question answering:

```
START → guard → language_normalizer → cache_lookup
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼ [hit]                                   ▼ [miss]
              cache_response                           intent_analyzer
                    │                                         │
                    │              ┌──────────────────────────┼──────────────────────────┐
                    │              ▼ [direct]                 ▼ [followup]               ▼ [retrieval]
                    │        response_synth           followup_transform           query_builder
                    │              │                          │                          │
                    │              │                          ▼                          ▼
                    │              │                    response_synth            tool_executor
                    │              │                          │                          │
                    │              │                          │                          ▼
                    │              │                          │                      reranker
                    │              │                          │                          │
                    │              │                          │                          ▼
                    │              │                          │                   chunk_expander
                    │              │                          │                          │
                    │              │                          │                          ▼
                    │              │                          │                  result_evaluator
                    │              │                          │                    │           │
                    │              │                          │              [retry]│           │[done]
                    │              │                          │                    ▼           │
                    │              │                          │              query_builder     │
                    │              │                          │                                │
                    │              │                          │                                ▼
                    │              └──────────────────────────┴───────────────────────→ response_synth
                    │                                                                         │
                    │                                                                         ▼
                    │                                                                    cache_store
                    │                                                                         │
                    └─────────────────────────────────────────────────────────────────────────┘
                                                              │
                                                              ▼
                                                          telemetry → END
```

**Graph nodes** (in `src/chatbot_rag/services/ask_stream/graph/nodes/`):

| Node | Purpose |
|------|---------|
| `guard.py` | Input safety checks and prompt injection detection |
| `language_normalizer.py` | Language detection, question normalization |
| `cache_lookup.py` | Semantic query cache lookup |
| `cache_response.py` | Serve cached responses |
| `intent_analyzer.py` | Intent analysis and routing (direct/followup/retrieval) |
| `followup_transform.py` | Follow-up question context handling |
| `query_builder.py` | LLM rewrites user question into retrieval query with decomposition |
| `tool_executor.py` | Executes tools (retrieve_documents, get_form_links) |
| `reranker.py` | Cross-encoder reranking of retrieved documents |
| `chunk_expander.py` | Adaptive chunk expansion for context enrichment |
| `result_evaluator.py` | Evaluates retrieval quality, triggers retry if needed |
| `response.py` | Generates final response with SSE streaming |
| `cache_store.py` | Store responses in semantic cache |
| `telemetry.py` | Tracing and observability via Langfuse |

### Service Layer (Singleton Pattern)

Key services in `src/chatbot_rag/services/`:

**Core RAG Services:**
- `retriever_service.py` - Document retrieval with progressive thresholds (0.65 → 0.50 → 0.35)
- `embedding_service.py` - Text embedding via OpenAI-compatible API
- `qdrant_service.py` - Vector database operations
- `semantic_cache_service.py` - Semantic query caching
- `reranker_service.py` - Cross-encoder reranking
- `contextual_chunking_service.py` - Context-aware document chunking

**LLM & Prompt Services:**
- `prompt_service.py` - Langfuse prompt management
- `query_variation.py` - Query variation generation
- `agent_tools.py` - Tool definitions for the agent

**Content Services:**
- `document_service.py` - Document management
- `web_scraper_service.py` - Web content scraping
- `crawl4ai_client.py` - Crawl4AI integration
- `markdown_parser.py` - Markdown parsing

**Evaluation & Reporting:**
- `evaluators.py` - Evaluation framework
- `dataset_cases.py` - Dataset test cases
- `trace_report_service.py` - Trace report generation
- `feedback_service.py` - User feedback handling

### API Layer

Routes in `src/chatbot_rag/api/`:
- `rag_routes.py` - RAG endpoints (`/api/v1/rag/ask`, `/api/v1/rag/ask-stream`)
- `admin_routes.py` - Admin endpoints (document management, Langfuse prompts, datasets, experiments)
- `file_routes.py` - File upload/download
- `cache_routes.py` - Cache management
- `scraper_routes.py` - Web scraping endpoints
- `report_routes.py` - Report generation endpoints

### Multi-Domain Support

Domain-specific configurations in `src/chatbot_rag/domains/`:
- `core/domain.py` - Base domain configuration
- `hospital/prompts.py` - Hospital domain prompts
- `hospital/fallbacks.py` - Hospital domain fallback responses

### LLM Factory System

LLM management in `src/chatbot_rag/llm/`:
- `factory.py` - LLM factory with multiple model backends
- `responses_chat_model.py` - Custom chat model implementation
- `graph_nodes.py` - Unified graph state management

### Configuration

All settings via environment variables, managed in `src/chatbot_rag/core/config.py` using Pydantic Settings.

Key env vars:
- `QDRANT_URL`, `QDRANT_COLLECTION_NAME` - Vector DB config
- `OPENAI_API_BASE`, `OPENAI_API_KEY` - LLM API config
- `EMBEDDING_MODEL`, `CHAT_MODEL` - Model names
- `LOG_LEVEL`, `LOG_TO_CONSOLE`, `LOG_TO_FILE` - Logging config
- `DOMAIN` - Domain configuration (e.g., `hospital`)

## Frontend

### Chat Widget (Embeddable)
Location: `frontend/chat-widget/`
- **Tech Stack**: Svelte + Vite + Tailwind CSS
- **Purpose**: Embeddable chat widget for external websites
- **Development**: `docker compose up chat-widget-dev` (port 4202)
- **Production**: `docker compose up chat-widget` (port 4203)

### Chatbot UI (Main Application)
Location: `frontend/chatbot-ui/`
- **Tech Stack**: Angular

## Key Design Decisions

1. **Progressive Retrieval**: Three threshold levels (0.65, 0.50, 0.35) with automatic retry on failure
2. **Semantic Caching**: Query-level caching with semantic similarity matching
3. **Cross-Encoder Reranking**: Improves retrieval precision after initial vector search
4. **Chunk Expansion**: Adaptive context expansion for better response quality
5. **Agentic Intent Analysis**: LLM-based intent analyzer decides routing (direct/followup/retrieval)
6. **SSE Streaming**: Real-time response streaming via `sse-starlette`
7. **ORJSON**: Used for 2-3x faster JSON serialization
8. **Langfuse Integration**: All LLM calls traced for observability
9. **Multi-Domain Architecture**: Configurable domain-specific prompts and fallbacks

## Testing

Pytest markers for targeted testing:
- `security` - All security tests
- `injection` - Prompt injection tests
- `jailbreak` - Jailbreak attack tests
- `legitimate` - Legitimate question tests (no false positives)
- `comprehensive` - Full attack test suite (60+ cases)

Test data in `tests/fixtures/attack_test_cases.json`.

## External Dependencies

- **Qdrant** - Vector database (port 6333)
- **LLM API** - OpenAI-compatible endpoint (LMStudio, TWCC, etc.)
- **Langfuse** - Observability platform (optional)
- **Crawl4AI** - Web scraping service (optional)
