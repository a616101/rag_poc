# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChatBot GraphRAG is a production-ready **GraphRAG** (Graph-enhanced Retrieval-Augmented Generation) chatbot API built with FastAPI, LangGraph, and multiple vector/graph databases. It combines:

- **Vector Retrieval** (Dense + Sparse + Full-text via Qdrant + OpenSearch)
- **Knowledge Graph** (Entity-Relation Graph via NebulaGraph)
- **Community Detection** (Leiden Algorithm)
- **Multi-mode Retrieval** (LOCAL / GLOBAL / DRIFT)

The system supports multi-domain configurations and includes an embeddable chat widget.

## Common Commands

```bash
# Development server (with auto-reload)
uv run graphrag-dev

# Production server
uv run graphrag-prod

# Database migrations
uv run graphrag-db upgrade

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
docker compose --profile development up -d
docker compose logs -f app-dev
docker compose --profile test run --rm test
```

## Architecture

### Core Flow: LangGraph Computational Graph

The system uses a 22-node LangGraph workflow for intelligent question answering:

```
START
  |
[guard] ─── OWASP LLM01 security check (prompt injection detection)
  |
[acl] ─── Multi-tenant access control (tenant_id + acl_groups)
  |
[normalize] ─── Language detection & question normalization
  |
[cache] ─── Version-aware semantic cache lookup
  |       |
  |   [HIT] → Return cached response
  |       |
  |   [MISS] ↓
  |
[intent] ─── Intent analysis (direct/followup/retrieval)
  |          + Query mode selection (local/global/drift)
  |
  ├─[direct]─────→ [output]
  |
  ├─[followup]───→ [context] → [output]
  |
  └─[retrieval]──→ [decompose] → [retrieve] → [graph_query]
                         |              |            |
                         ↓              ↓            ↓
                   Sub-queries    Vector+BM25   2-hop traversal
                         |              |            |
                         └──────────────┴────────────┘
                                        |
                                        ↓
                                   [rerank] ─── Cross-encoder reranking
                                        |
                                        ↓
                                   [quality] ─── Groundedness evaluation
                                        |
                                  ┌─────┴─────┐
                              [PASS]       [RETRY]
                                  |            |
                                  ↓            ↓
                              [output]    [decompose] (with different strategy)
                                  |
                                  ↓
                              [cache_store] → [telemetry] → END
```

**Graph nodes** (in `src/chatbot_graphrag/graph_workflow/nodes/`):

| Node | Purpose |
|------|---------|
| `guard.py` | OWASP LLM01 prompt injection detection (regex + LLM verification) |
| `acl.py` | Multi-tenant access control |
| `normalize.py` | Language detection, question normalization |
| `cache.py` | Version-aware semantic cache (index_version + prompt_version) |
| `intent.py` | Intent analysis and query mode routing |
| `context.py` | Conversation context management |
| `retrieval.py` | Hybrid search (Dense + Sparse + BM25 + RRF) |
| `graph.py` | NebulaGraph 2-hop traversal |
| `rerank.py` | Cross-encoder reranking |
| `quality.py` | Groundedness evaluation (heuristic + 10% Ragas sampling) |
| `output.py` | Final response generation with SSE streaming |
| `status.py` | Status transition tracking |

### Service Layer Architecture

Key services in `src/chatbot_graphrag/services/`:

**Graph Services (`graph/`)**:
- `nebula_client.py` - NebulaGraph connection and operations
- `entity_extractor.py` - LLM-driven entity extraction
- `relation_extractor.py` - Relation extraction
- `community_detector.py` - Leiden community detection
- `community_summarizer.py` - Community summary generation
- `batch_loader.py` - Batch loading to graph database

**Ingestion Services (`ingestion/`)**:
- `coordinator.py` - Ingestion workflow coordination
- `curated_pipeline.py` - YAML + Markdown file processing
- `raw_pipeline.py` - PDF/DOCX/HTML processing
- `schema_validator.py` - Schema validation

**LLM Services (`llm/`)**:
- `factory.py` - Multi-backend support (OpenAI, DeepSeek, etc.)
- `concurrent_llm.py` - Concurrency control with priority queue
- `responses_accumulator.py` - Streaming response accumulation

**Retrieval Services (`retrieval/`)**:
- `local_mode.py` - Entity-based local graph search
- `global_mode.py` - Community-based global retrieval
- `drift_mode.py` - Dynamic multi-round exploration

**Search Services (`search/`)**:
- `opensearch_service.py` - Full-text search backend
- `hybrid_search.py` - Vector + keyword hybrid search

**Cache Services (`cache/`)**:
- Semantic similarity-based query caching with version awareness

**Vector Services (`vector/`)**:
- `qdrant_service.py` - Vector database management (3 collections)

**Storage Services (`storage/`)**:
- `minio_service.py` - S3-compatible object storage

### API Layer

Routes in `src/chatbot_graphrag/api/routes/`:

| Route File | Endpoints | Purpose |
|------------|-----------|---------|
| `ask_stream.py` | `/api/v1/rag/ask/stream` | Responses API format streaming |
| `ask_stream_chat.py` | `/api/v1/rag/ask/stream_chat` | OpenAI Chat API compatible |
| `vectorize.py` | `/api/v1/rag/vectorize` | Async document ingestion |
| `cache_admin.py` | `/api/v1/admin/cache` | Cache management |

### Database Layer

**PostgreSQL** (`db/`):
- SQLAlchemy ORM models for documents, chunks, ACL
- Alembic migrations
- Repository pattern for data access

**NebulaGraph**:
- Entity-Relation graph with 14 entity types, 11 relation types
- Community structure (Level 0-3)

**Qdrant** (3 collections):
- `graphrag_chunks` - Document chunks
- `graphrag_entities` - Extracted entities
- `graphrag_communities` - Community summaries

**OpenSearch**:
- BM25 full-text search index

### Configuration

All settings via environment variables, managed in `src/chatbot_graphrag/core/config.py` using Pydantic Settings.

Key env vars:
- `QDRANT_URL`, `QDRANT_COLLECTION_*` - Vector DB config
- `NEBULA_HOST`, `NEBULA_PORT` - Graph DB config
- `OPENSEARCH_URL` - Full-text search config
- `POSTGRES_URL` - Relational DB config
- `OPENAI_API_BASE`, `OPENAI_API_KEY` - LLM API config
- `EMBEDDING_MODEL`, `CHAT_MODEL` - Model names
- `LOG_LEVEL`, `LOG_TO_CONSOLE`, `LOG_TO_FILE` - Logging config
- `DOMAIN` - Domain configuration (e.g., `hospital`)

## Frontend

### Chat Widget (Embeddable)
Location: `frontend/chat-widget/`
- **Tech Stack**: Svelte 4 + Vite + Tailwind CSS
- **Purpose**: Embeddable chat widget for external websites
- **Development**: `npm run dev` (port 4202)
- **Production**: `npm run build` → `dist/`

### Chatbot UI (Main Application)
Location: `frontend/chatbot-ui/`
- **Tech Stack**: Angular 20+
- **Purpose**: Full-featured chat interface
- **Development**: `ng serve` (port 4200)

## Key Design Decisions

1. **GraphRAG Architecture**: Combines vector retrieval with knowledge graph for better context understanding
2. **Multi-mode Retrieval**: LOCAL (entity-centric), GLOBAL (community-centric), DRIFT (exploratory)
3. **Hybrid Search**: Dense + Sparse + BM25 with Reciprocal Rank Fusion (RRF)
4. **LLM Concurrency Control**: Multi-backend semaphores with priority scheduling
5. **Version-aware Caching**: Cache invalidation based on index_version + prompt_version
6. **OWASP Security**: LLM01 (Prompt Injection) + LLM02 (Output Handling) protection
7. **Multi-tenant ACL**: tenant_id + acl_groups based access control
8. **SSE Streaming**: Real-time response streaming via `sse-starlette`
9. **ORJSON**: 2-3x faster JSON serialization
10. **Langfuse Integration**: All LLM calls traced for observability

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
- **NebulaGraph** - Graph database (port 9669)
- **OpenSearch** - Full-text search (port 9200)
- **PostgreSQL** - Relational database (port 5432)
- **Redis** - Caching (port 6379)
- **MinIO** - Object storage (port 9000)
- **LLM API** - OpenAI-compatible endpoint (LMStudio, TWCC, etc.)
- **Langfuse** - Observability platform (optional)
