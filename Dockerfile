# Build stage for dependencies
FROM python:3.13-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install system dependencies for Playwright (needed during build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set environment variables for uv
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

# Copy dependency files and README (README is required by hatchling)
COPY pyproject.toml uv.lock README.md ./

# Copy source code (required for editable install)
COPY src ./src

# Install dependencies with cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Install Playwright browsers with cache mount
RUN --mount=type=cache,target=/root/.cache/ms-playwright \
    uv run playwright install chromium

# Copy remaining files
COPY rag_test_data ./rag_test_data

# Development stage (no bytecode compilation)
FROM python:3.13-slim AS development

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Playwright system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set environment variables for development (disable bytecode compilation)
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0 \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy dependency files and README (README is required by hatchling)
COPY pyproject.toml uv.lock README.md ./

# Copy source code (required for editable install)
COPY src ./src

# Install dependencies with cache mount (survives layer rebuilds)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Install Playwright browsers with cache mount
# Cache mount speeds up downloads, files are installed to default location
RUN --mount=type=cache,target=/root/.cache/ms-playwright-download \
    PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright-download \
    uv run playwright install chromium && \
    mkdir -p /root/.cache/ms-playwright && \
    cp -r /root/.cache/ms-playwright-download/* /root/.cache/ms-playwright/

# Copy remaining files (will be overridden by volume mount in docker-compose)
COPY files ./files
COPY rag_test_data ./rag_test_data

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Default command for development
CMD ["uv", "run", "chatbot-dev"]

# Runtime stage (production)
FROM python:3.13-slim AS production

# Install Playwright system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy virtual environment from builder (includes all dependencies)
COPY --from=builder /app/.venv /app/.venv

# Copy Playwright browsers from builder
COPY --from=builder /root/.cache/ms-playwright /root/.cache/ms-playwright

# Copy application code
COPY src ./src
COPY files ./files
COPY rag_test_data ./rag_test_data
COPY pyproject.toml uv.lock README.md ./

# Create logs directory
RUN mkdir -p logs

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "chatbot_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
