# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — builder: install dependencies into a virtual environment
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /build

# Install build tools required by some packages (e.g. asyncpg, pymupdf, paddleocr)
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources \
 && apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime: lean final image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies for OCR, healthchecks, and BGE/torch runtime
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources \
 && apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Persistent, writable cache dir for the BGE model weights (downloaded from
# HuggingFace on first use). Mounted as a volume in docker-compose.yml so
# the ~1.3GB model isn't re-downloaded on every container restart.
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source and migrations
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
RUN chown -R appuser:appgroup /app/.cache
USER appuser

EXPOSE 8000

# Uvicorn with 1 worker to enforce process-level concurrency constraints
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]