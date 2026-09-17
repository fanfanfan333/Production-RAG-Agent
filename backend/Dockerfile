# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — builder: install dependencies into a virtual environment
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /build

# ── 让解析器优先用 IPv4 ──────────────────────────────────────────────────────
# 很多内网 / 家宽环境里容器的 IPv6 是**不可达**的，而镜像源（如清华 TUNA）会同时
# 返回 AAAA 记录。glibc 默认把 IPv6 排在前面，于是 `pip install` / `apt-get`
# 会先连一个连不通的地址 —— 实测表现为构建在 "Looking in indexes: ..." 之后
# 长时间毫无输出、CPU 归零、网络计数静止（一次卡了 40 分钟），而宿主机 curl
# 同一 URL 只要 1.9s。gai.conf 的 precedence 把 IPv4 映射地址提到最高优先级，
# 解析结果就变成 IPv4 优先，构建立刻恢复正常。
#
# 注意：这行必须放在任何 apt / pip 之前，否则网络请求已经发生了。
RUN echo 'precedence ::ffff:0:0/96  100' >> /etc/gai.conf

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
# 注意：不要动 builder 阶段的 apt 行 —— 它的下一层就是 `pip install -r
# requirements.txt`（十几分钟的 torch / docling / paddle）。改这里只让本阶段的
# 层失效，pip 层仍命中缓存。
#
# tesseract-ocr-chi-sim：中文 OCR 的**兜底**通道。主力是 PaddleOCR，但一旦
# PaddleOCR 因依赖缺失退化，只有 eng 的 Tesseract 会把中文图片读成空字符串 ——
# 而且是静默的（日志之外没有任何信号）。装上简体中文语言包，兜底就真的兜得住。
#
# 先落 gai.conf（同 builder 阶段，理由见上）：apt 同样会卡在不可达的 IPv6 上。
RUN echo 'precedence ::ffff:0:0/96  100' >> /etc/gai.conf \
 && sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources \
 && apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    libgl1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Persistent, writable cache dir for model weights (BGE / reranker / Table
# Transformer / PaddleOCR / Docling), downloaded on first use. Mounted as the
# `hf_cache` volume in docker-compose.yml so the ~1.5GB of weights isn't
# re-downloaded on every container restart.
#
# ⚠️ HF_HOME 必须指向卷的根 `/app/.cache`，不能是 `/app/.cache/huggingface`。
# huggingface_hub 把权重放在 `$HF_HOME/hub/`。指向子目录会得到
# `/app/.cache/huggingface/hub` —— 而权重实际在 `/app/.cache/hub/`（历史布局），
# 于是每次都判定"缓存不存在"→ 重新下载 → 若那个子目录恰好归 root（调试时
# `docker exec -u root` 留下的），写入被拒。症状不是报错，而是入库整批
# `N chunks failed to embed permanently.`，极难反查到缓存目录。
# 启动自检见 app/utils/hf_cache.py。
ENV HF_HOME=/app/.cache
RUN mkdir -p /app/.cache

# appuser 的 HOME 默认是 /nonexistent，会导致 PaddleOCR / Docling / HuggingFace
# 首次下载模型时写不进缓存。把 HOME 指到 /app/.cache 统一解决（docker-compose
# 已把 /app/.cache 挂成持久卷，模型不会每次重启重复下载）。
ENV HOME=/app/.cache \
    PADDLE_CACHE_DIR=/app/.cache/paddleocr
RUN mkdir -p /app/.cache/paddleocr

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source and migrations
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
RUN chown -R appuser:appgroup /app/.cache
# Writable storage for extracted images / archived originals / generated docs.
# IMAGE_STORAGE_DIR defaults to "uploads" (relative to WORKDIR /app), so this
# directory MUST exist and be owned by appuser — /app itself is root-owned and
# the app would otherwise silently fail to persist images (Errno 13).
ENV IMAGE_STORAGE_DIR=/app/uploads \
    DOCUMENT_OUTPUT_DIR=/app/uploads/_generated
RUN mkdir -p /app/uploads/_generated && chown -R appuser:appgroup /app/uploads

USER appuser

EXPOSE 8000

# Uvicorn with 1 worker to enforce process-level concurrency constraints
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]