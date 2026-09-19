# ─────────────────────────────────────────────────────────────────────────────
# ⚠️ 本机（Windows + Docker Desktop, 无公网/间歇公网）请**不要**直接用本文件：
#    `requirements.txt` 钉了 `torch==2.14.0+cpu` / `torchvision==0.29.0+cpu`，
#    `+cpu` 是 PEP 440 本地版本标识，**标准 PyPI 上没有**（下载会失败）。
#    正确做法是用下面的「离线 wheelhouse」配方：
#
#        docker build -f Dockerfile.local -t backend-backend:latest .   # 需先备好 _wheels/
#    或一键（构建失败即中止 + 构建后校验容器内代码与磁盘一致，防"假烤"）：
#        powershell -ExecutionPolicy Bypass -File build_local.ps1
#
#    也可以用覆盖文件（等价于上面的 -f Dockerfile.local）：
#        docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --build
#
#    本文件保留为「有公网/CI + 内部 wheelhouse 镜像」的通用配方；它仍需一个
#    能提供 `+cpu` wheel 的源或等价离线镜像，否则 pip 装不上 GPU-less torch。
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — builder: install dependencies into a virtual environment
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# ── 镜像源（构建期可覆盖 + 多镜像自动回退）──────────────────────────────────
# 实测教训：上游镜像会**整体或间歇性**故障 —— 清华 TUNA 全站 502（apt 与 PyPI
# 同时挂）；换成阿里云后又出现同一 URL「前一次 200、后一次 502 Bad Gateway」的
# 抖动。单镜像 + 无重试的写法会让构建随机失败；更糟的是同目录的
# `docker compose up -d` 仍会用**旧镜像**成功拉起容器 —— 形成"看起来烤好了、
# 其实跑的还是旧代码"的假象（只有容器内外 md5 比对才能揭穿）。
# 因此：按「首选镜像 + 内置回退列表」逐个尝试，取第一个 apt-get update 成功的。
# ⚠️ 回退列表刻意写成 RUN 里的**字面量**，而不是塞进 ARG 默认值 —— 实测
# `ARG APT_MIRRORS=a b c` 只取第一个词（构建日志里展开成 `for m in a`），
# 多镜像回退会静默失效（有防护的假象）。带空格的 ARG 默认值必须加引号。
# 单个镜像名仍可覆盖，且没有引号坑：
#   docker compose build --build-arg APT_MIRROR=mirrors.cloud.tencent.com
ARG APT_MIRROR=mirrors.aliyun.com
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL}

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
# 逐个镜像试到 apt-get update 成功为止；全部失败则让 install 显式失败（不静默降级）。
#
# ⚠️ 必须**连同 scheme 一起替换**：只换域名（`deb.debian.org` → `<mirror>`）时
# URI 仍是 `http://`，请求被发到 80 端口 —— 而国内镜像的 80 端口普遍返回 502，
# 表现为"源看着换了却依然 apt 失败"。Dockerfile.local 早已写明这个坑，这里同步修。
RUN set -eu; \
    for m in ${APT_MIRROR} mirrors.cloud.tencent.com mirrors.huaweicloud.com; do \
      sed -i "s|http://deb.debian.org|https://${m}|g" /etc/apt/sources.list.d/debian.sources; \
      if apt-get -o Acquire::Retries=3 update; then echo "APT_MIRROR_OK=${m}"; break; fi; \
      echo "APT_MIRROR_FAILED=${m}"; \
      sed -i "s|https://${m}|http://deb.debian.org|g" /etc/apt/sources.list.d/debian.sources; \
    done; \
    apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
    gcc \
    libpq-dev \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# 上游 PyPI 镜像同样会抖动（TUNA 直接连接被重置）：加 --retries 让 pip 自己重试，
# 而不是一次抖动就废掉十几分钟的依赖安装层。
#
# ⚠️ 这一步需要能解析 `torch==2.14.0+cpu`（见文件顶部说明）。标准 PyPI 没有
#    `+cpu` 轮子，因此本配方要么指向内部 wheelhouse 镜像，要么改用 Dockerfile.local
#    的离线 `--find-links /wheels` 路径。若在这里看到
#    "No matching distribution found for torch==2.14.0+cpu"，就是走错了配方。
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --retries 5 --timeout 60 --prefix=/install -r requirements.txt


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
# APT_MIRROR 在 runtime 阶段需重新声明（ARG 是 stage 作用域），回退逻辑同 builder。
ARG APT_MIRROR=mirrors.aliyun.com
RUN echo 'precedence ::ffff:0:0/96  100' >> /etc/gai.conf \
 && set -eu; \
    for m in ${APT_MIRROR} mirrors.cloud.tencent.com mirrors.huaweicloud.com; do \
      sed -i "s|http://deb.debian.org|https://${m}|g" /etc/apt/sources.list.d/debian.sources; \
      if apt-get -o Acquire::Retries=3 update; then echo "APT_MIRROR_OK=${m}"; break; fi; \
      echo "APT_MIRROR_FAILED=${m}"; \
      sed -i "s|https://${m}|http://deb.debian.org|g" /etc/apt/sources.list.d/debian.sources; \
    done; \
    apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
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