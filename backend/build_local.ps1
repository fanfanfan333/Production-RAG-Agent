# ─────────────────────────────────────────────────────────────────────────────
# 本机一键构建并校验（Windows / Docker Desktop）
#
#   powershell -ExecutionPolicy Bypass -File build_local.ps1
#
# 它解决两个反复踩到的坑：
#   1) 通用 Dockerfile 在本机装不上依赖（torch==2.14.0+cpu 不在标准 PyPI）→
#      本脚本固定用 Dockerfile.local 的离线 wheelhouse 配方。
#   2) 「假烤镜像」：build 失败时 docker compose 仍可能用同名旧镜像把容器拉起来，
#      看起来 healthy、跑的却是旧代码 → 本脚本 build 失败即 `exit 1`（绝不起旧镜像），
#      并在构建后逐个比对**容器内代码与磁盘源码的 md5**，不一致即 `exit 2`。
#
# 只有最后打印 "BUILD OK & VERIFIED" 才算真正烤进去。
# ─────────────────────────────────────────────────────────────────────────────
param(
    [string]$Image = "backend-backend:latest",
    [string]$Service = "backend"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# 离线 wheelhouse 是所有依赖的来源；没有它必然构建失败（COPY _wheels 会报错）。
if (-not (Test-Path (Join-Path $root "_wheels"))) {
    Write-Host "缺少 _wheels/ —— 请先运行 backend/_fetch_wheels.sh 预取离线 wheel" -ForegroundColor Red
    exit 1
}

Write-Host "==> building $Image from Dockerfile.local (offline wheelhouse)"
docker build -f Dockerfile.local -t $Image .
if ($LASTEXITCODE -ne 0) {
    Write-Host "构建失败：已中止，**不会**用旧镜像启动容器（避免假烤）" -ForegroundColor Red
    exit 1
}

Write-Host "==> recreating $Service from the freshly built image"
docker compose -f docker-compose.yml up -d --no-deps --force-recreate $Service
if ($LASTEXITCODE -ne 0) {
    Write-Host "容器启动失败" -ForegroundColor Red
    exit 3
}

# ── 容器内外 md5 比对：只有一致才算真烤进去 ─────────────────────────────────
$keyFiles = @(
    "app/main.py",
    "app/config.py",
    "app/services/tenancy.py",
    "app/services/company_registry.py",
    "app/services/retrieval_service.py",
    "app/services/document_query_service.py",
    "app/api/companies.py",
    "app/api/document_management.py",
    "app/api/query.py",
    "alembic.ini"
)

$mismatch = @()
foreach ($f in $keyFiles) {
    $diskPath = Join-Path $root ($f -replace "/", "\")
    if (-not (Test-Path $diskPath)) { $mismatch += "$f (磁盘缺文件)"; continue }
    $disk = (Get-FileHash -Algorithm MD5 $diskPath).Hash.ToLower()
    $cont = (docker compose -f docker-compose.yml exec -T $Service md5sum "/app/$f" 2>$null)
    $contHash = if ($cont) { ($cont -split "\s+")[0].Trim().ToLower() } else { "" }
    if ($disk -ne $contHash) {
        $mismatch += "$f disk=$disk cont=$contHash"
    } else {
        Write-Host "  ok  $f"
    }
}

if ($mismatch.Count -gt 0) {
    Write-Host "容器内代码与磁盘不一致（假烤！）：" -ForegroundColor Red
    $mismatch | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 2
}

docker compose -f docker-compose.yml ps $Service
Write-Host "BUILD OK & VERIFIED" -ForegroundColor Green
exit 0
