@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  RAG Agent — 一键启动（前端 + 后端依赖服务）
REM  说明见同目录 README-启动说明.md
REM ─────────────────────────────────────────────────────────────────────────────
setlocal
set "NODE_HOME=D:\RAG项目依赖和环境\node"
set "PROJ=D:\RAG\Production-RAG-Agent"

REM 让本窗口能用 node / npm（不依赖系统 PATH，换机器也能跑）
set "PATH=%NODE_HOME%;%PATH%"

cd /d "%PROJ%"

echo ============================================
echo   RAG Agent 开发环境启动
echo ============================================
echo.
echo [1/3] 启动后端依赖服务 (postgres / qdrant / keycloak / backend)
pushd "%PROJ%\backend"
docker compose up -d
popd
if errorlevel 1 (
  echo.
  echo [警告] docker compose 未成功。请确认 Docker Desktop 已启动。
  echo.
)

echo.
echo [2/3] 等待后端健康检查...
set /a _try=0
:waitloop
set /a _try+=1
if %_try% GTR 30 goto waited
curl -sf http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
  echo   ... 第 %_try% 次探测，后端尚未就绪
  timeout /t 3 /nobreak >nul
  goto waitloop
)
echo   后端就绪: http://localhost:8000
echo   接口文档: http://localhost:8000/docs

:waited
echo.
echo [3/3] 启动前端 dev server (http://localhost:3000)
echo.
echo   按 Ctrl+C 停止前端。后端容器仍在后台运行，
echo   需要停止请执行:  cd backend ^&^& docker compose down
echo.
node "%NODE_HOME%\node_modules\npm\bin\npm-cli.js" run dev

endlocal
