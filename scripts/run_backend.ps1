# 交互式启动后端服务（FastAPI，阶段4 完成后使用）
# 用法：powershell -ExecutionPolicy Bypass -File scripts\run_backend.ps1 [-Port 8000]

param([int]$Port = 8000)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$py   = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path $py)) { Write-Error "未找到虚拟环境: $py（先运行 python -m venv .venv 并安装依赖）" }

# Qdrant 是 /chat 的硬依赖：知识问答链路需要它。先探测，未启动则提示。
try {
    Invoke-WebRequest 'http://localhost:6333/collections' -UseBasicParsing -TimeoutSec 5 | Out-Null
    Write-Host '[backend] Qdrant 就绪'
} catch {
    Write-Warning '[backend] Qdrant 未就绪，知识问答链路会降级。请先执行: pwsh -File scripts\start_qdrant.ps1'
}

Write-Host "[backend] 启动 uvicorn -> http://127.0.0.1:$Port"
& $py -m uvicorn backend.main:app --host 127.0.0.1 --port $Port --reload
