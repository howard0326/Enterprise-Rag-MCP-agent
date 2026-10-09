# 一键启动演示环境（Qdrant + 后端服务 + 打开 Qdrant 控制台）
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1 -Stop   # 停止全部
#
# 说明：
#   * 两个服务都用 Start-Process 脱离启动，并把 stdout/stderr 重定向到文件。
#     必须显式重定向，否则子进程会继承调用方的控制台句柄，脚本 exit 后
#     调用方仍在等句柄关闭（表现为终端提示符回不来）。
#   * Qdrant 的 snapshots 目录放系统临时目录：受限环境下 qdrant 打开既有
#     snapshots 目录会因权限 panic（src\snapshots.rs PermissionDenied）。
#   * 端口占用会先检测，已在运行则跳过，避免重复启动导致端口冲突。

param(
    [switch]$Stop,
    [int]$BackendPort = 8000,
    [int]$QdrantPort = 6333
)

$ErrorActionPreference = 'Stop'

$root      = Split-Path -Parent $PSScriptRoot
$python    = $root + '\.venv\Scripts\python.exe'
$qdrantExe = 'D:\projects\.tools\qdrant\qdrant.exe'
$logs      = $root + '\.data\logs'
$storage   = $root + '\.data\qdrant\storage'
$snapshots = $env:TEMP + '\qdrant_snapshots'

function Test-Port {
    param([int]$Port)
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return [bool]$conn
}

function Wait-Endpoint {
    param([string]$Url, [int]$Seconds = 40)
    for ($i = 0; $i -lt $Seconds; $i++) {
        Start-Sleep -Milliseconds 800
        try {
            $r = Invoke-WebRequest $Url -UseBasicParsing -TimeoutSec 4
            if ($r.StatusCode -eq 200) { return $true }
        } catch { }
    }
    return $false
}

# ---------------- 停止 ----------------
if ($Stop) {
    Write-Host '[demo] 停止服务…' -ForegroundColor Yellow
    Get-Process qdrant -ErrorAction SilentlyContinue | Stop-Process -Force
    # 只杀本项目的后端（按命令行匹配，避免误杀其他 python）
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'uvicorn.*backend\.main' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'mcp_servers' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
    Write-Host '[demo] 已停止'
    exit 0
}

Write-Host '========================================' -ForegroundColor Cyan
Write-Host ' 企业级 RAG + MCP 智能助手 —— 演示环境' -ForegroundColor Cyan
Write-Host '========================================' -ForegroundColor Cyan
Write-Host ''

# ---------------- 前置检查 ----------------
New-Item -ItemType Directory -Force $storage, $snapshots, $logs | Out-Null

if (-not (Test-Path $python)) {
    throw "未找到虚拟环境: $python"
}
if (-not (Test-Path $qdrantExe)) {
    throw "未找到 qdrant.exe: $qdrantExe"
}

# ---------------- 1) Qdrant ----------------
Write-Host '[1/3] Qdrant 向量库' -ForegroundColor Green
if (Test-Port -Port $QdrantPort) {
    Write-Host "      已在运行（端口 $QdrantPort），跳过" -ForegroundColor DarkGray
} else {
    $env:QDRANT__STORAGE__STORAGE_PATH   = $storage
    $env:QDRANT__STORAGE__SNAPSHOTS_PATH = $snapshots
    $env:QDRANT__SERVICE__HTTP_PORT      = "$QdrantPort"
    $env:QDRANT__SERVICE__GRPC_PORT      = "$($QdrantPort + 1)"

    $q = Start-Process -FilePath $qdrantExe -WorkingDirectory $storage `
        -RedirectStandardOutput ($logs + '\qdrant.out.log') `
        -RedirectStandardError  ($logs + '\qdrant.err.log') `
        -WindowStyle Hidden -PassThru
    Write-Host "      已启动 PID=$($q.Id)"

    if (Wait-Endpoint -Url "http://127.0.0.1:$QdrantPort/collections" -Seconds 25) {
        Write-Host "      就绪: http://127.0.0.1:$QdrantPort/dashboard" -ForegroundColor Green
    } else {
        Write-Host '      未就绪，错误日志尾部：' -ForegroundColor Red
        Get-Content ($logs + '\qdrant.err.log') -Tail 10 -ErrorAction SilentlyContinue
        throw '  Qdrant 启动失败'
    }
}

# ---------------- 2) 后端服务 ----------------
Write-Host '[2/3] FastAPI 后端服务' -ForegroundColor Green
if (Test-Port -Port $BackendPort) {
    Write-Host "      已在运行（端口 $BackendPort），跳过" -ForegroundColor DarkGray
} else {
    $env:QDRANT_LOG_LEVEL = 'ERROR'
    $b = Start-Process -FilePath $python `
        -ArgumentList '-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', "$BackendPort" `
        -WorkingDirectory $root `
        -RedirectStandardOutput ($logs + '\uvicorn.out.log') `
        -RedirectStandardError  ($logs + '\uvicorn.err.log') `
        -WindowStyle Hidden -PassThru
    Write-Host "      已启动 PID=$($b.Id)（含启动预热，约 15 秒）"

    if (Wait-Endpoint -Url "http://127.0.0.1:$BackendPort/health" -Seconds 40) {
        Write-Host "      就绪: http://127.0.0.1:$BackendPort/docs" -ForegroundColor Green
    } else {
        Write-Host '      未就绪，错误日志尾部：' -ForegroundColor Red
        Get-Content ($logs + '\uvicorn.err.log') -Tail 15 -ErrorAction SilentlyContinue
        throw '  后端启动失败'
    }
}

# ---------------- 3) 演示入口 ----------------
Write-Host '[3/3] 演示入口' -ForegroundColor Green
Write-Host ''
Write-Host '  ┌─────────────────────────────────────────────────────────┐' -ForegroundColor Cyan
Write-Host '  │ ① 演示控制台（推荐主入口：对话 + 向量库状态一屏）        │' -ForegroundColor Cyan
Write-Host "  │    http://127.0.0.1:$BackendPort/demo                              │" -ForegroundColor White
Write-Host '  │                                                         │' -ForegroundColor Cyan
Write-Host '  │ ② 接口文档（可交互测试，看原始 JSON）                   │' -ForegroundColor Cyan
Write-Host "  │    http://127.0.0.1:$BackendPort/docs                              │" -ForegroundColor White
Write-Host '  │                                                         │' -ForegroundColor Cyan
Write-Host '  │ ③ 健康检查（MCP 工具目录与 LLM 状态）                   │' -ForegroundColor Cyan
Write-Host "  │    http://127.0.0.1:$BackendPort/health                            │" -ForegroundColor White
Write-Host '  └─────────────────────────────────────────────────────────┘' -ForegroundColor Cyan
Write-Host ''
Write-Host '  注意：Qdrant 的 /dashboard 打不开是正常的 —— Windows 单文件版不带' -ForegroundColor DarkYellow
Write-Host '        Web UI 静态资源（只有 Docker 镜像才有）。/demo 是它的替代品。' -ForegroundColor DarkYellow
Write-Host ''
Write-Host '  提示：两个服务都在后台运行，本窗口可以关闭。' -ForegroundColor DarkGray
Write-Host '        停止全部服务请执行：scripts\start_demo.ps1 -Stop' -ForegroundColor DarkGray
