# 本地启动 Qdrant（Windows 单文件二进制，无需 Docker）
#
# 为什么需要这个脚本：
#   DSH 沙箱会在命令结束时回收其派生的整个进程树，因此常驻服务必须在
#   "你自己的终端"里启动一次；之后所有命令（含 DSH 内的调用）都能连上
#   localhost:6333。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\start_qdrant.ps1            # 启动（已在运行则跳过）
#   powershell -ExecutionPolicy Bypass -File scripts\start_qdrant.ps1 -Restart   # 先杀旧进程再启动
#
# 注意：本机为 Windows PowerShell 5.1（无 pwsh），且 5.1 读取无 BOM 的 UTF-8
#       脚本会按 GBK 解码，导致中文乱码与引号失配的误导性语法错误。
#       本文件带 UTF-8 BOM 保存；修改后请运行 scripts\ensure_ps1_bom.py 复核。
#
# 设计要点（三版踩坑后的修正，每一条都有实测依据）：
#   1. Start-Process 必须同时重定向 stdout/stderr 到文件：否则派生的常驻子进程
#      会继承调用方的 stdio 句柄，脚本 exit 后调用方仍在等句柄关闭
#      （表现为终端提示符回不来、命令挂住）。
#   2. -WorkingDirectory 必须指向已确认存在的目录：5.1 上该参数不做校验，
#      目录不存在时报 DirectoryNotFoundException。启动前用 Assert-Dir 兜底。
#   3. 配置写进生成的 YAML 并由 QDRANT_CONFIG 指向，而不是用命令行参数传
#      Windows 路径（含冒号，经 cmd 转发易被空格切分）。
#   4. SNAPSHOTS 放系统临时目录：受限环境下 qdrant 打开既有 snapshots 目录会
#      因权限 panic（src\snapshots.rs PermissionDenied）。
#   5. 不用 WMI（Win32_Process.Create）拉起进程：部分受限环境会以
#      0x80041003（拒绝访问）失败。

param(
    [switch]$Restart,
    [int]$Port = 6333
)

$ErrorActionPreference = 'Stop'

function Assert-Dir {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "[$Label] 目录创建失败: $Path"
    }
    Write-Host "[$Label] $Path"
}

function Test-QdrantReady {
    param([int]$P)
    try {
        $r = Invoke-WebRequest "http://localhost:$P/collections" -UseBasicParsing -TimeoutSec 5
        return ($r.StatusCode -eq 200)
    } catch {
        return $false
    }
}

$root      = Split-Path -Parent $PSScriptRoot
$qdrantExe = 'D:\projects\.tools\qdrant\qdrant.exe'
$storage   = $root + '\.data\qdrant\storage'
$snapshots = $env:TEMP + '\qdrant_snapshots'
$logs      = $root + '\.data\logs'
$configYml = $logs + '\qdrant-config.yaml'

Write-Host "[qdrant] 项目根目录: $root"
Write-Host "[qdrant] 可执行文件: $qdrantExe"

if (-not (Test-Path -LiteralPath $qdrantExe -PathType Leaf)) {
    throw "未找到 qdrant.exe: $qdrantExe（请下载 qdrant-x86_64-pc-windows-msvc.zip 并解压到 D:\projects\.tools\qdrant\）"
}

Assert-Dir -Path $storage   -Label 'storage  '
Assert-Dir -Path $snapshots -Label 'snapshots'
Assert-Dir -Path $logs      -Label 'logs     '

if ($Restart) {
    Get-Process qdrant -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2
    Write-Host '[qdrant] 已停止旧进程'
}

if (Test-QdrantReady -P $Port) {
    Write-Host "[qdrant] 已在 http://localhost:$Port 运行，跳过启动"
    exit 0
}

# 生成 qdrant 配置：路径写进 YAML，避免命令行传 Windows 路径被 cmd 切分
$yaml = @"
storage:
  storage_path: '$storage'
  snapshots_path: '$snapshots'
service:
  http_port: $Port
  grpc_port: $($Port + 1)
  host: 127.0.0.1
log_level: INFO
"@
Set-Content -LiteralPath $configYml -Value $yaml -Encoding UTF8
Write-Host "[qdrant] 配置: $configYml"

$outLog = $logs + '\qdrant.out.log'
$errLog = $logs + '\qdrant.err.log'
$env:QDRANT_CONFIG    = $configYml
$env:QDRANT_EXE       = $qdrantExe
$env:QDRANT_STORAGE   = $storage
$env:QDRANT_OUT_LOG   = $outLog
$env:QDRANT_ERR_LOG   = $errLog

# 通过独立启动器拉起常驻进程：qdrant 会继承控制台句柄，直接在本进程里
# Start-Process 会让调用方一直等句柄关闭（提示符回不来）。多一层进程隔离后
# 本脚本可以立即返回。
$launcher = $PSScriptRoot + '\qdrant_launcher.ps1'
$nullOut  = $logs + '\launcher.out.log'
$nullErr  = $logs + '\launcher.err.log'
Start-Process -FilePath 'powershell.exe' `
    -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $launcher `
    -RedirectStandardOutput $nullOut `
    -RedirectStandardError  $nullErr `
    -WindowStyle Hidden | Out-Null

Write-Host "[qdrant] 启动器已触发，等待进程出现…"

# 只检查进程是否存在，不做网络探测：避免本脚本持有 socket / 句柄
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Milliseconds 800
    if (Get-Process qdrant -ErrorAction SilentlyContinue) {
        Write-Host "[qdrant] 进程已就绪: http://localhost:$Port/dashboard"
        Write-Host "[qdrant] 停止方式: Stop-Process -Name qdrant -Force"
        exit 0
    }
}

Write-Host "--- 24 秒内未见 qdrant 进程，日志尾部 ---"
if (Test-Path -LiteralPath $errLog) { Get-Content -LiteralPath $errLog -Tail 15 }
if (Test-Path -LiteralPath $outLog) { Get-Content -LiteralPath $outLog -Tail 15 }
if (Test-Path -LiteralPath $nullErr) { Get-Content -LiteralPath $nullErr -Tail 15 }
throw "[qdrant] 启动失败"
