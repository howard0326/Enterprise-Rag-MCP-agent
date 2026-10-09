# Qdrant 启动器（由 start_qdrant.ps1 派生，一般不需要手动运行）
#
# 职责：在读好环境变量后拉起 qdrant.exe，并把子进程的 stdout/stderr 全部
#       重定向到文件。
#
# 为什么要独立成一层：
#   qdrant.exe 会继承调用方的控制台/管道句柄，导致父脚本即使 exit，调用方
#   仍在等句柄关闭（终端提示符回不来）。把"真正拉起常驻进程"这一步放进
#   独立进程，父脚本就能立刻返回。
#
# 需要的环境变量（由 start_qdrant.ps1 设置）：
#   QDRANT_EXE       qdrant.exe 绝对路径
#   QDRANT_STORAGE   storage 目录
#   QDRANT_CONFIG    qdrant YAML 配置路径
#   QDRANT_OUT_LOG   子进程 stdout 日志
#   QDRANT_ERR_LOG   子进程 stderr 日志

$ErrorActionPreference = 'Stop'

$exe     = $env:QDRANT_EXE
$storage = $env:QDRANT_STORAGE
$config  = $env:QDRANT_CONFIG
$outLog  = $env:QDRANT_OUT_LOG
$errLog  = $env:QDRANT_ERR_LOG

if (-not $exe -or -not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "[launcher] QDRANT_EXE 无效: $exe"
}
if (-not $config -or -not (Test-Path -LiteralPath $config -PathType Leaf)) {
    throw "[launcher] QDRANT_CONFIG 无效: $config"
}

# 这里必须显式重定向：否则 qdrant 继承本进程句柄，句柄链一路传到调用方
Start-Process -FilePath $exe `
    -WorkingDirectory $storage `
    -RedirectStandardOutput $outLog `
    -RedirectStandardError  $errLog `
    -WindowStyle Hidden | Out-Null
