#!/usr/bin/env pwsh
# 阶段4 端到端 curl 测试：非流式 / SSE 流式 / 工具目录 / 会话管理
#
# 用法（需先在另一个窗口启动 Qdrant 与后端服务）：
#   powershell -ExecutionPolicy Bypass -File scripts\curl_demo.ps1
#
# 说明：请求体一律先写入 UTF-8 文件，再用 curl -d "@file" 提交。
# 直接把含中文的 JSON 放进 PowerShell 单引号字符串会被按 ANSI 编码传出，
# 服务端收到后报 JSON decode error（已实测踩过）。

param(
    [string]$Base = 'http://127.0.0.1:8000',
    [string]$TmpDir = "$PSScriptRoot\..\.tmp"
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force $TmpDir | Out-Null
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Write-Json {
    param([string]$Path, [string]$Content)
    [System.IO.File]::WriteAllText($Path, $Content, $utf8)
}

function Show-Json {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path $Path)) { Write-Host "  [缺失] $Path"; return }
    $text = [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
    Write-Host "  $Label ($($text.Length) 字符)"
}

Write-Host '=== 1) POST /chat  混合问题（订单 + 政策）===' -ForegroundColor Cyan
$body1 = Join-Path $TmpDir 'req_chat_mixed.json'
Write-Json $body1 '{"message":"订单 ORD-001 符合退货政策吗？","session_id":"demo-1"}'
Write-Host '  curl 命令:'
Write-Host "  curl -sS -X POST $Base/chat -H `"Content-Type: application/json`" -d `"@req_chat_mixed.json`""
$out1 = Join-Path $TmpDir 'resp_chat_mixed.json'
curl.exe -sS -X POST "$Base/chat" -H 'Content-Type: application/json' -d "@$body1" --max-time 180 -o $out1
Write-Host "  curl exit=$LASTEXITCODE"
Show-Json $out1 '响应'

Write-Host ''
Write-Host '=== 2) POST /chat  操作请求（创建工单）===' -ForegroundColor Cyan
$body2 = Join-Path $TmpDir 'req_chat_ticket.json'
Write-Json $body2 '{"message":"帮我为订单 ORD-001 退款未到账创建一个高优先级工单","session_id":"demo-2"}'
$out2 = Join-Path $TmpDir 'resp_chat_ticket.json'
curl.exe -sS -X POST "$Base/chat" -H 'Content-Type: application/json' -d "@$body2" --max-time 180 -o $out2
Write-Host "  curl exit=$LASTEXITCODE"
Show-Json $out2 '响应'

Write-Host ''
Write-Host '=== 3) POST /chat/stream  SSE 流式 ===' -ForegroundColor Cyan
$body3 = Join-Path $TmpDir 'req_chat_stream.json'
Write-Json $body3 '{"message":"退货政策是什么？","session_id":"demo-3"}'
$out3 = Join-Path $TmpDir 'resp_chat_stream.txt'
Write-Host "  curl -N -sS -X POST $Base/chat/stream -H `"Content-Type: application/json`" -d `"@req_chat_stream.json`""
curl.exe -N -sS -X POST "$Base/chat/stream" -H 'Content-Type: application/json' -d "@$body3" --max-time 180 -o $out3
Write-Host "  curl exit=$LASTEXITCODE"
if (Test-Path $out3) {
    $lines = [System.IO.File]::ReadAllLines($out3, [System.Text.Encoding]::UTF8)
    Write-Host "  SSE 行数: $($lines.Count)"
    Write-Host '  事件序列:'
    $lines | Where-Object { $_ -like 'event:*' } | ForEach-Object { Write-Host "    $_" }
}

Write-Host ''
Write-Host '=== 4) GET /tools ===' -ForegroundColor Cyan
$out4 = Join-Path $TmpDir 'resp_tools.json'
curl.exe -sS "$Base/tools" --max-time 60 -o $out4
Write-Host "  curl exit=$LASTEXITCODE"
if (Test-Path $out4) {
    $tools = [System.IO.File]::ReadAllText($out4, [System.Text.Encoding]::UTF8) | ConvertFrom-Json
    Write-Host "  工具数: $($tools.count) -> $($tools.tools.PSObject.Properties.Name -join ', ')"
}

Write-Host ''
Write-Host '=== 5) GET /sessions 与 GET /sessions/{id} ===' -ForegroundColor Cyan
$out5 = Join-Path $TmpDir 'resp_sessions.json'
curl.exe -sS "$Base/sessions?limit=5" --max-time 60 -o $out5
if (Test-Path $out5) {
    $sess = [System.IO.File]::ReadAllText($out5, [System.Text.Encoding]::UTF8) | ConvertFrom-Json
    Write-Host "  最近会话数: $($sess.count)"
    $sess.sessions | Select-Object -First 5 | ForEach-Object { Write-Host "    $($_.session_id) 消息 $($_.messages) 更新 $($_.updated_at)" }
}
$out6 = Join-Path $TmpDir 'resp_history.json'
curl.exe -sS "$Base/sessions/demo-1?turns=3" --max-time 60 -o $out6
if (Test-Path $out6) {
    $hist = [System.IO.File]::ReadAllText($out6, [System.Text.Encoding]::UTF8) | ConvertFrom-Json
    Write-Host "  demo-1 轮数: $($hist.turns.Count)"
}

Write-Host ''
Write-Host '产出文件：' -ForegroundColor Green
Get-ChildItem $TmpDir -Filter 'resp_*' | ForEach-Object { Write-Host "  $($_.Name)  $($_.Length) 字节" }
