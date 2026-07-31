<#
.SYNOPSIS
  AGI 研究完整套装 - Hugging Face 模型一键下载(约 30 GB)
.DESCRIPTION
  用纯 PowerShell 调 Invoke-WebRequest 走 hf-mirror / huggingface.co 官方下载。
  支持断点续传,默认放到 F:\agent to AGI\agi-research-kit\models
.NOTES
  - 用法: .\download_agi_set.ps1 -HFToken "hf_xxxx"
  - HFToken 在 https://huggingface.co/settings/tokens 申请(免费)
  - 部分量化或受限模型必须登录才能下
#>

[CmdletBinding()]
param(
    [string]$TargetRoot = "F:\agent to AGI\agi-research-kit\models",
    [string]$HFToken = "",
    [string]$Tier = "agi-complete"   # 备选: "minimal", "standard"
)

$ErrorActionPreference = "Stop"

# ---- 模型清单 ----
$tiers = @{
    "minimal"      = @()  # 仅 Qwen3-1.7B + 视觉
    "standard"     = @()
    "agi-complete" = @(
        @{ Id = "Qwen/Qwen3-1.7B";                  Sub = "main"; Tier = "主力 Agent LLM(推荐); ~3.8 GB"; Required = $true },
        @{ Id = "Qwen/Qwen3-0.6B";                  Sub = "main"; Tier = "最小通用 LLM; ~1.4 GB";       Required = $false },
        @{ Id = "HuggingFaceTB/SmolLM3-3B";         Sub = "main"; Tier = "HF 官方 3B; ~5.7 GB";          Required = $false },
        @{ Id = "HuggingFaceTB/SmolVLM-256M-Instruct"; Sub = "main"; Tier = "GUI/视觉 Agent; ~0.5 GB"; Required = $true },
        @{ Id = "Qwen/Qwen3-ASR-0.6B-hf";           Sub = "main"; Tier = "语音 Agent; ~1.4 GB";          Required = $false },
        @{ Id = "HuggingFaceTB/SmolLM2-135M-Instruct"; Sub = "main"; Tier = "极小模型研究; ~0.25 GB";   Required = $false },
        @{ Id = "HuggingFaceTB/SmolLM2-360M-Instruct"; Sub = "main"; Tier = "极小模型研究; ~0.5 GB";    Required = $false },
        @{ Id = "Qwen/Qwen3-4B-Instruct-2507";      Sub = "main"; Tier = "Agent 专用大模型(可选); ~8 GB"; Required = $false },
        @{ Id = "Qwen/Qwen-AgentWorld-35B-A3B";     Sub = "main"; Tier = "AGI 前沿 MoE-Agent 研究(可选); ~70 GB"; Required = $false }
    )
}

$models = $tiers[$Tier]
if (-not $models) { throw "未知 tier: $Tier (可选: minimal, standard, agi-complete)" }

# ---- 工具函数 ----
function Test-Tools {
    $hasPS51 = $PSVersionTable.PSVersion -ge [Version]"5.1"
    if (-not $hasPS51) { Write-Warning "建议 PowerShell 5.1+(Win10/11 自带)" }
}

function Get-Headers {
    $h = @{ "User-Agent" = "AGI-Research-Kit/1.0" }
    if ($HFToken) { $h["Authorization"] = "Bearer $HFToken" }
    return $h
}

function Get-FileList {
    param([string]$Repo, [string]$Revision)
    $api = "https://huggingface.co/api/models/$Repo/tree/$Revision"
    $j = (Invoke-WebRequest -Uri $api -UseBasicParsing -Headers (Get-Headers) -TimeoutSec 30).Content | ConvertFrom-Json
    $out = @()
    foreach ($f in $j) {
        if ($f.type -ne "file") { continue }
        $sz = $f.size
        if ($f.lfs -and $f.lfs.size) { $sz = $f.lfs.size }
        # 只要主权重 / 配置 / tokenizer
        if ($f.path -match "\.(safetensors|gguf|bin|json|txt|model)$" -or $f.path -match "tokenizer|vocab|merges|special_tokens") {
            $out += [PSCustomObject]@{ Path = $f.path; Size = [long]$sz }
        }
    }
    return $out
}

function Save-File {
    param([string]$Url, [string]$Dest, [long]$ExpectedBytes)
    if (Test-Path $Dest) {
        $cur = (Get-Item $Dest).Length
        if ($cur -eq $ExpectedBytes) { return @{ Skipped = $true; Bytes = $cur } }
    }
    Write-Host "  ↓ $Url" -ForegroundColor Cyan
    Invoke-WebRequest -Uri $Url -OutFile $Dest -Headers (Get-Headers) -UseBasicParsing -TimeoutSec 600 -Resume
    $got = (Get-Item $Dest).Length
    if ($ExpectedBytes -gt 0 -and $got -ne $ExpectedBytes) {
        Write-Warning "  ⚠ 大小不匹配: 期望 $ExpectedBytes / 实际 $got (可能需要 HF Token)"
    }
    return @{ Skipped = $false; Bytes = $got }
}

# ---- 主流程 ----
Test-Tools
New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null

$totalBytes = 0L
$downloadedBytes = 0L
$report = @()

# 先统计总大小(仅 LFS)
foreach ($m in $models) {
    Write-Host "[枚举] $($m.Id)" -ForegroundColor DarkGray
    try {
        $files = Get-FileList -Repo $m.Id -Revision $m.Sub
        $m | Add-Member -NotePropertyName Files -NotePropertyValue $files -Force
        $m | Add-Member -NotePropertyName Bytes -NotePropertyValue ($files | Measure-Object Size -Sum).Sum -Force
        $totalBytes += $m.Bytes
    } catch {
        Write-Warning "  枚举失败: $_"
    }
}

Write-Host ""
Write-Host "===== AGI 研究套装 - 下载计划 =====" -ForegroundColor Green
Write-Host ("目标目录: {0}" -f $TargetRoot)
Write-Host ("模型数量: {0}" -f $models.Count)
Write-Host ("总大小:   {0:N2} GB" -f ($totalBytes / 1GB))
Write-Host "=================================="
Write-Host ""

foreach ($m in $models) {
    Write-Host ("[{0}] {1}  - {2}" -f $m.Id, ("{0:N2} GB" -f ($m.Bytes/1GB)), $m.Tier) -ForegroundColor Yellow
    $dir = Join-Path $TargetRoot ($m.Id -replace '/','\')
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    foreach ($f in $m.Files) {
        $url = "https://huggingface.co/$($m.Id)/resolve/$($m.Sub)/$($f.Path)"
        $dest = Join-Path $dir $f.Path
        try {
            $r = Save-File -Url $url -Dest $dest -ExpectedBytes $f.Size
            if (-not $r.Skipped) { $downloadedBytes += $r.Bytes }
        } catch {
            Write-Warning "    ✗ 失败: $($f.Path) - $_"
        }
    }
}

Write-Host ""
Write-Host "===== 完成 =====" -ForegroundColor Green
Write-Host ("本次新增下载: {0:N2} GB" -f ($downloadedBytes / 1GB))
Write-Host ("总计需求:   {0:N2} GB" -f ($totalBytes / 1GB))
Write-Host "下一步: 运行 setup_env.ps1 安装 Python + Ollama"
