<# AGI 研究环境一键安装 #>
param([switch]$SkipPython,[switch]$SkipOllama,[switch]$SkipDeps)
$pythonDir = "F:\agent to AGI\agi-research-kit\python"
$venvDir   = "F:\agent to AGI\agi-research-kit\.venv"

function Test-Tool($n) { $null -ne (Get-Command $n -ErrorAction SilentlyContinue) }

if (-not $SkipPython) {
  if (Test-Tool python) {
    Write-Host "[1/3] Python 已存在" -ForegroundColor Green
  } else {
    Write-Host "[1/3] 下载 Python 3.12 安装包" -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $pythonDir | Out-Null
    $pyUrl = "https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe"
    $pyInstaller = Join-Path $pythonDir "py-installer.exe"
    Invoke-WebRequest -Uri $pyUrl -OutFile $pyInstaller -UseBasicParsing -TimeoutSec 600
    Write-Host "请手动运行(需要管理员):  $pyInstaller" -ForegroundColor Cyan
    Write-Host "安装选项: Add Python to PATH + Install for all users" -ForegroundColor Cyan
  }
}

if (-not $SkipOllama) {
  if (Test-Tool ollama) {
    Write-Host "[2/3] Ollama 已存在" -ForegroundColor Green
  } else {
    Write-Host "[2/3] 下载 Ollama 安装包" -ForegroundColor Yellow
    $ollUrl = "https://github.com/ollama/ollama/releases/latest/download/OllamaSetup.exe"
    $ollInstaller = Join-Path $env:TEMP "OllamaSetup.exe"
    Invoke-WebRequest -Uri $ollUrl -OutFile $ollInstaller -UseBasicParsing -TimeoutSec 600
    Write-Host "请手动运行(下一步安装):  $ollInstaller" -ForegroundColor Cyan
  }
}

if (-not $SkipDeps) {
  Write-Host "[3/3] 创建虚拟环境 + 安装依赖 ..." -ForegroundColor Yellow
  if (-not (Test-Path $venvDir)) { python -m venv $venvDir }
  & "$venvDir\Scripts\python.exe" -m pip install --upgrade pip
  & "$venvDir\Scripts\pip.exe" install transformers torch accelerate safetensors huggingface_hub datasets tokenizers openai httpx rich pydantic gradio pyyaml tiktoken ollama litellm sentence-transformers rank-bm25 faiss-cpu pypdf
}

Write-Host "" 
Write-Host "===== 环境准备完成 =====" -ForegroundColor Green
Write-Host "下一步: "
Write-Host "  1) 如果上一步提示手动安装 Python / Ollama,请先完成"
Write-Host "  2)  ollama serve    (开后台)"
Write-Host "  3)  ollama pull qwen3:1.7b"
Write-Host "  4)  python experiments\hello_agent.py"
