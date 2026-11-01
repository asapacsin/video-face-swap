# Prerequisites: Git, Python 3.12 (py launcher), FFmpeg on PATH, NVIDIA driver with CUDA 12 support.
# Repeatable. Downloads about 3.5 GB of packages and, with -Models, 1.47 GB of approved weights.
param([switch]$Models)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$revision = '72470819a0373be3388b3929c8f8f311f418fc3c'
$upstream = Join-Path $root 'upstream-facefusion'
$venv = Join-Path $root '.venv-gpu'

function Invoke-Checked { param($File, [string[]]$Arguments)
  & $File @Arguments
  if ($LASTEXITCODE -ne 0) { throw "$File failed with exit code $LASTEXITCODE" }
}

foreach ($tool in 'git', 'py', 'ffmpeg', 'ffprobe', 'nvidia-smi') { $null = Get-Command $tool }
if (-not (Test-Path -LiteralPath $upstream)) {
  # Keep LF endings: the worker checks the checkout is clean at the pinned revision.
  Invoke-Checked git @('clone', '-c', 'core.autocrlf=false', '--no-checkout',
    'https://github.com/facefusion/facefusion', $upstream)
  Invoke-Checked git @('-C', $upstream, 'checkout', '--detach', $revision)
}
if ((git -C $upstream rev-parse HEAD) -ne $revision) { throw 'Upstream revision mismatch; inspect manually.' }
if (git -C $upstream status --porcelain) { throw 'Upstream is modified; inspect manually.' }
if (-not (Test-Path -LiteralPath "$venv\Scripts\python.exe")) { Invoke-Checked py @('-3.12', '-m', 'venv', $venv) }
$python = "$venv\Scripts\python.exe"
$env:PATH = "$venv\Scripts;$env:PATH"
Push-Location $upstream
try { Invoke-Checked $python @('install.py', 'cuda@12', '--skip-conda') } finally { Pop-Location }
# The upstream installer ignores pip's exit status and does not fetch the CUDA and cuDNN DLLs.
Invoke-Checked $python @('-m', 'pip', 'install', 'onnxruntime-gpu[cuda,cudnn]==1.24.4')
Invoke-Checked $python @('-m', 'pip', 'install', '-e', "$root[dev]")
Invoke-Checked $python @('-m', 'pip', 'check')
if ($Models) { Invoke-Checked $python @((Join-Path $root 'scripts\fetch_models.py')) }
Invoke-Checked $python @('-m', 'face_replace', '--help') | Out-Null
'Setup complete.'
