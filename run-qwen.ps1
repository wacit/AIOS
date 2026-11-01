#Requires -Version 5.1
# aios-scripts-version: 6
<#
.SYNOPSIS
  Download (if needed) and launch Qwen3.8-27B on llama-server, tuned for the
  detected GPUs. Defaults are the "max decode speed" profile for 2x RTX 5060 Ti 16 GB.

.PARAMETER Quant      GGUF quant tag from the HF repo. Default: auto (UD-Q4_K_XL for 2x16 GB, Q8_0 for >= 40 GB VRAM, UD-Q3_K_XL for one 16 GB card).
.PARAMETER Ctx        Context length (tokens). Default 65536.
.PARAMETER KvType     KV-cache type: f16 | q8_0 | q4_0. Default q8_0 (q4_0 for 256K context).
.PARAMETER Split      Multi-GPU split mode: layer | row. Default layer. Benchmark both with -Bench.
.PARAMETER NoMtp      Disable MTP speculative decoding (needed for --parallel > 1).
.PARAMETER Parallel   Concurrent request slots. MTP requires 1. Default 1.
.PARAMETER Vision     Also load the mmproj vision encoder (image input).
.PARAMETER Port       Server port. Default 8080.
.PARAMETER BindHost   Bind address. Default 0.0.0.0 (LAN reachable). Use 127.0.0.1 for local only.
.PARAMETER Bench      Run llama-bench instead of the server.
#>
[CmdletBinding()]
param(
    [string]$Quant    = 'auto',
    [int]$Ctx         = 65536,
    [ValidateSet('f16','q8_0','q4_0')][string]$KvType = 'q8_0',
    [ValidateSet('layer','row')][string]$Split = 'layer',
    [switch]$NoMtp,
    [int]$Parallel    = 1,
    [switch]$Vision,
    [int]$Port        = 8080,
    [string]$BindHost = '0.0.0.0',
    [string]$BinDir   = '',        # default <script dir>\bin     (resolved below; $PSScriptRoot is empty in
    [string]$ModelDir = '',        # default <script dir>\models  param() defaults under powershell.exe -File)
    [switch]$Bench
)
$ErrorActionPreference = 'Stop'
$ScriptDir = $PSScriptRoot
if (-not $ScriptDir -and $MyInvocation.MyCommand.Path) { $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $ScriptDir) { $ScriptDir = (Get-Location).Path }
if (-not $BinDir)   { $BinDir   = Join-Path $ScriptDir 'bin' }
if (-not $ModelDir) { $ModelDir = Join-Path $ScriptDir 'models' }

$HF_REPO = 'unsloth/Qwen3.8-27B-GGUF'   # ships the MTP head tensors (blk.*.nextn.*) + fixed chat template

# --- binaries present? --------------------------------------------------------
$server = Join-Path $BinDir 'llama-server.exe'
$benchExe = Join-Path $BinDir 'llama-bench.exe'
if (-not (Test-Path $server)) {
    Write-Host "llama-server.exe not found in $BinDir" -ForegroundColor Red
    Write-Host "Build it first:  powershell -ExecutionPolicy Bypass -File `"$(Join-Path $ScriptDir 'build-llama.ps1')`"" -ForegroundColor Yellow
    exit 1
}

# --- hardware -----------------------------------------------------------------
$gpus = @()
$nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvsmi) {
    foreach ($p in "$env:SystemRoot\System32\nvidia-smi.exe", "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe") {
        if (Test-Path $p) { $nvsmi = Get-Item $p; break }
    }
}
if ($nvsmi) {
    $rows = @()
    $prevEap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { $rows = & $nvsmi.Source --query-gpu=index,name,memory.total --format=csv,noheader,nounits 2>&1 } catch { $rows = @() } finally { $ErrorActionPreference = $prevEap }
    foreach ($r in $rows) {
        $f = "$r" -split ',\s*'
        if ($f.Count -ge 3 -and $f[0] -match '^\d+$') { $gpus += [pscustomobject]@{ Index=[int]$f[0]; Name=$f[1]; VramGB=[math]::Round([int]$f[2]/1024,1) } }
    }
}
if ($gpus.Count -eq 0) { Write-Host 'nvidia-smi found no GPUs - assuming 2x16 GB; pass -Quant to override.' -ForegroundColor Yellow; $totalVram = 32 }
else { $totalVram = ($gpus | Measure-Object VramGB -Sum).Sum }
$physCores = (Get-CimInstance Win32_Processor | Select-Object -First 1).NumberOfCores
Write-Host "GPUs: $($gpus.Count) x [$($gpus.Name -join ', ')]  total VRAM $totalVram GB   CPU cores $physCores" -ForegroundColor Cyan

# --- pick quant ---------------------------------------------------------------
# Sizes (GB): Q3_K_M 13.8 | IQ4_XS 15.7 | Q4_K_M 17.1 | Q5_K_M 19.8 | Q6_K 22.9 | Q8_0 29.0
if ($Quant -eq 'auto') {
    if     ($totalVram -ge 40) { $Quant = 'Q8_0'       }
    elseif ($totalVram -ge 26) { $Quant = 'UD-Q4_K_XL' }   # 2x16 GB: Unsloth dynamic 4-bit, ~Q4_K_M size, better quality
    elseif ($totalVram -ge 15) { $Quant = 'UD-Q3_K_XL' }   # single 16 GB card
    else                       { $Quant = 'UD-IQ2_XXS' }
}
Write-Host "Quant: $Quant   ctx: $Ctx   kv: $KvType   split: $Split   mtp: $(-not $NoMtp)" -ForegroundColor Cyan

# --- model source -------------------------------------------------------------
# llama.cpp downloads (with resume) into $env:LLAMA_CACHE on first use and reuses it after.
# Requires a build with HTTPS support (build-llama.ps1 sets LLAMA_BUILD_BORINGSSL=ON).
New-Item -ItemType Directory -Force $ModelDir | Out-Null
$env:LLAMA_CACHE = $ModelDir
$hfModel = "${HF_REPO}:$Quant"
Write-Host "Model: -hf $hfModel  (cache: $ModelDir)" -ForegroundColor Cyan

# --- launch -------------------------------------------------------------------
$env:CUDA_MODULE_LOADING = 'LAZY'
if ($Bench) {
    # llama-bench takes a file path, so use the cached GGUF (download it first via a normal run)
    function Find-Cached { Get-ChildItem $ModelDir -Recurse -Filter "*$Quant*.gguf" -ErrorAction SilentlyContinue | Where-Object { $_.Name -notmatch 'mmproj' } | Select-Object -First 1 }
    $cached = Find-Cached
    if (-not $cached) {
        # No GGUF yet: let llama-server download it (resumable), then stop it once it reports healthy.
        Write-Host "Downloading $hfModel via llama-server (first time only, ~17 GB)..." -ForegroundColor Yellow
        $dl = Start-Process -FilePath $server -PassThru -NoNewWindow -ArgumentList @(
            '-hf', $hfModel, '--no-mmproj', '-ngl', '0', '-c', '512', '--host', '127.0.0.1', '--port', "$Port")
        while (-not $dl.HasExited) {
            Start-Sleep -Seconds 3
            try { $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2; if ($h.status -eq 'ok') { break } } catch { }
        }
        if (-not $dl.HasExited) { Stop-Process -Id $dl.Id -Force; Start-Sleep -Seconds 1 }
        $cached = Find-Cached
        if (-not $cached) { throw "Download did not produce a $Quant GGUF in $ModelDir - check the llama-server output above." }
    }
    & $benchExe -m $cached.FullName -ngl 999 -fa 1 -sm $Split -p 512 -n 128 -r 3
    exit $LASTEXITCODE
}

$srvArgs = @(
    '-hf', $hfModel,
    '--jinja',                       # mandatory: Qwen3.8 chat template / stop tokens
    '-ngl', '999',                   # everything on GPU
    '-sm', $Split,                   # layer split across both cards (row = try for extra bandwidth)
    '-c', "$Ctx",
    '-fa', 'on',
    '--cache-type-k', $KvType, '--cache-type-v', $KvType,
    '-b', '2048', '-ub', '512',      # prompt-processing batch sizes
    '-t', "$physCores",              # CPU threads only matter for the tiny non-offloaded bits
    '--parallel', "$Parallel",
    '--host', $BindHost, '--port', "$Port",
    '--metrics'
)
if (-not $NoMtp) {
    if ($Parallel -ne 1) { throw 'MTP speculative decoding requires -Parallel 1 (pass -NoMtp for multi-slot).' }
    $srvArgs += @('--spec-type', 'draft-mtp', '--spec-draft-n-max', '2')   # +33-39% decode on consumer GPUs
}
if (-not $Vision) { $srvArgs += '--no-mmproj' }   # -hf auto-downloads the vision encoder unless told not to

Write-Host "llama-server $($srvArgs -join ' ')" -ForegroundColor DarkGray
& $server @srvArgs
