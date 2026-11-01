#Requires -Version 5.1
# aios-scripts-version: 6
<#
.SYNOPSIS
  Build llama.cpp from source on Windows, tuned to the detected CPU / RAM / GPU(s).

.DESCRIPTION
  1. Detects CPU, RAM and every GPU: NVIDIA (VRAM + compute capability via nvidia-smi),
     Intel Arc / AMD discrete cards (real VRAM from the display-class registry, since
     Win32_VideoController.AdapterRAM is 32-bit and caps at 4 GB) and integrated GPUs.
  2. Installs missing toolchain pieces via winget (winget source only - the msstore source
     fails behind TLS-inspecting proxies): Git, CMake, Ninja, Visual Studio Build Tools
     (C++ workload), NVIDIA CUDA Toolkit, the Vulkan SDK or the Intel oneAPI Base Toolkit (SYCL).
  3. Clones / updates llama.cpp (full clone - a shallow clone breaks `git pull`).
  4. Configures with the CUDA arch list that matches the installed cards
     (RTX 5060 Ti = compute 12.0 -> sm_120) and builds with Ninja.
  5. Installs the binaries to $InstallDir and writes build-info.json there.

  Re-run at any time to pull the latest llama.cpp and rebuild.

.PARAMETER SrcDir      Where to clone llama.cpp.           Default: <script dir>\llama.cpp
.PARAMETER InstallDir  Where the built binaries go.        Default: <script dir>\bin
.PARAMETER SkipInstall Do not try to install toolchain pieces (just detect).
.PARAMETER Clean       Delete the build directory before configuring.
.PARAMETER Branch      Git ref to build (default master).
.PARAMETER Backend     auto | cuda | vulkan | sycl | cpu
                       auto = cuda if an NVIDIA GPU is present; else sycl if a discrete Intel Arc card AND the
                       Intel oneAPI toolkit are present; else vulkan for any discrete GPU; else cpu.
                       sycl = Intel's native backend for Arc (llama.cpp docs/backend/SYCL.md): needs the
                       Intel oneAPI Base Toolkit (DPC++ icx compiler + oneMKL + oneDNN, installed via winget
                       if missing - several GB). Vulkan works on Arc with no extra SDK beyond the Vulkan SDK.
                       Also settable via the LLAMA_BACKEND environment variable.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\build-llama.ps1
#>
[CmdletBinding()]
param(
    [string]$SrcDir     = '',      # default: <script dir>\llama.cpp  (resolved below - $PSScriptRoot is empty in
    [string]$InstallDir = '',      # default: <script dir>\bin         param() defaults under powershell.exe -File)
    [switch]$SkipInstall,
    [switch]$Clean,
    [string]$Branch     = 'master',
    [ValidateSet('auto','cuda','vulkan','sycl','cpu')][string]$Backend = $(if ($env:LLAMA_BACKEND) { $env:LLAMA_BACKEND } else { 'auto' })
)

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'

# Resolve the script folder in a way that works from a prompt, via -File, and when dot-sourced (PS 5.1 quirk)
$ScriptDir = $PSScriptRoot
if (-not $ScriptDir -and $MyInvocation.MyCommand.Path) { $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $ScriptDir) { $ScriptDir = (Get-Location).Path }
if (-not $SrcDir)     { $SrcDir     = Join-Path $ScriptDir 'llama.cpp' }
if (-not $InstallDir) { $InstallDir = Join-Path $ScriptDir 'bin' }

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Info($msg) { Write-Host "    $msg" -ForegroundColor Gray }
function Write-Warn($msg) { Write-Host "    ! $msg" -ForegroundColor Yellow }

# Native commands (git, nvcc, cmake ...) write progress to stderr; under Windows PowerShell 5.1 with
# ErrorActionPreference=Stop that turns into a terminating error as soon as stderr is redirected.
# Run them with Continue and check $LASTEXITCODE ourselves instead.
function Invoke-Native([scriptblock]$Cmd, [switch]$IgnoreExit) {
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { & $Cmd 2>&1 | ForEach-Object { "$_" }; $code = $LASTEXITCODE } finally { $ErrorActionPreference = $prev }
    if (-not $IgnoreExit -and $code -ne 0) { throw "command failed (exit $code): $Cmd" }
}
$env:GIT_REDIRECT_STDERR = '2>&1'      # git: send progress to stdout so PowerShell never sees it as an error
$env:GIT_TERMINAL_PROMPT = '0'

# ---------------------------------------------------------------------------
# 1. Hardware detection
# ---------------------------------------------------------------------------
Write-Step 'Detecting hardware'

$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$physCores = ($cpu.NumberOfCores)
$logCores  = ($cpu.NumberOfLogicalProcessors)
$ramGB     = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
Write-Info "CPU : $($cpu.Name.Trim())  ($physCores cores / $logCores threads)"
Write-Info "RAM : $ramGB GB"

# --- every display adapter, with real VRAM -----------------------------------------------------
# Win32_VideoController.AdapterRAM is a uint32 and caps at 4 GB, which is wrong for a 24 GB
# Arc Pro B60 (or a 16 GB RTX). The display-class registry keys carry the true size in
# HardwareInformation.qwMemorySize (QWORD, sometimes REG_BINARY); match them to WMI by DriverDesc.
function ConvertTo-Int64Value($v) {
    if ($null -eq $v) { return 0L }
    if ($v -is [byte[]]) {
        if ($v.Length -ge 8) { return [BitConverter]::ToInt64($v, 0) }
        if ($v.Length -ge 4) { return [int64][BitConverter]::ToUInt32($v, 0) }
        return 0L
    }
    try { return [int64]$v } catch { return 0L }
}
$regVram = @{}
$dispClass = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
Get-ChildItem $dispClass -ErrorAction SilentlyContinue | ForEach-Object {
    $p = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue
    if ($p -and $p.DriverDesc) {
        $mem = ConvertTo-Int64Value $p.'HardwareInformation.qwMemorySize'
        if ($mem -le 0) { $mem = ConvertTo-Int64Value $p.'HardwareInformation.MemorySize' }
        if ($mem -gt 0 -and -not $regVram.ContainsKey($p.DriverDesc)) { $regVram[$p.DriverDesc] = $mem }
    }
}
$allGpus = @()
Get-CimInstance Win32_VideoController | Where-Object { $_.Name -notmatch 'Microsoft Basic|Remote|Virtual|DisplayLink|ASPEED|Matrox' } | ForEach-Object {
    # (ASPEED / Matrox = the BMC's VGA on server boards such as the ASRock Rack ROMED8-2T - never a compute device)
    $name = $_.Name.Trim()
    $vendor = if ($name -match 'NVIDIA') { 'nvidia' } elseif ($name -match 'Intel') { 'intel' } elseif ($name -match 'AMD|Radeon|ATI') { 'amd' } else { 'other' }
    $mem = 0L
    if ($regVram.ContainsKey($name)) { $mem = $regVram[$name] } elseif ($_.AdapterRAM) { $mem = [int64]$_.AdapterRAM }
    # Integrated GPUs ("Intel(R) Graphics" on Core Ultra, UHD / Iris Xe, "AMD Radeon(TM) Graphics") are
    # display adapters, not compute targets. Anything discrete with a real frame buffer is a Vulkan target.
    $integrated = ($name -match '^Intel\(R\) (Graphics|UHD|Iris|HD)') -or ($name -match 'Radeon\(TM\) Graphics$') -or
                  ($vendor -ne 'nvidia' -and $mem -lt 3GB)
    $allGpus += [pscustomobject]@{
        Index = $allGpus.Count; Name = $name; Vendor = $vendor; VramMB = [int]($mem / 1MB)
        Integrated = $integrated; Driver = $_.DriverVersion
    }
}

# --- NVIDIA detail (compute capability) via nvidia-smi -------------------------------------------
$gpus = @()
$nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvsmi -and (Test-Path "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe")) {
    $nvsmi = Get-Item "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
}
if ($nvsmi) {
    $rows = Invoke-Native { & $nvsmi.Source --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv,noheader,nounits } -IgnoreExit
    foreach ($r in $rows) {
        $f = $r -split ',\s*'
        if ($f.Count -ge 4) {
            $gpus += [pscustomobject]@{
                Index = [int]$f[0]; Name = $f[1]; VramMB = [int]$f[2]
                ComputeCap = $f[3]; Driver = $f[4]
            }
        }
    }
}
if ($gpus.Count -eq 0) {
    # Fall back to the WMI/registry list (no compute cap available)
    $allGpus | Where-Object { $_.Vendor -eq 'nvidia' } | ForEach-Object {
        $gpus += [pscustomobject]@{ Index = $gpus.Count; Name = $_.Name; VramMB = $_.VramMB; ComputeCap = ''; Driver = $_.Driver }
    }
}
$vkGpus = @($allGpus | Where-Object { -not $_.Integrated })     # discrete cards a Vulkan build will run on
foreach ($g in $allGpus) {
    $cc = ($gpus | Where-Object { $_.Name -eq $g.Name } | Select-Object -First 1).ComputeCap
    $kind = if ($g.Integrated) { 'integrated' } else { 'discrete' }
    Write-Info ("GPU{0}: {1}  {2} GB  {3}/{4}{5}  driver={6}" -f $g.Index, $g.Name, [math]::Round($g.VramMB/1024,1), $g.Vendor, $kind, $(if ($cc) { "  cc=$cc" }), $g.Driver)
}
if ($allGpus.Count -eq 0) { Write-Warn 'No GPU detected at all.' }

# Map compute caps -> CMAKE_CUDA_ARCHITECTURES. Unknown caps fall back to a
# name-based guess; final fallback covers Ampere..Blackwell.
$archSet = New-Object System.Collections.Generic.HashSet[string]
foreach ($g in $gpus) {
    $cc = $g.ComputeCap -replace '\.', ''
    if ($cc -match '^\d{2,3}$') { [void]$archSet.Add($cc) }
    elseif ($g.Name -match 'RTX 50')      { [void]$archSet.Add('120') }
    elseif ($g.Name -match 'RTX 40')      { [void]$archSet.Add('89') }
    elseif ($g.Name -match 'RTX 30|A\d0') { [void]$archSet.Add('86') }
}
if ($gpus.Count -gt 0 -and $archSet.Count -eq 0) { '86','89','120' | ForEach-Object { [void]$archSet.Add($_) } }
$cudaArchs = ($archSet | Sort-Object) -join ';'
$needsCuda128 = $archSet -contains '120'   # Blackwell needs toolkit >= 12.8

# Intel oneAPI (DPC++ compiler + oneMKL + oneDNN) - needed for the SYCL backend
function Get-OneApiRoot {
    foreach ($r in @($env:ONEAPI_ROOT, "${env:ProgramFiles(x86)}\Intel\oneAPI", "$env:ProgramFiles\Intel\oneAPI")) {
        if ($r -and (Test-Path "$r\setvars.bat") -and (Test-Path "$r\compiler\latest\bin\icx.exe")) { return $r }
    }
    return $null
}
$intelGpus = @($vkGpus | Where-Object { $_.Vendor -eq 'intel' })     # discrete Intel Arc cards -> SYCL targets

# Backend choice: CUDA if an NVIDIA card is present; SYCL if a discrete Intel Arc card and oneAPI are both
# present; else Vulkan for any discrete GPU (Intel Arc / AMD); else CPU.
if ($Backend -eq 'auto') {
    if ($gpus.Count -gt 0) { $Backend = 'cuda' }
    elseif ($intelGpus.Count -gt 0 -and (Get-OneApiRoot)) { $Backend = 'sycl' }
    elseif ($vkGpus.Count -gt 0) { $Backend = 'vulkan' }
    else { $Backend = 'cpu' }
}
if ($Backend -eq 'sycl') {
    if ($intelGpus.Count -gt 0) {
        Write-Info "SYCL GPUs: $(($intelGpus | ForEach-Object { "$($_.Name) ($([math]::Round($_.VramMB/1024,1)) GB)" }) -join ', ')"
        Write-Info "Total SYCL VRAM: $([math]::Round((($intelGpus | Measure-Object VramMB -Sum).Sum)/1024,1)) GB across $($intelGpus.Count) card(s) - multi-card serving uses -sm layer; ONEAPI_DEVICE_SELECTOR=level_zero:N pins one card"
        if ((($intelGpus | Measure-Object VramMB -Sum).Sum) -gt ($ramGB * 1024)) { Write-Info "Note: VRAM ($([math]::Round((($intelGpus | Measure-Object VramMB -Sum).Sum)/1024)) GB) exceeds system RAM ($ramGB GB) - AIOS keeps mmap on so models larger than RAM still load straight into the cards." }
        if ($intelGpus.Count -lt $vkGpus.Count) { Write-Info 'Non-Intel discrete GPUs are ignored by the SYCL backend.' }
    } else {
        Write-Warn 'Backend sycl requested but no discrete Intel Arc GPU found - building anyway (it will run on integrated Intel graphics, if any).'
    }
}
if ($Backend -eq 'cuda' -and $gpus.Count -eq 0) { Write-Warn 'Backend cuda requested but no NVIDIA GPU found - building anyway.'; if (-not $cudaArchs) { $cudaArchs = '86;89;120' } }
Write-Info "Backend: $Backend$(if ($Backend -eq 'cuda') { "  (CUDA architectures: $cudaArchs)" })"
if ($Backend -eq 'vulkan') {
    if ($vkGpus.Count -gt 0) {
        Write-Info "Vulkan GPUs: $(($vkGpus | ForEach-Object { "$($_.Name) ($([math]::Round($_.VramMB/1024,1)) GB)" }) -join ', ')"
        Write-Info "Total Vulkan VRAM: $([math]::Round((($vkGpus | Measure-Object VramMB -Sum).Sum)/1024,1)) GB across $($vkGpus.Count) card(s); llama.cpp uses discrete GPUs only by default (GGML_VK_VISIBLE_DEVICES overrides)"
        if ((($vkGpus | Measure-Object VramMB -Sum).Sum) -gt ($ramGB * 1024)) { Write-Info "Note: VRAM ($([math]::Round((($vkGpus | Measure-Object VramMB -Sum).Sum)/1024)) GB) exceeds system RAM ($ramGB GB) - AIOS keeps mmap on so models larger than RAM still load straight into the cards." }
    } else {
        Write-Warn 'No discrete GPU found - the Vulkan build will run on integrated graphics (slow). Consider -Backend cpu.'
    }
}
if ($Backend -eq 'cpu') { Write-Warn 'Building CPU-only.' }

# ---------------------------------------------------------------------------
# 2. Toolchain
# ---------------------------------------------------------------------------
Write-Step 'Checking toolchain'

function Install-Winget($id, $extra) {
    if ($SkipInstall) { Write-Warn "Missing: $id (SkipInstall set)"; return }
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { throw "winget not found - install '$id' manually." }
    Write-Info "Installing $id via winget ..."
    # --source winget: without it winget also queries the msstore source, which fails with
    # 0x8a15005e ("server certificate did not match") behind TLS-inspecting proxies / on some
    # networks - and then aborts the whole install instead of using the winget source.
    $wg = @('install','--id',$id,'-e','--source','winget','--accept-source-agreements','--accept-package-agreements','--silent','--disable-interactivity')
    if ($extra) { $wg += '--override'; $wg += $extra }
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { & winget @wg 2>&1 | ForEach-Object { "$_" } | Out-Host; $code = $LASTEXITCODE } finally { $ErrorActionPreference = $prev }
    if ($code -ne 0 -and $code -ne -1978335189) {   # -1978335189 = 0x8A15002B "no applicable upgrade / already installed"
        Write-Warn "winget install $id exited with $code (0x$('{0:X8}' -f $code))"
    }
    # refresh PATH for this session
    $env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
}
function Require-Tool($name, $wingetId, $hint) {
    if (Get-Command $name -ErrorAction SilentlyContinue) { return }
    Install-Winget $wingetId
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        # winget adds the tool to the machine PATH but a running shell may not see it - look in the usual places
        $cands = @("$env:ProgramFiles\Git\cmd", "$env:ProgramFiles\CMake\bin", "$env:LOCALAPPDATA\Microsoft\WinGet\Links",
                   "$env:ProgramFiles\WinGet\Links", "$env:LOCALAPPDATA\Programs\Git\cmd")
        foreach ($c in $cands) { if (Test-Path (Join-Path $c "$name.exe")) { $env:Path = "$c;$env:Path"; break } }
    }
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) { throw "$name still not found after install. $hint" }
}

Require-Tool 'git'   'Git.Git'           'Install Git for Windows (winget install --id Git.Git --source winget) and re-run.'
Require-Tool 'cmake' 'Kitware.CMake'     'Install CMake (winget install --id Kitware.CMake --source winget) and re-run.'
Require-Tool 'ninja' 'Ninja-build.Ninja' 'Install Ninja (winget install --id Ninja-build.Ninja --source winget) and re-run.'

# Visual Studio Build Tools with the C++ workload (2022 or newer - vswhere finds any edition/version)
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
function Find-VsPath {
    if (Test-Path $vswhere) {
        $p = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($p) { return ($p | Select-Object -First 1) }
    }
    foreach ($base in "${env:ProgramFiles}\Microsoft Visual Studio", "${env:ProgramFiles(x86)}\Microsoft Visual Studio") {
        foreach ($ver in '18','2022') { foreach ($ed in 'BuildTools','Community','Professional','Enterprise') {
            $c = Join-Path $base "$ver\$ed"
            if (Test-Path (Join-Path $c 'VC\Tools\MSVC')) { return $c }
        } }
    }
    return $null
}
$vsPath = Find-VsPath
if (-not $vsPath) {
    Install-Winget 'Microsoft.VisualStudio.2022.BuildTools' '--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'
    $vsPath = Find-VsPath
    if (-not $vsPath) { throw 'Visual Studio Build Tools (C++ workload) still not found. Install "Desktop development with C++" from https://visualstudio.microsoft.com/visual-cpp-build-tools/ and re-run.' }
}
Write-Info "MSVC : $vsPath"

# CUDA toolkit
function Get-CudaRoot {
    if ($env:CUDA_PATH -and (Test-Path "$env:CUDA_PATH\bin\nvcc.exe")) { return $env:CUDA_PATH }
    $root = "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $root) {
        $latest = Get-ChildItem $root -Directory | Where-Object { Test-Path "$($_.FullName)\bin\nvcc.exe" } |
                  Sort-Object { [version]($_.Name.TrimStart('v')) } -Descending | Select-Object -First 1
        if ($latest) { return $latest.FullName }
    }
    return $null
}
$cudaRoot = $null
if ($Backend -eq 'cuda') {
    $cudaRoot = Get-CudaRoot
    $cudaVer  = $null
    if ($cudaRoot) {
        $nvccOut = (Invoke-Native { & "$cudaRoot\bin\nvcc.exe" --version } -IgnoreExit) -join "`n"
        if ($nvccOut -match 'release (\d+\.\d+)') { $cudaVer = [version]$Matches[1] }
    }
    if (-not $cudaRoot -or ($needsCuda128 -and $cudaVer -lt [version]'12.8')) {
        if ($cudaRoot) { Write-Warn "CUDA $cudaVer is too old for Blackwell (need >= 12.8)." }
        Install-Winget 'Nvidia.CUDA'
        $cudaRoot = Get-CudaRoot
        if (-not $cudaRoot) { throw 'CUDA Toolkit not found after install. Install from developer.nvidia.com/cuda-downloads and re-run.' }
        $nvccOut = (Invoke-Native { & "$cudaRoot\bin\nvcc.exe" --version } -IgnoreExit) -join "`n"
        if ($nvccOut -match 'release (\d+\.\d+)') { $cudaVer = [version]$Matches[1] }
    }
    Write-Info "CUDA : $cudaVer  ($cudaRoot)"
    $env:CUDA_PATH = $cudaRoot
    $env:Path = "$cudaRoot\bin;$env:Path"
}

# Vulkan SDK (Intel Arc / AMD / any GPU without CUDA)
$vulkanSdk = $null
if ($Backend -eq 'vulkan') {
    $vulkanSdk = $env:VULKAN_SDK
    if (-not $vulkanSdk -or -not (Test-Path "$vulkanSdk\Bin\glslc.exe")) {
        Install-Winget 'KhronosGroup.VulkanSDK'
        $vulkanSdk = [Environment]::GetEnvironmentVariable('VULKAN_SDK','Machine')
        if (-not $vulkanSdk) { $vulkanSdk = Get-ChildItem 'C:\VulkanSDK' -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName }
        if (-not $vulkanSdk) { throw 'Vulkan SDK not found after install. Install from vulkan.lunarg.com and re-run.' }
        $env:VULKAN_SDK = $vulkanSdk
    }
    $env:Path = "$vulkanSdk\Bin;$env:Path"
    Write-Info "Vulkan SDK: $vulkanSdk"
}

# Intel oneAPI (SYCL backend for Intel Arc) - llama.cpp docs/backend/SYCL.md
# llama.cpp's SYCL kernels name the Battlemage G31 die (Arc Pro B60 / B70, B580 = G21) via
# sycl::ext::oneapi::experimental::architecture::intel_gpu_bmg_g31, which only exists in DPC++ >= 2025.2 -
# 2025.1 fails with "no member named 'intel_gpu_bmg_g31'". winget's Intel.OneAPI.BaseToolkit is stuck at 2025.1,
# so a too-old/missing toolkit is fixed with Intel's own offline installer of "Deep Learning Essentials"
# (DPC++ compiler + oneMKL + oneDNN + oneDPL - the smaller bundle SYCL.md recommends) installed side by side.
$minOneApi = [version]'2025.2'
$oneApiInstallerUrl = 'https://registrationcenter-download.intel.com/akdlm/IRC_NAS/ffd3f1fb-09fa-4b43-9df3-955b0e29ada0/intel-deep-learning-essentials-2026.1.3.20_offline.exe'
function Get-OneApiVersion($root) {
    $vs = Get-ChildItem "$root\compiler" -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.Name -match '^(\d+\.\d+(\.\d+)?)') { try { [version]$Matches[1] } catch {} }
    }
    if ($vs) { return ($vs | Sort-Object -Descending | Select-Object -First 1) }
    return $null
}
function Install-OneApi {
    if ($SkipInstall) { Write-Warn 'Missing/outdated Intel oneAPI (SkipInstall set)'; return }
    $exe = Join-Path $env:TEMP (Split-Path $oneApiInstallerUrl -Leaf)
    if (-not (Test-Path $exe) -or (Get-Item $exe).Length -lt 100MB) {
        Write-Info "Downloading $oneApiInstallerUrl (~2 GB) ..."
        if (Get-Command curl.exe -ErrorAction SilentlyContinue) { Invoke-Native { & curl.exe -L --fail --retry 3 -o $exe $oneApiInstallerUrl } | Out-Null }
        else { Invoke-WebRequest -Uri $oneApiInstallerUrl -OutFile $exe -UseBasicParsing }
    }
    Write-Info 'Running the Intel installer silently - approve the UAC prompt; this takes 10-20 minutes ...'
    # -a passes the rest to the installer; skip VS integration (Build Tools has no IDE to integrate with)
    $p = Start-Process -FilePath $exe -ArgumentList '-a','--silent','--eula','accept','-p=NEED_VS2022_INTEGRATION=0' -Verb RunAs -Wait -PassThru
    if ($p.ExitCode -ne 0) { Write-Warn "Intel installer exited with $($p.ExitCode) - see %LOCALAPPDATA%\Temp\intel_oneapi_*.log" }
}
$oneApiRoot = $null; $oneApiVer = $null
if ($Backend -eq 'sycl') {
    $oneApiRoot = Get-OneApiRoot
    if ($oneApiRoot) { $oneApiVer = Get-OneApiVersion $oneApiRoot }
    if (-not $oneApiRoot -or -not $oneApiVer -or $oneApiVer -lt $minOneApi) {
        if ($oneApiRoot) { Write-Warn "oneAPI DPC++ $oneApiVer is too old for the Arc B-series (Battlemage G31) SYCL kernels - need >= $minOneApi. Installing a newer toolkit next to it." }
        else { Write-Info 'Intel oneAPI (DPC++ icx compiler, oneMKL, oneDNN) not found - installing Intel Deep Learning Essentials ...' }
        Install-OneApi
        $oneApiRoot = Get-OneApiRoot
        if ($oneApiRoot) { $oneApiVer = Get-OneApiVersion $oneApiRoot }
        if (-not $oneApiRoot -or -not $oneApiVer -or $oneApiVer -lt $minOneApi) {
            throw "Intel oneAPI DPC++ >= $minOneApi not found after install (found: $oneApiVer). Install 'Intel Deep Learning Essentials' or the 'Intel oneAPI Toolkit' from https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneapi-toolkit-download.html and re-run."
        }
    }
    Write-Info "oneAPI: $oneApiRoot  (DPC++ $oneApiVer)"
}

# ---------------------------------------------------------------------------
# 3. Source
# ---------------------------------------------------------------------------
Write-Step 'Fetching llama.cpp source'
if (-not (Test-Path (Join-Path $SrcDir '.git'))) {
    Invoke-Native { git clone --progress https://github.com/ggml-org/llama.cpp.git $SrcDir } | Out-Host
}
Push-Location $SrcDir
try {
    if (Test-Path '.git\shallow') { Write-Info 'Un-shallowing clone'; Invoke-Native { git fetch --unshallow } | Out-Host }
    Invoke-Native { git fetch --tags origin } | Out-Host
    Invoke-Native { git checkout -q $Branch } | Out-Host
    Invoke-Native { git reset -q --hard "origin/$Branch" } -IgnoreExit | Out-Null   # fails harmlessly for tags/commits
    $commit = (Invoke-Native { git rev-parse --short HEAD }) -join ''
    $tag    = (Invoke-Native { git describe --tags --always } -IgnoreExit) -join ''
    Write-Info "llama.cpp $tag ($commit)"
} finally { Pop-Location }

# ---------------------------------------------------------------------------
# 4. Configure + build (inside a VS developer environment, Ninja generator)
#    Ninja + nvcc avoids the "No CUDA toolset found" MSBuild-integration trap.
# ---------------------------------------------------------------------------
Write-Step 'Configuring'
$buildDir = Join-Path $SrcDir 'build'
if ($Clean -and (Test-Path $buildDir)) { Remove-Item $buildDir -Recurse -Force }

Import-Module (Join-Path $vsPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null

if ($Backend -eq 'sycl') {
    # setvars.bat puts icx / oneMKL / oneDNN on PATH, LIB, INCLUDE, CMPLR_ROOT ... - import its environment
    # into this PowerShell session (equivalent of the docs' cmd.exe "/K" setvars.bat && powershell trick).
    $envDump = & cmd.exe /s /c "`"$oneApiRoot\setvars.bat`" intel64 >nul 2>&1 && set"
    foreach ($line in $envDump) {
        if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
    }
    if (-not (Get-Command icx -ErrorAction SilentlyContinue)) { throw "icx.exe not on PATH after running $oneApiRoot\setvars.bat - check the oneAPI install." }
    $icxVer = ((Invoke-Native { & icx --version } -IgnoreExit) -join ' ')
    if ($icxVer -match 'Version (\d+\.\d+(\.\d+)?)' -and [version]$Matches[1] -lt $minOneApi) {
        throw "setvars.bat selected DPC++ $($Matches[1]) (< $minOneApi) although $oneApiVer is installed - the 'compiler\latest' link in $oneApiRoot points at the old version. Repair/reinstall the newer toolkit and re-run."
    }
    Write-Info "icx  : $((Get-Command icx).Source)  ($($icxVer -replace '\s+',' ' -replace '^(.{0,110}).*','$1'))"
    $syclLs = Invoke-Native { & sycl-ls } -IgnoreExit
    if ($syclLs) { $syclLs | Where-Object { $_ -match 'gpu' } | ForEach-Object { Write-Info "sycl-ls: $_" } }
    if (-not ($syclLs -match 'level_zero:gpu')) { Write-Warn 'sycl-ls shows no [level_zero:gpu] device - install/refresh the Intel Arc driver (intel.com/arc drivers) before running the build output.' }
}

$cmakeArgs = @(
    '-S', $SrcDir, '-B', $buildDir, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release',
    "-DBUILD_SHARED_LIBS=$(if ($Backend -eq 'sycl') { 'ON' } else { 'OFF' })",   # static .exe files (SYCL: shared, as in examples/sycl/win-build-sycl.bat)
    '-DGGML_NATIVE=ON',                  # -march=native equivalent (Zen5: AVX-512, Zen3 EPYC 7443 / Arrow Lake: AVX2)
    '-DLLAMA_BUILD_TESTS=OFF',
    '-DLLAMA_BUILD_EXAMPLES=ON',
    '-DLLAMA_BUILD_SERVER=ON'
)
if ($Backend -eq 'vulkan') { $cmakeArgs += '-DGGML_VULKAN=ON' }
if ($Backend -eq 'sycl') {
    # docs/backend/SYCL.md "Windows - Option 1: FP16" - cl for C, icx (DPC++) for C++, oneMKL + oneDNN on by default
    $cmakeArgs += @('-DGGML_SYCL=ON', '-DGGML_SYCL_TARGET=INTEL', '-DGGML_SYCL_F16=ON',
                    '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=icx')
}
if ($cudaRoot) {
    $cmakeArgs += @(
        '-DGGML_CUDA=ON',
        "-DCMAKE_CUDA_ARCHITECTURES=$cudaArchs",
        "-DCMAKE_CUDA_COMPILER=$cudaRoot\bin\nvcc.exe",
        '-DGGML_CUDA_FA_ALL_QUANTS=ON',  # flash-attn kernels for q4_0/q8_0 KV cache
        '-DGGML_CUDA_GRAPHS=ON'
    )
}
# HTTPS for -hf downloads: Windows has no system OpenSSL, so llama.cpp can fetch+build BoringSSL
# (boringssl.googlesource.com) or LibreSSL (cdn.openbsd.org). Try BoringSSL, then LibreSSL, then none.
$sslVariants = @(
    @{ name = 'BoringSSL'; args = @('-DLLAMA_BUILD_BORINGSSL=ON', '-DLLAMA_BUILD_LIBRESSL=OFF') },
    @{ name = 'LibreSSL';  args = @('-DLLAMA_BUILD_BORINGSSL=OFF', '-DLLAMA_BUILD_LIBRESSL=ON') },
    @{ name = 'no HTTPS';  args = @('-DLLAMA_BUILD_BORINGSSL=OFF', '-DLLAMA_BUILD_LIBRESSL=OFF', '-DLLAMA_OPENSSL=OFF') }
)
$configured = $false
foreach ($v in $sslVariants) {
    $allArgs = $cmakeArgs + $v.args
    Write-Info "cmake ($($v.name)) $($allArgs -join ' ')"
    try {
        Invoke-Native { & cmake @allArgs } | Out-Host
        $configured = $true; $sslUsed = $v.name; break
    } catch {
        Write-Warn "configure with $($v.name) failed - $($_.Exception.Message)"
        Remove-Item (Join-Path $buildDir 'CMakeCache.txt') -ErrorAction SilentlyContinue
        Remove-Item (Join-Path $buildDir '_deps') -Recurse -Force -ErrorAction SilentlyContinue
    }
}
if (-not $configured) { throw 'CMake configure failed with every SSL variant - see log above' }
if ($sslUsed -eq 'no HTTPS') { Write-Warn 'Built without HTTPS: -hf downloads will not work in llama-server (AIOS downloads models itself, so the GUI is unaffected).' }

Write-Step "Building with $logCores threads (this takes a while with CUDA / SYCL)"
Invoke-Native { & cmake --build $buildDir --config Release -j $logCores } | Out-Host

# ---------------------------------------------------------------------------
# 5. Install
# ---------------------------------------------------------------------------
Write-Step "Installing to $InstallDir"
New-Item -ItemType Directory -Force $InstallDir | Out-Null
Get-ChildItem (Join-Path $buildDir 'bin') -Filter '*.exe' | Copy-Item -Destination $InstallDir -Force
Get-ChildItem (Join-Path $buildDir 'bin') -Filter '*.dll' -ErrorAction SilentlyContinue | Copy-Item -Destination $InstallDir -Force
# CUDA runtime DLLs so the binaries run on machines without the toolkit
if ($cudaRoot) {
    Get-ChildItem "$cudaRoot\bin" -Include 'cudart64_*.dll','cublas64_*.dll','cublasLt64_*.dll' -Recurse -ErrorAction SilentlyContinue |
        Copy-Item -Destination $InstallDir -Force
}

[ordered]@{
    built_at    = (Get-Date).ToString('s')
    backend     = $Backend
    ssl         = $sslUsed
    llama_tag   = $tag
    llama_commit= $commit
    cuda        = "$cudaVer"
    cuda_archs  = $cudaArchs
    cpu         = $cpu.Name.Trim()
    cores       = $physCores
    ram_gb      = $ramGB
    gpus        = $(if ($Backend -eq 'cuda') { $gpus } elseif ($Backend -eq 'sycl') { $intelGpus } else { $vkGpus })   # the cards this build targets
    all_gpus    = $allGpus
    vulkan_sdk  = "$vulkanSdk"
    oneapi_root = "$oneApiRoot"    # SYCL builds need setvars.bat in the environment at run time (aios.py does this)
    oneapi_ver  = "$oneApiVer"
} | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $InstallDir 'build-info.json')

Write-Step 'Smoke test'
$cli = Join-Path $InstallDir 'llama-cli.exe'; $srv = Join-Path $InstallDir 'llama-server.exe'
if (-not (Test-Path $srv)) { throw "llama-server.exe was not produced - check the build log above" }
if (Test-Path $cli) { Invoke-Native { & $cli --version } -IgnoreExit | Out-Host }
if ($Backend -eq 'sycl') {
    $lsdev = Join-Path $InstallDir 'llama-ls-sycl-device.exe'
    if (Test-Path $lsdev) { Invoke-Native { & $lsdev } -IgnoreExit | Out-Host }
}
if ($Backend -ne 'cpu') { Invoke-Native { & $srv --list-devices } -IgnoreExit | Out-Host }
if ($Backend -eq 'sycl') { Write-Host "Note: SYCL binaries need the oneAPI runtime: run `"$oneApiRoot\setvars.bat`" first when launching llama-server by hand (AIOS does it for you)." -ForegroundColor Yellow }

Write-Host "`nDone. Binaries in $InstallDir" -ForegroundColor Green
Write-Host "Next: .\run-qwen.ps1 to download + launch Qwen3.8-27B tuned for this machine." -ForegroundColor Green
