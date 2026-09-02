#Requires -Version 5.1
# ASCII-only. Windows port of fetch-cudnn.sh.
#
# Linux pip wheel: libcudnn.so.9 (ELF). MSVC cannot link it (LNK1107).
# Windows pip wheel: cudnn*.dll only, no cudnn.lib. This script extracts the
# win_amd64 wheel and builds an MSVC import lib with dumpbin + lib.exe.
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$CudnnDir = Join-Path $Root "third_party\cudnn"
$FeDir = Join-Path $Root "third_party\cudnn-frontend"
$WheelRoot = Join-Path $CudnnDir "nvidia\cudnn"
$WinLib = Join-Path $WheelRoot "lib\cudnn.lib"
$WinDll = Join-Path $WheelRoot "bin\cudnn64_9.dll"

function Find-Python {
    foreach ($c in @("python", "py", "python3")) {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }
    $p314 = Join-Path $env:LOCALAPPDATA "Programs\Python\Python314\python.exe"
    if (Test-Path $p314) { return $p314 }
    return $null
}

function Get-CudnnPipPackage {
    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    $nvccExe = $null
    if ($nvcc) { $nvccExe = $nvcc.Source }
    elseif ($env:CUDA_PATH) {
        $p = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
        if (Test-Path $p) { $nvccExe = $p }
    }
    $text = ""
    if ($nvccExe) {
        $prev = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try { $text = & $nvccExe --version 2>&1 | ForEach-Object { "$_" } | Out-String } finally { $ErrorActionPreference = $prev }
    }
    if ($text -match 'release\s+(\d+)') {
        $maj = [int]$Matches[1]
        if ($maj -ge 13) { return "nvidia-cudnn-cu13" }
        if ($maj -eq 12) { return "nvidia-cudnn-cu12" }
    }
    return "nvidia-cudnn-cu13"
}

function Find-MsvcTool([string]$Name) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return $null }
    $found = @(& $vswhere -latest -products * -find "VC\Tools\MSVC\*\bin\Hostx64\x64\$Name" 2>$null |
        Where-Object { $_ })
    if ($found.Count -gt 0) { return ([string]$found[0]).Trim() }
    return $null
}

function New-ImportLibFromDll([string]$DllPath, [string]$OutLib) {
    $dumpbin = Find-MsvcTool "dumpbin.exe"
    $libexe = Find-MsvcTool "lib.exe"
    if (-not $dumpbin -or -not $libexe) {
        throw "dumpbin.exe / lib.exe not found (need VS Build Tools C++ workload)"
    }
    $outDir = Split-Path $OutLib -Parent
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $exp = & $dumpbin /exports $DllPath 2>&1 | ForEach-Object { "$_" }
    $ErrorActionPreference = $prev
    $names = New-Object System.Collections.Generic.List[string]
    $inTable = $false
    foreach ($line in $exp) {
        if ($line -match '^\s*ordinal\s+hint') { $inTable = $true; continue }
        if ($inTable -and $line -match '^\s*$') {
            if ($names.Count -gt 0) { break } else { continue }
        }
        if ($inTable -and $line -match '^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)') {
            [void]$names.Add($Matches[1])
        }
    }
    if ($names.Count -lt 10) {
        throw "dumpbin /exports parsed $($names.Count) symbols from $DllPath"
    }
    $def = Join-Path $outDir "cudnn.def"
    @(
        "LIBRARY cudnn64_9.dll"
        "EXPORTS"
    ) + $names | Set-Content -Encoding ascii -LiteralPath $def
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $libexe /nologo "/def:$def" /machine:x64 "/out:$OutLib"
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($code -ne 0 -or -not (Test-Path $OutLib)) {
        throw "lib.exe failed to write $OutLib (exit $code)"
    }
    Write-Host "[fetch-cudnn] import lib: $OutLib ($($names.Count) exports)"
}

function Install-WinCudnnWheel {
    $py = Find-Python
    if (-not $py) {
        Write-Host "[fetch-cudnn] WARN: python not found; cannot download Windows cuDNN wheel"
        return
    }
    $pkg = Get-CudnnPipPackage
    $whlDir = Join-Path $env:TEMP "jgpt-cudnn-whl"
    New-Item -ItemType Directory -Force -Path $whlDir | Out-Null
    Write-Host "[fetch-cudnn] pip download $pkg (win_amd64)..."
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $py -m pip download $pkg -d $whlDir --no-deps
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($code -ne 0) {
        Write-Host "[fetch-cudnn] WARN: pip download $pkg failed"
        return
    }
    $whl = Get-ChildItem -LiteralPath $whlDir -Filter "*win_amd64.whl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $whl) {
        Write-Host "[fetch-cudnn] WARN: no win_amd64 wheel in $whlDir"
        return
    }
    $extract = Join-Path $env:TEMP "jgpt-cudnn-extract"
    if (Test-Path $extract) { Remove-Item -Recurse -Force $extract }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($whl.FullName, $extract)
    $src = Join-Path $extract "nvidia\cudnn"
    if (-not (Test-Path (Join-Path $src "bin\cudnn64_9.dll"))) {
        Write-Host "[fetch-cudnn] WARN: wheel missing bin\cudnn64_9.dll"
        return
    }
    foreach ($sub in @("bin", "include")) {
        $from = Join-Path $src $sub
        $to = Join-Path $WheelRoot $sub
        New-Item -ItemType Directory -Force -Path $to | Out-Null
        Copy-Item -Path (Join-Path $from "*") -Destination $to -Recurse -Force
    }
    Write-Host "[fetch-cudnn] extracted $($whl.Name) -> $WheelRoot"
}

if (-not (Test-Path $WinDll)) {
    Install-WinCudnnWheel
}

if ((Test-Path $WinDll) -and -not (Test-Path $WinLib)) {
    try {
        New-ImportLibFromDll $WinDll $WinLib
    } catch {
        Write-Host "[fetch-cudnn] WARN: $($_.Exception.Message)"
    }
}

$feHeader = Join-Path $FeDir "include\cudnn_frontend.h"
if (-not (Test-Path $feHeader)) {
    Write-Host "[fetch-cudnn] clone NVIDIA/cudnn-frontend v1.16.0"
    git clone --depth 1 --filter=blob:none --sparse --branch v1.16.0 `
        https://github.com/NVIDIA/cudnn-frontend.git $FeDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[fetch-cudnn] WARN: cudnn-frontend clone failed"
    } else {
        git -C $FeDir sparse-checkout set include
        if (Test-Path (Join-Path $FeDir ".git")) {
            Remove-Item -Recurse -Force (Join-Path $FeDir ".git")
        }
    }
}

if (Test-Path $WinLib) {
    Write-Host "[fetch-cudnn] OK: $WinLib"
} elseif (Test-Path $WinDll) {
    Write-Host "[fetch-cudnn] DLL present but no cudnn.lib (SDPA link skipped)"
} else {
    Write-Host "[fetch-cudnn] cuDNN not installed (optional; FA WMMA fallback is slower)"
}
