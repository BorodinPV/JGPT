#Requires -Version 5.1
# ASCII-only: Windows PowerShell 5.1 parses UTF-8 without BOM as system ANSI,
# which breaks quoted strings that contain "(6 GB)" and Cyrillic.
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Src = Join-Path $Root "src\main\cpp"
$BuildDir = Join-Path $Root "build"

function Write-Need($title, $install) {
    Write-Host ""
    Write-Host "MISSING: $title" -ForegroundColor Red
    Write-Host $install
}

function Find-CMake {
    $cmd = Get-Command cmake -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @(
        "${env:ProgramFiles}\CMake\bin\cmake.exe",
        "${env:ProgramFiles(x86)}\CMake\bin\cmake.exe"
    )
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsCmake = & $vswhere -latest -products * -find "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" 2>$null
        if ($vsCmake) { $candidates = @($vsCmake) + $candidates }
    }
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    return $null
}

function Find-Nvcc {
    $cmd = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    if ($env:CUDA_PATH) {
        $p = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
        if (Test-Path $p) { return $p }
    }
    $roots = Get-ChildItem "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA" -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending
    foreach ($r in $roots) {
        $p = Join-Path $r.FullName "bin\nvcc.exe"
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Find-Vswhere {
    $p = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $p) { return $p }
    return $null
}

function Find-Ninja([string]$VsInstall) {
    $cmd = Get-Command ninja -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $cands = @(
        (Join-Path $VsInstall "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"),
        "${env:ProgramFiles}\CMake\bin\ninja.exe"
    )
    foreach ($c in $cands) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    return $null
}

# VS generator looks for CUDA*.props in MSBuild (often missing if CUDA was
# installed before Build Tools). Ninja + nvcc + cl.exe does not need that.
function Import-VsDevEnvironment([string]$VsInstall) {
    $vsDevCmd = Join-Path $VsInstall "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path $vsDevCmd)) {
        throw "VsDevCmd.bat not found: $vsDevCmd"
    }
    $cmdLine = "`"$vsDevCmd`" -no_logo -arch=amd64 -host_arch=amd64 && set"
    $lines = & cmd.exe /c $cmdLine
    foreach ($line in $lines) {
        if ([string]::IsNullOrEmpty($line)) { continue }
        $eq = $line.IndexOf("=")
        if ($eq -lt 1) { continue }
        $name = $line.Substring(0, $eq)
        $value = $line.Substring($eq + 1)
        [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
    if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
        throw "cl.exe not on PATH after VsDevCmd. Repair VS Build Tools C++ workload."
    }
}

function Clear-CMakeCache([string]$Dir) {
    $cache = Join-Path $Dir "CMakeCache.txt"
    if (Test-Path $cache) {
        Write-Host "Clearing stale CMake cache: $cache"
        Remove-Item $cache -Force
    }
    $cf = Join-Path $Dir "CMakeFiles"
    if (Test-Path $cf) {
        Remove-Item $cf -Recurse -Force
    }
}

function Find-Mvn {
    $cmd = Get-Command mvn -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $cmdBat = Get-Command mvn.cmd -ErrorAction SilentlyContinue
    if ($cmdBat) { return $cmdBat.Source }
    $candidates = New-Object System.Collections.Generic.List[string]
    if ($env:M2_HOME) {
        [void]$candidates.Add((Join-Path $env:M2_HOME "bin\mvn.cmd"))
    }
    [void]$candidates.Add("${env:ProgramFiles}\Apache\maven\bin\mvn.cmd")
    [void]$candidates.Add("${env:ProgramFiles}\Maven\bin\mvn.cmd")
    $ideaRoot = Get-ChildItem "${env:ProgramFiles}\JetBrains" -Directory -ErrorAction SilentlyContinue
    foreach ($dir in $ideaRoot) {
        $ideaMvn = Join-Path $dir.FullName "plugins\maven-plugin\lib\maven3\bin\mvn.cmd"
        if (Test-Path $ideaMvn) {
            [void]$candidates.Add($ideaMvn)
        }
    }
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    return $null
}

$missing = $false

$cmake = Find-CMake
if (-not $cmake) {
    $missing = $true
    Write-Need "CMake" @"
Install CMake, then open a NEW PowerShell:
  winget install Kitware.CMake
"@
}

$nvcc = Find-Nvcc
if (-not $nvcc) {
    $missing = $true
    Write-Need "CUDA Toolkit (nvcc)" @"
nvcc is not on PATH. Typical location:
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin
Add that bin folder to the user PATH, then open a NEW PowerShell.
"@
}

$vswhere = Find-Vswhere
$vsPath = $null
if ($vswhere) {
    $vsPath = @(& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null) |
        Where-Object { $_ } | Select-Object -First 1
    $vsPath = "$vsPath".Trim()
}
if (-not $vsPath) {
    $missing = $true
    Write-Need "Visual Studio Build Tools (MSVC / cl.exe)" @"
CUDA on Windows needs cl.exe. Install Build Tools 2022:
  winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
Then open a NEW PowerShell and run:
  .\scripts\build-cuda.ps1
"@
}

$javaHome = $env:JAVA_HOME
if (-not $javaHome -or -not (Test-Path (Join-Path $javaHome "bin\java.exe"))) {
    $missing = $true
    Write-Need "JAVA_HOME (JDK 25+)" @"
Set JAVA_HOME to JDK 25 or 26, for example:
  `$env:JAVA_HOME = 'C:\Users\pc\.jdks\liberica-full-26.0.1'
"@
}

if ($missing) {
    Write-Host ""
    Write-Host "Native JNI lib is not built until cmake + MSVC + nvcc are available." -ForegroundColor Yellow
    Write-Host "Linux: ./scripts/build-cuda.sh   or   ./scripts/jgpt-smart.sh" -ForegroundColor Yellow
    exit 1
}

$cudaBin = Split-Path $nvcc -Parent
$cudaRoot = (Resolve-Path (Join-Path $cudaBin "..")).Path
$env:CUDA_PATH = $cudaRoot
$env:CUDACXX = $nvcc
$cudaRuntimeDir = $cudaBin
$cudaBinX64 = Join-Path $cudaBin "x64"
if (Test-Path $cudaBinX64) {
    $cudaRuntimeDir = $cudaBinX64
}

Write-Host "Loading MSVC environment (cl.exe)..."
Import-VsDevEnvironment $vsPath

$ninja = Find-Ninja $vsPath
if (-not $ninja) {
    Write-Need "Ninja" "Expected at:`n  $vsPath\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
    exit 1
}
$env:PATH = "$(Split-Path $ninja -Parent);$cudaRuntimeDir;$cudaBin;$env:PATH"

$cap = $null
try {
    $cap = (nvidia-smi --query-gpu=compute_cap --format=csv,noheader).Trim().Split("`n")[0].Trim()
} catch { }

# Turing sm_75: FA tile 128 needs ~90 KiB smem; limit is 64 KiB.
if (-not $env:JGPT_FA_TILE_SIZE -and $cap -eq "7.5") {
    $env:JGPT_FA_TILE_SIZE = "64"
    Write-Host "GPU compute $cap (Turing): JGPT_FA_TILE_SIZE=64"
}

Write-Host "cmake:  $cmake"
Write-Host "ninja:  $ninja"
Write-Host "nvcc:   $nvcc"
Write-Host "cl:     $((Get-Command cl).Source)"
Write-Host "VS:     $vsPath"
Write-Host "arch:   native"
if ($env:JGPT_FA_TILE_SIZE) { Write-Host "FA tile: $($env:JGPT_FA_TILE_SIZE)" }

Clear-CMakeCache $BuildDir

$cmakeArgs = @(
    "-G", "Ninja",
    "-B", $BuildDir,
    "-S", $Src,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_CUDA_COMPILER=$nvcc",
    "-DCMAKE_CUDA_HOST_COMPILER=cl",
    "-DCMAKE_CXX_COMPILER=cl",
    "-DCMAKE_C_COMPILER=cl",
    "-DCMAKE_CUDA_ARCHITECTURES=native"
)

& $cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $cmake --build $BuildDir --parallel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$dll = Join-Path $Root "build\jgpt_cuda.dll"
if (-not (Test-Path $dll)) {
    Write-Host "Build finished but DLL not found: $dll  (check build\Release)" -ForegroundColor Red
    exit 1
}

foreach ($srcDir in @($cudaRuntimeDir, $cudaBin)) {
    if (-not (Test-Path $srcDir)) { continue }
    Get-ChildItem $srcDir -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^(cudart64_|cublas64_|cublasLt64_|nvJitLink)' -and $_.Extension -eq '.dll' } |
        ForEach-Object { Copy-Item $_.FullName -Destination $BuildDir -Force }
}

$envFile = Join-Path $Root "build\jgpt-cuda-env.ps1"
$envBody = @"
# ASCII-only. Dot-source before Maven/IDEA:
#   . .\build\jgpt-cuda-env.ps1
`$env:CUDA_PATH = '$cudaRoot'
`$env:PATH = '$cudaRuntimeDir;$cudaBin;$Root\build;' + `$env:PATH
`$env:JGPT_CUDA_LIB = '$dll'
Write-Host "JGPT_CUDA_LIB=`$env:JGPT_CUDA_LIB"
Write-Host "CUDA_PATH=`$env:CUDA_PATH"
Write-Host "Next: mvn test  (same PowerShell window)"
"@
[System.IO.File]::WriteAllText($envFile, $envBody)

Write-Host ""
Write-Host "OK: $dll" -ForegroundColor Green
Write-Host "Next, in this PowerShell:"
Write-Host "  . .\build\jgpt-cuda-env.ps1"

$mvn = Find-Mvn
if ($mvn) {
    Write-Host "  & '$mvn' -q test"
} else {
    Write-Host "  Maven is not on PATH. Install it, then open a NEW PowerShell:"
    Write-Host "    winget install Apache.Maven"
    Write-Host "  Or run tests from IntelliJ: Maven tool window -> Lifecycle -> test"
}

Write-Host ""
Write-Host "This PC is RTX 2060 6GB VRAM: start with preset 04-minimal or 03-recovery, not 00."
