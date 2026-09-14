#Requires -Version 5.1
# ASCII-only: Windows PowerShell 5.1 parses UTF-8 without BOM as system ANSI.
#
# jgpt-gui.ps1 - JGPT desktop GUI (JavaFX bundled with Liberica Full JDK).
# Training tab starts/stops jgpt-train-*.cmd as a child process and reads state\stats.json;
# Chat tab loads model weights into this JVM (needs build\jgpt_cuda.dll).
#
# Usage:
#   .\scripts\windows\jgpt-gui.cmd            compile only the GUI sources with javac (safe while training runs)
#   .\scripts\windows\jgpt-gui.cmd --mvn      full "mvn compile" first (do NOT use while a trainer is running:
#                                             it rewrites target\classes under the live JVM)
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

$UseMvn = $false
foreach ($a in $args) {
    if ($a -eq "--mvn") { $UseMvn = $true }
    elseif ($a -in @("-h", "--help")) {
        Write-Host "Usage: .\scripts\windows\jgpt-gui.cmd [--mvn]"
        exit 0
    }
}

function Get-NativeOutput {
    param([scriptblock]$Command)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $lines = & $Command 2>&1 | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { "$_" }
        }
        return ($lines -join "`n")
    } finally {
        $ErrorActionPreference = $prev
    }
}

function Get-JdkMajor([string]$JavaHome) {
    $release = Join-Path $JavaHome "release"
    if (Test-Path $release) {
        foreach ($line in Get-Content -LiteralPath $release) {
            if ($line -match 'JAVA_VERSION="(\d+)') { return [int]$Matches[1] }
        }
    }
    return 0
}

function Has-JavaFx([string]$JavaHome) {
    return (Test-Path (Join-Path $JavaHome "jmods\javafx.controls.jmod")) -or
           (Test-Path (Join-Path $JavaHome "lib\javafx.properties"))
}

function Find-JavaHome {
    $homes = New-Object System.Collections.Generic.List[string]
    if ($env:JAVA_HOME) { [void]$homes.Add($env:JAVA_HOME) }
    $jdksRoot = Join-Path $env:USERPROFILE ".jdks"
    if (Test-Path $jdksRoot) {
        Get-ChildItem $jdksRoot -Directory -ErrorAction SilentlyContinue | ForEach-Object { [void]$homes.Add($_.FullName) }
    }
    foreach ($root in @("${env:ProgramFiles}\BellSoft", "${env:ProgramFiles}\Java")) {
        if (Test-Path $root) {
            Get-ChildItem $root -Directory -ErrorAction SilentlyContinue | ForEach-Object { [void]$homes.Add($_.FullName) }
        }
    }
    $best = $null
    $bestMajor = 0
    foreach ($h in $homes) {
        $m = Get-JdkMajor $h
        if ($m -ge 25 -and (Has-JavaFx $h) -and $m -gt $bestMajor) { $bestMajor = $m; $best = $h }
    }
    return $best
}

function Find-Mvn {
    foreach ($name in @("mvn", "mvn.cmd")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }
    $candidates = @()
    foreach ($root in @("${env:ProgramFiles}\JetBrains", "${env:LOCALAPPDATA}\Programs")) {
        if (Test-Path $root) {
            Get-ChildItem $root -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -like "IntelliJ*" -or $_.Name -like "idea*" } |
                ForEach-Object { $candidates += (Join-Path $_.FullName "plugins\maven-plugin\lib\maven3\bin\mvn.cmd") }
        }
    }
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    return $null
}

$javaHome = Find-JavaHome
if (-not $javaHome) {
    Write-Host "[GUI] ERROR: need JDK 25+ WITH JavaFX (Liberica Full). Found none under JAVA_HOME / ~\.jdks." -ForegroundColor Red
    Write-Host "  e.g. `$env:JAVA_HOME = '$env:USERPROFILE\.jdks\liberica-full-26.0.1'"
    exit 1
}
$env:JAVA_HOME = $javaHome
$javaExe = Join-Path $javaHome "bin\java.exe"
$javacExe = Join-Path $javaHome "bin\javac.exe"

$m2 = Join-Path $env:USERPROFILE ".m2\repository"
$jars = @(
    (Join-Path $m2 "org\slf4j\slf4j-api\2.0.9\slf4j-api-2.0.9.jar"),
    (Join-Path $m2 "ch\qos\logback\logback-classic\1.5.14\logback-classic-1.5.14.jar"),
    (Join-Path $m2 "ch\qos\logback\logback-core\1.5.14\logback-core-1.5.14.jar")
)
$classes = Join-Path $Root "target\classes"
$guiClasses = Join-Path $Root "target\gui-classes"

if ($UseMvn) {
    $mvn = Find-Mvn
    if (-not $mvn) { Write-Host "[GUI] ERROR: mvn not found" -ForegroundColor Red; exit 1 }
    if ($env:MAVEN_OPTS -notmatch "enable-native-access") {
        $env:MAVEN_OPTS = "--enable-native-access=ALL-UNNAMED $($env:MAVEN_OPTS)".Trim()
    }
    Write-Host "[GUI] mvn -q compile ..."
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & $mvn -q compile
    $code = $LASTEXITCODE; $ErrorActionPreference = $prev
    if ($code -ne 0) { exit $code }
    $cp = (@($classes) + $jars) -join ";"
} else {
    if (-not (Test-Path (Join-Path $classes "com\veles\llm\jgpt\model\GPTModel.class"))) {
        Write-Host "[GUI] ERROR: target\classes is empty - run once with --mvn (or mvn compile)." -ForegroundColor Red
        exit 1
    }
    foreach ($j in $jars) {
        if (-not (Test-Path $j)) { Write-Host "[GUI] ERROR: missing $j (run mvn compile once)" -ForegroundColor Red; exit 1 }
    }
    $srcDir = Join-Path $Root "src\main\java\com\veles\llm\jgpt\gui"
    $sources = @(Get-ChildItem -LiteralPath $srcDir -Filter "*.java" -File | ForEach-Object { $_.FullName })
    $newest = ($sources | ForEach-Object { (Get-Item $_).LastWriteTimeUtc } | Measure-Object -Maximum).Maximum
    $marker = Join-Path $guiClasses "com\veles\llm\jgpt\gui\JgptGui.class"
    $needBuild = -not (Test-Path $marker) -or ((Get-Item $marker).LastWriteTimeUtc -lt $newest)
    if ($needBuild) {
        New-Item -ItemType Directory -Force -Path $guiClasses | Out-Null
        Write-Host "[GUI] javac gui sources -> target\gui-classes"
        $cpCompile = (@($classes) + $jars) -join ";"
        $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        & $javacExe --enable-preview --release ((Get-JdkMajor $javaHome).ToString()) -encoding UTF-8 -nowarn `
            -cp $cpCompile -d $guiClasses @sources
        $code = $LASTEXITCODE; $ErrorActionPreference = $prev
        if ($code -ne 0) { Write-Host "[GUI] javac failed" -ForegroundColor Red; exit $code }
    }
    $cp = (@($guiClasses, $classes) + $jars) -join ";"
}

$cudaEnv = Join-Path $Root "build\jgpt-cuda-env.ps1"
if (Test-Path $cudaEnv) { . $cudaEnv | Out-Null }
$dll = Join-Path $Root "build\jgpt_cuda.dll"
if (-not $env:JGPT_CUDA_LIB -and (Test-Path $dll)) { $env:JGPT_CUDA_LIB = $dll }
if ($env:JGPT_CUDA_LIB) { $env:PATH = "$(Split-Path $env:JGPT_CUDA_LIB);$env:PATH" }

# Chat is inference only; the FP16 GEMM / FlashAttention defaults are the same as for jgpt-chat-*.cmd.
if (-not $env:JGPT_FP16_MATMUL) { $env:JGPT_FP16_MATMUL = "1" }
if (-not $env:JGPT_FLASH_ATTENTION) { $env:JGPT_FLASH_ATTENTION = "1" }

Write-Host "[GUI] JAVA_HOME=$javaHome"
Write-Host "[GUI] JGPT_CUDA_LIB=$env:JGPT_CUDA_LIB"
$javaArgs = @(
    "--enable-native-access=ALL-UNNAMED,javafx.graphics",
    "--sun-misc-unsafe-memory-access=allow",
    "--add-modules=jdk.incubator.vector,javafx.controls,javafx.graphics",
    "--enable-preview",
    "-Dfile.encoding=UTF-8",
    "-Dstdout.encoding=UTF-8",
    "-Dstderr.encoding=UTF-8",
    "-Djgpt.root=$Root",
    "-Dprism.order=d3d,sw",
    "-cp", $cp,
    "com.veles.llm.jgpt.gui.JgptGui"
)
& $javaExe @javaArgs
exit $LASTEXITCODE
