#Requires -Version 5.1
# ASCII-only: Windows PowerShell 5.1 parses UTF-8 without BOM as system ANSI.
#
# Interactive InferChat for 20L-wide (SFT model_best, else pretrain).
#   .\scripts\windows\jgpt-chat-20L-wide.cmd
#   .\scripts\windows\jgpt-chat-20L-wide.cmd --prompt "Privet"
# Extra args after the script name go to InferChat (--temperature, --max-new-tokens, ...).
$ErrorActionPreference = "Stop"

try {
    chcp 65001 | Out-Null
    [Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch { }

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

$EnvFile = Join-Path $Root "env\20L-wide-sft.env"
$TokDefault = "checkpoints\tokenizer_wide_16k.bin"
$ModelCandidates = @(
    "checkpoints\wide_20L_sft\model_best.bin",
    "checkpoints\wide_20L_sft\model_final.bin",
    "checkpoints\wide_20L_16k_1024\model_best.bin",
    "checkpoints\wide_20L_16k_1024\model_final.bin"
)
$ModelDefault = $null
foreach ($c in $ModelCandidates) {
    if (Test-Path (Join-Path $Root $c)) {
        $ModelDefault = $c
        break
    }
}

function Import-BashEnvFile([string]$Path) {
    Get-Content -LiteralPath $Path | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#")) { return }
        if ($line -notmatch '^export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$') { return }
        $name = $Matches[1]
        $val = $Matches[2].Trim()
        if ($val.Length -ge 2) {
            $q = $val[0]
            if (($q -eq [char]34 -or $q -eq [char]39) -and $val[-1] -eq $q) {
                $val = $val.Substring(1, $val.Length - 2)
            }
        }
        [System.Environment]::SetEnvironmentVariable($name, $val, "Process")
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
    $java = Join-Path $JavaHome "bin\java.exe"
    if (-not (Test-Path $java)) { return 0 }
    $text = Get-NativeOutput { & $java -version }
    if ($text -match 'version "(\d+)') { return [int]$Matches[1] }
    return 0
}

function Find-JavaHome {
    if ($env:JAVA_HOME) {
        $major = Get-JdkMajor $env:JAVA_HOME
        if ($major -ge 25) { return $env:JAVA_HOME }
    }
    $homes = New-Object System.Collections.Generic.List[string]
    $jdksRoot = Join-Path $env:USERPROFILE ".jdks"
    if (Test-Path $jdksRoot) {
        Get-ChildItem $jdksRoot -Directory -ErrorAction SilentlyContinue | ForEach-Object {
            [void]$homes.Add($_.FullName)
        }
    }
    $bestHome = $null
    $bestMajor = 0
    foreach ($h in $homes) {
        $m = Get-JdkMajor $h
        if ($m -ge 25 -and $m -gt $bestMajor) {
            $bestMajor = $m
            $bestHome = $h
        }
    }
    return $bestHome
}

function Find-Mvn {
    foreach ($name in @("mvn", "mvn.cmd")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }
    $candidates = New-Object System.Collections.Generic.List[string]
    if ($env:M2_HOME) {
        [void]$candidates.Add((Join-Path $env:M2_HOME "bin\mvn.cmd"))
    }
    [void]$candidates.Add("${env:ProgramFiles}\Apache\maven\bin\mvn.cmd")
    [void]$candidates.Add("${env:ProgramFiles}\Maven\bin\mvn.cmd")
    $ideaHomes = @()
    foreach ($root in @(
            "${env:ProgramFiles}\JetBrains",
            "${env:LOCALAPPDATA}\Programs"
        )) {
        if (Test-Path $root) {
            $ideaHomes += Get-ChildItem $root -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -like "IntelliJ*" -or $_.Name -like "idea*" }
        }
    }
    foreach ($dir in $ideaHomes) {
        $ideaMvn = Join-Path $dir.FullName "plugins\maven-plugin\lib\maven3\bin\mvn.cmd"
        if (Test-Path $ideaMvn) { [void]$candidates.Add($ideaMvn) }
    }
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    return $null
}

function Get-FallbackClasspath([string]$ProjectRoot) {
    $classes = Join-Path $ProjectRoot "target\classes"
    if (-not (Test-Path (Join-Path $classes "com\veles\llm\jgpt\app\InferChat.class"))) {
        return $null
    }
    $m2 = Join-Path $env:USERPROFILE ".m2\repository"
    $jars = @(
        (Join-Path $m2 "org\slf4j\slf4j-api\2.0.9\slf4j-api-2.0.9.jar"),
        (Join-Path $m2 "ch\qos\logback\logback-classic\1.5.14\logback-classic-1.5.14.jar"),
        (Join-Path $m2 "ch\qos\logback\logback-core\1.5.14\logback-core-1.5.14.jar")
    )
    foreach ($j in $jars) {
        if (-not (Test-Path $j)) { return $null }
    }
    return (@($classes) + $jars) -join ";"
}

if (-not (Test-Path $EnvFile)) {
    Write-Host "[20L-CHAT] ERROR: missing $EnvFile" -ForegroundColor Red
    exit 1
}
Import-BashEnvFile $EnvFile
$env:JGPT_SFT = "1"
$env:JGPT_SFT_CHAT_TEMPLATE = "1"

$javaHome = Find-JavaHome
if (-not $javaHome) {
    Write-Host "[20L-CHAT] ERROR: JDK 25+ not found. Set JAVA_HOME." -ForegroundColor Red
    exit 1
}
$env:JAVA_HOME = $javaHome
$javaExe = Join-Path $env:JAVA_HOME "bin\java.exe"

$dll = Join-Path $Root "build\jgpt_cuda.dll"
if (-not $env:JGPT_CUDA_LIB -and (Test-Path $dll)) {
    $env:JGPT_CUDA_LIB = $dll
}
if (-not $env:JGPT_CUDA_LIB -or -not (Test-Path $env:JGPT_CUDA_LIB)) {
    Write-Host "[20L-CHAT] ERROR: no CUDA JNI. Expected $dll" -ForegroundColor Red
    exit 1
}

if (-not $ModelDefault) {
    Write-Host "[20L-CHAT] ERROR: no 20L-wide weights (train pretrain/SFT first)" -ForegroundColor Red
    exit 1
}

$modelAbs = Join-Path $Root $ModelDefault
$tokAbs = Join-Path $Root $TokDefault
if (-not (Test-Path $modelAbs)) {
    Write-Host "[20L-CHAT] ERROR: no weights: $modelAbs" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $tokAbs)) {
    Write-Host "[20L-CHAT] ERROR: no tokenizer: $tokAbs" -ForegroundColor Red
    exit 1
}

if ($env:MAVEN_OPTS -notmatch "enable-native-access") {
    $env:MAVEN_OPTS = "--enable-native-access=ALL-UNNAMED $($env:MAVEN_OPTS)".Trim()
}

$cp = $null
$mvn = Find-Mvn
if ($mvn) {
    Write-Host "[20L-CHAT] Maven: $mvn"
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $mvn -q compile
    $mvnCode = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    if ($mvnCode -ne 0) { exit $mvnCode }
    $cpFile = Join-Path $env:TEMP ("jgpt-chat-cp-" + [guid]::NewGuid().ToString() + ".txt")
    try {
        $ErrorActionPreference = "Continue"
        & $mvn -q dependency:build-classpath "-DincludeScope=runtime" "-Dmdep.outputFile=$cpFile"
        $mvnCode = $LASTEXITCODE
        $ErrorActionPreference = $prevEap
        if ($mvnCode -ne 0) { exit $mvnCode }
        $depCp = ((Get-Content -LiteralPath $cpFile -Raw).Trim() -replace "[\r\n]", "")
        $cp = "$(Join-Path $Root 'target\classes');$depCp"
    } finally {
        if (Test-Path $cpFile) { Remove-Item -LiteralPath $cpFile -Force }
    }
} else {
    Write-Host "[20L-CHAT] mvn not on PATH; using target\classes + .m2 jars"
    $cp = Get-FallbackClasspath $Root
    if (-not $cp) {
        Write-Host "[20L-CHAT] ERROR: no Maven and no compiled classes/jars." -ForegroundColor Red
        Write-Host "  winget install Apache.Maven"
        Write-Host "  then open a NEW PowerShell, or compile once from IDEA."
        exit 1
    }
}

$inferArgs = @(
    "--boo", $Root,
    "--layers", "20",
    "--seq-len", "1024",
    "--model", $ModelDefault,
    "--tokenizer", $TokDefault
) + @($args)

$javaArgs = @()
if (-not [string]::IsNullOrWhiteSpace($env:MAVEN_OPTS)) {
    $javaArgs += @($env:MAVEN_OPTS -split '\s+' | Where-Object { $_ })
}
$javaArgs += @(
    "-Dfile.encoding=UTF-8",
    "-Dstdout.encoding=UTF-8",
    "-Dstderr.encoding=UTF-8",
    "--sun-misc-unsafe-memory-access=allow",
    "--add-modules=jdk.incubator.vector",
    "--enable-preview",
    "-cp", $cp,
    "com.veles.llm.jgpt.app.InferChat"
) + $inferArgs

Write-Host "[20L-CHAT] model=$ModelDefault  (empty line / quit = exit)"
& $javaExe @javaArgs
exit $LASTEXITCODE
