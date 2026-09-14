#Requires -Version 5.1
# ASCII-only: Windows PowerShell 5.1 parses UTF-8 without BOM as system ANSI.
#
# jgpt-train-28L-wide-sft.ps1 - SFT after 28L-wide pretrain. Fresh Adam.
# Same geometry. New dir: checkpoints\wide_28L_sft
# Does NOT touch 37L or the pretrain dir (except reading model_best/final).
#
# Usage:
#   .\scripts\windows\jgpt-train-28L-wide-sft.cmd --no-build
#
# Chat (after this run):
#   .\scripts\windows\jgpt-chat-28L-wide.cmd
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

$LogFile = Join-Path $Root "training_28L_wide_sft.log"
$EnvFile = Join-Path $Root "env\28L-wide-sft.env"
$CkptDir = Join-Path $Root "checkpoints\wide_28L_sft"
$TokenizerFile = Join-Path $Root "checkpoints\tokenizer_wide_16k.bin"
$SrcBest = Join-Path $Root "checkpoints\wide_28L_16k_1024\model_best.bin"
$SrcFinal = Join-Path $Root "checkpoints\wide_28L_16k_1024\model_final.bin"

$DataDir = $env:JGPT_DATA_DIR
if (-not $DataDir) { $DataDir = "data\sft\clean" }
$DoFresh = $false
$SkipBuild = $false
$RestartPlan = $false
$CkptSubdirArg = $null
$SeedFrom = $null

function Show-Usage {
    Write-Host @"
Usage: .\scripts\windows\jgpt-train-28L-wide-sft.cmd [OPTIONS]

SFT after 28L-wide pretrain (one dialog per window, role tokens).
Data: data\sft\clean (built from data\sft\raw by scripts\sft-filter-clean.py if empty).
      --data-dir data\sft\short -> old length-only filter.
      --data-dir data\sft\exam -> synthetic exam.jsonl (tiny; memorizes, not a real SFT).
Preset: env\28L-wide-sft.env (full-vocab CE, LR 5e-5, 2 epochs)
Checkpoints: checkpoints\wide_28L_sft
Seeds from checkpoints\wide_28L_16k_1024\model_best.bin (or model_final.bin), fresh Adam.

Options:
  --data-dir PATH   directory with .jsonl (default: data\sft\clean)
  --env PATH        preset env file (default: env\28L-wide-sft.env)
  --ckpt-dir NAME   checkpoints\<NAME> (default: JGPT_CHECKPOINT_SUBDIR / wide_28L_sft)
  --seed-from PATH  copy these weights to model_final.bin when the ckpt dir has no Adam
  --restart-plan    keep weights + Adam from the newest checkpoint, reset step/LR/epoch/best
                    (= JGPT_FINETUNE=1 for this run only)
  --fresh           archive ONLY the target ckpt dir (tokenizer stays)
  --no-build        skip CUDA rebuild (need build\jgpt_cuda.dll)
  -h, --help        this help

Stop (do NOT Ctrl+C):
  .\scripts\windows\jgpt-stop-train.cmd

Examples:
  .\scripts\windows\jgpt-train-28L-wide-sft.cmd --no-build
  .\scripts\windows\jgpt-train-28L-wide-sft.cmd --no-build --env env\28L-wide-sft-overfit.env --data-dir data\sft\diag_overfit --ckpt-dir wide_28L_sft_overfit --seed-from checkpoints\wide_28L_sft\model_best.bin
"@
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

function Get-JgptUserOverrides {
    $saved = @{}
    Get-ChildItem Env: | Where-Object { $_.Name -like "JGPT_*" } | ForEach-Object {
        $saved[$_.Name] = $_.Value
    }
    return $saved
}

function Restore-JgptUserOverrides($saved) {
    foreach ($k in $saved.Keys) {
        [System.Environment]::SetEnvironmentVariable($k, $saved[$k], "Process")
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
    foreach ($root in @(
            "${env:ProgramFiles}\Java",
            "${env:ProgramFiles}\Eclipse Adoptium",
            "${env:ProgramFiles}\BellSoft",
            "${env:ProgramFiles}\Microsoft",
            "${env:ProgramFiles}\Amazon Corretto"
        )) {
        if (-not (Test-Path $root)) { continue }
        Get-ChildItem $root -Directory -ErrorAction SilentlyContinue | ForEach-Object {
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

function Split-MavenOpts([string]$Raw) {
    if ([string]::IsNullOrWhiteSpace($Raw)) { return @() }
    return @($Raw -split '\s+' | Where-Object { $_ })
}

function Get-JsonlCount([string]$Dir) {
    if (-not (Test-Path $Dir)) { return 0 }
    return @(Get-ChildItem -LiteralPath $Dir -Filter "*.jsonl" -File -Recurse -ErrorAction SilentlyContinue).Count
}

function Invoke-LoggedJava([string]$JavaExe, [string[]]$JavaArgs, [string]$LogPath) {
    $utf8 = New-Object System.Text.UTF8Encoding $false
    $writer = New-Object System.IO.StreamWriter($LogPath, $true, $utf8)
    $writer.AutoFlush = $true
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $JavaExe @JavaArgs 2>&1 | ForEach-Object {
            $line = "$_"
            [Console]::Out.WriteLine($line)
            $writer.WriteLine($line)
        }
        if ($null -ne $LASTEXITCODE) { return $LASTEXITCODE }
        if (-not $?) { return 1 }
        return 0
    } finally {
        $writer.Flush()
        $writer.Dispose()
        $ErrorActionPreference = $prevEap
    }
}

try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch { }

$argList = @($args)
$i = 0
while ($i -lt $argList.Count) {
    switch ($argList[$i]) {
        "--data-dir" {
            if ($i + 1 -ge $argList.Count) {
                Write-Host "[28L-WIDE-SFT] ERROR: --data-dir requires a path" -ForegroundColor Red
                Show-Usage
                exit 1
            }
            $DataDir = $argList[$i + 1]
            $i += 2
        }
        "--env" {
            if ($i + 1 -ge $argList.Count) {
                Write-Host "[28L-WIDE-SFT] ERROR: --env requires a path" -ForegroundColor Red
                Show-Usage
                exit 1
            }
            $EnvFile = $argList[$i + 1]
            $i += 2
        }
        "--ckpt-dir" {
            if ($i + 1 -ge $argList.Count) {
                Write-Host "[28L-WIDE-SFT] ERROR: --ckpt-dir requires a directory name" -ForegroundColor Red
                Show-Usage
                exit 1
            }
            $CkptSubdirArg = $argList[$i + 1]
            $i += 2
        }
        "--seed-from" {
            if ($i + 1 -ge $argList.Count) {
                Write-Host "[28L-WIDE-SFT] ERROR: --seed-from requires a path" -ForegroundColor Red
                Show-Usage
                exit 1
            }
            $SeedFrom = $argList[$i + 1]
            $i += 2
        }
        "--fresh" { $DoFresh = $true; $i += 1 }
        "--no-build" { $SkipBuild = $true; $i += 1 }
        "--restart-plan" { $RestartPlan = $true; $i += 1 }
        { $_ -in @("-h", "--help") } { Show-Usage; exit 0 }
        default {
            Write-Host "Unknown argument: $($argList[$i])" -ForegroundColor Red
            Show-Usage
            exit 1
        }
    }
}

if (-not [System.IO.Path]::IsPathRooted($EnvFile)) {
    $EnvFile = Join-Path $Root $EnvFile
}
if (-not (Test-Path -LiteralPath $EnvFile)) {
    Write-Host "[28L-WIDE-SFT] ERROR: missing preset: $EnvFile" -ForegroundColor Red
    exit 1
}

if ($RestartPlan) {
    $env:JGPT_FINETUNE = "1"
    Write-Host "[28L-WIDE-SFT] --restart-plan: JGPT_FINETUNE=1 (weights+Adam kept, step/LR/epoch/best reset)"
} elseif ($env:JGPT_FINETUNE -and $env:JGPT_FINETUNE -ne "0") {
    Write-Host "[28L-WIDE-SFT] WARNING: JGPT_FINETUNE=$($env:JGPT_FINETUNE) is set in this shell - step counter will be reset AGAIN." -ForegroundColor Yellow
    Write-Host "  For a plain resume run: Remove-Item Env:JGPT_FINETUNE  (or open a new PowerShell)" -ForegroundColor Yellow
}

if (-not [System.IO.Path]::IsPathRooted($DataDir)) {
    $DataDir = Join-Path $Root $DataDir
}

$jsonlBefore = Get-JsonlCount $DataDir
$dataLeaf = (Split-Path -Leaf $DataDir).ToLowerInvariant()
if ($jsonlBefore -eq 0) {
    # Empty data dir: build it. "exam" -> synthetic exam.jsonl (tiny, memorizes; not for real SFT).
    # Anything else (default "short") -> filter data\sft\raw with sft-filter-short.py.
    $py = $null
    foreach ($c in @("python", "py", "python3")) {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if ($cmd) { $py = $cmd.Source; break }
    }
    if (-not $py) {
        Write-Host "[28L-WIDE-SFT] ERROR: python not found (need it to build $DataDir)" -ForegroundColor Red
        exit 1
    }
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    if ($dataLeaf -eq "exam") {
        $makePy = Join-Path $Root "scripts\sft-make-exam.py"
        Write-Host "[28L-WIDE-SFT] generating exam JSONL -> $DataDir"
        & $py $makePy --dst (Join-Path $DataDir "exam.jsonl")
    } elseif ($dataLeaf -eq "short") {
        $filterPy = Join-Path $Root "scripts\sft-filter-short.py"
        $rawDir = Join-Path $Root "data\sft\raw"
        if (-not (Test-Path -LiteralPath $rawDir)) {
            Write-Host "[28L-WIDE-SFT] ERROR: $DataDir is empty and $rawDir is missing (nothing to filter)" -ForegroundColor Red
            exit 1
        }
        Write-Host "[28L-WIDE-SFT] filtering $rawDir -> $DataDir (sft-filter-short.py)"
        & $py $filterPy --src $rawDir --dst $DataDir
    } else {
        $filterPy = Join-Path $Root "scripts\sft-filter-clean.py"
        $rawDir = Join-Path $Root "data\sft\raw"
        if (-not (Test-Path -LiteralPath $rawDir)) {
            Write-Host "[28L-WIDE-SFT] ERROR: $DataDir is empty and $rawDir is missing (nothing to filter)" -ForegroundColor Red
            exit 1
        }
        Write-Host "[28L-WIDE-SFT] filtering $rawDir -> $DataDir (sft-filter-clean.py)"
        & $py $filterPy --src $rawDir --dst $DataDir
    }
    $pyCode = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    if ($pyCode -ne 0) { exit $pyCode }
} else {
    Write-Host "[28L-WIDE-SFT] using existing jsonl in $DataDir"
}

if (-not (Test-Path -LiteralPath $DataDir)) {
    Write-Host "[28L-WIDE-SFT] ERROR: data dir missing: $DataDir" -ForegroundColor Red
    exit 1
}
$DataDir = (Resolve-Path -LiteralPath $DataDir).Path

$jsonlCount = Get-JsonlCount $DataDir
if ($jsonlCount -eq 0) {
    Write-Host "[28L-WIDE-SFT] ERROR: no .jsonl files in $DataDir (JGPT_SFT=1)" -ForegroundColor Red
    exit 1
}

$userOverrides = Get-JgptUserOverrides
$userFaTile = $userOverrides.ContainsKey("JGPT_FA_TILE_SIZE")
Import-BashEnvFile $EnvFile
Restore-JgptUserOverrides $userOverrides
$env:JGPT_DATA_DIR = $DataDir
if ($CkptSubdirArg) {
    if ($CkptSubdirArg -match '[\\/]|\.\.') {
        Write-Host "[28L-WIDE-SFT] ERROR: --ckpt-dir must be a single name under checkpoints\" -ForegroundColor Red
        exit 1
    }
    $env:JGPT_CHECKPOINT_SUBDIR = $CkptSubdirArg
}
if (-not $env:JGPT_CHECKPOINT_SUBDIR) {
    $env:JGPT_CHECKPOINT_SUBDIR = "wide_28L_sft"
}
# Concatenation (not a quoted Join-Path) so the GUI TrainRun parser keeps the
# default $CkptDir above instead of treating $($env:...) as a Windows path.
$CkptDir = Join-Path $Root ("checkpoints\" + $env:JGPT_CHECKPOINT_SUBDIR)
$CkptBackup = Join-Path $Root ("checkpoints\" + $env:JGPT_CHECKPOINT_SUBDIR + "_prev_backup")
$LogFile = Join-Path $Root ("training_" + $env:JGPT_CHECKPOINT_SUBDIR + ".log")

$javaHome = Find-JavaHome
if (-not $javaHome) {
    Write-Host "[28L-WIDE-SFT] ERROR: JDK 25+ not found. Set JAVA_HOME, for example:" -ForegroundColor Red
    Write-Host "  `$env:JAVA_HOME = '$env:USERPROFILE\.jdks\liberica-full-26.0.1'"
    exit 1
}
$env:JAVA_HOME = $javaHome
$javaExe = Join-Path $env:JAVA_HOME "bin\java.exe"

if ($env:MAVEN_OPTS -notmatch "enable-native-access") {
    $env:MAVEN_OPTS = "--enable-native-access=ALL-UNNAMED $($env:MAVEN_OPTS)".Trim()
}
if ($env:JGPT_JAVA_MEM) {
    $env:MAVEN_OPTS = "$($env:JGPT_JAVA_MEM) $($env:MAVEN_OPTS)".Trim()
}

if (-not $env:JGPT_IF_STEP_BEYOND_PLAN) {
    $env:JGPT_IF_STEP_BEYOND_PLAN = "restart_schedule"
}

if (-not $userFaTile) {
    $smi = Get-NativeOutput { nvidia-smi --query-gpu=compute_cap --format=csv,noheader }
    $cap = ($smi -split "[\r\n]+" | Where-Object { $_.Trim() } | Select-Object -First 1)
    if ($cap) { $cap = $cap.Trim() }
    if ($cap -eq "7.5") {
        $env:JGPT_FA_TILE_SIZE = "64"
        Write-Host "[28L-WIDE-SFT] GPU compute $cap (Turing): JGPT_FA_TILE_SIZE=64"
    }
}

if ($DoFresh) {
    New-Item -ItemType Directory -Force -Path $CkptBackup | Out-Null
    $moved = 0
    if (Test-Path $CkptDir) {
        Get-ChildItem -LiteralPath $CkptDir -Force -ErrorAction SilentlyContinue | ForEach-Object {
            Move-Item -LiteralPath $_.FullName -Destination $CkptBackup -Force
            $moved += 1
        }
    }
    Write-Host "[28L-WIDE-SFT] --fresh: moved $moved file(s) to $CkptBackup (tokenizer untouched)"
}

$hasCkpt = $false
if (Test-Path $CkptDir) {
    # AllBooksTrain resumes from the newest of final/step_N/epoch_N/best (by globalStep in the header)
    $anyCk = @(Get-ChildItem -LiteralPath $CkptDir -Filter "checkpoint_*.bin" -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^checkpoint_(final|best|step_\d+|epoch_\d+)\.bin$' })
    if ($anyCk.Count -gt 0) { $hasCkpt = $true }
}
if ($hasCkpt) {
    Write-Host "[28L-WIDE-SFT] NOTE: found Adam checkpoint(s) in $CkptDir - resume from the newest step"
} else {
    New-Item -ItemType Directory -Force -Path $CkptDir | Out-Null
    $dstFinal = Join-Path $CkptDir "model_final.bin"
    if (-not (Test-Path -LiteralPath $dstFinal)) {
        $seed = $null
        if ($SeedFrom) {
            if (-not [System.IO.Path]::IsPathRooted($SeedFrom)) {
                $SeedFrom = Join-Path $Root $SeedFrom
            }
            if (-not (Test-Path -LiteralPath $SeedFrom)) {
                Write-Host "[28L-WIDE-SFT] ERROR: --seed-from missing: $SeedFrom" -ForegroundColor Red
                exit 1
            }
            $seed = $SeedFrom
        } elseif (Test-Path -LiteralPath $SrcBest) {
            $seed = $SrcBest
        } elseif (Test-Path -LiteralPath $SrcFinal) {
            $seed = $SrcFinal
        }
        if (-not $seed) {
            Write-Host "[28L-WIDE-SFT] ERROR: missing seed weights: $SrcBest" -ForegroundColor Red
            Write-Host "  Run pretrain first: .\scripts\windows\jgpt-train-28L-wide.cmd --no-build"
            exit 1
        }
        Copy-Item -LiteralPath $seed -Destination $dstFinal -Force
        Write-Host "[28L-WIDE-SFT] seeded weights: $seed -> $dstFinal (fresh Adam)"
    } else {
        Write-Host "[28L-WIDE-SFT] NOTE: $dstFinal exists, no Adam checkpoint - continue from weights, step 0"
        if ($SeedFrom) {
            Write-Host "[28L-WIDE-SFT] NOTE: --seed-from ignored because $dstFinal already exists"
        }
    }
}
if (-not (Test-Path $TokenizerFile)) {
    Write-Host "[28L-WIDE-SFT] ERROR: missing tokenizer: $TokenizerFile" -ForegroundColor Red
    exit 1
}

$bs = 0
$acc = 0
[void][int]::TryParse($env:JGPT_BATCH_SIZE, [ref]$bs)
[void][int]::TryParse($env:JGPT_ACCUMULATION_STEPS, [ref]$acc)
$eff = $bs * $acc

Write-Host ""
Write-Host "============================================================"
Write-Host " JGPT Train 28L-wide SFT  |  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host " layers=$($env:JGPT_PRESET_NUM_LAYERS)  d=$($env:JGPT_D_MODEL)  heads=$($env:JGPT_NUM_HEADS)  seq=$($env:JGPT_MAX_SEQ_LEN)  vocab=$($env:JGPT_VOCAB_SIZE)"
Write-Host " batch=$($env:JGPT_BATCH_SIZE)  accum=$($env:JGPT_ACCUMULATION_STEPS)  eff_batch=$eff  lr=$($env:JGPT_LEARNING_RATE)  pack=$($env:JGPT_SFT_PACK)"
Write-Host " data=$DataDir  ($jsonlCount jsonl)"
Write-Host " ckpt=$CkptDir"
Write-Host " tok=$TokenizerFile"
Write-Host " log=$LogFile"
Write-Host " JAVA_HOME=$($env:JAVA_HOME)"
Write-Host " stop=.\scripts\windows\jgpt-stop-train.cmd  (NOT Ctrl+C)"
Write-Host "============================================================"
Write-Host ""

$dll = Join-Path $Root "build\jgpt_cuda.dll"
if (-not $SkipBuild) {
    $buildPs1 = Join-Path $Root "scripts\windows\build-cuda.ps1"
    & $buildPs1
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    if (-not (Test-Path $dll)) {
        Write-Host "[28L-WIDE-SFT] ERROR: --no-build but no lib in build\ ($dll)" -ForegroundColor Red
        exit 1
    }
}

$cudaEnv = Join-Path $Root "build\jgpt-cuda-env.ps1"
if (Test-Path $cudaEnv) {
    . $cudaEnv
}
if (Test-Path $dll) {
    $env:JGPT_CUDA_LIB = $dll
    $env:PATH = "$(Join-Path $Root 'build');$env:PATH"
}

$mvn = Find-Mvn
if (-not $mvn) {
    Write-Host "[28L-WIDE-SFT] ERROR: Maven not found. Install it, then open a NEW PowerShell:" -ForegroundColor Red
    Write-Host "  winget install Apache.Maven"
    Write-Host "Or use IntelliJ bundled Maven (mvn.cmd under plugins\maven-plugin\lib\maven3\bin)."
    exit 1
}

Write-Host "[28L-WIDE-SFT] mvn compile..."
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $mvn -q compile
$mvnCode = $LASTEXITCODE
$ErrorActionPreference = $prevEap
if ($mvnCode -ne 0) { exit $mvnCode }

$cpFile = Join-Path $env:TEMP ("jgpt-cp-" + [guid]::NewGuid().ToString() + ".txt")
try {
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $mvn -q dependency:build-classpath "-DincludeScope=runtime" "-Dmdep.outputFile=$cpFile"
    $mvnCode = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    if ($mvnCode -ne 0) { exit $mvnCode }
    $depCp = (Get-Content -LiteralPath $cpFile -Raw).Trim()
    $depCp = $depCp -replace "[\r\n]", ""
    $cp = "$(Join-Path $Root 'target\classes');$depCp"
} finally {
    if (Test-Path $cpFile) { Remove-Item -LiteralPath $cpFile -Force }
}

$javaArgs = @()
$javaArgs += Split-MavenOpts $env:MAVEN_OPTS
$javaArgs += @(
    "--sun-misc-unsafe-memory-access=allow",
    "--add-modules=jdk.incubator.vector",
    "--enable-preview",
    "-cp", $cp,
    "com.veles.llm.jgpt.app.AllBooksTrain",
    "--boo", $Root,
    "--data-dir", $DataDir
)

Write-Host "[28L-WIDE-SFT] starting AllBooksTrain (resume from newest checkpoint_* in wide_28L_sft; stop: jgpt-stop-train.cmd)..."
$exitCode = Invoke-LoggedJava $javaExe $javaArgs $LogFile
exit $exitCode
