#Requires -Version 5.1
# ASCII-only. Point AllBooksTrain resume at best after a hard kill
# (resume prefers checkpoint_final.bin / model_final.bin).
#
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\Users\pc\Desktop\JGPT\scripts\windows\jgpt-resume-37L-from-best.ps1
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$d = Join-Path $Root "checkpoints\sft_37L_16k_2048"

if (-not (Test-Path -LiteralPath $d)) {
    Write-Host "ERROR: no checkpoint dir: $d" -ForegroundColor Red
    exit 1
}

Write-Host "dir: $d"
Write-Host ""
Get-ChildItem -LiteralPath $d -File |
    Where-Object { $_.Name -match '^(model_|checkpoint_)' } |
    Sort-Object LastWriteTime -Descending |
    Format-Table Name, LastWriteTime, Length -AutoSize

$bestC = Join-Path $d "checkpoint_best.bin"
$bestM = Join-Path $d "model_best.bin"
$finC = Join-Path $d "checkpoint_final.bin"
$finM = Join-Path $d "model_final.bin"

if (-not (Test-Path -LiteralPath $bestC) -or -not (Test-Path -LiteralPath $bestM)) {
    Write-Host "ERROR: need checkpoint_best.bin and model_best.bin" -ForegroundColor Red
    exit 1
}

$bestTime = (Get-Item -LiteralPath $bestC).LastWriteTime
$need = $true
if (Test-Path -LiteralPath $finC) {
    $finTime = (Get-Item -LiteralPath $finC).LastWriteTime
    if ($finTime -ge $bestTime) {
        $need = $false
        Write-Host "final is newer or same as best ($finTime). no copy."
    } else {
        Write-Host "final is older than best ($finTime < $bestTime). will replace."
    }
} else {
    Write-Host "no checkpoint_final.bin. will copy from best."
}

if ($need) {
    if (Test-Path -LiteralPath $finC) {
        Move-Item -LiteralPath $finC -Destination ($finC + ".stale") -Force
    }
    if (Test-Path -LiteralPath $finM) {
        Move-Item -LiteralPath $finM -Destination ($finM + ".stale") -Force
    }
    Copy-Item -LiteralPath $bestC -Destination $finC
    Copy-Item -LiteralPath $bestM -Destination $finM
    Write-Host "OK: final is a copy of best"
}

Write-Host ""
Write-Host "Next (one line):"
Write-Host "powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Root\scripts\windows\jgpt-train-37L-sft.ps1 --no-build"
