@echo off
REM Tiny exam finetune of 37L SFT (does not touch sft_37L_16k_2048 / short_ft weights dir).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0jgpt-train-37L-sft-exam.ps1" %*
exit /b %ERRORLEVEL%
