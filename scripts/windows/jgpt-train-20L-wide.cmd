@echo off
REM 20L-wide books pretrain (does not touch sft_37L_* checkpoints).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0jgpt-train-20L-wide.ps1" %*
exit /b %ERRORLEVEL%
