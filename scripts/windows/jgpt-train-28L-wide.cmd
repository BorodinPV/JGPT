@echo off
REM 28L-wide books pretrain ~134M (does not touch sft_37L_* or wide_20L_*).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0jgpt-train-28L-wide.ps1" %*
exit /b %ERRORLEVEL%
