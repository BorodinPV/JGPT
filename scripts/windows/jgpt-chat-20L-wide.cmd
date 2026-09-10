@echo off
REM Interactive InferChat for 20L-wide (SFT if present, else pretrain).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0jgpt-chat-20L-wide.ps1" %*
exit /b %ERRORLEVEL%
