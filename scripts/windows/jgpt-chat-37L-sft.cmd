@echo off
REM Interactive InferChat for 37L SFT (model_best.bin).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0jgpt-chat-37L-sft.ps1" %*
exit /b %ERRORLEVEL%
