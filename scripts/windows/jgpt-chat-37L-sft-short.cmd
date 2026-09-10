@echo off
REM Interactive InferChat for 37L short SFT finetune (model_best.bin).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0jgpt-chat-37L-sft-short.ps1" %*
exit /b %ERRORLEVEL%
