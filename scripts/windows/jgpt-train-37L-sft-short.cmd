@echo off
REM Short Q&A finetune of 37L SFT (does not touch sft_37L_16k_2048).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0jgpt-train-37L-sft-short.ps1" %*
exit /b %ERRORLEVEL%
