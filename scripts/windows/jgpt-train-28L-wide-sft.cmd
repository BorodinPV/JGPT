@echo off
REM 28L-wide SFT after pretrain (does not overwrite pretrain, 20L, or 37L dirs).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0jgpt-train-28L-wide-sft.ps1" %*
exit /b %ERRORLEVEL%
