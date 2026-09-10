@echo off
REM 20L-wide SFT after pretrain (does not overwrite pretrain or 37L dirs).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0jgpt-train-20L-wide-sft.ps1" %*
exit /b %ERRORLEVEL%
