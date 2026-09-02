@echo off
REM Windows launcher for jgpt-train-37L-sft.ps1 (same flags as the .sh / .ps1).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0jgpt-train-37L-sft.ps1" %*
exit /b %ERRORLEVEL%
