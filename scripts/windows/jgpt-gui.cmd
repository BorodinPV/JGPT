@echo off
REM JGPT desktop GUI (training control + live charts + log + checkpoints + chat).
REM   jgpt-gui.cmd         compile GUI sources only (safe while a trainer is running)
REM   jgpt-gui.cmd --mvn   full mvn compile first
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0jgpt-gui.ps1" %*
exit /b %ERRORLEVEL%
