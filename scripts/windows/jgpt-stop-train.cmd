@echo off
REM Soft-stop the running AllBooksTrain process (pretrain or SFT).
REM Does NOT kill java. Creates state\STOP; the trainer writes checkpoint_final.bin
REM and exits. Do NOT answer Yes to "Terminate batch job" / "завершить пакет".
cd /d "%~dp0..\.."
if not exist state mkdir state
echo stop> state\STOP
echo.
echo Stop requested: state\STOP
echo Wait in the training window for [STOP] then:
echo   [SHUTDOWN] checkpoint сохранён
echo Then you can close the window.
echo.
