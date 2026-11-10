@echo off
rem Double-click to start the CO2 backfill predictor. The first start installs it (one time only).
title CO2 backfill predictor
cd /d "%~dp0app"
if exist ".venv\installed.ok" goto run

set PY=
py -3 -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1 && set PY=py -3
if not defined PY python -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1 && set PY=python
if not defined PY (
  echo.
  echo  Python 3.11 or newer was not found on this computer.
  echo  Please install Python 3.12 from https://www.python.org/downloads/
  echo  and tick "Add python.exe to PATH" on the first screen of the installer.
  echo  Then double-click this file again.
  echo.
  pause
  exit /b 1
)
echo.
echo  First start: installing the app. This happens only once and takes 5-10 minutes.
echo.
%PY% -m venv .venv || goto fail
.venv\Scripts\python -m pip install -r requirements.txt --timeout 30 || .venv\Scripts\python -m pip install -r requirements.txt --timeout 30 -i https://pypi.tuna.tsinghua.edu.cn/simple || goto fail
echo ok> .venv\installed.ok

:run
echo.
echo  The app will open in your web browser in a few seconds (http://localhost:8501).
echo  The first page takes about 1 minute while the models train.
echo  Keep this window open while you use the app. Close it to stop the app.
echo.
start "" cmd /c "timeout /t 8 /nobreak >nul & start http://localhost:8501"
.venv\Scripts\python -m streamlit run app.py --server.port 8501
pause
exit /b 0

:fail
echo.
echo  Installation failed. Please check the internet connection and double-click this file again.
echo  If it fails again, send Zayn a photo of this window.
if exist .venv rmdir /s /q .venv
pause
exit /b 1
