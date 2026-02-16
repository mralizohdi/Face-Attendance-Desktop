@echo off
setlocal
cd /d %~dp0

where py >nul 2>nul
if %errorlevel%==0 (
  py -3.11 -c "import sys;print(sys.version)" >nul 2>nul
  if %errorlevel%==0 (
    set PY=py -3.11
  ) else (
    py -3.10 -c "import sys;print(sys.version)" >nul 2>nul
    if %errorlevel%==0 (
      set PY=py -3.10
    ) else (
      set PY=py
    )
  )
) else (
  set PY=python
)

if not exist venv\Scripts\python.exe (
  echo [1/3] Creating venv...
  %PY% -m venv venv
  if errorlevel 1 goto :ERR
)

echo [2/3] Installing requirements (first time only)...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto :ERR

echo [3/3] Starting Desktop App...
python app_tk.py
if errorlevel 1 goto :ERR

pause
exit /b 0

:ERR
echo.
echo ❌ /  .    ‌   .
pause
exit /b 1
