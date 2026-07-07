@echo off
setlocal

set "PROJECT_ROOT=C:\Users\sanggil\stopworking"
set "PYTHON_EXE=python"
set "LOG_DIR=%PROJECT_ROOT%\logs"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for /f %%i in ('%PYTHON_EXE% -c "import datetime; print(datetime.date.today().isoformat().replace('-', ''))"') do set "LOG_DATE=%%i"
set "LOG_FILE=%LOG_DIR%\daily_prediction_%LOG_DATE%.log"

cd /d "%PROJECT_ROOT%"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo [%date% %time%] daily prediction start >> "%LOG_FILE%"
%PYTHON_EXE% "%PROJECT_ROOT%\run_daily_prediction.py" --config "%PROJECT_ROOT%\config.ini" >> "%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"
echo [%date% %time%] daily prediction end exit_code=%EXIT_CODE% >> "%LOG_FILE%"

exit /b %EXIT_CODE%
