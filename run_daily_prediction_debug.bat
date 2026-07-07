@echo off
setlocal

set "PROJECT_ROOT=C:\Users\sanggil\stopworking"
set "PYTHON_EXE=python"
set "LOG_DIR=%PROJECT_ROOT%\logs"

echo [debug] PROJECT_ROOT=%PROJECT_ROOT%
echo [debug] PYTHON_EXE=%PYTHON_EXE%

if not exist "%PROJECT_ROOT%" (
    echo [error] project root not found: %PROJECT_ROOT%
    pause
    exit /b 1
)

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for /f %%i in ('%PYTHON_EXE% -c "import datetime; print(datetime.date.today().isoformat().replace('-', ''))"') do set "LOG_DATE=%%i"
if "%LOG_DATE%"=="" set "LOG_DATE=unknown_date"

set "LOG_FILE=%LOG_DIR%\daily_prediction_%LOG_DATE%.log"

cd /d "%PROJECT_ROOT%"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo [debug] LOG_FILE=%LOG_FILE%
echo [debug] starting python...
echo [%date% %time%] daily prediction debug start >> "%LOG_FILE%"

%PYTHON_EXE% "%PROJECT_ROOT%\run_daily_prediction.py" --config "%PROJECT_ROOT%\config.ini" >> "%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

echo [%date% %time%] daily prediction debug end exit_code=%EXIT_CODE% >> "%LOG_FILE%"
echo [debug] python exit_code=%EXIT_CODE%
echo [debug] log file:
echo %LOG_FILE%
echo.
echo Last 20 log lines:
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Test-Path '%LOG_FILE%') { Get-Content '%LOG_FILE%' -Tail 20 } else { Write-Host 'log file not found' }"
echo.
pause
exit /b %EXIT_CODE%
