$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\sanggil\stopworking"
$PythonExe = "python"
$LogDir = Join-Path $ProjectRoot "logs"
$LogFile = Join-Path $LogDir ("daily_prediction_{0}.log" -f (Get-Date -Format "yyyyMMdd"))

if (!(Test-Path -LiteralPath $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

Set-Location -LiteralPath $ProjectRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] daily prediction start" |
    Tee-Object -FilePath $LogFile -Append

& $PythonExe ".\run_daily_prediction.py" --config ".\config.ini" 2>&1 |
    Tee-Object -FilePath $LogFile -Append

$ExitCode = $LASTEXITCODE

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] daily prediction end exit_code=$ExitCode" |
    Tee-Object -FilePath $LogFile -Append

exit $ExitCode
