@echo off
REM ModelShift - portable local launcher (Windows).
REM One command: creates a local venv, installs deps, and starts the server.
REM
REM   run.bat            LIVE mode (real Bedrock via your local AWS creds), :8971
REM   run.bat --demo     OFFLINE demo (seeded sample run, no AWS needed)
REM
REM Optional overrides via environment variables: PORT, HOST, REGION, PYTHON.
setlocal enabledelayedexpansion

set "REPO_DIR=%~dp0"
cd /d "%REPO_DIR%"

if "%VENV%"==""   set "VENV=%REPO_DIR%.venv"
if "%PORT%"==""   set "PORT=8971"
if "%HOST%"==""   set "HOST=127.0.0.1"
if "%REGION%"=="" set "REGION=us-east-1"
if "%PYTHON%"=="" set "PYTHON=python"

REM Create the venv on first run.
if not exist "%VENV%\Scripts\python.exe" (
  echo ==^> Creating virtualenv at %VENV% ...
  "%PYTHON%" -m venv "%VENV%"
  if errorlevel 1 (
    echo !! Could not create venv. Install Python 3.11+ and ensure it is on PATH. 1>&2
    exit /b 1
  )
)
set "PY=%VENV%\Scripts\python.exe"

echo ==^> Installing dependencies (requirements.txt) ...
"%PY%" -m pip install --quiet --upgrade pip
"%PY%" -m pip install --quiet -r "%REPO_DIR%requirements.txt"

REM Default mode is live; first --demo/--live arg overrides. Remaining args pass through.
set "MODE=--live"
set "PASS_ARGS="
for %%A in (%*) do (
  if "%%A"=="--demo" ( set "MODE=--demo"
  ) else if "%%A"=="--live" ( set "MODE=--live"
  ) else ( set "PASS_ARGS=!PASS_ARGS! %%A" )
)

echo ==^> Starting ModelShift %MODE% on http://%HOST%:%PORT%  (region=%REGION%)
echo     Open http://%HOST%:%PORT% in your browser. Press Ctrl-C to stop.
echo.

"%PY%" -m modelshift.server %MODE% --region "%REGION%" --host "%HOST%" --port "%PORT%"%PASS_ARGS%
endlocal
