@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv" (
  py -3.12 -m venv .venv || py -3.11 -m venv .venv || py -3 -m venv .venv
  if errorlevel 1 (
    echo Failed to create Python virtual environment.
    pause
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-build.txt
if errorlevel 1 (
  echo Failed to install dependencies.
  pause
  exit /b 1
)

pyinstaller --clean --noconfirm ColonyCounter.spec
if errorlevel 1 (
  echo Build failed.
  pause
  exit /b 1
)

echo.
echo Build complete:
echo %cd%\dist\ColonyCounter\ColonyCounter.exe
echo.
echo Zip the whole dist\ColonyCounter folder when distributing it.
pause
