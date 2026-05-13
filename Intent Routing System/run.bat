@echo off
cd /d "%~dp0"
title Intent Classifier
color 0B

echo.
echo ================================================
echo    INTENT ROUTING SYSTEM
echo ================================================
echo.

:: ==============================================
:: CHECK PYTHON 3.11
:: ==============================================
echo [1/5] Checking Python 3.11...
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Python 3.11 is required but not found.
    echo        Download from: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
    echo        Make sure to check "Add Python to PATH"
    pause
    exit /b 1
)
py -3.11 --version
echo [OK] Python 3.11 found.
echo.

:: ==============================================
:: CREATE VIRTUAL ENV WITH PYTHON 3.11
:: ==============================================
echo [2/5] Setting up virtual environment...
if exist "venv\" (
    echo Removing old venv...
    rmdir /s /q venv
)
echo Creating venv with Python 3.11...
py -3.11 -m venv venv
if errorlevel 1 (
    echo [FAIL] Could not create virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment created with Python 3.11.
echo.

:: ==============================================
:: ACTIVATE
:: ==============================================
echo Activating virtual environment...
call "venv\Scripts\activate.bat"
echo [OK] Activated.
echo.

:: ==============================================
:: INSTALL DEPENDENCIES
:: ==============================================
echo [3/5] Installing dependencies...
python -m pip install --upgrade pip
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [FAIL] Installation failed.
    pause
    exit /b 1
)
echo.
echo [OK] All packages installed.
echo.

:: ==============================================
:: TRAIN MODEL
:: ==============================================
echo [4/5] Checking model...
if not exist "model\intent_classifier.pkl" (
    echo.
    echo Training model...
    python train.py
    if errorlevel 1 (
        echo [FAIL] Training failed.
        pause
        exit /b 1
    )
    echo [OK] Training complete.
) else (
    echo [OK] Model found.
)
echo.

:: ==============================================
:: LAUNCH
:: ==============================================
echo [5/5] Starting server...
echo.
echo ================================================
echo   http://127.0.0.1:5000
echo   Press Ctrl+C to stop
echo ================================================
echo.
python app.py
pause