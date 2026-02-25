@echo off
setlocal enabledelayedexpansion

:: Change to project root directory (parent of scripts\)
cd /d "%~dp0\.."

echo ============================================
echo   Workshop Setup - Let AI Handle the Mess
echo ============================================
echo.

:: Find a compatible Python version (3.12 or 3.13)
set "PYTHON_CMD="

:: Try py launcher first (most reliable on Windows)
where py >nul 2>&1
if %errorlevel%==0 (
    for %%V in (3.13 3.12) do (
        if not defined PYTHON_CMD (
            py -%%V --version >nul 2>&1
            if !errorlevel!==0 (
                set "PYTHON_CMD=py -%%V"
                for /f "tokens=*" %%A in ('py -%%V --version 2^>^&1') do set "FULL_VERSION=%%A"
            )
        )
    )
)

:: Fallback: try python3 and python directly
if not defined PYTHON_CMD (
    for %%C in (python3 python) do (
        if not defined PYTHON_CMD (
            where %%C >nul 2>&1
            if !errorlevel!==0 (
                for /f "tokens=2 delims= " %%A in ('%%C --version 2^>^&1') do (
                    for /f "tokens=1,2 delims=." %%M in ("%%A") do (
                        if "%%M"=="3" (
                            if "%%N"=="12" (
                                set "PYTHON_CMD=%%C"
                                set "FULL_VERSION=Python %%A"
                            )
                            if "%%N"=="13" (
                                set "PYTHON_CMD=%%C"
                                set "FULL_VERSION=Python %%A"
                            )
                        )
                    )
                )
            )
        )
    )
)

if not defined PYTHON_CMD (
    echo [WARNING] Could not find Python 3.12 or 3.13.
    echo    Python 3.14 is NOT supported by the packages used in this workshop.
    echo.

    :: Try to install via winget
    where winget >nul 2>&1
    if not !errorlevel!==0 (
        echo    winget is not available. Please install Python 3.13 manually from:
        echo    https://www.python.org/downloads/
        echo.
        exit /b 1
    )

    set /p "confirm=   Install Python 3.13 automatically via winget? (Y/n): "
    if /i "!confirm!"=="" set "confirm=Y"
    if /i "!confirm!"=="n" (
        echo.
        echo    Please install Python 3.13 manually from:
        echo    https://www.python.org/downloads/
        exit /b 1
    )

    echo.
    echo [SETUP] Installing Python 3.13 via winget...
    winget install Python.Python.3.13 --accept-package-agreements --accept-source-agreements
    if !errorlevel! neq 0 (
        echo [ERROR] Installation failed. Please install Python 3.13 manually from:
        echo    https://www.python.org/downloads/
        exit /b 1
    )

    echo [OK] Python 3.13 installed.
    echo.
    echo [NOTE] You may need to restart your terminal for Python to be available.
    echo        Please close this window, open a new Command Prompt, and run setup.bat again.
    echo.
    exit /b 0
)

echo [OK] Found compatible Python: %FULL_VERSION% (%PYTHON_CMD%)
echo.

:: Create virtual environment
if exist "venv" (
    echo [WARNING] A 'venv' folder already exists.
    set /p "confirm=   Do you want to delete it and create a new one? (y/N): "
    if /i "!confirm!"=="y" (
        rmdir /s /q venv
        echo    Removed old venv.
    ) else (
        echo    Keeping existing venv.
    )
)

if not exist "venv" (
    echo [SETUP] Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    echo [OK] Virtual environment created.
)

echo.

:: Activate virtual environment
echo [SETUP] Activating virtual environment...
call venv\Scripts\activate.bat

:: Install dependencies
echo [SETUP] Installing required packages...
echo.
pip install --upgrade pip -q
pip install -r requirements.txt
echo.

echo ============================================
echo   [OK] Setup complete!
echo ============================================
echo.
echo To activate the environment in the future, run:
echo.
echo   venv\Scripts\activate.bat
echo.
echo To start Jupyter Lab, run:
echo.
echo   jupyter lab
echo.
