@echo off
REM AI Rapper System - Windows Installation Script

echo AI Rapper System - Installation
echo ==================================
echo.

REM Check Python
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+ from python.org
    pause
    exit /b 1
)
echo Python detected
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
echo Virtual environment created
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo Pip upgraded
echo.

REM Install requirements
echo Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt
echo Dependencies installed
echo.

REM Download NLTK data
echo Downloading NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('cmudict', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)"
echo NLTK data downloaded
echo.

REM Run setup
echo Running system setup...
python utils.py setup
echo System setup complete
echo.

REM Create .env
if not exist .env (
    echo Creating .env file...
    copy .env.example .env
    echo .env created - please configure your API keys
) else (
    echo .env already exists - skipping
)
echo.

REM Final message
echo ==================================
echo Installation complete!
echo.
echo Next steps:
echo    1. Configure .env file with your settings
echo    2. Add your lyrics to data\training_lyrics.json
echo    3. Start the server: python main.py
echo.
echo Documentation:
echo    - README.md: Complete documentation
echo    - QUICKSTART.md: Quick start guide
echo    - PROJECT_SUMMARY.md: System overview
echo.
echo Quick commands:
echo    python main.py           # Start server
echo    python example.py        # Run examples
echo    python utils.py health   # System check
echo.
echo Happy creating!
echo.
pause
