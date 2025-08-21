@echo off
title Mochi Production Video Generator

echo.
echo ================================================
echo      Mochi Production Video Generator
echo         Optimized for RTX 3070 Laptop
echo ================================================
echo.
echo BENCHMARK RESULTS:
echo   CPU Mode: 130.6 min (16f/16s) - FASTER + RECOMMENDED
echo   GPU Mode: 214.1 min (16f/32s) - Available but slower  
echo   Winner: CPU Mode (40%% faster on your system!)
echo.

REM Check virtual environment
if not exist "env" (
    echo ERROR: Virtual environment not found!
    pause
    exit /b 1
)

REM Check required files
if not exist "config.py" (
    echo ERROR: Production config not found!
    echo Please save the config.py file first.
    pause
    exit /b 1
)

if not exist "app_huggingface.py" (
    echo ERROR: Production app not found!
    echo Please save the app_huggingface.py file first.
    pause
    exit /b 1
)

echo Activating virtual environment...
call env\Scripts\activate

echo.
echo Checking Redis...
python -c "import redis; redis.Redis().ping(); print('Redis OK')" || (
    echo ERROR: Redis not running
    echo Please start Redis service first
    pause
    exit /b 1
)

echo.
echo Testing HuggingFace Diffusers...
python -c "import diffusers; print('Diffusers version:', diffusers.__version__)" || (
    echo ERROR: Diffusers not installed
    echo Installing diffusers...
    pip install diffusers[torch] accelerate transformers
)

echo.
echo Cleaning up any existing processes...
taskkill /F /IM celery.exe 2>nul >nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Mochi Production Worker*" 2>nul >nul

echo.
echo Starting Production Celery Worker...
start "Mochi Production Worker" cmd /k "title Mochi Production Worker && color 0B && echo PRODUCTION WORKER STARTING... && echo Keep this window open! && echo. && call env\Scripts\activate && celery -A app_huggingface.celery worker --loglevel=info --pool=solo --concurrency=1"

echo.
echo Waiting for worker to start...
timeout /t 10 >nul

echo.
echo Testing worker connection...
python -c "from app import celery; result=celery.control.inspect().ping(); print('Worker ready!' if result and len(result) > 0 else 'Worker starting...')" 2>nul || echo Worker initializing...

echo.
echo ================================================
echo        PRODUCTION SYSTEM READY!
echo ================================================
echo.
echo Web Interface: http://localhost:5000
echo.
echo OPTIMIZED PERFORMANCE:
echo   Quick Test:    8f/8s   -> ~33 minutes
echo   Standard:     16f/16s  -> ~131 minutes (benchmarked)
echo   High Quality: 24f/24s  -> ~295 minutes
echo   Maximum:      31f/32s  -> ~521 minutes (overnight)
echo.
echo RECOMMENDATIONS:
echo   - Start with Quick Test to verify everything works
echo   - Use Standard for regular videos
echo   - Run High/Maximum quality overnight
echo   - CPU mode is faster on your RTX 3070 Laptop!
echo.
echo Controls:
echo   - Press Ctrl+C here to stop Flask
echo   - Close worker window to stop processing
echo ================================================
echo.

echo Starting Production Flask App...
python app_huggingface.py

echo.
echo Production system stopped.
echo Cleaning up...
taskkill /F /IM celery.exe 2>nul >nul
echo Done!
pause