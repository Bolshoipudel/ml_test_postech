@echo off
REM Script to rebuild and restart the application

echo ========================================
echo Rebuilding LLM Assistant
echo ========================================
echo.

echo [1/4] Stopping existing containers...
docker-compose down
echo.

echo [2/4] Removing old app image...
docker rmi ml_test_postech-app 2>nul
echo.

echo [3/4] Building new image (this may take 3-5 minutes)...
docker-compose build --no-cache app
if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    echo Check the error messages above.
    pause
    exit /b 1
)
echo.

echo [4/4] Starting containers...
docker-compose up -d
echo.

echo Waiting for services to start...
timeout /t 10 /nobreak >nul
echo.

echo Checking status...
docker-compose ps
echo.

echo ========================================
echo Rebuild Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Check logs: docker-compose logs -f app
echo 2. Open Swagger UI: http://localhost:8000/docs
echo 3. Test health: curl http://localhost:8000/api/v1/health
echo.
pause
