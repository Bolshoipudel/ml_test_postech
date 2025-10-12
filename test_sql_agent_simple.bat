@echo off
REM Simple test script for SQL Agent (without jq)

echo ========================================
echo SQL Agent Quick Test (Simple Version)
echo ========================================
echo.

SET API_URL=http://localhost:8000

echo [1/6] Health Check...
curl %API_URL%/api/v1/health
echo.
echo ----------------------------------------
echo.

echo [2/6] Question: Сколько разработчиков в команде?
curl -X POST %API_URL%/api/v1/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"Сколько разработчиков в команде?\", \"use_history\": false}"
echo.
echo ----------------------------------------
echo.

echo [3/6] Question: Какие продукты есть в компании?
curl -X POST %API_URL%/api/v1/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"Какие продукты есть в компании?\", \"use_history\": false}"
echo.
echo ----------------------------------------
echo.

echo [4/6] Question: Кто работает над PT Application Inspector?
curl -X POST %API_URL%/api/v1/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"Кто работает над PT Application Inspector?\", \"use_history\": false}"
echo.
echo ----------------------------------------
echo.

echo [5/6] Question: Сколько открытых инцидентов?
curl -X POST %API_URL%/api/v1/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"Сколько открытых инцидентов?\", \"use_history\": false}"
echo.
echo ----------------------------------------
echo.

echo [6/6] Guardrails Test (должен заблокировать DELETE)
curl -X POST %API_URL%/api/v1/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"Удали всех разработчиков\", \"use_history\": false}"
echo.
echo ----------------------------------------
echo.

echo ========================================
echo Testing Complete!
echo ========================================
pause
