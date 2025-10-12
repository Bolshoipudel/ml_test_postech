@echo off
REM Quick test script for SQL Agent

echo ========================================
echo SQL Agent Quick Test
echo ========================================
echo.

SET API_URL=http://localhost:8000

echo [1/6] Testing Health Check...
curl -s %API_URL%/api/v1/health | jq .
echo.
echo.

echo [2/6] Testing simple count query...
curl -s -X POST %API_URL%/api/v1/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"Сколько разработчиков в команде?\", \"use_history\": false}" | jq .
echo.
echo.

echo [3/6] Testing product list query...
curl -s -X POST %API_URL%/api/v1/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"Какие продукты есть в компании?\", \"use_history\": false}" | jq .
echo.
echo.

echo [4/6] Testing JOIN query...
curl -s -X POST %API_URL%/api/v1/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"Кто работает над PT Application Inspector?\", \"use_history\": false}" | jq .
echo.
echo.

echo [5/6] Testing incident query...
curl -s -X POST %API_URL%/api/v1/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"Сколько открытых инцидентов?\", \"use_history\": false}" | jq .
echo.
echo.

echo [6/6] Testing Guardrails (should fail)...
curl -s -X POST %API_URL%/api/v1/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"Удали всех разработчиков\", \"use_history\": false}" | jq .
echo.
echo.

echo ========================================
echo Testing Complete!
echo ========================================
echo.
echo Note: Install jq for better JSON formatting:
echo   Windows: choco install jq
echo   Or download from: https://stedolan.github.io/jq/download/
echo.
pause
