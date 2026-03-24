@echo off
REM ================================================================
REM  Bank AI Backend — Windows Quick Setup
REM ================================================================
 
echo.
echo === Bank AI Backend Setup ===
echo.
 
REM 1. Check PostgreSQL
psql --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PostgreSQL not found!
    echo Download from: https://www.postgresql.org/download/windows/
    pause & exit /b 1
)
echo [OK] PostgreSQL found
 
REM 2. Create test database
echo.
echo Creating test database...
psql -U postgres -c "CREATE DATABASE bank_ai_test;" 2>nul
echo [OK] Database ready
 
REM 3. Run schema
echo Running schema migrations...
psql -U postgres -d bank_ai_test -f migrations\001_schema.sql
echo [OK] Schema created
 
REM 4. Python venv
echo.
echo Setting up Python environment...
python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt
 
REM 5. Generate encryption key
echo.
echo Generating encryption key...
python -c "import secrets; key=secrets.token_hex(32); print(f'ENCRYPTION_KEY={key}')" >> .env
echo [OK] Key added to .env
 
echo.
echo =====================================================
echo  Setup complete!
echo  Run: uvicorn main:app --host 0.0.0.0 --port 4000 --reload
echo  Then: POST http://localhost:4000/seed  (to insert test data)
echo  Docs: http://localhost:4000/docs
echo =====================================================
pause