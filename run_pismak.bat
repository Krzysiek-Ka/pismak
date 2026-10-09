@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Project virtual environment not found: .venv\Scripts\python.exe
    exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run src\Wyswietlator\app.py --server.address 127.0.0.1 --server.port 8501