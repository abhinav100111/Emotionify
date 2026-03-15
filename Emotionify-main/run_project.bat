@echo off
setlocal

if not exist .venv\Scripts\python.exe (
    echo Missing virtual environment. Run setup_project.bat first.
    exit /b 1
)

call .\.venv\Scripts\python.exe -m streamlit run music.py
