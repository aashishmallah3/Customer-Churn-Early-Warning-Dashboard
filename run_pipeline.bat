@echo off
echo ========================================================
echo Running Model Training & SHAP Attribution Pipeline
echo ========================================================
cd /d "%~dp0"
if exist ".venv312\Scripts\python.exe" (
    ".venv312\Scripts\python.exe" src\pipeline.py
) else (
    python src\pipeline.py
)
pause
