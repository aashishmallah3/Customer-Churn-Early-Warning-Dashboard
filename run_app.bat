@echo off
echo ========================================================
echo Launching Customer Churn Early-Warning Dashboard
echo ========================================================
cd /d "%~dp0"
if exist ".venv312\Scripts\streamlit.exe" (
    ".venv312\Scripts\streamlit.exe" run app.py
) else (
    streamlit run app.py
)
pause
