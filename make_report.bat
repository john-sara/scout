@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe evals\report.py && start "" results\report.html
pause
