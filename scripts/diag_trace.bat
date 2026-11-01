@echo off
rem headless closed-loop trace -> logs\diag_trace.txt
set PYTHONUTF8=1
"%~dp0..\sim\.venv\Scripts\python.exe" -u "%~dp0..\sim\diag_trace.py"
pause
