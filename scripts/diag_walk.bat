@echo off
rem headless walking diagnostic -> logs\diag_walk.txt
set PYTHONUTF8=1
"%~dp0..\sim\.venv\Scripts\python.exe" -u "%~dp0..\sim\diag_walk.py"
pause
