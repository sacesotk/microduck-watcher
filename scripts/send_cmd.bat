@echo off
rem UDP test sender for the sim: send_cmd.bat demo | walk 0.2 2 | sit | stand | head 0.5 | stop
"%~dp0..\sim\.venv\Scripts\python.exe" "%~dp0..\sim\send_cmd.py" %*
