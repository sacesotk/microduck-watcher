@echo off
rem cmd.exe wrapper: runs run_demo.ps1 with the same arguments
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_demo.ps1" %*
