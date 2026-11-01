@echo off
rem cmd.exe wrapper: runs run_watcher.ps1 with the same arguments
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_watcher.ps1" %*
