@echo off
rem cmd.exe wrapper: runs setup_watcher.ps1 with the same arguments
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_watcher.ps1" %*
