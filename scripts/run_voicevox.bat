@echo off
rem cmd.exe wrapper: runs run_voicevox.ps1 with the same arguments
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_voicevox.ps1" %*
