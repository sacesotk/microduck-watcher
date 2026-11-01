@echo off
rem cmd.exe wrapper: runs setup_voicevox.ps1 with the same arguments
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_voicevox.ps1" %*
