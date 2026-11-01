@echo off
rem cmd.exe wrapper: runs setup_sim.ps1 with the same arguments
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_sim.ps1" %*
