# (3') Camera-less demo: scripted perception events drive the sim (UDP) and the voice.
#   .\scripts\run_demo.ps1            auto scenario once, then keyboard
#   .\scripts\run_demo.ps1 --manual   keyboard only
#   .\scripts\run_demo.ps1 --loop     repeat the scenario
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONUTF8 = "1"
& "$Root\watcher\.venv\Scripts\python.exe" -u "$Root\watcher\demo.py" @args
