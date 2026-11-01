# (2) MuJoCo simulation. Listens for UDP/JSON commands on 127.0.0.1:5005.
# Extra arguments are passed to sim_runner.py, e.g.:
#   .\scripts\run_sim.ps1 --velstand     (walk/stand with velstand.onnx: can get up after a fall)
#   .\scripts\run_sim.ps1 --no-bam       (plain position actuators, lighter)
# Console output is also written to logs\sim.log (overwritten on each start).
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:MUJOCO_GL = "glfw"
$env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force (Join-Path $Root "logs") | Out-Null
& "$Root\sim\.venv\Scripts\python.exe" -u "$Root\sim\sim_runner.py" @args 2>&1 | Tee-Object -FilePath (Join-Path $Root "logs\sim.log")
