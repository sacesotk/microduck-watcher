# One-shot launcher: VOICEVOX Engine -> MuJoCo sim -> demo (this window).
#   scripts\run_all.bat               front-facing person
#   scripts\run_all.bat --behind      person behind the duck (search, find, greet)
#   scripts\run_all.bat --loop        repeat the scenario
#   scripts\run_all.bat --camera      real webcam watcher (main.py) instead of the demo
# VOICEVOX and the sim are reused if they are already running, and are left running on exit.
param([Parameter(ValueFromRemainingArguments = $true)] $Rest)
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONUTF8 = "1"
$useCamera = $false
$demoArgs = @()
foreach ($a in $Rest) { if ($a -eq "--camera") { $useCamera = $true } else { $demoArgs += $a } }

function Test-Voicevox {
    try { $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:50021/version" -TimeoutSec 2; return $r.StatusCode -eq 200 } catch { return $false }
}
function Test-Sim {
    return [bool](Get-NetUDPEndpoint -LocalPort 5005 -ErrorAction SilentlyContinue)
}

# 1) VOICEVOX
if (Test-Voicevox) {
    Write-Host "[run_all] VOICEVOX already running"
} else {
    $exe = Get-ChildItem -Path (Join-Path $Root "tools\voicevox\engine") -Recurse -Filter run.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $exe) {
        Write-Host "[run_all] VOICEVOX not installed (scripts\setup_voicevox.bat). Falling back to pyttsx3."
    } else {
        Write-Host "[run_all] starting VOICEVOX ..."
        Start-Process -FilePath $exe.FullName -ArgumentList "--host 127.0.0.1 --port 50021" -WorkingDirectory $exe.DirectoryName -WindowStyle Minimized
        $t0 = Get-Date
        while (-not (Test-Voicevox) -and ((Get-Date) - $t0).TotalSeconds -lt 120) { Start-Sleep -Seconds 2; Write-Host -NoNewline "." }
        Write-Host ""
        if (Test-Voicevox) { Write-Host "[run_all] VOICEVOX ready" } else { Write-Host "[run_all] VOICEVOX did not answer in 120 s; continuing (pyttsx3 fallback)" }
    }
}

# 2) MuJoCo sim
if (Test-Sim) {
    Write-Host "[run_all] sim already running (UDP 5005)"
} else {
    Write-Host "[run_all] starting sim ..."
    Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "$Root\scripts\run_sim.ps1"
    $t0 = Get-Date
    while (-not (Test-Sim) -and ((Get-Date) - $t0).TotalSeconds -lt 90) { Start-Sleep -Seconds 1; Write-Host -NoNewline "." }
    Write-Host ""
    if (Test-Sim) { Write-Host "[run_all] sim ready" } else { Write-Host "[run_all] sim did not start in 90 s; check the sim window" }
    Start-Sleep -Seconds 2
}

# 3) demo (or real camera watcher) in this window
if ($useCamera) {
    Write-Host "[run_all] starting watcher (real camera) ..."
    & "$Root\watcher\.venv\Scripts\python.exe" -u "$Root\watcher\main.py" @demoArgs
} else {
    Write-Host "[run_all] starting demo $demoArgs ..."
    & "$Root\watcher\.venv\Scripts\python.exe" -u "$Root\watcher\demo.py" @demoArgs
}
Write-Host "[run_all] demo finished. VOICEVOX and sim are left running (close their windows to stop)."
