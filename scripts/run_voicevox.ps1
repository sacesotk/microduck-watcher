# ① VOICEVOX Engine を起動（HTTP API :50021）。別ターミナルで動かしたままにする。
$Root = Split-Path -Parent $PSScriptRoot
$exe = Get-ChildItem -Path (Join-Path $Root "tools\voicevox\engine") -Recurse -Filter run.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $exe) { Write-Host "VOICEVOX が未展開です。scripts\setup_voicevox.ps1 を先に実行してください"; exit 1 }
Set-Location $exe.DirectoryName
& $exe.FullName --host 127.0.0.1 --port 50021
