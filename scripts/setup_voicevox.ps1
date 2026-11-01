# VOICEVOX Engine 0.26.0 (Windows CPU 版) を tools\voicevox\ に展開する。
# .vvpp は zip 形式なので拡張子を変えて Expand-Archive する。約 1.9 GB → 展開後 約 2.5 GB。
# GPU 版 (directml / nvidia) にしたい場合は $Asset を変える（CPU 版で十分速い）。
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Dir = Join-Path $Root "tools\voicevox"
$Ver = "0.26.0"
$Asset = "voicevox_engine-windows-cpu-$Ver.vvpp"
New-Item -ItemType Directory -Force $Dir | Out-Null
Set-Location $Dir

if (-not (Test-Path $Asset)) {
    curl.exe -L --retry 3 -o $Asset "https://github.com/VOICEVOX/voicevox_engine/releases/download/$Ver/$Asset"
}
if (-not (Test-Path "engine")) {
    Copy-Item $Asset "$Asset.zip" -Force
    Expand-Archive -Path "$Asset.zip" -DestinationPath "engine" -Force
    Remove-Item "$Asset.zip"
}
$exe = Get-ChildItem -Path engine -Recurse -Filter run.exe | Select-Object -First 1
Write-Host "run.exe: $($exe.FullName)"
Write-Host "起動は scripts\run_voicevox.ps1"
