# VOICEVOX Engine (Windows CPU build). Version, URL and SHA-256 are pinned in pins.json.
# The .vvpp is a zip archive; it is extracted to tools\voicevox\engine only after the digest matches.
. "$PSScriptRoot\pins.ps1"
$Dir = Join-Path $Root "tools\voicevox"
New-Item -ItemType Directory -Force $Dir | Out-Null
$asset = Join-Path $Dir $Pins.voicevox.asset

Get-VerifiedFile $Pins.voicevox.url $asset $Pins.voicevox.sha256

if (-not (Test-Path (Join-Path $Dir "engine"))) {
    Copy-Item $asset "$asset.zip" -Force
    Expand-Archive -Path "$asset.zip" -DestinationPath (Join-Path $Dir "engine") -Force
    Remove-Item "$asset.zip"
}
$exe = Get-ChildItem -Path (Join-Path $Dir "engine") -Recurse -Filter run.exe | Select-Object -First 1
Write-Host "run.exe: $($exe.FullName)"
Write-Host "start with scripts\run_voicevox.ps1"
