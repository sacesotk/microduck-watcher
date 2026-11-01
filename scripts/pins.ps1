# Shared helpers for the setup scripts: read pins.json and fetch files only when the digest matches.
# Dot-source this file:  . "$PSScriptRoot\pins.ps1"
$ErrorActionPreference = "Stop"
$script:Root = Split-Path -Parent $PSScriptRoot
$script:Pins = Get-Content (Join-Path $script:Root "pins.json") -Raw -Encoding UTF8 | ConvertFrom-Json

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLowerInvariant()
}

function Get-VerifiedFile([string]$Url, [string]$Dest, [string]$Sha256) {
    # Keep an existing file only if its digest matches; otherwise download to a temp path,
    # verify, and move into place. Never leave an unverified file at $Dest.
    $Sha256 = $Sha256.ToLowerInvariant()
    if (Test-Path $Dest) {
        if ((Get-Sha256 $Dest) -eq $Sha256) { Write-Host "[pins] ok      $Dest"; return }
        Write-Host "[pins] digest mismatch, re-downloading: $Dest"
        Remove-Item $Dest -Force
    }
    $dir = Split-Path -Parent $Dest
    if ($dir) { New-Item -ItemType Directory -Force $dir | Out-Null }
    $tmp = "$Dest.part"
    if (Test-Path $tmp) { Remove-Item $tmp -Force }
    Write-Host "[pins] fetch   $Url"
    & curl.exe -fsSL --retry 3 -o $tmp $Url
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $tmp)) { throw "download failed: $Url" }
    $got = Get-Sha256 $tmp
    if ($got -ne $Sha256) {
        Remove-Item $tmp -Force
        throw "SHA-256 mismatch for $Url`n  expected $Sha256`n  got      $got"
    }
    Move-Item $tmp $Dest -Force
    Write-Host "[pins] verified $Dest"
}

function Get-PinnedGitCheckout([string]$Url, [string]$Commit, [string]$Dest) {
    # Shallow-fetch exactly one commit and verify HEAD. An existing checkout at a different
    # commit is an error (never silently move it).
    if (Test-Path (Join-Path $Dest ".git")) {
        $head = (& git -C $Dest rev-parse HEAD).Trim()
        if ($head -ne $Commit) { throw "$Dest is at $head but pins.json requires $Commit. Remove the folder or update pins.json after review." }
        Write-Host "[pins] ok      $Dest @ $Commit"
        return
    }
    New-Item -ItemType Directory -Force $Dest | Out-Null
    & git -C $Dest init -q
    & git -C $Dest remote add origin $Url
    & git -C $Dest fetch -q --depth 1 origin $Commit
    if ($LASTEXITCODE -ne 0) { throw "git fetch of $Commit from $Url failed" }
    & git -C $Dest checkout -q --detach FETCH_HEAD
    $head = (& git -C $Dest rev-parse HEAD).Trim()
    if ($head -ne $Commit) { throw "checkout verification failed: HEAD=$head expected=$Commit" }
    Write-Host "[pins] fetched $Dest @ $Commit"
}
