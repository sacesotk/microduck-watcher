# watcher side setup (Windows native, Python 3.12, uv). Python packages come from the hash-pinned
# lock (watcher/requirements.lock, --require-hashes); model files are verified against pins.json.
. "$PSScriptRoot\pins.ps1"
Set-Location $Root

uv venv --python 3.12 watcher\.venv
uv pip install --python watcher\.venv --require-hashes -r watcher\requirements.lock

foreach ($m in $Pins.models.PSObject.Properties) {
    Get-VerifiedFile $m.Value.url (Join-Path $Root "assets\models\$($m.Name)") $m.Value.sha256
}
if (-not (Test-Path "assets\quack.wav")) {
    watcher\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'watcher'); from speech import make_quack_wav; print(make_quack_wav('assets/quack.wav'))"
}

watcher\.venv\Scripts\python.exe watcher\tests\test_udp_schema.py
watcher\.venv\Scripts\python.exe watcher\tests\test_behavior.py
