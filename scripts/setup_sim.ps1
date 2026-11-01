# sim side setup (Windows native, Python 3.12, uv). Everything external is pinned in pins.json:
#  - microduck_rl: exact commit (shallow fetch of that commit, HEAD verified)
#  - policies: Hugging Face revision + SHA-256 per file
#  - python packages: hash-pinned lock (sim/requirements.lock) installed with --require-hashes
#  - bam: git dependency pinned by commit SHA (VCS deps cannot be hash-pinned)
. "$PSScriptRoot\pins.ps1"
Set-Location $Root

Get-PinnedGitCheckout $Pins.microduck_rl.url $Pins.microduck_rl.commit (Join-Path $Root "microduck_rl")

uv venv --python 3.12 sim\.venv
uv pip install --python sim\.venv --require-hashes -r sim\requirements.lock
uv pip install --python sim\.venv --no-deps "better-actuator-models @ git+$($Pins.bam.url)@$($Pins.bam.commit)"

$rev = $Pins.policies.revision
foreach ($f in $Pins.policies.files.PSObject.Properties) {
    Get-VerifiedFile "https://huggingface.co/$($Pins.policies.repo)/resolve/$rev/$($f.Name)" (Join-Path $Root "policies\$($f.Name)") $f.Value
}

sim\.venv\Scripts\python.exe sim\check_env.py
