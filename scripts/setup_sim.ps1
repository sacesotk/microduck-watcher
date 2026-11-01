# sim 側の環境構築（Windows ネイティブ, Python 3.12, uv）
# 公式 microduck_rl は「変更しない」。uv sync（CUDA 約 2 GB）は使わず、推論に必要な最小構成だけ入れる。
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path "$Root\microduck_rl\.git")) {
    git clone --depth 1 --branch develop https://github.com/pollen-robotics/microduck_rl microduck_rl
}

uv venv --python 3.12 sim\.venv
uv pip install --python sim\.venv -r sim\requirements.txt

# 学習済みポリシー（Hugging Face pollen-robotics/microduck-policies, Apache-2.0）
New-Item -ItemType Directory -Force policies | Out-Null
foreach ($f in @("alpha_walking.onnx", "alpha_stand.onnx", "alpha_sitstand.onnx", "velstand.onnx", "manifest.json")) {
    if (-not (Test-Path "policies\$f")) {
        curl.exe -sL -o "policies\$f" "https://huggingface.co/pollen-robotics/microduck-policies/resolve/main/$f"
    }
}

sim\.venv\Scripts\python.exe sim\check_env.py
