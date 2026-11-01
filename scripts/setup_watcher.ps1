# 見守り側の環境構築（Windows ネイティブ, Python 3.12, uv）
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

uv venv --python 3.12 watcher\.venv
uv pip install --python watcher\.venv -r watcher\requirements.txt

# 学習済みモデル（追加学習なし）
New-Item -ItemType Directory -Force assets\models | Out-Null
if (-not (Test-Path "assets\models\face_landmarker.task")) {
    curl.exe -sL -o assets\models\face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
}
if (-not (Test-Path "assets\models\yolo11n.pt")) {
    curl.exe -sL -o assets\models\yolo11n.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
}
# 鳴き声（素材が無ければ合成音を生成）
if (-not (Test-Path "assets\quack.wav")) {
    watcher\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'watcher'); from speech import make_quack_wav; print(make_quack_wav('assets/quack.wav'))"
}

watcher\.venv\Scripts\python.exe watcher\tests\test_behavior.py
