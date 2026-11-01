# ③ 見守り（カメラ → 判定 → UDP/音声）。引数は main.py にそのまま渡る。
#    例: .\scripts\run_watcher.ps1 --no-sim --no-audio   （知覚だけ確認）
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
& "$Root\watcher\.venv\Scripts\python.exe" "$Root\watcher\main.py" @args
