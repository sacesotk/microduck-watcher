"""起動時の固定リビジョン検証（標準ライブラリのみ。sim venv / watcher venv の両方から使う）。

pins.json に書かれたコミット SHA と SHA-256 を、実際に import / 読み込みする前に照合する。
不一致は PinMismatch を投げ、呼び出し側は起動を中止する（--allow-unpinned で警告に落とせる）。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINS_PATH = os.path.join(ROOT, "pins.json")


class PinMismatch(RuntimeError):
    pass


def load_pins(path: str = PINS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check_file(path: str, expected_sha256: str, label: str = "") -> None:
    if not os.path.isfile(path):
        raise PinMismatch(f"{label or path}: file missing")
    got = sha256_file(path)
    if got.lower() != expected_sha256.lower():
        raise PinMismatch(f"{label or path}: SHA-256 mismatch\n  expected {expected_sha256}\n  got      {got}")


def repo_head(repo_dir: str) -> str:
    try:
        out = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise PinMismatch(f"cannot read git HEAD of {repo_dir}: {e}") from e
    if out.returncode != 0:
        raise PinMismatch(f"cannot read git HEAD of {repo_dir}: {out.stderr.strip()}")
    return out.stdout.strip()


def check_repo(repo_dir: str, expected_commit: str, label: str = "microduck_rl") -> None:
    head = repo_head(repo_dir)
    if head != expected_commit:
        raise PinMismatch(f"{label}: checked-out commit {head} != pinned {expected_commit}")
    # 作業ツリーの改変も検出（公式コードは無変更で読み込む前提）
    st = subprocess.run(["git", "-C", repo_dir, "status", "--porcelain", "--untracked-files=no"],
                        capture_output=True, text=True, timeout=20)
    if st.returncode == 0 and st.stdout.strip():
        raise PinMismatch(f"{label}: working tree has local modifications:\n{st.stdout.strip()}")


def verify_sim(pins: dict | None = None, policy_files=("alpha_walking.onnx", "alpha_stand.onnx", "alpha_sitstand.onnx")) -> list[str]:
    """sim_runner 起動時: 公式リポジトリのコミットと、読み込む ONNX のダイジェストを照合する。"""
    pins = pins or load_pins()
    check_repo(os.path.join(ROOT, "microduck_rl"), pins["microduck_rl"]["commit"])
    checked = [f"microduck_rl @ {pins['microduck_rl']['commit'][:12]}"]
    for name in policy_files:
        exp = pins["policies"]["files"].get(name)
        if exp is None:
            raise PinMismatch(f"{name}: not listed in pins.json")
        check_file(os.path.join(ROOT, "policies", name), exp, f"policies/{name}")
        checked.append(name)
    return checked


def verify_watcher_models(pins: dict | None = None) -> list[str]:
    """watcher/main.py 起動時: YOLO と Face Landmarker のモデルファイルを照合する。"""
    pins = pins or load_pins()
    checked = []
    for name, info in pins["models"].items():
        check_file(os.path.join(ROOT, "assets", "models", name), info["sha256"], f"assets/models/{name}")
        checked.append(name)
    return checked


if __name__ == "__main__":
    import sys
    try:
        print("sim:", verify_sim())
        print("watcher models:", verify_watcher_models())
        print("OK")
    except PinMismatch as e:
        print("PIN MISMATCH:", e)
        sys.exit(1)
