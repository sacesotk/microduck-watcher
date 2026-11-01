"""UDP/JSON メッセージの厳密な検証（sim_runner の指令受信と watcher/pov.py の姿勢受信で共用）。

方針:
- ループバック以外の送信元は捨てる（bind は 127.0.0.1 だが、送信元も明示的に確認する）
- データグラムの上限サイズを設け、UTF-8 / JSON として正しく、トップレベルが辞書であることを要求する
- JSON の NaN / Infinity は拒否する（Python の json.loads は既定で受理してしまう）
- 数値はすべて有限の float に正規化し、範囲でクリップする。未知のキーは無視、型違いは拒否
- 姿勢 (qpos) は長さが期待値と一致し、すべて有限で絶対値が上限以下であることを要求する
依存は標準ライブラリのみ（sim venv / watcher venv の両方から import できる）。
"""
from __future__ import annotations

import json
import math
from typing import Any, Optional

MAX_DATAGRAM = 8192
MAX_PER_TICK = 32            # 1 ティックに読むデータグラム数の上限（それ以上は次ティック以降に回す）
RCVBUF_BYTES = 64 * 1024     # ソケット受信バッファ（滞留できる量の上限）
ALLOWED_SENDERS = ("127.0.0.1",)
TOKEN_FILE = "logs/.udp_token"   # sim_runner が起動時に生成する共有トークン（同一ユーザーのローカルプロセスだけが読める前提）
TOKEN_BYTES = 32
POLICIES = ("walking", "standing", "sitstand", "stop", "reset", "scene")
VX_MAX, VY_MAX, WZ_MAX = 0.4, 0.2, 1.5
HEAD_YAW_MAX = 1.4
DURATION_MAX = 30.0
QPOS_ABS_MAX = 100.0


class Rejected(ValueError):
    pass


# ----------------------------------------------------------------- 共有トークン
def new_token() -> str:
    import secrets
    return secrets.token_hex(TOKEN_BYTES)


def token_path(root: str) -> str:
    import os
    return os.path.join(root, *TOKEN_FILE.split("/"))


def write_token(root: str, token: str) -> str:
    """トークンファイルを所有者のみ読める権限で作る（POSIX は 0600。Windows は同一ユーザー前提）。"""
    import os
    path = token_path(root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as f:
        f.write(token)
    return path


def read_token(root: str) -> Optional[str]:
    import os
    path = token_path(root)
    try:
        with open(path, encoding="ascii") as f:
            t = f.read().strip()
    except OSError:
        return None
    return t if _looks_like_token(t) else None


def _looks_like_token(t: Any) -> bool:
    return isinstance(t, str) and len(t) == TOKEN_BYTES * 2 and all(c in "0123456789abcdef" for c in t)


def check_token(msg: dict, expected: Optional[str]) -> None:
    """expected が None（トークン無効）なら何もしない。そうでなければ定数時間比較で一致を要求する。"""
    if expected is None:
        return
    import hmac
    t = msg.get("token")
    if not _looks_like_token(t) or not hmac.compare_digest(t, expected):
        raise Rejected("bad or missing token")


def _reject_constant(name: str):
    raise Rejected(f"non-finite JSON constant: {name}")


def loads_strict(data: bytes, sender: Optional[tuple] = None) -> dict:
    """バイト列 → 辞書。送信元・サイズ・文字コード・JSON 構文・NaN/Infinity・型を検査する。"""
    if sender is not None and sender[0] not in ALLOWED_SENDERS:
        raise Rejected(f"sender not allowed: {sender[0]}")
    if not isinstance(data, (bytes, bytearray)) or len(data) == 0 or len(data) > MAX_DATAGRAM:
        raise Rejected(f"bad datagram size: {len(data) if data is not None else None}")
    try:
        text = bytes(data).decode("utf-8")
    except UnicodeDecodeError as e:
        raise Rejected(f"not utf-8: {e}") from e
    try:
        obj = json.loads(text, parse_constant=_reject_constant)
    except (json.JSONDecodeError, RecursionError) as e:
        raise Rejected(f"bad json: {e}") from e
    if not isinstance(obj, dict):
        raise Rejected(f"top-level is not an object: {type(obj).__name__}")
    return obj


def finite(v: Any, lo: float, hi: float, name: str) -> float:
    """bool を除く実数のみ受け、有限であることを確認して [lo, hi] にクリップする。"""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise Rejected(f"{name}: not a number ({type(v).__name__})")
    f = float(v)
    if not math.isfinite(f):
        raise Rejected(f"{name}: not finite")
    return max(lo, min(hi, f))


def _bool(v: Any, name: str) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    raise Rejected(f"{name}: not a bool")


def validate_command(msg: dict) -> dict:
    """sim への指令を正規化する。戻り値のキー:
    policy, twist[3], head_yaw (float|None), duration, sit (bool|None),
    scene: {person?, phone?, eyes_closed?, mouth_open?, side?}"""
    if not isinstance(msg, dict):
        raise Rejected("command is not an object")
    policy = msg.get("policy", "stop")
    if not isinstance(policy, str) or policy.lower() not in POLICIES:
        raise Rejected(f"unknown policy: {policy!r}")
    policy = policy.lower()
    out: dict = {"policy": policy, "twist": [0.0, 0.0, 0.0], "head_yaw": None, "duration": 2.0, "sit": None, "scene": {}}

    if "head_yaw" in msg and msg["head_yaw"] is not None:
        out["head_yaw"] = finite(msg["head_yaw"], -HEAD_YAW_MAX, HEAD_YAW_MAX, "head_yaw")
    if "duration" in msg:
        out["duration"] = finite(msg["duration"], 0.0, DURATION_MAX, "duration")
    if policy == "walking":
        tw = msg.get("twist", [0.0, 0.0, 0.0])
        if not isinstance(tw, (list, tuple)) or len(tw) != 3:
            raise Rejected("twist must be a list of 3 numbers")
        out["twist"] = [finite(tw[0], -VX_MAX, VX_MAX, "twist[0]"),
                        finite(tw[1], -VY_MAX, VY_MAX, "twist[1]"),
                        finite(tw[2], -WZ_MAX, WZ_MAX, "twist[2]")]
    if policy == "sitstand":
        out["sit"] = _bool(msg.get("sit", True), "sit")
    if policy == "scene":
        sc = {}
        for k in ("person", "phone", "eyes_closed", "mouth_open"):
            if k in msg:
                sc[k] = _bool(msg[k], k)
        if "side" in msg:
            if msg["side"] not in ("front", "back"):
                raise Rejected(f"side must be front/back: {msg['side']!r}")
            sc["side"] = msg["side"]
        out["scene"] = sc
    return out


def validate_state(msg: dict, expected_len: int) -> list[float]:
    """sim が配信する姿勢。qpos は長さ expected_len、全要素が有限で |v| <= QPOS_ABS_MAX。"""
    if not isinstance(msg, dict):
        raise Rejected("state is not an object")
    q = msg.get("qpos")
    if not isinstance(q, list) or len(q) != expected_len:
        raise Rejected(f"qpos must be a list of {expected_len} numbers")
    out = []
    for i, v in enumerate(q):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise Rejected(f"qpos[{i}] not a number")
        f = float(v)
        if not math.isfinite(f) or abs(f) > QPOS_ABS_MAX:
            raise Rejected(f"qpos[{i}] out of range")
        out.append(f)
    return out
