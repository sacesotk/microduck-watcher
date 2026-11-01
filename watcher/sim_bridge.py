"""シミュレーション (sim/sim_runner.py) への UDP/JSON 送信（仕様書 6 章）。"""
from __future__ import annotations

import json
import os
import socket
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
import udp_schema  # noqa: E402


class SimBridge:
    """sim_runner が起動時に書く共有トークン (logs/.udp_token) を毎回の送信前に確認して付与する。
    sim を再起動するとトークンが変わるため、ファイルの更新時刻が変わったら読み直す。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 5005, verbose: bool = True, token_root: str = ROOT):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.verbose = verbose
        self.last_cmd: dict | None = None
        self.last_t = 0.0
        self.token_root = token_root
        self._token = None
        self._token_mtime = None
        self._warned = False

    def _refresh_token(self):
        path = udp_schema.token_path(self.token_root)
        try:
            m = os.stat(path).st_mtime
        except OSError:
            self._token, self._token_mtime = None, None
            return
        if m != self._token_mtime:
            self._token = udp_schema.read_token(self.token_root)
            self._token_mtime = m

    def send(self, msg: dict):
        self._refresh_token()
        out = dict(msg)
        if self._token:
            out["token"] = self._token
        elif not self._warned:
            print("[sim_bridge] no session token found (sim not started?) — commands will be rejected by the sim")
            self._warned = True
        try:
            self.sock.sendto(json.dumps(out).encode("utf-8"), self.addr)
        except OSError as e:
            print(f"[sim_bridge] send failed: {e}")
        self.last_cmd, self.last_t = msg, time.time()
        if self.verbose:
            print(f"[sim_bridge] {msg}")

    # 便利関数 ---------------------------------------------------------
    def walk(self, vx: float, duration: float, head_yaw: float | None = None, wz: float = 0.0):
        m = {"policy": "walking", "twist": [vx, 0.0, wz], "duration": duration}
        if head_yaw is not None:
            m["head_yaw"] = head_yaw
        self.send(m)

    def stop(self, head_yaw: float | None = None):
        m = {"policy": "stop"}
        if head_yaw is not None:
            m["head_yaw"] = head_yaw
        self.send(m)

    def look(self, head_yaw: float):
        self.send({"policy": "standing", "head_yaw": head_yaw})

    def sit(self, head_yaw: float | None = None):
        m = {"policy": "sitstand", "sit": True}
        if head_yaw is not None:
            m["head_yaw"] = head_yaw
        self.send(m)

    def stand(self, head_yaw: float | None = None):
        m = {"policy": "sitstand", "sit": False}
        if head_yaw is not None:
            m["head_yaw"] = head_yaw
        self.send(m)
