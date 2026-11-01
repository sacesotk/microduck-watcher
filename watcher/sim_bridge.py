"""シミュレーション (sim/sim_runner.py) への UDP/JSON 送信（仕様書 6 章）。"""
from __future__ import annotations

import json
import socket
import time


class SimBridge:
    def __init__(self, host: str = "127.0.0.1", port: int = 5005, verbose: bool = True):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.verbose = verbose
        self.last_cmd: dict | None = None
        self.last_t = 0.0

    def send(self, msg: dict):
        try:
            self.sock.sendto(json.dumps(msg).encode("utf-8"), self.addr)
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
