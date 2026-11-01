"""ダック視点（head_camera）のレンダラ。

sim_runner が UDP :5006 で配信する qpos を受け取り、同じシーン（公式 scene.xml + 椅子に座る人）を
watcher 側で読み込んで head_camera から描画する。物理は回さない（mj_forward で姿勢だけ更新）。
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
SCENE_XML = os.path.join(ROOT, "microduck_rl", "src", "mjlab_microduck", "robot", "microduck", "scene.xml")


class StateReceiver(threading.Thread):
    """sim_runner が配信する姿勢 (qpos) を受け取る。
    ループバック以外の送信元・サイズ超過・不正な JSON・NaN/Infinity・長さ違い・非有限値は捨て、
    検証を通った最新の qpos だけを latest に保持する（sim/udp_schema.py）。"""

    def __init__(self, expected_len: int, host="127.0.0.1", port=5006):
        super().__init__(daemon=True)
        from udp_schema import MAX_DATAGRAM, Rejected, loads_strict, validate_state
        self._loads, self._validate, self._Rejected, self._max = loads_strict, validate_state, Rejected, MAX_DATAGRAM
        self.expected_len = int(expected_len)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.settimeout(0.5)
        self.latest = None           # {"qpos": [...]} 検証済みのみ
        self.last_t = 0.0
        self.n_rejected = 0
        self._closed = False
        self.start()

    def close(self):
        self._closed = True
        try:
            self.sock.close()
        except OSError:
            pass

    def run(self):
        while not self._closed:
            try:
                data, sender = self.sock.recvfrom(self._max + 1)
            except socket.timeout:
                continue
            except OSError:
                # Windows では上限超のデータグラムが OSError(WSAEMSGSIZE) になる。
                # ソケットが閉じられた場合以外は「拒否」として受信を続ける。
                if self._closed:
                    break
                self.n_rejected += 1
                continue
            try:
                msg = self._loads(data, sender)
                qpos = self._validate(msg, self.expected_len)
            except self._Rejected:
                self.n_rejected += 1
                continue
            except Exception:
                self.n_rejected += 1
                continue
            self.latest = {"qpos": qpos}
            self.last_t = time.time()

    @property
    def alive(self):
        return time.time() - self.last_t < 1.0


class DuckPOV:
    def __init__(self, width=640, height=480, person=True, camera="head_camera"):
        import mujoco
        import demo_scene
        self.mujoco = mujoco
        self.demo_scene = demo_scene
        spec = mujoco.MjSpec.from_file(SCENE_XML)
        if person:
            demo_scene.add_person(spec)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.model.vis.global_.fovy = 90          # 小型カメラらしい広角
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[1] = 1   # 人・椅子
        self.opt.geomgroup[2] = 1   # ロボットの見た目メッシュ
        self.scene_flags = None
        # 公式モデルの <camera name="head_camera"> は頭の内部を向いているため使わず、
        # head_camera サイトの位置から、くちばし (mouth_tip) 方向を前方とする自由カメラで描く。
        self.site_cam = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "head_camera")
        self.site_tip = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "mouth_tip")
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.pitch_offset_deg = 25.0   # 少し上向き（座っている人の顔が入るように）
        self.forward_offset = 0.03     # レンズ位置から 3 cm 前（自分の頭のメッシュに埋もれないように）
        # 前方軸: STAND 姿勢で (mouth_tip - head_camera) に最も近い head_camera フレームの軸を使う
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
        self.data.qpos[:] = self.model.key_qpos[key]
        mujoco.mj_forward(self.model, self.data)
        R = self.data.site_xmat[self.site_cam].reshape(3, 3)
        d = self.data.site_xpos[self.site_tip] - self.data.site_xpos[self.site_cam]
        d[2] = 0.0
        d /= (np.linalg.norm(d) + 1e-9)
        proj = R.T @ d                              # ローカル軸成分
        ax = int(np.argmax(np.abs(proj)))
        self.fwd_local = np.zeros(3)
        self.fwd_local[ax] = np.sign(proj[ax])

    def set_scene(self, person: bool, phone: bool, eyes_closed: bool):
        flags = (person, phone, eyes_closed)
        if flags == self.scene_flags:
            return
        self.scene_flags = flags
        self.demo_scene.set_person_visible(self.model, person)
        self.demo_scene.set_phone_visible(self.model, phone and person)
        self.demo_scene.set_eyes_closed(self.model, eyes_closed)

    def render(self, qpos) -> np.ndarray:
        """qpos（sim から受信、検証済み）を反映して描画。BGR 画像を返す。
        長さ違い・非有限値は ValueError（呼び出し側は前回の画像を使う）。"""
        q = np.asarray(qpos, dtype=np.float64)
        if q.ndim != 1 or q.shape[0] != self.model.nq or not np.all(np.isfinite(q)):
            raise ValueError(f"invalid qpos: shape={q.shape}, finite={bool(np.all(np.isfinite(q))) if q.size else False}")
        self.data.qpos[:] = q
        self.mujoco.mj_forward(self.model, self.data)
        R = self.data.site_xmat[self.site_cam].reshape(3, 3)
        f = R @ self.fwd_local
        f /= (np.linalg.norm(f) + 1e-9)
        az = np.degrees(np.arctan2(f[1], f[0]))
        el = np.degrees(np.arcsin(np.clip(f[2], -1, 1))) + self.pitch_offset_deg
        pos = self.data.site_xpos[self.site_cam] + f * self.forward_offset
        dist = 0.5
        fwd = np.array([np.cos(np.radians(el)) * np.cos(np.radians(az)),
                        np.cos(np.radians(el)) * np.sin(np.radians(az)),
                        np.sin(np.radians(el))])
        self.cam.lookat[:] = pos + fwd * dist
        self.cam.distance = dist
        self.cam.azimuth = az
        self.cam.elevation = el
        self.renderer.update_scene(self.data, camera=self.cam, scene_option=self.opt)
        rgb = self.renderer.render()
        return rgb[:, :, ::-1].copy()


if __name__ == "__main__":  # 単体確認: 立ち姿勢で 1 枚描画して保存
    import cv2
    pov = DuckPOV()
    key = pov.mujoco.mj_name2id(pov.model, pov.mujoco.mjtObj.mjOBJ_KEY, "STAND")
    qpos = pov.model.key_qpos[key].copy()
    pov.set_scene(True, True, False)
    img = pov.render(qpos)
    out = os.path.join(ROOT, "logs", "pov_test.png")
    cv2.imwrite(out, img)
    print("saved", out, img.shape)
