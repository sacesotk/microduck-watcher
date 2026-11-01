"""ダック視点（シミュレーション映像）用の知覚。

YOLO11n / MediaPipe は単純形状で作った人形を認識できないため（2026-11-01 確認）、
色に基づく画像処理で perception.py / fatigue.py と同じ出力型（Detections, FaceResult）を作る。
判定ロジック（在席 1 秒、スマホが顔より下で 5 秒、PERCLOS・あくびによる疲労スコア）は実カメラ版と共通。

検出対象の色は sim/demo_scene.py と対応:
  シャツ = 暖色(赤橙)・高彩度 / 肌 = 低彩度の橙 / ズボン = 無彩色グレー / 目 = 黒 / 口 = 暗い赤 / スマホ = 黄色
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from fatigue import FaceResult
from perception import Box, Detections


class SimPerception:
    def __init__(self, min_shirt_px=150, min_pants_px=1500, eye_open_ratio=0.012, eye_closed_ratio=0.003,
                 mouth_closed_ratio=0.022, mouth_open_ratio=0.050, phone_min_ratio=0.12):
        self.min_shirt_px = min_shirt_px
        self.min_pants_px = min_pants_px
        self.eye_open_ratio = eye_open_ratio
        self.eye_closed_ratio = eye_closed_ratio
        self.mouth_closed_ratio = mouth_closed_ratio
        self.mouth_open_ratio = mouth_open_ratio
        self.phone_min_ratio = phone_min_ratio
        # 近さの校正（2026-11-01 計測, PERSON_X=1.2, POV 640x480 fovy90 pitch25）:
        #   顔枠中心 y/H: 腰まで 1.2 m → 0.32, 1.05 m → 0.28, 0.95 m → 0.24, 0.85 m → 0.20, 0.7 m → 0.11
        #   far_cy 以上なら遠い（近づく）、near_cy 以下なら十分近い（止まる）。足先はさらに 0.55 m 手前にある。
        self.far_cy = 0.28
        self.near_cy = 0.245
        self.last_debug = {}

    def closeness(self, face, img_h: int):
        """0 以下 = 遠い, 1 以上 = 十分近い, None = 顔が取れない。"""
        if not (face.found and face.bbox):
            return None
        cy = (face.bbox[1] + face.bbox[3]) / 2.0 / img_h
        return float((self.far_cy - cy) / (self.far_cy - self.near_cy))

    @staticmethod
    def _bbox(mask):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    @staticmethod
    def _largest_component(mask):
        n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        if n <= 1:
            return None, 0
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (lab == i), int(stats[i, cv2.CC_STAT_AREA])

    @staticmethod
    def _topmost_component(mask, min_area=40):
        """十分な面積がある成分のうち、最も上（重心 y が最小）のものを返す（頭 = 一番上の肌色）。"""
        n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        best, best_y = None, 1e9
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area and cent[i][1] < best_y:
                best, best_y = i, cent[i][1]
        if best is None:
            return None, 0
        return (lab == best), int(stats[best, cv2.CC_STAT_AREA])

    def analyze(self, img_bgr: np.ndarray):
        t0 = time.perf_counter()
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0].astype(int), hsv[:, :, 1] / 255.0, hsv[:, :, 2] / 255.0
        shirt = (H <= 10) & (S > 0.55) & (V > 0.25)
        skin = (H >= 4) & (H <= 25) & (S > 0.10) & (S < 0.50) & (V > 0.35)
        gray = (S < 0.15) & (V > 0.12) & (V < 0.55)
        dark = V < 0.16
        yellow = (H >= 20) & (H <= 38) & (S > 0.55) & (V > 0.45)   # スマホ（黄色のケース）
        mouthc = ((H <= 6) | (H >= 170)) & (S > 0.5) & (V > 0.08) & (V < 0.62)

        det = Detections()
        face = FaceResult()
        w = img_bgr.shape[1]
        has_shirt = shirt.sum() >= self.min_shirt_px
        pants_blob, pants_area = self._largest_component(gray) if gray.sum() >= self.min_pants_px else (None, 0)
        dbg = {"shirt_px": int(shirt.sum()), "pants_px": int(pants_area)}
        if has_shirt or pants_blob is not None:
            # 列範囲の基準: シャツがあればシャツ、無ければズボン（接近時はシャツが膝に隠れる）
            if has_shirt:
                anchor, _ = self._largest_component(shirt)
                sx1, sy1, sx2, sy2 = self._bbox(anchor)
                margin = (sx2 - sx1) * 0.6
            else:
                sx1, sy1, sx2, sy2 = self._bbox(pants_blob)
                margin = (sx2 - sx1) * 0.15
            col = np.zeros_like(shirt)
            col[:, int(max(0, sx1 - margin)):int(min(w, sx2 + margin))] = True
            body = (shirt | skin | gray) & col
            px1, py1, px2, py2 = self._bbox(body)
            det.persons.append(Box(px1, py1, px2, py2, 0.9 if has_shirt else 0.7))

            # 顔: シャツ上端より上にある肌色のうち最も上の成分（シャツが見えない近距離では顔は探さない）
            fb = None
            if has_shirt:
                # 目・口で肌色が分断されるので、閉処理で穴を埋めてから頭全体を 1 成分として取る
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                skin_closed = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
                above = skin_closed.copy()
                above[int(sy1) + 2:, :] = False
                fb, farea = self._topmost_component(above & col, min_area=40)
            if fb is not None:
                fx1, fy1, fx2, fy2 = self._bbox(fb)
                face.found = True
                face.bbox = (fx1, fy1, fx2, fy2)
                fw, fh = max(1.0, fx2 - fx1), max(1.0, fy2 - fy1)
                farea_box = fw * fh
                # 目: 顔枠内の黒（下から見上げる視点なので上下で区切らない。髪のある最上部だけ除く）
                eye_region = np.zeros_like(dark)
                eye_region[int(fy1 + fh * 0.12):int(fy2), int(fx1):int(fx2)] = True
                eye_ratio = float((dark & eye_region).sum()) / farea_box
                face.eye_close = float(np.clip((self.eye_open_ratio - eye_ratio) /
                                               (self.eye_open_ratio - self.eye_closed_ratio), 0.0, 1.0))
                # 口: 顔枠内の暗い赤（見上げる視点では口が枠の上寄りに来るので枠全体で探す）
                mouth_region = np.zeros_like(mouthc)
                mouth_region[int(fy1):int(fy2), int(fx1):int(fx2)] = True
                mouth_ratio = float((mouthc & mouth_region).sum()) / farea_box
                face.jaw_open = float(np.clip((mouth_ratio - self.mouth_closed_ratio) /
                                              (self.mouth_open_ratio - self.mouth_closed_ratio), 0.0, 1.0))
                dbg.update(face_area=int(farea_box), eye_ratio=round(eye_ratio, 4), mouth_ratio=round(mouth_ratio, 4))

                # スマホ: 人の列範囲にある黄色のまとまり（顔の中心より下にあるかは呼び出し側 phone_below_face が判定）
                pb, parea = self._largest_component(yellow & col)
                dbg["phone_ratio"] = round(parea / farea_box, 3)
                if pb is not None and parea / farea_box >= self.phone_min_ratio:
                    bx1, by1, bx2, by2 = self._bbox(pb)
                    det.phones.append(Box(bx1, by1, bx2, by2, 0.7))
        det.infer_ms = (time.perf_counter() - t0) * 1000
        face.infer_ms = det.infer_ms
        self.last_debug = dbg
        return det, face


if __name__ == "__main__":  # 校正用: 各状態・距離で生の比率と検出結果を表示
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pov import DuckPOV
    pov = DuckPOV()
    sp = SimPerception()
    key = pov.mujoco.mj_name2id(pov.model, pov.mujoco.mjtObj.mjOBJ_KEY, "STAND")
    q0 = pov.model.key_qpos[key].copy()
    for dx in (0.0, 0.35):
        q = q0.copy()
        q[0] = dx
        for person, phone, eyes, mouth in [(True, False, False, False), (True, True, False, False),
                                           (True, False, True, False), (True, False, True, True),
                                           (False, False, False, False)]:
            pov.set_scene(person, phone, eyes)
            pov.demo_scene.set_mouth_open(pov.model, mouth)
            img = pov.render(q)
            det, face = sp.analyze(img)
            print(f"dx={dx} person={int(person)} phone={int(phone)} eyes_closed={int(eyes)} mouth_open={int(mouth)} -> "
                  f"persons={len(det.persons)} phones={len(det.phones)} face={int(face.found)} "
                  f"eye_close={face.eye_close:.2f} jaw_open={face.jaw_open:.2f} dbg={sp.last_debug} ({det.infer_ms:.1f}ms)")
            cv2.imwrite(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs",
                                     f"simperc_{dx}_{int(person)}{int(phone)}{int(eyes)}{int(mouth)}.png"), img)
