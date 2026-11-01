"""MediaPipe Face Landmarker による閉眼・あくび・うつむき検出と疲労スコア（仕様書 5 章）。

S = w_perclos * PERCLOS + w_yawn * min(1, yawn / yawn_norm) + w_droop * droop
  PERCLOS : 直近 window_s 秒で eyeBlinkLeft/Right 平均 > eye_closed_thresh のフレーム割合
  yawn    : 直近 window_s 秒のあくび回数（jawOpen > yawn_jaw_thresh が yawn_min_s 以上続いたら 1 回）
  droop   : 頭の下向きピッチ角 > droop_pitch_deg のフレーム割合（スマホ検出中は除外可）
医学的評価ではなく、デモ用の目安。
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FaceResult:
    found: bool = False
    bbox: Optional[tuple] = None        # (x1,y1,x2,y2) ピクセル
    eye_close: float = 0.0              # 左右 eyeBlink 平均
    jaw_open: float = 0.0
    pitch_deg: float = 0.0              # 下向きが正
    infer_ms: float = 0.0


@dataclass
class FatigueState:
    score: float = 0.0
    perclos: float = 0.0
    yawn_count: int = 0
    droop: float = 0.0
    yawning_now: bool = False


class FatigueScorer:
    """疲労スコアの計算だけ（顔推定器は持たない）。実カメラ版 Fatigue とシミュレーション版 (demo) で共用。"""

    def __init__(self, cfg: dict):
        f = cfg["fatigue"]
        self.window_s = float(f["window_s"])
        self.eye_thr = float(f["eye_closed_thresh"])
        self.jaw_thr = float(f["yawn_jaw_thresh"])
        self.yawn_min_s = float(f["yawn_min_s"])
        self.droop_deg = float(f["droop_pitch_deg"])
        self.droop_sign = float(f.get("droop_pitch_sign", 1))
        w = f["weights"]
        self.w = (float(w["perclos"]), float(w["yawn"]), float(w["droop"]))
        self.yawn_norm = float(f.get("yawn_norm", 2))
        self.exclude_droop_phone = bool(f.get("exclude_droop_when_phone", True))

        self.frames: deque = deque()        # (t, closed:bool, droop:Optional[bool])
        self.yawns: deque = deque()         # t
        self.jaw_open_since: Optional[float] = None
        self.yawn_counted = False
        self._t0 = time.time()
        self._last_ts_ms = -1

    # ---------------------------------------------------------- スコア
    def update(self, face: FaceResult, now: float, phone_detected: bool = False) -> FatigueState:
        return _update(self, face, now, phone_detected)


class Fatigue(FatigueScorer):
    """MediaPipe Face Landmarker で顔を推定し、FatigueScorer でスコアを出す（実カメラ用）。"""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        f = cfg["fatigue"]
        self.mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=f["face_model"]),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,                        # 1 人だけ（最も顕著な顔）
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.lm = vision.FaceLandmarker.create_from_options(opts)

    # ------------------------------------------------------------ 顔推定
    def analyze(self, frame_bgr: np.ndarray, now: float) -> FaceResult:
        import cv2
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        ts = int((now - self._t0) * 1000)
        if ts <= self._last_ts_ms:             # VIDEO モードは単調増加が必須
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        res = self.lm.detect_for_video(img, ts)
        out = FaceResult(infer_ms=(time.perf_counter() - t0) * 1000)
        if not res.face_landmarks:
            return out
        h, w = frame_bgr.shape[:2]
        pts = res.face_landmarks[0]
        xs = [p.x * w for p in pts]
        ys = [p.y * h for p in pts]
        out.bbox = (min(xs), min(ys), max(xs), max(ys))
        out.found = True
        if res.face_blendshapes:
            bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
            out.eye_close = 0.5 * (bs.get("eyeBlinkLeft", 0.0) + bs.get("eyeBlinkRight", 0.0))
            out.jaw_open = bs.get("jawOpen", 0.0)
        if res.facial_transformation_matrixes:
            M = np.asarray(res.facial_transformation_matrixes[0])
            # 顔の前方ベクトル (+z) のカメラ座標 y 成分。頭が下がると y<0 になる。
            fwd_y = float(np.clip(M[1, 2], -1.0, 1.0))
            out.pitch_deg = self.droop_sign * (-math.degrees(math.asin(fwd_y)))
        return out



def _update(self, face: FaceResult, now: float, phone_detected: bool = False) -> FatigueState:
    if True:
        if face.found:
            closed = face.eye_close > self.eye_thr
            droop: Optional[bool] = None
            if not (self.exclude_droop_phone and phone_detected):
                droop = face.pitch_deg > self.droop_deg
            self.frames.append((now, closed, droop))
            # あくび: jawOpen が閾値超で yawn_min_s 以上継続 → 1 回
            if face.jaw_open > self.jaw_thr:
                if self.jaw_open_since is None:
                    self.jaw_open_since, self.yawn_counted = now, False
                elif not self.yawn_counted and now - self.jaw_open_since >= self.yawn_min_s:
                    self.yawns.append(now)
                    self.yawn_counted = True
            else:
                self.jaw_open_since, self.yawn_counted = None, False
        # 窓の外を捨てる
        cut = now - self.window_s
        while self.frames and self.frames[0][0] < cut:
            self.frames.popleft()
        while self.yawns and self.yawns[0] < cut:
            self.yawns.popleft()

        st = FatigueState(yawn_count=len(self.yawns), yawning_now=self.jaw_open_since is not None)
        if self.frames:
            st.perclos = sum(1 for _, c, _ in self.frames if c) / len(self.frames)
            dr = [d for _, _, d in self.frames if d is not None]
            st.droop = (sum(1 for d in dr if d) / len(dr)) if dr else 0.0
        wp, wy, wd = self.w
        st.score = float(min(1.0, wp * st.perclos
                             + wy * min(1.0, st.yawn_count / self.yawn_norm)
                             + wd * st.droop))
        return st
