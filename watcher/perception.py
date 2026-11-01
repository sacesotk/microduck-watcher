"""YOLO11n (COCO 学習済み) による人・スマホ検出（仕様書 5 章）。追加学習なし。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

PERSON_CLS = 0
PHONE_CLS = 67   # COCO "cell phone"


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def cx(self):
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self):
        return 0.5 * (self.y1 + self.y2)

    @property
    def area(self):
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def contains(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class Detections:
    persons: list = field(default_factory=list)
    phones: list = field(default_factory=list)
    infer_ms: float = 0.0

    @property
    def main_person(self) -> Optional[Box]:
        return max(self.persons, key=lambda b: b.area) if self.persons else None


class Perception:
    def __init__(self, cfg: dict):
        from ultralytics import YOLO
        p = cfg["perception"]
        self.model = YOLO(p["yolo_model"])
        self.imgsz = int(p.get("imgsz", 480))
        self.device = p.get("device", "cpu")
        self.person_conf = float(p["person_conf"])
        self.phone_conf = float(p["phone_conf"])
        self.every_n = max(1, int(p.get("detect_every_n", 1)))
        self._i = 0
        self._last = Detections()

    def detect(self, frame_bgr: np.ndarray) -> Detections:
        self._i += 1
        if (self._i - 1) % self.every_n != 0:
            return self._last
        t0 = time.perf_counter()
        r = self.model.predict(frame_bgr, imgsz=self.imgsz, device=self.device,
                               conf=min(self.person_conf, self.phone_conf),
                               classes=[PERSON_CLS, PHONE_CLS], verbose=False)[0]
        det = Detections(infer_ms=(time.perf_counter() - t0) * 1000)
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                b = Box(float(x1), float(y1), float(x2), float(y2), float(c))
                if k == PERSON_CLS and c >= self.person_conf:
                    det.persons.append(b)
                elif k == PHONE_CLS and c >= self.phone_conf:
                    det.phones.append(b)
        self._last = det
        return det


def phone_below_face(det: Detections, face_bottom_y: Optional[float]) -> bool:
    """スマホの中心が顔の下端より下、かつ人の枠と重なる（中心が人枠内）。
    顔が取れていない時は人枠の上から 1/3 の位置を顔下端の代わりに使う。"""
    person = det.main_person
    if person is None or not det.phones:
        return False
    if face_bottom_y is None:
        face_bottom_y = person.y1 + (person.y2 - person.y1) / 3.0
    for ph in det.phones:
        if ph.cy > face_bottom_y and person.contains(ph.cx, ph.cy):
            return True
    return False
