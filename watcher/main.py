"""見守り側メインループ: カメラ取得 → 知覚 (YOLO) → 疲労 (MediaPipe) → 状態遷移 → 表示・ログ。

起動:  watcher\\.venv\\Scripts\\python.exe watcher\\main.py [--config config.yaml] [--no-sim] [--no-audio]
終了:  画面で q / ESC、または Ctrl+C
プライバシー: 画像は保存しない。logs/ には時刻・検出有無・スコア・状態の CSV だけを書く。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(ROOT, "sim"))
import pins_check  # noqa: E402

from behavior import Actions, Behavior, Observation  # noqa: E402
from fatigue import Fatigue  # noqa: E402
from perception import Perception, phone_below_face  # noqa: E402
from sim_bridge import SimBridge  # noqa: E402
from speech import Speech  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 相対パスはプロジェクトルート基準にする
    for sec, key in [("perception", "yolo_model"), ("fatigue", "face_model"),
                     ("speech", "quack_wav"), ("logging", "dir")]:
        p = cfg[sec][key]
        if not os.path.isabs(p):
            cfg[sec][key] = os.path.join(ROOT, p)
    return cfg


class Overlay:
    """判定結果と字幕の描画。日本語は PIL (meiryo) で描く。"""

    def __init__(self, cfg: dict):
        d = cfg["display"]
        self.subtitle_s = float(d.get("subtitle_s", 4.0))
        self.font = None
        try:
            from PIL import ImageFont
            self.font = ImageFont.truetype(d.get("font_path", "C:/Windows/Fonts/meiryo.ttc"),
                                           int(d.get("font_size", 28)))
            self.font_small = ImageFont.truetype(d.get("font_path", "C:/Windows/Fonts/meiryo.ttc"), 16)
        except Exception as e:
            print(f"[overlay] 日本語フォントが使えません ({e})。字幕は英数字のみ")

    def draw_text_ja(self, img, text, xy, fill=(255, 255, 255), small=False):
        if self.font is None:
            cv2.putText(img, text.encode("ascii", "ignore").decode(), xy,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, fill, 2)
            return img
        from PIL import Image, ImageDraw
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(pil).text(xy, text, font=self.font_small if small else self.font, fill=fill[::-1])
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

    def render(self, frame, det, face, fat, st, fps, now, credit, speech_backend, mirror):
        img = frame.copy()
        # 枠
        for b in det.persons:
            cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 200, 0), 2)
            cv2.putText(img, f"person {b.conf:.2f}", (int(b.x1), int(b.y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        for b in det.phones:
            cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 140, 255), 2)
            cv2.putText(img, f"phone {b.conf:.2f}", (int(b.x1), int(b.y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
        if face.found and face.bbox:
            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 1)
            cv2.line(img, (x1, y2), (x2, y2), (255, 200, 0), 2)   # 顔の下端
        if mirror:
            img = cv2.flip(img, 1)

        # ステータス（左上）
        lines = [
            f"FPS {fps:4.1f}  yolo {det.infer_ms:4.0f}ms  face {face.infer_ms:4.0f}ms",
            f"person {'YES' if det.persons else 'no '}  phone_below_face {'YES' if st.get('phone_now') else 'no '}  face {'YES' if face.found else 'no '}",
            f"S={fat.score:.2f}  PERCLOS={fat.perclos:.2f}  yawn={fat.yawn_count}{'*' if fat.yawning_now else ''}  droop={fat.droop:.2f}",
            f"eye={face.eye_close:.2f} jaw={face.jaw_open:.2f} pitch={face.pitch_deg:+5.1f}deg",
            f"state {st['state']}  present {int(st['present'])}  greeted {int(st['greeted'])}  seated {int(st['seated'])}",
            f"hold phone {st['phone_hold']:4.1f}s fatigue {st['fatigue_hold']:4.1f}s | cd phone {st['phone_cd']:3.0f}s fatigue {st['fatigue_cd']:3.0f}s",
            f"speech {speech_backend}",
        ]
        pad = img.copy()
        cv2.rectangle(pad, (0, 0), (img.shape[1], 20 * len(lines) + 10), (0, 0, 0), -1)
        img = cv2.addWeighted(pad, 0.55, img, 0.45, 0)
        for i, s in enumerate(lines):
            color = (0, 255, 255) if (i == 4 and st["state"] != "IDLE") else (255, 255, 255)
            cv2.putText(img, s, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # 字幕（下）
        if st["subtitle"] and now - st["subtitle_t"] <= self.subtitle_s:
            h, w = img.shape[:2]
            cv2.rectangle(img, (0, h - 60), (w, h), (0, 0, 0), -1)
            img = self.draw_text_ja(img, st["subtitle"], (14, h - 50))
        if credit:
            img = self.draw_text_ja(img, credit, (img.shape[1] - 170, img.shape[0] - 22), (200, 200, 200), small=True)
        return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--no-sim", action="store_true", help="UDP 送信を標準出力に出すだけにする")
    ap.add_argument("--no-audio", action="store_true", help="音声を鳴らさない")
    ap.add_argument("--camera", type=int, default=None, help="カメラ番号を上書き")
    ap.add_argument("--no-window", action="store_true", help="画面を出さない（fps 計測用）")
    ap.add_argument("--duration", type=float, default=None, help="指定秒数で自動終了（計測用）")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.no_window:
        cfg["display"]["show_window"] = False

    cam = cfg["camera"]
    idx = args.camera if args.camera is not None else int(cam["index"])
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)       # Windows は DSHOW が起動が速く安定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam["width"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam["height"]))
    if not cap.isOpened():
        print(f"カメラ {idx} を開けません。config.yaml の camera.index を変えてください")
        sys.exit(1)

    try:  # 読み込む前にモデルファイルのダイジェストを pins.json と照合
        print("[main] model pins verified:", ", ".join(pins_check.verify_watcher_models()))
    except pins_check.PinMismatch as e:
        print(f"[main] REFUSING TO START: {e}  (scripts\\setup_watcher.bat で取り直してください)")
        sys.exit(2)
    perception = Perception(cfg)
    fatigue = Fatigue(cfg)
    bridge = SimBridge(cfg["robot"]["udp_host"], int(cfg["robot"]["udp_port"]), verbose=True)
    speech = None if args.no_audio else Speech(cfg)

    def do_send(m):
        if args.no_sim:
            print("[sim(off)]", m)
        else:
            bridge.send(m)

    def do_speak(key, text):
        print(f"[speak:{key}] {text}")
        if speech:
            speech.say(text)

    def do_quack():
        print("[quack]")
        if speech:
            speech.quack()

    behavior = Behavior(cfg, Actions(send=do_send, speak=do_speak, quack=do_quack))
    overlay = Overlay(cfg)

    # ログ（CSV のみ）
    os.makedirs(cfg["logging"]["dir"], exist_ok=True)
    log_path = os.path.join(cfg["logging"]["dir"], datetime.now().strftime("watch_%Y%m%d_%H%M%S.csv"))
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    log.writerow(["time", "fps", "person", "phone", "phone_below_face", "face",
                  "perclos", "yawn", "droop", "score", "state", "event"])
    log_interval = float(cfg["logging"]["interval_s"])
    last_log = 0.0
    print(f"[main] logging to {log_path}")

    show = bool(cfg["display"].get("show_window", True))
    win = cfg["display"].get("window_name", "Microduck Watcher")
    mirror = bool(cam.get("mirror_display", True))
    credit = (cfg["speech"].get("credit") or "") if (speech and speech.backend == "voicevox") else ""
    fps, t_prev = 0.0, time.time()
    t_start = time.time()
    n_frames = 0
    try:
        while True:
            if args.duration is not None and time.time() - t_start >= args.duration:
                print(f"[main] {n_frames} frames in {args.duration:.0f}s = {n_frames / args.duration:.1f} fps (avg)")
                break
            ok, frame = cap.read()
            n_frames += 1
            if not ok:
                print("フレーム取得失敗")
                time.sleep(0.05)
                continue
            now = time.time()
            det = perception.detect(frame)
            face = fatigue.analyze(frame, now)
            face_bottom = face.bbox[3] if (face.found and face.bbox) else None
            phone_now = phone_below_face(det, face_bottom)
            fat = fatigue.update(face, now, phone_detected=bool(det.phones))

            person = det.main_person
            person_x = 0.0
            if person is not None:
                person_x = (person.cx / frame.shape[1]) * 2.0 - 1.0
            closeness = None
            if face.found and face.bbox:
                fh = (face.bbox[3] - face.bbox[1]) / frame.shape[0]
                far_h, near_h = float(cfg["perception"].get("far_face_h", 0.12)), float(cfg["perception"].get("near_face_h", 0.18))
                closeness = (fh - far_h) / max(1e-6, near_h - far_h)
            obs = Observation(t=now, person=person is not None, person_x=person_x,
                              phone_below_face=phone_now, fatigue=fat.score, closeness=closeness)
            event = behavior.update(obs)
            if event:
                print(f"[main] *** {event} fired ***")

            dt = now - t_prev
            t_prev = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

            if now - last_log >= log_interval or event:
                last_log = now
                log.writerow([f"{now:.3f}", f"{fps:.1f}", int(person is not None), int(bool(det.phones)),
                              int(phone_now), int(face.found), f"{fat.perclos:.3f}", fat.yawn_count,
                              f"{fat.droop:.3f}", f"{fat.score:.3f}", behavior.state, event or ""])
                log_f.flush()

            if show:
                st = behavior.status(now)
                st["phone_now"] = phone_now
                img = overlay.render(frame, det, face, fat, st, fps, now, credit,
                                     speech.backend if speech else "off", mirror)
                cv2.imshow(win, img)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log_f.close()
        print("[main] bye")


if __name__ == "__main__":
    main()
