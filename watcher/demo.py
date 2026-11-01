"""カメラなしデモ（知覚ループ入り）。

  台本 → シミュレーション内の人（在/不在・スマホ・目・口） → ダック頭部カメラの映像
      → 知覚 (sim_perception: 人・顔・目・口・スマホ) → 疲労スコア (fatigue.FatigueScorer)
      → 状態遷移 (behavior) → sim への指令 (UDP) と音声 (VOICEVOX)

台本はシーンの「見た目」だけを変える。F1/F2/F3 の発火はダック視点の画像から判定した結果で決まる
（実カメラ版 main.py と同じ behavior.py / 疲労スコア式を使う。知覚だけ色ベース: YOLO/MediaPipe は
単純形状の人形を認識できないため）。

起動:  scripts\\run_demo.bat            （自動シナリオを 1 回再生。終了後はキー操作で続行可）
       scripts\\run_demo.bat --manual   （最初からキー操作のみ）
       scripts\\run_demo.bat --loop     （自動シナリオを繰り返す）
キー:  p 人の在/不在   s スマホ   f 疲労（目を閉じ・周期的にあくび）   b 人を前/後ろに移す   a 自動再生   r リセット   q 終了
       scripts\\run_demo.bat --behind   人をダックの背後に置いて開始（首振り → 歩き回り → 発見 → F1）
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from behavior import Actions, Behavior, Observation  # noqa: E402
from fatigue import FatigueScorer  # noqa: E402
from perception import phone_below_face  # noqa: E402
from pov import DuckPOV, StateReceiver  # noqa: E402
from sim_bridge import SimBridge  # noqa: E402
from sim_perception import SimPerception  # noqa: E402
from speech import Speech  # noqa: E402

W, H = 1120, 660
PANEL = (16, 16, 656, 496)   # 左パネル 640x480 = ダック視点

# 前半は開始からの秒、F1（挨拶）が発火した後は「F1 からの秒」で進む。
# 人が背後にいる場合はダックが首振り→歩き回りで見つけるまで F1 が遅れるため、残りは F1 基準にする。
SCENARIO_PRE = [  # (開始秒, person, phone, sleepy, 説明)
    (0.0, False, False, False, "待機中（不在）"),
    (2.0, True, False, False, "人が椅子に座った（ダックが見つけるのを待つ）"),
]
SCENARIO_POST = [  # (F1 からの秒, person, phone, sleepy, 説明)
    (0.0, True, False, False, "挨拶済み。普通に座っている"),
    (9.0, True, True, False, "スマホを顔より下で操作中"),
    (27.0, True, False, True, "目を閉じ気味・あくび"),
    (63.0, False, False, False, "人が席を外した（15 秒ごとに首振り、60 秒後に歩いて探す）"),
    (167.0, None, None, None, "シナリオ終了"),
]
YAWN_PERIOD_S, YAWN_OPEN_S = 5.0, 2.0


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for sec, key in [("speech", "quack_wav"), ("logging", "dir")]:
        p = cfg[sec][key]
        if not os.path.isabs(p):
            cfg[sec][key] = os.path.join(ROOT, p)
    return cfg


class Canvas:
    def __init__(self, cfg):
        d = cfg["display"]
        self.font = self.font_s = None
        try:
            from PIL import ImageFont
            fp = d.get("font_path", "C:/Windows/Fonts/meiryo.ttc")
            self.font = ImageFont.truetype(fp, 34)
            self.font_s = ImageFont.truetype(fp, 19)
        except Exception as e:
            print(f"[demo] 日本語フォントが使えません ({e})")

    def text(self, img, s, xy, color=(255, 255, 255), small=False):
        if self.font is None:
            cv2.putText(img, s.encode("ascii", "ignore").decode(), xy, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            return img
        from PIL import Image, ImageDraw
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(pil).text(xy, s, font=self.font_s if small else self.font, fill=color[::-1])
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

    def render(self, pov_img, det, face, fat, phone_now, st, now, credit, note, elapsed, mode, subtitle_s, pov_alive):
        img = np.full((H, W, 3), (40, 36, 32), np.uint8)
        x1, y1, x2, y2 = PANEL
        view = pov_img.copy()
        for b in det.persons:
            cv2.rectangle(view, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 200, 0), 2)
            cv2.putText(view, f"person {b.conf:.1f}", (int(b.x1), max(14, int(b.y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        for b in det.phones:
            cv2.rectangle(view, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 140, 255), 2)
            cv2.putText(view, f"phone {b.conf:.1f}", (int(b.x1), max(14, int(b.y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
        if face.found and face.bbox:
            fx1, fy1, fx2, fy2 = map(int, face.bbox)
            cv2.rectangle(view, (fx1, fy1), (fx2, fy2), (255, 200, 0), 1)
            cv2.putText(view, f"face eye={face.eye_close:.1f} jaw={face.jaw_open:.1f}", (fx1, fy2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        img[y1:y2, x1:x2] = cv2.resize(view, (x2 - x1, y2 - y1))
        cv2.rectangle(img, (x1, y1), (x2, y2), (120, 120, 120), 1)
        label = "Microduck head camera (sim)" + ("" if pov_alive else "   [waiting for sim: run_sim.bat]")
        cv2.putText(img, label, (x1 + 8, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        # 知覚の数値
        s1 = f"person {'YES' if det.persons else 'no '}  phone_below_face {'YES' if phone_now else 'no '}  face {'YES' if face.found else 'no '}   perception {det.infer_ms:4.1f} ms"
        s2 = f"S={fat.score:.2f}  PERCLOS={fat.perclos:.2f}  yawn={fat.yawn_count}{'*' if fat.yawning_now else ''}  droop={fat.droop:.2f}   (threshold {st['fatigue_thresh']:.2f})"
        cv2.putText(img, s1, (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(img, s2, (x1, y2 + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255) if fat.score > st["fatigue_thresh"] else (230, 230, 230), 1, cv2.LINE_AA)
        img = self.text(img, "知覚はこの映像（ダック視点）から色で判定。判定ルールと疲労スコア式は実カメラ版と共通", (x1, y2 + 54), (170, 170, 170), small=True)

        # 右: 状態
        x0 = 680
        lines = [
            f"mode: {mode}   t={elapsed:5.1f}s",
            f"state {st['state']} {st['phase']}   present {int(st['present'])}   greeted {int(st['greeted'])}   seated {int(st['seated'])}",
            f"distance: far={st['far']}  closeness={('%.2f' % st['closeness']) if st['closeness'] is not None else '-'}   (<=0 far, >=1 near)",
            f"hold  phone {st['phone_hold']:4.1f}s   fatigue {st['fatigue_hold']:4.1f}s",
            f"cooldown  phone {st['phone_cd']:3.0f}s   fatigue {st['fatigue_cd']:3.0f}s",
            f"last event: {st['last_event'] or '-'}",
        ]
        for i, s in enumerate(lines):
            color = (0, 255, 255) if (i == 1 and st["state"] not in ("IDLE", "SCAN")) else (230, 230, 230)
            cv2.putText(img, s, (x0, 46 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        img = self.text(img, "台本: " + note, (x0, 222), (255, 220, 120), small=True)
        img = self.text(img, "p:人 s:スマホ f:疲労 b:人を前/後ろ a:自動再生 r:リセット q:終了", (x0, 258), (160, 160, 160), small=True)
        rows = ["F1 挨拶: 人を1秒連続検出", "F2 スマホ注意: 顔より下で5秒", "F3 疲労: S>0.35 が10秒",
                "SCAN: 15秒ごとに首を振って探す", "ROAM: 1分見当たらなければ歩いて探す"]
        for i, s in enumerate(rows):
            key = f"F{i+1}" if i < 3 else ("SCAN" if i == 3 else "ROAM")
            on = (st["last_event"] == key or st["state"] == key) and st["state"] != "IDLE"
            img = self.text(img, ("▶ " if on else "   ") + s, (x0, 300 + 30 * i), (0, 255, 255) if on else (170, 170, 170), small=True)

        if st["subtitle"] and now - st["subtitle_t"] <= subtitle_s:
            cv2.rectangle(img, (0, H - 80), (W, H), (0, 0, 0), -1)
            img = self.text(img, st["subtitle"], (24, H - 64))
        if credit:
            img = self.text(img, credit, (W - 200, H - 26), (200, 200, 200), small=True)
        return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--manual", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--no-sim", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--behind", action="store_true", help="人をダックの背後に置く（探して見つけてから F1）")
    ap.add_argument("--duration", type=float, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    bridge = SimBridge(cfg["robot"]["udp_host"], int(cfg["robot"]["udp_port"]))
    speech = None if args.no_audio else Speech(cfg)
    pov = DuckPOV()
    rx = StateReceiver(expected_len=pov.model.nq)
    perception = SimPerception()
    scorer = FatigueScorer(cfg)

    def do_send(m):
        print("[sim(off)]", m) if args.no_sim else bridge.send(m)

    def do_speak(key, text):
        print(f"[speak:{key}] {text}")
        if speech:
            speech.say(text)

    def do_quack():
        print("[quack]")
        if speech:
            speech.quack()

    behavior = Behavior(cfg, Actions(send=do_send, speak=do_speak, quack=do_quack))
    canvas = Canvas(cfg)
    credit = (cfg["speech"].get("credit") or "") if (speech and speech.backend == "voicevox") else ""
    subtitle_s = float(cfg["display"].get("subtitle_s", 4.0))
    fatigue_thresh = float(cfg["events"]["fatigue_thresh"])

    scene = {"person": False, "phone": False, "sleepy": False}
    scene_sent = None
    side = "back" if args.behind else "front"
    side_sent = None
    t_f1 = None
    auto = not args.manual
    t_scn = time.time()
    t_start = t_scn
    note = "手動モード"
    win = "Microduck Demo"
    if not args.no_window:
        cv2.namedWindow(win)
    stand_key = pov.mujoco.mj_name2id(pov.model, pov.mujoco.mjtObj.mjOBJ_KEY, "STAND")
    last_qpos = pov.model.key_qpos[stand_key].tolist()
    pov_img = None
    print("[demo] start. keys: p s f a r q")
    try:
        while True:
            now = time.time()
            if args.duration is not None and now - t_start >= args.duration:
                break
            el = now - t_scn
            if auto:
                cur = None
                if t_f1 is None:
                    for t, person, phone, sleepy, desc in SCENARIO_PRE:
                        if el >= t:
                            cur = (t, person, phone, sleepy, desc)
                else:
                    for t, person, phone, sleepy, desc in SCENARIO_POST:
                        if now - t_f1 >= t:
                            cur = (t, person, phone, sleepy, desc)
                if cur and cur[1] is None:
                    if args.loop:
                        t_scn, t_f1 = now, None
                        behavior.greeted = False
                    else:
                        auto = False
                        note = "シナリオ終了（キーで続行できます）"
                elif cur:
                    scene.update(person=cur[1], phone=cur[2], sleepy=cur[3])
                    note = f"自動シナリオ: {cur[4]}" + ("  [人は背後]" if side == "back" else "")

            if side != side_sent:
                if not args.no_sim:
                    bridge.send({"policy": "scene", "side": side})
                pov.demo_scene.set_person_side(pov.model, side)
                side_sent = side

            # --- 台本 → シーンの見た目（sim の MuJoCo 画面と POV レンダラの両方） ---
            yawning = scene["sleepy"] and ((now % YAWN_PERIOD_S) < YAWN_OPEN_S)
            flags = (scene["person"], scene["phone"], scene["sleepy"], yawning)
            if flags != scene_sent:
                if not args.no_sim:
                    bridge.send({"policy": "scene", "person": flags[0], "phone": flags[1],
                                 "eyes_closed": flags[2], "mouth_open": flags[3]})
                pov.set_scene(flags[0], flags[1], flags[2])
                pov.demo_scene.set_mouth_open(pov.model, flags[3])
                scene_sent = flags

            # --- ダック視点を描画 → 知覚 → 疲労スコア ---
            if rx.latest is not None:
                last_qpos = rx.latest["qpos"]
            try:
                pov_img = pov.render(last_qpos)
            except ValueError as e:          # 検証をすり抜けた不正値は描画を飛ばす（前回の画像を保持）
                print(f"[demo] pov render skipped: {e}")
                if pov_img is None:
                    pov_img = np.zeros((480, 640, 3), np.uint8)
            det, face = perception.analyze(pov_img)
            face_mid = (face.bbox[1] + face.bbox[3]) / 2 if (face.found and face.bbox) else None
            phone_now = phone_below_face(det, face_mid)
            fat = scorer.update(face, now, phone_detected=bool(det.phones))
            person = det.main_person
            person_x = (person.cx / pov_img.shape[1]) * 2.0 - 1.0 if person is not None else 0.0

            obs = Observation(t=now, person=person is not None, person_x=person_x,
                              phone_below_face=phone_now, fatigue=fat.score,
                              closeness=perception.closeness(face, pov_img.shape[0]))
            ev = behavior.update(obs)
            if ev:
                print(f"[demo] *** {ev} fired at t={el:.1f}s  (S={fat.score:.2f})***")
                if ev == "F1" and t_f1 is None:
                    t_f1 = now
            st = behavior.status(now)
            st["fatigue_thresh"] = fatigue_thresh

            if not args.no_window:
                img = canvas.render(pov_img, det, face, fat, phone_now, st, now, credit, note, el,
                                    "auto" if auto else "manual", subtitle_s, rx.alive)
                cv2.imshow(win, img)
                k = cv2.waitKey(20) & 0xFF
            else:
                time.sleep(0.02)
                k = 255
            if k in (27, ord("q")):
                break
            elif k == ord("p"):
                auto = False; scene["person"] = not scene["person"]; note = "手動モード"
            elif k == ord("s"):
                auto = False; scene["phone"] = not scene["phone"]; note = "手動モード"
            elif k == ord("f"):
                auto = False; scene["sleepy"] = not scene["sleepy"]; note = "手動モード"
            elif k == ord("r"):
                auto = False; scene.update(person=False, phone=False, sleepy=False); note = "手動モード（リセット）"
            elif k == ord("a"):
                auto = True; t_scn = time.time(); t_f1 = None; behavior.greeted = False
            elif k == ord("b"):
                side = "back" if side == "front" else "front"
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("[demo] bye")


if __name__ == "__main__":
    main()
