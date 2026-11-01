"""sim_runner.py — 公式 scripts/infer_policy.py を「複製せず」にモジュールとして読み込み、
キーボード入力 (TerminalInput, Unix 専用 termios) だけを UDP/JSON 受信に差し替えて動かす。

Windows ネイティブ (mujoco + onnxruntime + bam) で動く。公式リポジトリは変更しない。

UDP (localhost:5005) で受け取る JSON（仕様書 6 章）:
  {"policy": "walking",  "twist": [0.2, 0.0, 0.0], "head_yaw": 0.3, "duration": 2.0}
  {"policy": "standing", "head_yaw": 0.3}
  {"policy": "sitstand", "sit": true}        # true=座る / false=立つ
  {"policy": "stop"}                         # 速度 0
  twist     : [vx, vy, wz]  vx は ±0.2 m/s にクリップ
  head_yaw  : rad。head_pose の並びは [neck_pitch, head_pitch, head_yaw, head_roll]
              （infer_policy.py の _update_command / handle_key で確認済み。角度 rad、クォータニオンではない）
  duration  : 秒。過ぎたら速度 0 に戻す（省略時は 2.0）

使い方（プロジェクトルートで）:
  sim\\.venv\\Scripts\\python.exe sim\\sim_runner.py            # 既定 3 ポリシーを読み込む
  sim\\.venv\\Scripts\\python.exe sim\\sim_runner.py --no-bam   # 位置アクチュエータで動かす（軽い）
  それ以外の引数は infer_policy.py にそのまま渡る（--foot-friction など）
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_DIR = os.path.join(ROOT, "microduck_rl")
SCRIPTS_DIR = os.path.join(RL_DIR, "scripts")
POLICY_DIR = os.path.join(ROOT, "policies")

UDP_HOST = os.environ.get("MICRODUCK_UDP_HOST", "127.0.0.1")
UDP_PORT = int(os.environ.get("MICRODUCK_UDP_PORT", "5005"))
STATE_PORT = int(os.environ.get("MICRODUCK_STATE_PORT", "5006"))   # 姿勢 (qpos) を配信 → demo の「ダック視点」描画用
STATE_EVERY = 3                                                       # 50 Hz ループの 3 回に 1 回 (~17 Hz)
# 学習済みポリシーには静止デッドバンドがある（2026-11-01 計測, sim/diag_minimal.py）:
#   前進は alpha_walking >= 0.25 m/s, velstand >= 0.20 m/s で歩き出す。後退は |vx| >= 0.4 m/s が必要。
#   旋回は wz >= 1.0 rad/s で反応（0.5 では静止）。仕様書の「前進 0.2 m/s 以下」はこのポリシーでは成立しない。
VX_MAX = 0.4
VY_MAX = 0.2
WZ_MAX = 1.5
HEAD_YAW_MAX = 1.4    # infer_policy.py の head_max（new_cmd_obs）
DEFAULT_DURATION = 2.0

# ---- Windows 用シム: infer_policy.py は先頭で termios/tty を import するが、
#      TerminalInput 以外では使わないので空モジュールで代替する。
#      Windows には termios が無い一方 tty.py は存在し、中で `from termios import *` を行うため、
#      termios が無い環境では tty も偽モジュールにする（実 tty を import してはいけない）。
try:
    import termios  # noqa: F401
except ImportError:
    for _name in ("termios", "tty"):
        sys.modules[_name] = types.ModuleType(_name)

sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(RL_DIR)  # MICRODUCK_XML は microduck_rl 直下からの相対パス
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import demo_scene  # noqa: E402  (椅子に座る人を実行時にシーンへ追加)
import infer_policy  # noqa: E402  (公式スクリプトをモジュールとして読み込む)

ADD_PERSON = True   # --no-person で無効化


def _install_scene_patches():
    """公式 scene.xml を変更せずに、読み込み時に椅子と人を足す。
    infer_policy.main は --no-bam なら MjModel.from_xml_path、BAM なら MjSpec.from_file を使う。"""
    orig_from_file = mujoco.MjSpec.from_file
    orig_launch = mujoco.viewer.launch_passive

    def spec_from_file(path, *a, **kw):
        spec = orig_from_file(path, *a, **kw)
        if ADD_PERSON:
            demo_scene.add_person(spec)
        return spec

    def model_from_xml_path(path, *a, **kw):
        spec = orig_from_file(path)
        if ADD_PERSON:
            demo_scene.add_person(spec)
        return spec.compile()

    def launch_passive(model, data, *a, **kw):
        h = orig_launch(model, data, *a, **kw)
        try:  # ダックと人が両方入る視点
            h.cam.lookat[:] = [0.45, 0.0, 0.35]
            h.cam.distance = 2.2
            h.cam.azimuth = 150
            h.cam.elevation = -12
        except Exception as e:
            print(f"[sim_runner] camera preset failed: {e}")
        return h

    mujoco.MjSpec.from_file = staticmethod(spec_from_file)
    mujoco.MjModel.from_xml_path = staticmethod(model_from_xml_path)
    mujoco.viewer.launch_passive = launch_passive


class UdpCommandInput:
    """TerminalInput 互換 (with 文 + get_keys())。
    get_keys() は 50 Hz ループの先頭で毎回呼ばれるので、ここで UDP を読み PolicyInference に直接反映する。
    キーは返さない（空リスト）。
    """

    policy = None  # PatchedPolicyInference が自身を登録する

    def __init__(self):
        self.sock = None
        self.deadline = None       # 速度指令の期限
        self.last_msg_t = 0.0
        self.n_msgs = 0
        self._init_qpos = None     # reset 用: main() が初期姿勢を作った直後の状態
        self._init_ctrl = None
        self._tick = 0
        self._state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _publish_state(self):
        """qpos を demo（ダック視点レンダラ）へ配信する。受け手がいなくても害はない。"""
        p = self.policy
        if p is None:
            return
        self._tick += 1
        if self._tick % STATE_EVERY:
            return
        try:
            msg = {"t": time.time(), "qpos": [round(float(v), 5) for v in p.data.qpos],
                   "policy": p.current_policy, "sit": bool(p.sit_mode)}
            self._state_sock.sendto(json.dumps(msg).encode("utf-8"), (UDP_HOST, STATE_PORT))
        except OSError:
            pass

    def _snapshot(self):
        p = self.policy
        if p is not None and self._init_qpos is None:
            self._init_qpos = p.data.qpos.copy()
            self._init_ctrl = p.data.ctrl.copy()

    def reset(self):
        """転倒などからの復帰: 初期姿勢に戻し、指令と内部状態をクリアする。"""
        p = self.policy
        if p is None or self._init_qpos is None:
            return
        import mujoco
        import numpy as np
        p.data.qpos[:] = self._init_qpos
        p.data.qvel[:] = 0.0
        p.data.qacc_warmstart[:] = 0.0
        p.data.ctrl[:] = self._init_ctrl
        p.last_action[:] = 0.0
        p.head_offset[:] = 0.0
        p.body_cmd[:] = 0.0
        p.sit_mode = False
        if p.bam_ctrl is not None and hasattr(p.bam_ctrl, "q_target"):
            p.bam_ctrl.q_target[:] = np.asarray(p.default_pose, dtype=np.float64)
        p.set_vel_cmd(0.0, 0.0, 0.0)
        mujoco.mj_forward(p.model, p.data)
        self.deadline = None
        print("[sim_runner] RESET to initial standing pose", flush=True)

    def __enter__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((UDP_HOST, UDP_PORT))
        self.sock.setblocking(False)
        print(f"[sim_runner] UDP listening on {UDP_HOST}:{UDP_PORT}  (Ctrl+C or close viewer to quit)")
        return self

    def __exit__(self, *exc):
        if self.sock:
            self.sock.close()

    # -- 内部 ----------------------------------------------------------------
    def _recv_latest(self):
        latest = None
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                break
            try:
                latest = json.loads(data.decode("utf-8"))
            except Exception as e:  # 壊れた JSON は捨てる
                print(f"[sim_runner] bad JSON: {e}")
        return latest

    @staticmethod
    def _clip(v, lim):
        return max(-lim, min(lim, float(v)))

    def _apply(self, msg: dict, now: float):
        p = self.policy
        if p is None:
            return
        kind = str(msg.get("policy", "stop")).lower()
        if kind == "reset":
            self.reset()
            return
        if kind == "scene":  # 舞台装置の見た目: {"policy":"scene","person":true,"phone":false,"eyes_closed":false}
            if "person" in msg:
                demo_scene.set_person_visible(p.model, bool(msg["person"]))
            if "phone" in msg:
                demo_scene.set_phone_visible(p.model, bool(msg["phone"]))
            if "eyes_closed" in msg:
                demo_scene.set_eyes_closed(p.model, bool(msg["eyes_closed"]))
            if "mouth_open" in msg:
                demo_scene.set_mouth_open(p.model, bool(msg["mouth_open"]))
            if "side" in msg:
                demo_scene.set_person_side(p.model, str(msg["side"]))
            return
        head_yaw = msg.get("head_yaw")
        if head_yaw is not None:
            p.head_offset[2] = self._clip(head_yaw, HEAD_YAW_MAX)
            p._update_command()

        if kind == "sitstand":
            want_sit = bool(msg.get("sit", True))
            if p.sit_session is None:
                print("[sim_runner] sitstand policy not loaded")
                return
            if want_sit != p.sit_mode:
                p.vel_cmd[:] = 0.0
                p.toggle_sit()
            self.deadline = None
            return

        # walking / standing / stop → まず座っていれば立たせる
        if p.sit_mode:
            p.toggle_sit()
        if kind in ("walking", "standing", "stop"):
            tw = msg.get("twist", [0.0, 0.0, 0.0]) if kind == "walking" else [0.0, 0.0, 0.0]
            vx = self._clip(tw[0] if len(tw) > 0 else 0.0, VX_MAX)
            vy = self._clip(tw[1] if len(tw) > 1 else 0.0, VY_MAX)
            wz = self._clip(tw[2] if len(tw) > 2 else 0.0, WZ_MAX)
            p.set_vel_cmd(vx, vy, wz)
            moving = (abs(vx) + abs(vy) + abs(wz)) > 1e-6
            self.deadline = now + float(msg.get("duration", DEFAULT_DURATION)) if moving else None
        else:
            print(f"[sim_runner] unknown policy {kind!r}")

    # -- TerminalInput 互換 API ------------------------------------------------
    def get_keys(self):
        now = time.time()
        self._snapshot()
        msg = self._recv_latest()
        if msg is not None:
            self.n_msgs += 1
            self.last_msg_t = now
            print(f"[sim_runner] recv #{self.n_msgs}: {msg}", flush=True)
            try:
                self._apply(msg, now)
            except Exception as e:
                print(f"[sim_runner] apply failed: {e}", flush=True)
        if self.deadline is not None and now >= self.deadline:
            self.deadline = None
            if self.policy is not None:
                self.policy.set_vel_cmd(0.0, 0.0, 0.0)
        self._publish_state()
        return []


class PatchedPolicyInference(infer_policy.PolicyInference):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        UdpCommandInput.policy = self


def main():
    global ADD_PERSON
    argv = sys.argv[1:]
    if "--no-person" in argv:
        argv.remove("--no-person")
        ADD_PERSON = False
    _install_scene_patches()
    # 既定ポリシー（指定が無ければ補う）
    defaults = {
        "--walking": os.path.join(POLICY_DIR, "alpha_walking.onnx"),
        "--standing": os.path.join(POLICY_DIR, "alpha_stand.onnx"),
        "--sitstand": os.path.join(POLICY_DIR, "alpha_sitstand.onnx"),
    }
    # --velstand: 歩行も静止も velstand.onnx（転倒から起き上がれる。仕様書 11 章の代替案）
    if "--velstand" in argv:
        argv.remove("--velstand")
        vs = os.path.join(POLICY_DIR, "velstand.onnx")
        defaults["--walking"] = vs
        defaults["--standing"] = vs
    for flag, path in defaults.items():
        if flag not in argv and not (flag == "--standing" and "-s" in argv):
            argv += [flag, path]
    if "--new-cmd-obs" not in argv:
        argv.append("--new-cmd-obs")
    # 既定は XML の位置アクチュエータ (--no-bam)。BAM では着座からの起立と後退が不安定だった一方、
    # XML アクチュエータは公式 docs の sim 評価値 (0.4 m/s 指令 → 0.17 m/s) を再現した (2026-11-01)。
    # BAM を使うときは --bam を付ける。
    if "--bam" in argv:
        argv.remove("--bam")
    elif "--no-bam" not in argv:
        argv.append("--no-bam")

    infer_policy.TerminalInput = UdpCommandInput
    infer_policy.PolicyInference = PatchedPolicyInference
    sys.argv = ["infer_policy.py"] + argv
    print("[sim_runner] args:", " ".join(argv))
    infer_policy.main()


if __name__ == "__main__":
    main()
