"""状態遷移とイベント判定（仕様書 2 章・6 章）。

知覚結果 Observation を毎フレーム update() に渡すと、トリガ条件・優先度 (F3 > F2 > F1)・
クールダウンを評価し、発火した反応を「一気通貫」で実行する:

    [座っていれば立つ] → [遠ければ近づく（人を画面中央に保ちながら）] → 声かけ（F3 は座ってから）→ [F2/F3 は少し下がる]

- 短い未検出（lost_tolerance_s 以内）では在席・スマホ・疲労の判定タイマーを切らさない。
- 反応はいったん始まれば、途中で人を見失っても発話まで必ず進める（接近だけは見失えば打ち切る）。
- 人がいない間は 15 秒ごとに首を振って探し（SCAN）、roam_after_s 以上見当たらなければ歩いて探し回る（ROAM）。
ロボット・音声は Actions のコールバック経由なので、テスト時は偽の Actions を渡して sim/音声なしで検証できる。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Observation:
    t: float
    person: bool = False             # 人の枠がある
    person_x: float = 0.0            # 画面横位置 -1(左)〜+1(右)
    phone_below_face: bool = False   # スマホが顔より下かつ人の枠と重なる
    fatigue: float = 0.0             # 疲労スコア S (0〜1)
    closeness: Optional[float] = None  # 近さ 0..1（0 以下 = 遠い → 近づく, 1 以上 = 十分近い → 止まる）。不明なら None
                                       # 呼び出し側（demo.py / main.py）がカメラごとの校正値で正規化する


@dataclass
class Actions:
    send: Callable[[dict], None]            # UDP 送信
    speak: Callable[[str, str], None]       # (phrase_key, text)
    quack: Callable[[], None]


@dataclass
class Step:
    at: float
    fn: Callable[[], None]
    label: str = ""


@dataclass
class Behavior:
    cfg: dict
    actions: Actions
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self):
        ev = self.cfg["events"]
        self.presence_min_s = float(ev["presence_min_s"])
        self.absence_reset_s = float(ev["absence_reset_s"])
        self.phone_hold_s = float(ev["phone_hold_s"])
        self.phone_cd_s = float(ev["phone_cooldown_s"])
        self.fatigue_thresh = float(ev["fatigue_thresh"])
        self.fatigue_hold_s = float(ev["fatigue_hold_s"])
        self.fatigue_cd_s = float(ev["fatigue_cooldown_s"])
        rb = self.cfg["robot"]
        self.v = float(rb["forward_speed"])
        self.v_back = float(rb.get("backward_speed", rb["forward_speed"]))
        self.approach_s = float(rb["approach_s"])                       # 距離不明のときの前進時間
        self.approach_max_s = float(rb.get("approach_max_s", 4.0))      # 近づく最大時間
        self.retreat_s = float(rb["retreat_s"])
        self.head_yaw_max = float(rb["head_yaw_max"])
        self.head_yaw_sign = float(rb.get("head_yaw_sign", 1))
        self.sit_glide_s = float(rb.get("sit_glide_s", 2.5))
        self.idle_sit = bool(rb.get("idle_sit", True))
        self.idle_sit_after_s = float(rb.get("idle_sit_after_s", 6.0))
        self.lost_tol_s = float(rb.get("lost_tolerance_s", 1.5))        # この時間までの未検出は無視
        self.scan_interval_s = float(rb.get("scan_interval_s", 0.0))
        self.scan_yaw = float(rb.get("scan_yaw", 0.9))
        self.roam_after_s = float(rb.get("roam_after_s", 0.0))
        self.roam_max_s = float(rb.get("roam_max_s", 36.0))
        self.roam_turn_wz = float(rb.get("roam_turn_wz", 1.2))
        self.roam_turn_s = float(rb.get("roam_turn_s", 1.5))
        self.roam_walk_s = float(rb.get("roam_walk_s", 1.5))
        self.steer_x_thresh = float(rb.get("steer_x_thresh", 0.35))
        self.steer_wz = float(rb.get("steer_wz", 1.0))
        self.phrases = self.cfg["speech"]["phrases"]

        # 追跡（未検出に寛容）
        self.t0: Optional[float] = None
        self.last_seen: Optional[float] = None       # 人を最後に検出した時刻
        self.person_since: Optional[float] = None    # （寛容判定で）人が居続けている開始時刻
        self.present = False
        self.greeted = False
        self.last_phone_t: Optional[float] = None
        self.phone_since: Optional[float] = None
        self.fatigue_since: Optional[float] = None
        self.phone_cd_until = 0.0
        self.fatigue_cd_until = 0.0
        self.last_closeness: Optional[float] = None
        self.last_head_yaw = 0.0
        # ロボット
        self.seated = False
        self.robot_initialized = False
        # 反応（フェーズ機械）と探索（ステップ列）
        self.state = "IDLE"              # IDLE / F1 / F2 / F3 / SCAN / ROAM
        self.phase = ""                  # 反応中: STAND / APPROACH / SETTLE / ACT / RETREAT
        self.phase_t0 = 0.0
        self.act_steps: list[Step] = []
        self.seq: list[Step] = []        # SCAN / ROAM 用
        self.seq_t0 = 0.0
        self.seq_done_t: Optional[float] = None
        self.last_event: Optional[str] = None
        self.last_event_t: Optional[float] = None
        self.last_scan_t = 0.0
        self.last_roam_t = -1e9
        self.rescan_pending = False
        self._steer_wz_last = 0.0
        self._last_steer_t = 0.0
        self._approach_lost_t: Optional[float] = None
        self.last_text = ""
        self.last_text_t = 0.0

    # ------------------------------------------------------------------ utils
    def head_yaw_for(self, person_x: float) -> float:
        return float(max(-1.0, min(1.0, person_x)) * self.head_yaw_max * self.head_yaw_sign)

    def _pick(self, key: str) -> str:
        return self.rng.choice(list(self.phrases[key]))

    def _say(self, key: str, now: float):
        text = self._pick(key)
        self.last_text, self.last_text_t = text, now
        self.actions.speak(key, text)

    def _sit(self):
        self.actions.send({"policy": "sitstand", "sit": True, "head_yaw": self.last_head_yaw})
        self.seated = True

    def _stand(self):
        self.actions.send({"policy": "sitstand", "sit": False, "head_yaw": self.last_head_yaw})
        self.seated = False

    def _look(self):
        self.actions.send({"policy": "standing", "head_yaw": self.last_head_yaw})

    def _head(self, yaw: float):
        self.actions.send({"policy": "sitstand", "sit": self.seated, "head_yaw": float(yaw)})

    def _walk(self, vx: float, dur: float, wz: float = 0.0, head_yaw: Optional[float] = None):
        self.seated = False
        self.actions.send({"policy": "walking", "twist": [vx, 0.0, wz],
                           "head_yaw": self.last_head_yaw if head_yaw is None else head_yaw,
                           "duration": round(dur, 2)})

    def _stop(self):
        self.actions.send({"policy": "stop", "head_yaw": self.last_head_yaw})

    def _is_far(self) -> Optional[bool]:
        """直近の近さから「遠い（接近が必要）」か。不明なら None。"""
        if self.last_closeness is None:
            return None
        return self.last_closeness <= 0.0

    def _is_near(self) -> bool:
        return self.last_closeness is not None and self.last_closeness >= 1.0

    # ------------------------------------------------------------ 反応の開始
    def _begin_reaction(self, kind: str, now: float):
        if self.state == "SCAN":
            self._head(self.last_head_yaw)
        if self.state == "ROAM":
            self._stop()
        self.seq = []
        self.state = kind
        self.last_event, self.last_event_t = kind, now
        self.seq_done_t = None
        self._enter_phase("STAND", now)

    def _enter_phase(self, phase: str, now: float):
        self.phase, self.phase_t0 = phase, now
        if phase == "STAND":
            if self.seated:
                self._stand()
            else:
                self._enter_phase("APPROACH", now)
        elif phase == "APPROACH":
            far = self._is_far()
            if far is False:
                self._enter_phase("SETTLE", now)
                return
            self._walk(self.v, self.approach_max_s if far else self.approach_s, head_yaw=0.0)
            self._steer_wz_last = 0.0
            self._last_steer_t = now
            self._approach_lost_t = None
        elif phase == "SETTLE":
            self._stop()
        elif phase == "ACT":
            self.act_steps = self._act_steps(self.state, now)
        elif phase == "RETREAT":
            self._walk(-self.v_back, self.retreat_s)
        elif phase == "DONE":
            if self.state in ("F2", "F3"):
                self._stop()
            self.rescan_pending = self.state in ("F2", "F3")
            self.state, self.phase = "IDLE", ""
            self.seq_done_t = now

    def _act_steps(self, kind: str, now: float) -> list[Step]:
        g = self.sit_glide_s
        if kind == "F1":   # 向く → 鳴く → 発話
            return [Step(0.0, self._look, "look"), Step(0.5, self.actions.quack, "quack"),
                    Step(1.2, lambda: self._say("greet", now), "speak"), Step(4.0, lambda: None, "end")]
        if kind == "F2":   # 発話
            return [Step(0.3, lambda: self._say("phone", now), "speak"), Step(3.0, lambda: None, "end")]
        # F3: 座る → 鳴く → 発話 → 立つ
        return [Step(0.0, self._sit, "sit"), Step(g * 0.6, self.actions.quack, "quack"),
                Step(g * 0.6 + 0.6, lambda: self._say("fatigue", now), "speak"),
                Step(g + 3.0, self._stand, "stand"), Step(2 * g + 3.0, lambda: None, "end")]

    def _reaction_tick(self, obs: Observation):
        now = obs.t
        el = now - self.phase_t0
        if self.phase == "STAND":
            if el >= self.sit_glide_s:
                self._enter_phase("APPROACH", now)
        elif self.phase == "APPROACH":
            known = self.last_closeness is not None
            limit = self.approach_max_s if known else self.approach_s   # 距離不明なら approach_s だけ前進
            if (known and self._is_near()) or el >= limit:              # 十分近くなったら止まる
                self._enter_phase("SETTLE", now)
                return
            # 見失ったら少し待ち、戻らなければ接近を打ち切る（発話には進む）
            if obs.person:
                self._approach_lost_t = None
                if now - self._last_steer_t >= 0.5:
                    wz = 0.0
                    if abs(obs.person_x) > self.steer_x_thresh:
                        wz = -self.steer_wz if obs.person_x > 0 else self.steer_wz   # 画面右の人 → 右旋回
                    if wz != self._steer_wz_last:
                        self._walk(self.v, max(0.4, limit - el) + 0.1, wz=wz, head_yaw=0.0)
                        self._steer_wz_last = wz
                    self._last_steer_t = now
            else:
                if self._approach_lost_t is None:
                    self._approach_lost_t = now
                elif now - self._approach_lost_t > self.lost_tol_s:
                    self._enter_phase("SETTLE", now)
        elif self.phase == "SETTLE":
            if el >= 0.5:
                self._enter_phase("ACT", now)
        elif self.phase == "ACT":
            while self.act_steps and self.act_steps[0].at <= el:
                step = self.act_steps.pop(0)
                try:
                    step.fn()
                except Exception as e:
                    print(f"[behavior] step {step.label} failed: {e}")
            if not self.act_steps:
                self._enter_phase("RETREAT" if self.state in ("F2", "F3") else "DONE", now)
        elif self.phase == "RETREAT":
            if el >= self.retreat_s:
                self._enter_phase("DONE", now)

    # --------------------------------------------------------- 探索（SCAN/ROAM）
    def _start_seq(self, name: str, steps: list[Step], now: float):
        self.state = name
        self.seq = sorted(steps, key=lambda s: s.at)
        self.seq_t0 = now

    def _seq_scan(self, now: float):
        y = self.scan_yaw
        self._start_seq("SCAN", [Step(0.0, lambda: self._head(+y), "scan_left"),
                                 Step(1.4, lambda: self._head(-y), "scan_right"),
                                 Step(2.8, lambda: self._head(0.0), "scan_center"),
                                 Step(3.6, lambda: None, "end")], now)
        self.last_scan_t = now

    def _seq_roam(self, now: float):
        self.last_head_yaw = 0.0
        steps, t = [], 0.0
        if self.seated:
            steps.append(Step(0.0, self._stand, "stand"))
            t = self.sit_glide_s
        while t + self.roam_turn_s + self.roam_walk_s + 2.0 <= self.roam_max_s:
            steps.append(Step(t, lambda: self._walk(0.0, self.roam_turn_s, wz=self.roam_turn_wz, head_yaw=0.0), "turn"))
            t += self.roam_turn_s + 1.0
            steps.append(Step(t, lambda: self._walk(self.v, self.roam_walk_s, head_yaw=0.0), "walk"))
            t += self.roam_walk_s + 1.0
        steps.append(Step(t, self._stop, "stop"))
        steps.append(Step(t + 0.5, lambda: None, "end"))
        self._start_seq("ROAM", steps, now)
        self.last_roam_t = now

    # ---------------------------------------------------------------- update
    def update(self, obs: Observation) -> Optional[str]:
        """毎フレーム呼ぶ。発火したイベント名 (F1/F2/F3) または None を返す。"""
        now = obs.t
        if self.t0 is None:
            self.t0 = now
            self.last_scan_t = now
        if not self.robot_initialized:
            self.robot_initialized = True
            if self.idle_sit:
                self._sit()

        # --- 追跡（短い未検出は無視） ---
        if obs.person:
            self.last_seen = now
            self.last_head_yaw = self.head_yaw_for(obs.person_x)
            if obs.closeness is not None:
                self.last_closeness = obs.closeness
            elif self.last_closeness is not None:
                # 人は見えるのに顔が取れない = 顔が画角の上に外れるほど近い → 近いとみなす
                self.last_closeness = 1.0
        seen = obs.person or (self.last_seen is not None and now - self.last_seen <= self.lost_tol_s)
        if seen:
            if self.person_since is None:
                self.person_since = now
            if not self.present and now - self.person_since >= self.presence_min_s:
                self.present = True
        else:
            self.person_since = None
            self.present = False
            if self.last_seen is not None and now - self.last_seen >= self.absence_reset_s:
                self.greeted = False

        if obs.person and obs.phone_below_face:
            self.last_phone_t = now
        phone_eff = seen and self.last_phone_t is not None and now - self.last_phone_t <= self.lost_tol_s
        if phone_eff:
            if self.phone_since is None:
                self.phone_since = now
        else:
            self.phone_since = None

        if seen and obs.fatigue > self.fatigue_thresh:
            if self.fatigue_since is None:
                self.fatigue_since = now
        else:
            self.fatigue_since = None

        # --- 発火判定（反応中は新規発火しない。SCAN/ROAM は割り込める） ---
        fired = None
        if self.state in ("IDLE", "SCAN", "ROAM") and self.present:
            f3 = (self.fatigue_since is not None and now - self.fatigue_since >= self.fatigue_hold_s
                  and now >= self.fatigue_cd_until)
            f2 = (self.phone_since is not None and now - self.phone_since >= self.phone_hold_s
                  and now >= self.phone_cd_until)
            f1 = not self.greeted
            if f3:
                self.fatigue_cd_until = now + self.fatigue_cd_s
                self.fatigue_since = None
                self.greeted = True
                self._begin_reaction("F3", now)
                fired = "F3"
            elif f2:
                self.phone_cd_until = now + self.phone_cd_s
                self.phone_since = None
                self.greeted = True
                self._begin_reaction("F2", now)
                fired = "F2"
            elif f1:
                self.greeted = True
                self._begin_reaction("F1", now)
                fired = "F1"

        # --- 反応の進行 ---
        if self.state in ("F1", "F2", "F3"):
            self._reaction_tick(obs)

        # --- 人がいない間の探索 ---
        if fired is None and self.state == "IDLE" and not seen and self.robot_initialized:
            absent_for = now - (self.last_seen if self.last_seen is not None else self.t0)
            if (self.roam_after_s > 0 and absent_for >= self.roam_after_s
                    and now - self.last_roam_t >= self.roam_after_s):
                self._seq_roam(now)
            elif self.rescan_pending and self.seq_done_t is not None and now - self.seq_done_t >= 1.0:
                self.rescan_pending = False
                self._seq_scan(now)
            elif (self.scan_interval_s > 0
                  and now - max(self.last_scan_t, self.seq_done_t or 0.0) >= self.scan_interval_s):
                self._seq_scan(now)
        if seen:
            self.rescan_pending = False

        self.tick(now)
        return fired

    def tick(self, now: float):
        """SCAN/ROAM ステップ列の進行と、待機時の着座。"""
        if self.state in ("SCAN", "ROAM"):
            el = now - self.seq_t0
            while self.seq and self.seq[0].at <= el:
                step = self.seq.pop(0)
                try:
                    step.fn()
                except Exception as e:
                    print(f"[behavior] step {step.label} failed: {e}")
            if not self.seq:
                self.state = "IDLE"
                self.seq_done_t = now
        elif (self.state == "IDLE" and self.idle_sit and not self.seated and self.seq_done_t is not None
              and now - self.seq_done_t >= self.idle_sit_after_s):
            self._sit()
            self.seq_done_t = None

    # ------------------------------------------------------------- 表示用
    def status(self, now: float) -> dict:
        def left(until):
            return max(0.0, until - now)
        return {
            "state": self.state,
            "phase": self.phase,
            "present": self.present,
            "greeted": self.greeted,
            "phone_hold": 0.0 if self.phone_since is None else now - self.phone_since,
            "fatigue_hold": 0.0 if self.fatigue_since is None else now - self.fatigue_since,
            "phone_cd": left(self.phone_cd_until),
            "fatigue_cd": left(self.fatigue_cd_until),
            "seated": self.seated,
            "far": self._is_far(),
            "closeness": self.last_closeness,
            "last_event": self.last_event,
            "subtitle": self.last_text,
            "subtitle_t": self.last_text_t,
        }
