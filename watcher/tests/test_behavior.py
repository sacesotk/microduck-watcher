"""behavior.py の単体確認（仕様書 10 章）: 知覚の代わりに偽のイベント列を流し、
状態遷移・優先度・クールダウン・再発火抑制・未検出への寛容さ・接近の有無を確認する。sim/音声は不要。

実行:  watcher\\.venv\\Scripts\\python.exe watcher\\tests\\test_behavior.py
"""
import copy
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from behavior import Actions, Behavior, Observation  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))


def make(cfg):
    sent, spoken, quacks = [], [], []
    b = Behavior(cfg, Actions(send=lambda m: sent.append(m),
                              speak=lambda k, t: spoken.append((k, t)),
                              quack=lambda: quacks.append(1)))
    return b, sent, spoken, quacks


def run(b, timeline, dt=0.1, flicker=None):
    """timeline: [(t_end, dict(person=..., phone=..., fatigue=..., closeness=...)), ...] を dt 刻みで流す。
    各要素の状態は「前の要素の t_end から、その要素の t_end まで」続く。
    flicker=(period, gap): 周期 period ごとに gap 秒だけ未検出にする（短い取りこぼしの模擬）。"""
    fired = []
    t = 0.0
    for t_end, o in timeline:
        while t < t_end:
            person = o.get("person", False)
            phone = o.get("phone", False)
            if flicker and person and (t % flicker[0]) < flicker[1]:
                person, phone = False, False
            ev = b.update(Observation(t=t, person=person, person_x=o.get("x", 0.0),
                                      phone_below_face=phone and person, fatigue=o.get("fatigue", 0.0),
                                      closeness=o.get("closeness", 1.0) if person else None))
            if ev:
                fired.append((round(t, 1), ev))
            t += dt
    return fired


def check(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    return bool(cond)


def main():
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ok = True

    print("case1: 入室 → 1 秒で F1。5 秒離席では再挨拶なし、10 秒以上で再挨拶。近い(closeness=1.0)ので接近しない")
    b, sent, spoken, quacks = make(cfg)
    fired = run(b, [(3.0, dict(person=True)), (8.0, dict(person=False)), (14.0, dict(person=True)),
                    (26.0, dict(person=False)), (32.0, dict(person=True))])
    print("   ", fired)
    ok &= check("F1 が 2 回、最初は 1.0 s 付近", [e for _, e in fired] == ["F1", "F1"] and abs(fired[0][0] - 1.0) < 0.2)
    ok &= check("greet 発話と鳴き声", spoken[0][0] == "greet" and len(quacks) >= 1)
    ok &= check("待機は座る", sent[0] == {"policy": "sitstand", "sit": True, "head_yaw": 0.0})
    ok &= check("近いので walking なし", not any(m.get("policy") == "walking" for m in sent))

    print("case2: スマホ 5 秒で F2、60 秒クールダウン。遠い(closeness=0.0)ので近づいてから話し、下がる")
    b, sent, spoken, quacks = make(cfg)
    fired = run(b, [(2.0, dict(person=True, closeness=0.0)), (82.0, dict(person=True, phone=True, closeness=0.0))])
    print("   ", fired)
    f2 = [t for t, e in fired if e == "F2"]
    ok &= check("F2 が 2 回、間隔は約 60 s", len(f2) == 2 and 59.5 <= f2[1] - f2[0] <= 66.0)
    ok &= check("phone 発話", any(k == "phone" for k, _ in spoken))
    walks = [m for m in sent if m.get("policy") == "walking"]
    ok &= check("前進(接近)してから後退", bool(walks) and walks[0]["twist"][0] == cfg["robot"]["forward_speed"]
                and any(m["twist"][0] < 0 for m in walks))

    print("case3: 疲労とスマホが同時成立 → F3 優先。F3 中は F2 が発火しない。疲労時は座る")
    cfg3 = copy.deepcopy(cfg)
    cfg3["events"]["fatigue_hold_s"] = cfg3["events"]["phone_hold_s"]
    b, sent, spoken, quacks = make(cfg3)
    fired = run(b, [(2.0, dict(person=True)), (40.0, dict(person=True, phone=True, fatigue=0.5))])
    print("   ", fired)
    ok &= check("F1 → F3 の順", [e for _, e in fired][:2] == ["F1", "F3"])
    ok &= check("疲労時に座る", any(m.get("policy") == "sitstand" and m.get("sit") for m in sent[1:]))
    ok &= check("F3 中は F2 なし", all(e != "F2" or t > fired[1][0] + 10 for t, e in fired))
    ok &= check("fatigue 発話", any(k == "fatigue" for k, _ in spoken))

    print("case4: 人がいなければ何も発火しない")
    b, sent, spoken, quacks = make(cfg)
    fired = run(b, [(30.0, dict(person=False, phone=True, fatigue=0.9))])
    ok &= check("発火なし", fired == [])

    print("case5: 0.5 秒おきに 0.3 秒の未検出があっても F1/F2 は成立し、発話まで完了する")
    b, sent, spoken, quacks = make(cfg)
    fired = run(b, [(2.0, dict(person=True)), (20.0, dict(person=True, phone=True))], flicker=(0.5, 0.3))
    print("   ", fired)
    ok &= check("F1 と F2 が発火", [e for _, e in fired] == ["F1", "F2"])
    ok &= check("greet と phone を発話", [k for k, _ in spoken] == ["greet", "phone"])

    print("case6: 接近の途中で人を完全に見失っても、発話までは進む")
    b, sent, spoken, quacks = make(cfg)
    # 0-2 s 人(遠い) → 2-14 s スマホ → 14 s 以降は完全に見失う（F2 は 12 s 頃に発火し、接近中に見失う）
    fired = run(b, [(2.0, dict(person=True, closeness=0.0)), (14.0, dict(person=True, phone=True, closeness=0.0)),
                    (26.0, dict(person=False))])
    print("   ", fired, [k for k, _ in spoken])
    ok &= check("F2 発火後に見失っても phone を発話", "F2" in [e for _, e in fired] and any(k == "phone" for k, _ in spoken))

    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
