"""UDP/JSON 検証の単体テスト（sim/udp_schema.py）。sim・音声・カメラは不要。

実行:  watcher\\.venv\\Scripts\\python.exe watcher\\tests\\test_udp_schema.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "sim"))
from udp_schema import (MAX_DATAGRAM, Rejected, loads_strict, validate_command,  # noqa: E402
                        validate_state)

ok = True


def check(name, cond):
    global ok
    print(("  ok   " if cond else "  FAIL ") + name)
    ok = ok and bool(cond)


def rejected(fn, *a, **kw):
    try:
        fn(*a, **kw)
        return False
    except Rejected:
        return True


LOOP = ("127.0.0.1", 50000)
print("loads_strict")
check("正常な辞書を受理", loads_strict(b'{"policy":"stop"}', LOOP) == {"policy": "stop"})
check("ループバック以外の送信元は拒否", rejected(loads_strict, b'{"policy":"stop"}', ("192.168.1.5", 1)))
check("空データは拒否", rejected(loads_strict, b"", LOOP))
check("上限超のサイズは拒否", rejected(loads_strict, b'{"a":"' + b"x" * MAX_DATAGRAM + b'"}', LOOP))
check("UTF-8 でないものは拒否", rejected(loads_strict, b"\xff\xfe{}", LOOP))
check("壊れた JSON は拒否", rejected(loads_strict, b'{"policy":', LOOP))
check("配列（辞書でない）は拒否", rejected(loads_strict, b"[1,2,3]", LOOP))
check("文字列（辞書でない）は拒否", rejected(loads_strict, b'"stop"', LOOP))
check("NaN は拒否", rejected(loads_strict, b'{"qpos":[NaN]}', LOOP))
check("Infinity は拒否", rejected(loads_strict, b'{"head_yaw":-Infinity}', LOOP))

print("validate_command")
c = validate_command({"policy": "walking", "twist": [0.3, 0, 0], "head_yaw": 0.3, "duration": 2.0})
check("walking を正規化", c["policy"] == "walking" and c["twist"] == [0.3, 0.0, 0.0] and c["head_yaw"] == 0.3 and c["duration"] == 2.0)
check("速度は上限にクリップ", validate_command({"policy": "walking", "twist": [9, 9, 9]})["twist"] == [0.4, 0.2, 1.5])
check("duration は上限にクリップ", validate_command({"policy": "walking", "twist": [0.3, 0, 0], "duration": 1e9})["duration"] == 30.0)
check("未知の policy は拒否", rejected(validate_command, {"policy": "selfdestruct"}))
check("policy が文字列でなければ拒否", rejected(validate_command, {"policy": 1}))
check("twist の長さ違いは拒否", rejected(validate_command, {"policy": "walking", "twist": [0.3]}))
check("twist に文字列は拒否", rejected(validate_command, {"policy": "walking", "twist": ["0.3", 0, 0]}))
check("twist に bool は拒否", rejected(validate_command, {"policy": "walking", "twist": [True, 0, 0]}))
check("twist に nan は拒否", rejected(validate_command, {"policy": "walking", "twist": [float("nan"), 0, 0]}))
check("head_yaw に inf は拒否", rejected(validate_command, {"policy": "standing", "head_yaw": float("inf")}))
check("head_yaw はクリップ", validate_command({"policy": "standing", "head_yaw": 5})["head_yaw"] == 1.4)
check("sitstand の sit は bool", validate_command({"policy": "sitstand", "sit": False})["sit"] is False)
check("sit に文字列は拒否", rejected(validate_command, {"policy": "sitstand", "sit": "yes"}))
check("scene の side は front/back のみ", rejected(validate_command, {"policy": "scene", "side": "left"}))
check("scene のフラグ", validate_command({"policy": "scene", "person": True, "phone": 0})["scene"] == {"person": True, "phone": False})
check("未知のキーは無視", "foo" not in validate_command({"policy": "stop", "foo": 1}))

print("validate_state")
q = [0.0] * 21
check("正しい長さの qpos を受理", validate_state({"qpos": q}, 21) == q)
check("長さ違いは拒否", rejected(validate_state, {"qpos": q[:-1]}, 21))
check("qpos 無しは拒否", rejected(validate_state, {"t": 1.0}, 21))
check("qpos が配列でなければ拒否", rejected(validate_state, {"qpos": "0,0,0"}, 21))
check("文字列要素は拒否", rejected(validate_state, {"qpos": ["0"] + q[1:]}, 21))
check("bool 要素は拒否", rejected(validate_state, {"qpos": [True] + q[1:]}, 21))
check("nan 要素は拒否", rejected(validate_state, {"qpos": [float("nan")] + q[1:]}, 21))
check("巨大な値は拒否", rejected(validate_state, {"qpos": [1e6] + q[1:]}, 21))
check("入れ子の配列は拒否", rejected(validate_state, {"qpos": [[0.0]] + q[1:]}, 21))

print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
