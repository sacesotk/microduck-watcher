"""手動テスト用: sim_runner に UDP/JSON を送る（仕様書 10 章「sim_runner.py の単体確認」）。
例:
  python sim/send_cmd.py walk 0.2 2        # 0.2 m/s で 2 秒前進
  python sim/send_cmd.py back 0.2 2        # 後退
  python sim/send_cmd.py turn 0.5 2        # 旋回 0.5 rad/s
  python sim/send_cmd.py stop
  python sim/send_cmd.py sit  /  stand
  python sim/send_cmd.py head 0.5          # 首を 0.5 rad
  python sim/send_cmd.py raw "{\"policy\":\"walking\",\"twist\":[0.1,0,0],\"duration\":1}"
  python sim/send_cmd.py demo              # 立つ→前進→停止→座る→立つ→後退 を順に流す
"""
import json
import socket
import sys
import time

HOST, PORT = "127.0.0.1", 5005


def send(d: dict):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(json.dumps(d).encode(), (HOST, PORT))
    print("sent", d)


def main(a):
    if not a:
        print(__doc__)
        return
    c = a[0]

    def f(i, d):
        return float(a[i]) if len(a) > i else d

    # 既定値はポリシーのデッドバンドより上（前進 0.3 / 後退 0.4 / 旋回 1.0）
    if c == "walk":
        send({"policy": "walking", "twist": [f(1, 0.3), 0, 0], "duration": f(2, 2.0)})
    elif c == "back":
        send({"policy": "walking", "twist": [-f(1, 0.4), 0, 0], "duration": f(2, 1.5)})
    elif c == "turn":
        send({"policy": "walking", "twist": [0, 0, f(1, 1.0)], "duration": f(2, 2.0)})
    elif c == "stop":
        send({"policy": "stop"})
    elif c == "reset":
        send({"policy": "reset"})      # 転倒したときに初期の立ち姿勢へ戻す
    elif c == "scene":                 # 例: scene person=1 phone=1 eyes=0
        m = {"policy": "scene"}
        for kv in a[1:]:
            k, v = kv.split("=")
            m[{"eyes": "eyes_closed"}.get(k, k)] = v not in ("0", "false", "False")
        send(m)
    elif c == "sit":
        send({"policy": "sitstand", "sit": True})
    elif c == "stand":
        send({"policy": "sitstand", "sit": False})
    elif c == "head":
        send({"policy": "standing", "head_yaw": f(1, 0.5)})
    elif c == "raw":
        send(json.loads(a[1]))
    elif c == "demo":
        for d, wait in [
            ({"policy": "sitstand", "sit": False}, 4),
            ({"policy": "walking", "twist": [0.3, 0, 0], "duration": 3.0}, 4),
            ({"policy": "standing", "head_yaw": 0.6}, 2),
            ({"policy": "sitstand", "sit": True}, 4),
            ({"policy": "sitstand", "sit": False}, 4),
            ({"policy": "walking", "twist": [-0.4, 0, 0], "duration": 1.5}, 3),
            ({"policy": "stop", "head_yaw": 0.0}, 0),
        ]:
            send(d)
            time.sleep(wait)
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
