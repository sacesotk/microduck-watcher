# Microduck 見守り・声かけシミュレーション（このPC向け構成）

仕様書: `docs/（非公開のためリポジトリには含めない）Microduck 見守り・声かけシミュレーション 仕様書.pdf`（2026-11-01）

Web カメラで実在のユーザーを認識し、MuJoCo 上の Microduck が「挨拶 (F1)」「スマホ注意 (F2)」
「疲労への声かけ (F3)」を返すデモ。知覚は実カメラ、ロボットはシミュレーション、声は PC スピーカー。

## このPCに合わせた判断

| 項目 | 仕様書の想定 | このPCでの構成 | 理由 |
|---|---|---|---|
| OS | Ubuntu 22.04/24.04 | **Windows 11 ネイティブで 3 プロセスすべて動かす** | カメラ・UDP・MuJoCo ビューアが最も単純。WSL2 はカメラ共有(usbipd)と UDP 経路が増えるだけで利点がない |
| sim 環境 | `uv sync`（CUDA 系 約 2 GB） | `sim/.venv`: **mujoco 3.10.0 + onnxruntime + bam** だけ（公式 uv.lock と同じ版） | 推論は CPU 版 MuJoCo で十分。仕様書 11 章の「最小環境」案を最初から採用 |
| infer_policy.py | 複製してキー入力を差し替え | **複製せず** `sim/sim_runner.py` がモジュールとして読み込み、`TerminalInput`(termios, Unix 専用) だけを UDP 受信クラスに差し替え | 公式リポジトリは無変更のまま更新に追従できる |
| 学習済みポリシー | `pretrained/pollen-robotics/` | リポジトリ(develop)には同梱されていないため **Hugging Face `pollen-robotics/microduck-policies`** から `policies/` に取得 | Apache-2.0。`velstand.onnx`（転倒から起き上がれる）も予備として取得済み |
| Python | 3.10 以上 | **3.12**（両 venv とも） | bam と公式 pyproject が `>=3.12,<3.13`。mediapipe 0.10 系の Windows wheel も 3.12 まで |
| 音声再生 | simpleaudio | **sounddevice + soundfile** | simpleaudio は Windows/py3.12 用 wheel が無くビルドが必要 |
| 音声合成 | VOICEVOX Engine | VOICEVOX Engine 0.26.0 **Windows CPU 版**を `tools/voicevox/engine/` に展開 | GPU 不要。使えない時は pyttsx3 に自動フォールバック |
| GPU | 不要 | 使わない（torch は CPU 版） | RTX 3060 Laptop 6 GB はあるが、YOLO11n + MediaPipe は CPU で 10 fps 以上出る見込み |
| head_pose の形式 | README に記載なし | `infer_policy.py` で確認: **角度 [rad]、並びは [neck_pitch, head_pitch, head_yaw, head_roll]**、上限 ±1.4 | 首の向きはそのまま使える（体ごと旋回させる代替は不要） |

PC: Ryzen 7 6800HS (8C/16T), RAM 15 GB, RTX 3060 Laptop 6 GB, Windows 11 25H2, 内蔵 UVC Web カメラ (HD + IR)。

## ディレクトリ

```
microduck-watcher/
├── microduck_rl/            # 公式リポジトリ (develop, shallow clone) — 変更しない。.gitignore 済み
├── policies/                # alpha_walking / alpha_stand / alpha_sitstand / velstand .onnx + manifest.json
├── sim/
│   ├── sim_runner.py        # infer_policy.py を読み込み、キー入力を UDP/JSON 受信に差し替え
│   ├── send_cmd.py          # 手動 UDP テスト (walk/back/turn/stop/sit/stand/head/demo)
│   ├── check_env.py         # シーンと ONNX の読み込み確認
│   ├── requirements.txt
│   └── .venv/
├── watcher/
│   ├── main.py              # カメラ取得・全体ループ・画面表示・CSV ログ
│   ├── perception.py        # YOLO11n: person / cell phone
│   ├── fatigue.py           # MediaPipe Face Landmarker: PERCLOS・あくび・うつむき → 疲労スコア S
│   ├── behavior.py          # 状態遷移・優先度 (F3>F2>F1)・クールダウン・反応シーケンス
│   ├── speech.py            # VOICEVOX (HTTP) / pyttsx3、鳴き声再生、起動時プリキャッシュ
│   ├── sim_bridge.py        # UDP 送信
│   ├── tests/test_behavior.py   # 偽イベント列で状態遷移を確認（sim/音声不要）
│   ├── requirements.txt
│   └── .venv/
├── assets/
│   ├── quack.wav            # 合成した仮の鳴き声（差し替え可）
│   └── models/              # yolo11n.pt, face_landmarker.task（追加学習なし）
├── tools/voicevox/engine/   # VOICEVOX Engine（.gitignore 済み）
├── scripts/                 # setup_*.ps1, run_*.ps1, run_all.ps1
├── config.yaml              # 閾値・時間・発話文（すべてここ）
└── logs/                    # 時刻・検出有無・スコア・状態の CSV（画像は保存しない）
```

## セットアップ（済みのものは再実行不要）

```powershell
cd <clone した場所>\microduck-watcher
.\scripts\setup_sim.ps1        # clone + sim/.venv + policies/  → check_env.py が OK を出す
.\scripts\setup_watcher.ps1    # watcher/.venv + モデル + quack.wav → test_behavior.py が PASS
.\scripts\setup_voicevox.ps1   # VOICEVOX Engine 1.9 GB の取得と展開
```

## 起動（3 ターミナル）

cmd.exe からは同名の `.bat`（例 `scripts\run_sim.bat`）を使う。`.ps1` は PowerShell 専用で、cmd から叩いても実行されない。

```powershell
.\scripts\run_voicevox.ps1     # ① http://127.0.0.1:50021/version で JSON が返れば OK
.\scripts\run_sim.ps1          # ② MuJoCo ビューアが開き "UDP listening on 127.0.0.1:5005"
.\scripts\run_watcher.ps1      # ③ カメラ画面に枠・スコア・状態が重なる。q / ESC で終了
```

**まとめて起動**: `scripts\run_all.bat`（VOICEVOX → sim → デモを順に起動し、起動済みなら再利用）。
`--behind` / `--loop` などはそのままデモに渡る。`--camera` を付けると実カメラ版 watcher/main.py を起動する。

### カメラなしデモ（録画用）

```powershell
.\scripts\run_demo.ps1           # ③の代わりに。偽の知覚イベント列で F1→F2→F3 を自動再生（約 75 秒）
.\scripts\run_demo.ps1 --loop    # 繰り返し
.\scripts\run_demo.ps1 --manual  # キー操作のみ（p:人 s:スマホ f:疲労 a:自動再生 r:リセット q:終了）
```

`watcher/demo.py` の流れ:
台本 → シミュレーション内の人の見た目（在/不在・スマホ・目・口）→ **ダック頭部カメラの映像**（`watcher/pov.py`:
sim_runner が UDP :5006 で配信する qpos を受け、同じシーンを頭部カメラ位置から描画）→ **知覚**
（`watcher/sim_perception.py`: 色ベースで人・顔・目の開閉・口の開き・スマホを検出）→ 疲労スコア（`fatigue.FatigueScorer`、
実カメラ版と同じ式）→ 状態遷移（behavior.py、実カメラ版と共通）→ sim への指令と音声。
台本は見た目を変えるだけで、F1/F2/F3 の発火はダック視点の画像の判定結果で決まる。
YOLO11n / MediaPipe は単純形状の人形を認識できないため（2026-11-01 確認）、sim 用の知覚だけ色ベースにしている。
人がいない間は 15 秒ごと（`robot.scan_interval_s`）に首を左右に振って探し、60 秒以上見当たらなければ
（`robot.roam_after_s`）立ち上がって「旋回 → 少し前進」を繰り返しながら最大 36 秒歩いて探し回る（ROAM）。
人を見つければ探索を中断して挨拶に移る。見つからなければ停止して座り、さらに 60 秒後にまた探す。
`run_demo --behind`（またはデモ画面で `b`）で人をダックの**背後**に置ける。ダックには見えないので
首振り → 60 秒後に歩き回り → 旋回中にカメラが人を捉える → F1、という「探して見つける」流れになる。
自動シナリオは F1 発火後の経過時間で進む（スマホ +9 s、疲労 +27 s、退席 +63 s）。
**反応は一気通貫**（behavior.py のフェーズ機械）: `[座っていれば立つ] → [遠ければ近づく] → 声かけ → [F2/F3 は下がる]`。
- 近さは呼び出し側がカメラごとに 0..1 に正規化して渡す（Observation.closeness: 0 以下 = 遠い → 近づく、1 以上 = 十分近い → 止まる）。
  sim 版は顔枠中心の画面内高さで校正（`sim_perception.far_cy/near_cy`: 腰まで 1.05 m 以上で遠い、0.95 m で停止 = 足先から約 0.4 m）。
  実カメラ版は顔枠の高さ（`perception.far_face_h/near_face_h`）。距離不明（顔が取れない）なら `approach_s` だけ前進。
  遠ければ近くなるまで最大 `approach_max_s` 前進し、人が画面中央から外れたら体ごと向き直す（`steer_x_thresh`, `steer_wz`）。
  人（腰）はダックから 1.2 m（`demo_scene.PERSON_X`）に置く。0.85 m だと足先が 0.3 m しか離れておらず近づきすぎた。
- いったん発火した反応は、途中で人を見失っても発話まで必ず進む（接近だけは `lost_tolerance_s` 以上見失うと打ち切る）。
- `lost_tolerance_s`（1.5 s）以内の未検出では在席・スマホ・疲労のタイマーを切らさない。
- 近距離でシャツが膝に隠れてもズボン（灰色）で人を追い続ける。反応後に見失っていれば 1 秒後に首振りで探し直す。
F1 も遠ければ近づいてから挨拶する（仕様書は「立つ → 向く → 鳴く → 発話」だが、ユーザー要望で接近を追加）。
自動シナリオの実測: F1 3.0 s, F2 17.1 s, F3 46.1 s（S=0.48）。

sim 側は `sim/demo_scene.py` で **椅子に座る人**（見た目だけ、衝突なし）をダックの正面 1 m に実行時追加する
（公式 scene.xml は無変更）。demo の進行に合わせて `{"policy":"scene", person/phone/eyes_closed}` で
人の在/不在・手元のスマホ・目の開閉が切り替わる。人が不要なら `run_sim --no-person`。
公式モデルの `<camera name="head_camera">` は頭の内部を向いているため、POV はくちばし方向を前方とする自由カメラで描く。

## 単体確認（仕様書 10 章）

| モジュール | 方法 |
|---|---|
| sim_runner.py | ②を起動した状態で `scripts\send_cmd.bat demo`（= `sim\.venv\Scripts\python.exe sim\send_cmd.py demo`）（立つ→前進→首→座る→立つ→後退） |
| perception.py / fatigue.py | `.\scripts\run_watcher.ps1 --no-sim --no-audio` で枠・信頼度・S の内訳を目視。閉眼・あくびで S が上がること |
| behavior.py | `watcher\.venv\Scripts\python.exe watcher\tests\test_behavior.py` |
| speech.py | ①を起動後 `watcher\.venv\Scripts\python.exe -c "import sys,yaml;sys.path.insert(0,'watcher');from speech import Speech;print(Speech(yaml.safe_load(open('config.yaml',encoding='utf-8'))).test_all())"` → 各 1 秒未満 |
| fps 計測 | `.\scripts\run_watcher.ps1 --no-sim --no-audio --no-window --duration 20` |

## UDP/JSON（watcher → sim, localhost:5005）

```json
{"policy": "walking",  "twist": [0.2, 0.0, 0.0], "head_yaw": 0.3, "duration": 2.0}
{"policy": "standing", "head_yaw": 0.3}
{"policy": "sitstand", "sit": true}
{"policy": "stop"}
```

`policy` は walking / standing / sitstand / stop / reset / scene。`duration` 経過で速度 0。
受信側（sim の指令 :5005、demo の姿勢 :5006）は `sim/udp_schema.py` で厳密に検証する:
ループバック以外の送信元・8 KiB 超・UTF-8/JSON として不正・`NaN`/`Infinity`・辞書でないトップレベル・未知の policy・
型違い（文字列や bool の数値）・twist の長さ違い・qpos の長さ違いや非有限値はすべて捨て、数値は範囲内にクリップする。
テスト: `watcher\tests\test_udp_schema.py`。
`sitstand` は `sit` フラグで座る/立つ（同じポリシーが約 2 秒で遷移）。walking/standing/stop を送ると座っていれば先に立つ。

## 歩行ポリシーの制約（2026-11-01 計測）

`sim/diag_minimal.py`（公式スクリプト不使用、mujoco + onnxruntime のみ）で指令速度を掃引した結果:

| 指令 | alpha_walking | velstand |
|---|---|---|
| 前進 0.20 m/s | 歩かない | 0.08 m/s |
| 前進 0.25 / 0.30 / 0.40 m/s | 0.12 / 0.13 / 0.18 m/s | 0.10 / 0.11 / 0.15 m/s |
| 後退 0.15〜0.30 m/s | 歩かない | 歩かない |
| 後退 0.40 m/s | 0.23 m/s | 0.19 m/s |
| 旋回 0.5 rad/s | 回らない | — |
| 旋回 1.0 rad/s | 0.43 rad/s | — |

ポリシーに静止デッドバンドがあり、仕様書の「前進 0.2 m/s 以下」では歩かない。
そのため `config.yaml` は forward_speed 0.3 / backward_speed 0.4 を既定にした。
また `run_sim` の既定は `--no-bam`（XML 位置アクチュエータ）。BAM 有効時は着座からの起立と後退が不安定で、
XML アクチュエータは公式 docs の sim 評価値（0.4 m/s 指令 → 0.17 m/s）を再現したため。BAM に戻すには `--bam`。

切り分け用スクリプト: `scripts\diag_walk.bat`（公式経路・組み合わせ比較）、`scripts\diag_trace.bat`（閉ループ詳細）、
`sim\.venv\Scripts\python.exe sim\diag_minimal.py`（最小実装）。

## 調整のポイント

- **カメラ番号**: IR カメラが 0 に来る場合は `config.yaml` の `camera.index` を 1 に。
- **疲労の閾値**: 自分の顔で「普通」「目を閉じ気味・あくび」を各 1 分記録し、`fatigue.*` を調整。画面に eye/jaw/pitch の生値が出る。
  うつむきの符号が逆なら `droop_pitch_sign: -1`。
- **首の向き**: 画面右の人に対して首が左へ向くなら `robot.head_yaw_sign` を反転。
- **転倒する**: `run_sim --velstand`（転倒から起き上がれる velstand を歩行・静止に使う）。`scripts\send_cmd.bat reset` で初期姿勢に戻せる。速度を下げるのは逆効果（0.25 m/s 未満では歩かない）。
- **ビューアが開かない**: `run_sim.ps1` は `MUJOCO_GL=glfw` 済み。`--no-bam` で軽くもできる。
- **VOICEVOX クレジット**: 画面右下に `speech.credit`（既定 "VOICEVOX:ずんだもん"）を表示。録画・公開物にも表記する。

## プライバシー

映像は PC 内のみで処理。画像は保存せず、`logs/*.csv` には時刻・検出有無・スコア・状態だけを書く。
疲労スコアは目と口の動きから作った目安で、医学的評価ではない。
