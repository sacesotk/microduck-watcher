# サードパーティのソフトウェア・モデル・音声

このリポジトリに含まれるのは本プロジェクトで書いたコードのみで、以下はセットアップスクリプトが取得する。
それぞれのライセンス・利用規約に従うこと。

| 名称 | 用途 | 取得元 | ライセンス |
|---|---|---|---|
| microduck_rl（Pollen Robotics） | MuJoCo モデル、`scripts/infer_policy.py`（無変更で読み込む） | github.com/pollen-robotics/microduck_rl | Apache-2.0（モデル資産は LICENSE-MODELS.md 参照） |
| microduck-policies（Pollen Robotics） | 学習済み歩行・起立・着座 ONNX | huggingface.co/pollen-robotics/microduck-policies | Apache-2.0 |
| Better Actuator Models (bam)（Rhoban） | XL330 アクチュエータモデル（`--bam` 時のみ） | github.com/Rhoban/bam | Apache-2.0 |
| MuJoCo（Google DeepMind） | 物理シミュレーション・描画 | PyPI `mujoco` | Apache-2.0 |
| ONNX Runtime（Microsoft） | ポリシー推論 | PyPI `onnxruntime` | MIT |
| Ultralytics YOLO11n | 実カメラ版の人・スマホ検出（`yolo11n.pt`） | PyPI `ultralytics` / GitHub releases | AGPL-3.0（商用は Enterprise License） |
| MediaPipe Face Landmarker（Google） | 実カメラ版の閉眼・あくび推定（`face_landmarker.task`） | PyPI `mediapipe` / storage.googleapis.com | Apache-2.0 |
| OpenCV | 画像処理・表示 | PyPI `opencv-python` | Apache-2.0 |
| VOICEVOX Engine / VOICEVOX 音声ライブラリ | 日本語音声合成 | github.com/VOICEVOX/voicevox_engine | Engine は LGPL-3.0。音声はキャラクターごとの利用規約に従い、生成音声を含む成果物に「VOICEVOX:<キャラクター名>」のクレジット表記が必要 |
| pyttsx3 | VOICEVOX が無い時の代替音声 | PyPI | MPL-2.0 |
| sounddevice / soundfile / PyYAML / requests / Pillow / NumPy | 補助ライブラリ | PyPI | 各ライブラリのライセンス |

本デモの音声は VOICEVOX の「青山龍星」（既定。config.yaml で変更可）で合成している。録画・公開物には画面右下と同じ
クレジット「VOICEVOX:青山龍星」を表記する。
