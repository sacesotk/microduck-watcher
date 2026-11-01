"""音声: VOICEVOX Engine (HTTP :50021) → 失敗時は pyttsx3。鳴き声 wav の再生（仕様書 6 章）。

- 再生はすべて別スレッドで行い、メインの知覚ループを止めない。
- prewarm=True なら起動時に全発話文を合成してメモリに保持し、発火から 1 秒以内に再生を始める。
- VOICEVOX を使う場合、録画・公開物には config の credit（例 "VOICEVOX:ずんだもん"）を表記する。
"""
from __future__ import annotations

import io
import queue
import threading
import time
from typing import Optional

import numpy as np


def _play_wav_bytes(wav_bytes: bytes, blocking: bool = True):
    import sounddevice as sd
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    sd.play(data, sr)
    if blocking:
        sd.wait()


class Speech:
    def __init__(self, cfg: dict):
        s = cfg["speech"]
        self.url = s["voicevox_url"].rstrip("/")
        self.speaker = int(s.get("speaker_id", 3))
        self.credit = s.get("credit", "")
        self.quack_path = s.get("quack_wav", "assets/quack.wav")
        self.phrases = s["phrases"]
        self.cache: dict[str, bytes] = {}
        self.backend = "none"
        self.q: queue.Queue = queue.Queue()
        self.busy = False
        self.last_error: Optional[str] = None
        self._quack_bytes: Optional[bytes] = None
        try:
            with open(self.quack_path, "rb") as f:
                self._quack_bytes = f.read()
        except OSError as e:
            self.last_error = f"quack.wav not found: {e}"

        self.backend = "voicevox" if self._voicevox_alive() else "pyttsx3"
        print(f"[speech] backend = {self.backend}")
        if self.backend == "voicevox" and s.get("prewarm", True):
            self.prewarm()
        threading.Thread(target=self._worker, daemon=True).start()

    # ---------------------------------------------------------- VOICEVOX
    def _voicevox_alive(self) -> bool:
        try:
            import requests
            r = requests.get(self.url + "/version", timeout=1.5)
            return r.ok
        except Exception as e:
            self.last_error = f"VOICEVOX not reachable: {e}"
            return False

    def synth_voicevox(self, text: str) -> bytes:
        import requests
        q = requests.post(self.url + "/audio_query", params={"text": text, "speaker": self.speaker}, timeout=10)
        q.raise_for_status()
        r = requests.post(self.url + "/synthesis", params={"speaker": self.speaker},
                          json=q.json(), timeout=30)
        r.raise_for_status()
        return r.content

    def prewarm(self):
        t0 = time.time()
        n = 0
        for key, texts in self.phrases.items():
            for t in texts:
                if t not in self.cache:
                    try:
                        self.cache[t] = self.synth_voicevox(t)
                        n += 1
                    except Exception as e:
                        self.last_error = f"prewarm failed: {e}"
                        print("[speech]", self.last_error)
        print(f"[speech] prewarmed {n} phrases in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------ 再生
    def _worker(self):
        while True:
            kind, payload = self.q.get()
            self.busy = True
            try:
                if kind == "wav":
                    _play_wav_bytes(payload)
                elif kind == "text":
                    self._speak_blocking(payload)
            except Exception as e:
                self.last_error = f"play failed: {e}"
                print("[speech]", self.last_error)
            finally:
                self.busy = False
                self.q.task_done()

    def _speak_blocking(self, text: str):
        if self.backend == "voicevox":
            wav = self.cache.get(text)
            if wav is None:
                try:
                    wav = self.synth_voicevox(text)
                    self.cache[text] = wav
                except Exception as e:
                    self.last_error = f"VOICEVOX synth failed, fallback pyttsx3: {e}"
                    print("[speech]", self.last_error)
                    wav = None
            if wav is not None:
                _play_wav_bytes(wav)
                return
        import pyttsx3
        eng = pyttsx3.init()
        # 日本語音声 (Microsoft Haruka/Ayumi/Sayaka 等) があれば選ぶ
        for v in eng.getProperty("voices"):
            if "ja" in str(getattr(v, "languages", "")).lower() or "Haruka" in v.name or "Ayumi" in v.name or "Sayaka" in v.name:
                eng.setProperty("voice", v.id)
                break
        eng.say(text)
        eng.runAndWait()

    # -------------------------------------------------------------- API
    def say(self, text: str):
        """非同期。発話開始までの遅延は VOICEVOX キャッシュ済みなら数十 ms。"""
        self.q.put(("text", text))

    def quack(self):
        if self._quack_bytes:
            self.q.put(("wav", self._quack_bytes))

    def test_all(self) -> dict:
        """3 種の発話がそれぞれ 1 秒以内に再生を始めるかを計測する（仕様書 10 章）。"""
        import sounddevice as sd
        res = {}
        for key, texts in self.phrases.items():
            t0 = time.perf_counter()
            wav = self.cache.get(texts[0])
            if wav is None and self.backend == "voicevox":
                wav = self.synth_voicevox(texts[0])
            if wav is not None:
                import soundfile as sf
                data, sr = sf.read(io.BytesIO(wav), dtype="float32")
                sd.play(data, sr)
                res[key] = time.perf_counter() - t0
                sd.wait()
            else:
                self._speak_blocking(texts[0])
                res[key] = time.perf_counter() - t0
        return res


def make_quack_wav(path: str, sr: int = 22050):
    """合成のアヒル風効果音（素材が無い時の代替）。assets/quack.wav を生成する。"""
    import soundfile as sf
    t = np.linspace(0, 0.18, int(sr * 0.18), endpoint=False)
    f0 = 420 * (1 - 0.35 * t / t[-1])                 # 下降するピッチ
    sig = np.zeros_like(t)
    for k, a in [(1, 1.0), (2, 0.6), (3, 0.45), (4, 0.3), (5, 0.2), (6, 0.12)]:
        sig += a * np.sin(2 * np.pi * k * np.cumsum(f0) / sr)
    sig *= (1 + 0.6 * np.sin(2 * np.pi * 70 * t))     # ガラガラ感
    env = np.minimum(1, t / 0.015) * np.exp(-6 * t)
    one = sig * env
    gap = np.zeros(int(sr * 0.07))
    out = np.concatenate([one, gap, one * 0.85])
    out = 0.6 * out / np.max(np.abs(out))
    sf.write(path, out.astype(np.float32), sr)
    return path
