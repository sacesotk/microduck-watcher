"""sim 環境の疎通確認: MuJoCo でシーンを読み、ONNX ポリシーの入出力形状を表示する。"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import mujoco, onnxruntime as ort, numpy as np
print("python", sys.version.split()[0], "mujoco", mujoco.__version__, "onnxruntime", ort.__version__, "numpy", np.__version__)
try:
    import bam; print("bam ok")
except Exception as e:
    print("bam NOT available:", e, "-> sim_runner は --no-bam で動かす")
xml = os.path.join(ROOT, "microduck_rl", "src", "mjlab_microduck", "robot", "microduck", "scene.xml")
m = mujoco.MjModel.from_xml_path(xml)
print("scene ok: nq", m.nq, "nu", m.nu, "actuators:", [m.actuator(i).name for i in range(m.nu)])
for f in ["alpha_walking", "alpha_stand", "alpha_sitstand", "velstand"]:
    s = ort.InferenceSession(os.path.join(ROOT, "policies", f + ".onnx"))
    i, o = s.get_inputs()[0], s.get_outputs()[0]
    print(f"{f}: {i.name}{i.shape} -> {o.name}{o.shape} meta={s.get_modelmeta().custom_metadata_map}")
print("OK")
