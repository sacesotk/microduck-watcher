"""閉ループの詳細トレース（ビューアなし）。歩行指令中と着座指令中に、観測・行動・足の高さ・体幹の姿勢を
0.25 秒ごとに logs/diag_trace.txt へ書く。diag_walk で「全部動かない」場合の次の切り分け用。

実行: scripts\\diag_trace.bat
"""
from __future__ import annotations

import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_DIR = os.path.join(ROOT, "microduck_rl")
POLICY_DIR = os.path.join(ROOT, "policies")
REPORT = os.path.join(ROOT, "logs", "diag_trace.txt")

try:
    import termios  # noqa: F401
except ImportError:
    for _n in ("termios", "tty"):
        sys.modules[_n] = types.ModuleType(_n)
sys.path.insert(0, os.path.join(RL_DIR, "scripts"))
os.chdir(RL_DIR)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import infer_policy as ip  # noqa: E402

np.set_printoptions(precision=3, suppress=True, linewidth=200)
LINES = []


def log(s=""):
    print(s, flush=True)
    LINES.append(s)


def build(walk, stand, use_bam):
    xml = ip.MICRODUCK_XML
    if use_bam:
        bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, 0.0)
        model, data, bam_ctrl, _ = ip.load_mujoco_with_bam(xml, bam_model, 0.005, 0.1, ip.BAM_VIN_MIN)
    else:
        model = mujoco.MjModel.from_xml_path(xml)
        model.opt.timestep = 0.005
        data = mujoco.MjData(model)
        bam_ctrl = None
    pol = ip.PolicyInference(
        model, data, bam_ctrl=bam_ctrl,
        walking_onnx_path=os.path.join(POLICY_DIR, walk),
        standing_onnx_path=os.path.join(POLICY_DIR, stand),
        sitstand_onnx_path=os.path.join(POLICY_DIR, "alpha_sitstand.onnx"),
        new_cmd_obs=True, use_projected_gravity=True, switch_threshold=0.05,
    )
    pol.set_vel_cmd(0.0, 0.0, 0.0)
    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qadr = model.jnt_qposadr[fj]
    data.qpos[qadr:qadr + 3] = [0.0, 0.0, 0.125]
    data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
    for i, qi in enumerate(pol.joint_qpos_indices):
        data.qpos[qi] = pol.default_pose[i]
    if bam_ctrl is not None:
        bam_ctrl.reset(data.qpos)
    pol.set_position_targets(pol.default_pose)
    mujoco.mj_forward(model, data)
    lf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    rf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    return model, data, pol, bam_ctrl, qadr, lf, rf


def trace(model, data, pol, bam_ctrl, qadr, lf, rf, seconds, label, every=0.25):
    log(f"--- {label} ---")
    log("   t   | angvel(3) gravity(3) | cmd(48:51) | Ljoints rel(5) | action L(5) | ctrl/torque L(5) | footz L/R mm | trunk z mm pitch deg | ncon")
    n = int(seconds * 50)
    for k in range(n):
        obs = pol.get_observations()
        a = pol.infer()
        pol.apply_action(a)
        for _ in range(4):
            if bam_ctrl is not None:
                bam_ctrl.update()
            mujoco.mj_step(model, data)
        if k % int(every * 50) == 0:
            qw, qx, qy, qz = data.qpos[qadr + 3:qadr + 7]
            pitch = np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1)))
            log(f"{k/50:6.2f} | {obs[0:3]} {obs[3:6]} | {obs[48:51]} | {obs[6:11]} | {a[0:5]} | {data.ctrl[0:5]} | "
                f"{data.site_xpos[lf][2]*1000:5.1f}/{data.site_xpos[rf][2]*1000:5.1f} | "
                f"{data.qpos[qadr+2]*1000:5.1f} {pitch:+6.1f} | {data.ncon}")
    log()


def main():
    log(f"diag_trace mujoco={mujoco.__version__} ort={__import__('onnxruntime').__version__} numpy={np.__version__}")
    log(f"gravity opt: {ip.mujoco.MjModel.from_xml_path(ip.MICRODUCK_XML).opt.gravity}  timestep after load: see below")
    for use_bam in (True, False):
        tag = "BAM" if use_bam else "no-bam"
        m, d, pol, bc, qadr, lf, rf = build("alpha_walking.onnx", "alpha_stand.onnx", use_bam)
        log(f"[{tag}] timestep={m.opt.timestep} gravity={m.opt.gravity} actuator gainprm[0]={m.actuator_gainprm[0,:3]} "
            f"biasprm[0]={m.actuator_biasprm[0,:3]} forcerange[0]={m.actuator_forcerange[0]} ctrlrange[0]={m.actuator_ctrlrange[0]} "
            f"dyntype={m.actuator_dyntype[0]} gaintype={m.actuator_gaintype[0]} biastype={m.actuator_biastype[0]}")
        trace(m, d, pol, bc, qadr, lf, rf, 1.0, f"{tag} alpha stand (cmd 0)", every=0.25)
        pol.set_vel_cmd(0.3, 0.0, 0.0)
        trace(m, d, pol, bc, qadr, lf, rf, 4.0, f"{tag} alpha walk vx=0.3", every=0.25)
        pol.set_vel_cmd(0.0, 0.0, 0.0)
        pol.toggle_sit()
        trace(m, d, pol, bc, qadr, lf, rf, 4.0, f"{tag} sitstand sit=1", every=0.5)
        pol.toggle_sit()
        trace(m, d, pol, bc, qadr, lf, rf, 4.0, f"{tag} sitstand sit=0 (stand up)", every=0.5)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    log(f"report -> {REPORT}")


if __name__ == "__main__":
    main()
