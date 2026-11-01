"""公式スクリプトを使わない最小実装での歩行確認（mujoco + onnxruntime のみ）。

infer_policy.py と同じ観測（gyro, 体幹の投影重力, 関節角-既定姿勢, 関節速度, 前回行動, 13 次元指令）を
自前で組み、アクチュエータだけを
  xml      : scene.xml の position アクチュエータ (kp=0.55)
  pd_K     : トルクモータ + 自前 PD (kp=K Nm/rad, 力上限 ±0.96 Nm)  ※BAM の実効剛性の見積り ~1.1
で動かし、歩くかどうかを見る。公式スクリプトの配線か、物理/アクチュエータ側かを切り分ける。
"""
from __future__ import annotations

import os
import sys

import mujoco
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(ROOT, "microduck_rl", "src", "mjlab_microduck", "robot", "microduck", "scene.xml")
POL = os.path.join(ROOT, "policies")
DEFAULT_POSE = np.array([0.0, -0.0873, -0.4579, -0.0049, 0.453, 0.3491, 0.3491, 0.0, 0.0,
                         0.0, 0.0873, 0.4579, 0.0049, -0.453], dtype=np.float32)
FORCE_LIM = 0.963


def quat_rot_inv(q, v):
    w, xyz = q[0], q[1:4]
    t = np.cross(xyz, v) * 2
    return v - w * t + np.cross(xyz, t)


def build(actuator: str):
    if actuator == "xml":
        model = mujoco.MjModel.from_xml_path(XML)
    else:
        spec = mujoco.MjSpec.from_file(XML)
        for act in spec.actuators:
            act.set_to_motor()
            act.forcelimited = True
            act.forcerange = (-FORCE_LIM, FORCE_LIM)
            act.ctrllimited = False
            act.gear = [1.0, 0, 0, 0, 0, 0]
        model = spec.compile()
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    return model, data


def run(policy_file: str, actuator: str, vx: float, seconds: float = 6.0, kp: float = 1.1, kd: float = 0.02):
    model, data = build(actuator)
    sess = ort.InferenceSession(os.path.join(POL, policy_file))
    inp = sess.get_inputs()[0].name
    nu = model.nu
    qpos_idx = [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(nu)]
    qvel_idx = [int(model.jnt_dofadr[model.actuator_trnid[i, 0]]) for i in range(nu)]
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    gyro = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")]
    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qadr, vadr = model.jnt_qposadr[fj], model.jnt_dofadr[fj]
    lf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    rf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    data.qpos[qadr:qadr + 3] = [0, 0, 0.125]
    data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
    for i, qi in enumerate(qpos_idx):
        data.qpos[qi] = DEFAULT_POSE[i]
    mujoco.mj_forward(model, data)

    last_action = np.zeros(nu, np.float32)
    cmd = np.zeros(13, np.float32)
    target = DEFAULT_POSE.copy()

    def obs():
        q = data.xquat[trunk].astype(np.float32)
        return np.concatenate([
            data.sensordata[gyro:gyro + 3].astype(np.float32),
            quat_rot_inv(q, np.array([0, 0, -1], np.float32)),
            data.qpos[qpos_idx].astype(np.float32) - DEFAULT_POSE,
            data.qvel[qvel_idx].astype(np.float32),
            last_action, cmd]).astype(np.float32)

    fwd, footz, acts, fell = [], [], [], None
    n_stand = 100
    n = n_stand + int(seconds * 50)
    for k in range(n):
        cmd[0] = vx if k >= n_stand else 0.0
        a = sess.run(None, {inp: obs().reshape(1, -1)})[0].ravel().astype(np.float32)
        last_action = a.copy()
        target = DEFAULT_POSE + a
        for _ in range(4):
            if actuator == "xml":
                data.ctrl[:] = target
            else:
                q = data.qpos[qpos_idx]
                dq = data.qvel[qvel_idx]
                data.ctrl[:] = np.clip(kp * (target - q) - kd * dq, -FORCE_LIM, FORCE_LIM)
            mujoco.mj_step(model, data)
        if k >= n_stand:
            qw, qx, qy, qz = data.qpos[qadr + 3:qadr + 7]
            yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            fwd.append(np.cos(yaw) * data.qvel[vadr] + np.sin(yaw) * data.qvel[vadr + 1])
            footz.append([data.site_xpos[lf][2], data.site_xpos[rf][2]])
            acts.append(np.abs(a).mean())
            up_z = 1 - 2 * (qx * qx + qy * qy)
            if fell is None and (up_z < 0.3 or data.qpos[qadr + 2] < 0.05):
                fell = (k - n_stand) / 50
    fwd = np.array(fwd); footz = np.array(footz)
    base = np.median(footz, axis=0)
    lifts = int(np.sum(np.diff((footz > base + 0.008).astype(int), axis=0) == 1))
    amp = (footz.max(axis=0) - footz.min(axis=0)) * 1000
    tag = actuator if actuator == "xml" else f"pd kp={kp} kd={kd}"
    print(f"{policy_file:<20} {tag:<18} vx={vx:+.2f}  achieved(後半)={fwd[len(fwd)//2:].mean():+.3f} m/s  "
          f"足振幅 L/R={amp[0]:.0f}/{amp[1]:.0f} mm  足上げ={lifts}  |a|={np.mean(acts):.3f}  "
          f"転倒={'なし' if fell is None else f'{fell:.1f}s'}", flush=True)


if __name__ == "__main__":
    print(f"mujoco {mujoco.__version__} ort {ort.__version__}")
    for pf in ("alpha_walking.onnx", "velstand.onnx"):
        run(pf, "xml", 0.3)
        for kp in (1.1, 2.0, 4.0):
            run(pf, "pd", 0.3, kp=kp)
        run(pf, "pd", 0.0, kp=2.0)
