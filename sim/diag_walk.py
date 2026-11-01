"""歩行が出ない原因の切り分け（ビューアなし・自動）。

公式 infer_policy.py と同じ初期化で PolicyInference を作り、
  [ポリシー: alpha / velstand] × [アクチュエータ: BAM / --no-bam 相当] × [遅延あり/なし]
の組み合わせで「3 秒静止 → 8 秒前進 0.2 m/s → 2 秒停止」を実行し、
達成速度・足の上下動（足を上げているか）・転倒の有無を logs/diag_walk.txt に書く。

実行:  scripts\\diag_walk.bat   （所要 1〜2 分、画面は出ない）
"""
from __future__ import annotations

import os
import sys
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_DIR = os.path.join(ROOT, "microduck_rl")
POLICY_DIR = os.path.join(ROOT, "policies")
REPORT = os.path.join(ROOT, "logs", "diag_walk.txt")

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

LINES = []


def log(s=""):
    print(s, flush=True)
    LINES.append(s)


def build(policy_set: str, use_bam: bool, delay: tuple[int, int]):
    xml = ip.MICRODUCK_XML
    if use_bam:
        bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, 0.0)
        model, data, bam_ctrl, _ = ip.load_mujoco_with_bam(xml, bam_model, 0.005, 0.1, ip.BAM_VIN_MIN)
    else:
        model = mujoco.MjModel.from_xml_path(xml)
        model.opt.timestep = 0.005
        data = mujoco.MjData(model)
        bam_ctrl = None
    if policy_set == "alpha":
        walk, stand = "alpha_walking.onnx", "alpha_stand.onnx"
    else:
        walk, stand = "velstand.onnx", "velstand.onnx"
    pol = ip.PolicyInference(
        model, data, bam_ctrl=bam_ctrl,
        walking_onnx_path=os.path.join(POLICY_DIR, walk),
        standing_onnx_path=os.path.join(POLICY_DIR, stand),
        sitstand_onnx_path=os.path.join(POLICY_DIR, "alpha_sitstand.onnx"),
        new_cmd_obs=True, use_projected_gravity=True, switch_threshold=0.05,
        delay_min_lag=delay[0], delay_max_lag=delay[1],
    )
    pol.set_vel_cmd(0.0, 0.0, 0.0)
    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qadr = model.jnt_qposadr[fj]
    vadr = model.jnt_dofadr[fj]
    data.qpos[qadr:qadr + 3] = [0.0, 0.0, 0.125]
    data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
    for i, qi in enumerate(pol.joint_qpos_indices):
        data.qpos[qi] = pol.default_pose[i]
    if bam_ctrl is not None:
        bam_ctrl.reset(data.qpos)
    pol.set_position_targets(pol.default_pose)
    mujoco.mj_forward(model, data)
    feet = [b for b in range(model.nbody) if "foot" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")]
    return model, data, pol, bam_ctrl, qadr, vadr, feet


def run_phase(model, data, pol, bam_ctrl, qadr, vadr, feet, seconds, vx, label):
    pol.set_vel_cmd(vx, 0.0, 0.0)
    n = int(seconds * 50)
    fwd, footz, act, trunk_z = [], [], [], []
    fell_at = None
    for k in range(n):
        a = pol.infer()
        pol.apply_action(a)
        for _ in range(4):
            if bam_ctrl is not None:
                bam_ctrl.update()
            mujoco.mj_step(model, data)
        # 体幹座標系の前進速度（ヨーだけ考慮）
        qw, qx, qy, qz = data.qpos[qadr + 3:qadr + 7]
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        vxw, vyw = data.qvel[vadr], data.qvel[vadr + 1]
        fwd.append(np.cos(yaw) * vxw + np.sin(yaw) * vyw)
        footz.append([data.xpos[b][2] for b in feet])
        act.append(np.abs(a).mean())
        z = data.qpos[qadr + 2]
        trunk_z.append(z)
        # 上向きベクトルの z 成分（1=直立, <0.3 で転倒とみなす）
        up_z = 1 - 2 * (qx * qx + qy * qy)
        if fell_at is None and (up_z < 0.3 or z < 0.05):
            fell_at = k / 50
    fwd = np.array(fwd)
    footz = np.array(footz)
    tail = fwd[len(fwd) // 2:]
    lifts = 0
    if footz.size:
        base = np.median(footz, axis=0)
        above = footz > base + 0.008      # 8 mm 以上持ち上げ
        lifts = int(np.sum(np.diff(above.astype(int), axis=0) == 1))
        amp = (footz.max(axis=0) - footz.min(axis=0)) * 1000
    else:
        amp = np.array([0.0])
    log(f"  {label:<22} cmd vx={vx:+.2f}  achieved(後半平均)={tail.mean():+.3f} m/s  "
        f"足の上下振幅={'/'.join(f'{x:.0f}' for x in amp)} mm  足上げ回数={lifts}  "
        f"|action|平均={np.mean(act):.3f}  trunk_z={np.mean(trunk_z)*1000:.0f} mm  "
        f"policy={pol.current_policy}  転倒={'なし' if fell_at is None else f'{fell_at:.1f}s'}")
    return fell_at is not None


def scenario(policy_set, use_bam, delay):
    name = f"{policy_set:<8} {'BAM   ' if use_bam else 'no-bam'} delay={delay}"
    log(f"[{name}]")
    try:
        m, d, pol, bc, qadr, vadr, feet = build(policy_set, use_bam, delay)
    except Exception as e:
        log(f"  build failed: {e!r}")
        return
    t0 = time.time()
    fell = run_phase(m, d, pol, bc, qadr, vadr, feet, 3.0, 0.0, "stand 3s")
    if not fell:
        fell = run_phase(m, d, pol, bc, qadr, vadr, feet, 8.0, 0.2, "walk +0.2 8s")
    if not fell:
        fell = run_phase(m, d, pol, bc, qadr, vadr, feet, 2.0, 0.0, "stop 2s")
    if not fell:
        fell = run_phase(m, d, pol, bc, qadr, vadr, feet, 6.0, -0.15, "walk -0.15 6s")
    if not fell:
        run_phase(m, d, pol, bc, qadr, vadr, feet, 2.0, 0.0, "stop 2s")
    log(f"  ({time.time() - t0:.1f}s 実時間)")
    log()


def main():
    log(f"diag_walk  mujoco={mujoco.__version__}  numpy={np.__version__}  python={sys.version.split()[0]}  "
        f"platform={sys.platform}")
    log(f"infer_policy: {ip.__file__}")
    log()
    for policy_set in ("alpha", "velstand"):
        for use_bam in (True, False):
            scenario(policy_set, use_bam, (0, 0))
    scenario("alpha", True, (3, 6))      # 学習時の遅延 3〜6 ステップを模擬
    scenario("velstand", True, (3, 6))
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    log(f"report -> {REPORT}")


if __name__ == "__main__":
    main()
