"""デモ用の舞台装置: 椅子に座る人（見た目だけ、衝突なし）を MuJoCo シーンに実行時に追加する。

公式の scene.xml は変更せず、MjSpec でワールドに静的ジオメトリを足してからコンパイルする。
人はダックの正面（+x 方向）に置き、ダックの方を向いて座っている。実寸（座高 約 1.2 m）なので
ダック（高さ 約 12 cm）は机の上のペットのような見え方になる。
"""
from __future__ import annotations

import math

import mujoco

SKIN = (0.93, 0.78, 0.68, 1.0)
SHIRT = (0.85, 0.33, 0.18, 1.0)   # 背景(青)と区別しやすい暖色
PANTS = (0.30, 0.30, 0.30, 1.0)
SHOES = (0.40, 0.24, 0.12, 1.0)
MOUTH = (0.45, 0.10, 0.10, 1.0)
HAIR = (0.20, 0.14, 0.10, 1.0)
WOOD = (0.30, 0.46, 0.36, 1.0)    # 椅子: 緑系（シャツ/肌の色判定と混ざらないように）
PHONE = (0.95, 0.85, 0.10, 1.0)   # スマホ: 黄色のケース（色で検出しやすい）

PERSON_X = 1.2       # 人（お尻）の x 位置 [m]。ダックは原点から +x に歩く
SEAT_H = 0.45        # 座面高さ


def _geom(body, name, gtype, size, pos, rgba, quat=None, euler=None, fromto=None):
    g = body.add_geom()
    g.name = name
    g.type = gtype
    g.size = list(size) + [0.0] * (3 - len(size))
    g.rgba = list(rgba)
    g.contype = 0        # 見た目だけ（ダックや床と衝突しない）
    g.conaffinity = 0
    g.group = 1
    if fromto is not None:
        g.fromto = list(fromto)
    else:
        g.pos = list(pos)
    if quat is not None:
        g.quat = list(quat)
    if euler is not None:
        # euler (deg, xyz) → quat
        r, p, y = [math.radians(a) for a in euler]
        cr, sr, cp, sp, cy, sy = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2), math.sin(y / 2)
        g.quat = [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                  cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]
    return g


def add_person(spec: mujoco.MjSpec, x: float = PERSON_X, phone: bool = False):
    """椅子と、ダック（原点, +x を向く）の方を向いて座る人を追加する。"""
    w = spec.worldbody
    B = mujoco.mjtGeom.mjGEOM_BOX
    C = mujoco.mjtGeom.mjGEOM_CAPSULE
    S = mujoco.mjtGeom.mjGEOM_SPHERE
    CY = mujoco.mjtGeom.mjGEOM_CYLINDER

    # --- 椅子 ---
    _geom(w, "chair_seat", B, (0.22, 0.22, 0.02), (x, 0, SEAT_H - 0.02), WOOD)
    _geom(w, "chair_back", B, (0.02, 0.22, 0.25), (x + 0.20, 0, SEAT_H + 0.25), WOOD)
    for i, (dx, dy) in enumerate([(-0.18, -0.18), (-0.18, 0.18), (0.18, -0.18), (0.18, 0.18)]):
        _geom(w, f"chair_leg{i}", CY, (0.02, (SEAT_H - 0.04) / 2), (x + dx, dy, (SEAT_H - 0.04) / 2), WOOD)

    # --- 人（ダック側 = -x を向く） ---
    hip_z = SEAT_H + 0.08
    # 太もも: 腰からダック側へ水平
    for s, dy in (("l", 0.09), ("r", -0.09)):
        _geom(w, f"thigh_{s}", C, (0.07,), None, PANTS, fromto=(x + 0.02, dy, hip_z, x - 0.40, dy, hip_z))
        _geom(w, f"shin_{s}", C, (0.055,), None, PANTS, fromto=(x - 0.40, dy, hip_z - 0.02, x - 0.45, dy, 0.06))
        _geom(w, f"foot_{s}", B, (0.12, 0.05, 0.03), (x - 0.55, dy, 0.03), SHOES)
    # 胴体
    _geom(w, "torso", C, (0.14,), None, SHIRT, fromto=(x + 0.02, 0, hip_z + 0.05, x + 0.00, 0, hip_z + 0.50))
    _geom(w, "neck", C, (0.045,), None, SKIN, fromto=(x, 0, hip_z + 0.55, x, 0, hip_z + 0.66))
    # 頭（少し下を向いている）
    head_z = hip_z + 0.76
    _geom(w, "head", S, (0.10,), (x - 0.02, 0, head_z), SKIN)
    _geom(w, "hair", S, (0.102,), (x + 0.01, 0, head_z + 0.03), HAIR)
    _geom(w, "eye_l", S, (0.018,), (x - 0.112, 0.036, head_z + 0.012), (0.05, 0.05, 0.05, 1))
    _geom(w, "eye_r", S, (0.018,), (x - 0.112, -0.036, head_z + 0.012), (0.05, 0.05, 0.05, 1))
    # 口（あくびで大きく開く）: 顔の前面やや下
    _geom(w, "mouth", mujoco.mjtGeom.mjGEOM_ELLIPSOID, (0.006, 0.024, 0.009), (x - 0.120, 0, head_z - 0.042), MOUTH)
    # 腕: 肩から前へ、膝の上で手を合わせる
    for s, dy in (("l", 0.16), ("r", -0.16)):
        _geom(w, f"upper_arm_{s}", C, (0.045,), None, SHIRT, fromto=(x, dy, hip_z + 0.48, x - 0.12, dy + (-0.02 if s == "l" else 0.02), hip_z + 0.22))
        _geom(w, f"forearm_{s}", C, (0.04,), None, SKIN, fromto=(x - 0.12, dy + (-0.02 if s == "l" else 0.02), hip_z + 0.22, x - 0.34, dy * 0.45, hip_z + 0.14))
        _geom(w, f"hand_{s}", S, (0.045,), (x - 0.36, dy * 0.45, hip_z + 0.13), SKIN)
    # スマホ（手の間）。phone=False でも geom は作り、rgba の alpha で表示/非表示を切り替える
    g = _geom(w, "phone", B, (0.035, 0.07, 0.005), (x - 0.38, 0, hip_z + 0.16), PHONE, euler=(0, 60, 0))
    g.rgba[3] = 1.0 if phone else 0.0
    return spec


PERSON_GEOMS = ["thigh_l", "thigh_r", "shin_l", "shin_r", "foot_l", "foot_r", "torso", "neck", "head", "hair",
                "eye_l", "eye_r", "mouth", "upper_arm_l", "upper_arm_r", "forearm_l", "forearm_r", "hand_l", "hand_r"]


def set_person_visible(model: mujoco.MjModel, visible: bool):
    """人の在/不在: 人のジオメトリの alpha を切り替える（椅子は残す）。"""
    for n in PERSON_GEOMS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        if gid >= 0:
            model.geom_rgba[gid, 3] = 1.0 if visible else 0.0
    if not visible:
        set_phone_visible(model, False)


CHAIR_GEOMS = ["chair_seat", "chair_back", "chair_leg0", "chair_leg1", "chair_leg2", "chair_leg3"]
_ORIG = {}   # geom 名 -> (pos, quat)  正面配置の値（初回に記録）


def set_person_side(model: mujoco.MjModel, side: str = "front"):
    """人と椅子をダックの正面 (+x) か背後 (-x) に置く。背後は原点まわりに 180° 回した配置。"""
    import numpy as np
    names = PERSON_GEOMS + ["phone"] + CHAIR_GEOMS
    for n in names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        if gid < 0:
            continue
        if n not in _ORIG:
            _ORIG[n] = (model.geom_pos[gid].copy(), model.geom_quat[gid].copy())
        pos, quat = _ORIG[n]
        if side == "back":
            model.geom_pos[gid] = [-pos[0], -pos[1], pos[2]]
            # q' = Rz(180°) * q ,  Rz(180°) = (0, 0, 0, 1)
            w, x, y, z = quat
            model.geom_quat[gid] = np.array([-z, -y, x, w])
        else:
            model.geom_pos[gid] = pos
            model.geom_quat[gid] = quat


def set_phone_visible(model: mujoco.MjModel, visible: bool):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "phone")
    if gid >= 0:
        model.geom_rgba[gid, 3] = 1.0 if visible else 0.0


def set_mouth_open(model: mujoco.MjModel, open_: bool):
    """あくび: 口を大きく開ける。"""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mouth")
    if gid >= 0:
        model.geom_size[gid] = [0.006, 0.034, 0.032] if open_ else [0.006, 0.024, 0.009]


def set_eyes_closed(model: mujoco.MjModel, closed: bool):
    """眠そうな表情の代わり: 目を小さく（閉じたように）する。"""
    for n in ("eye_l", "eye_r"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        if gid >= 0:
            model.geom_size[gid, 0] = 0.005 if closed else 0.018


if __name__ == "__main__":  # 単体確認: コンパイルできるか
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml = os.path.join(root, "microduck_rl", "src", "mjlab_microduck", "robot", "microduck", "scene.xml")
    spec = mujoco.MjSpec.from_file(xml)
    n0 = len(spec.geoms)
    add_person(spec, phone=True)
    m = spec.compile()
    print(f"ok: geoms {n0} -> {m.ngeom}, bodies {m.nbody}, nu {m.nu}")
