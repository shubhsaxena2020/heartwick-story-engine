"""Auto-rig a single character image into a raster part sheet (the AI-image source of engine v2).

Pipeline (research/RESEARCH_REPORT.md section 5): generate a rig-ready figure (side/3-4 view, arms clear of the body,
plain background) -> BiRefNet cut-out -> RTMPose COCO-17 keypoints -> this module:
  * bones at the keypoints, rest angles measured from the image;
  * every opaque pixel goes to the nearest limb capsule (near-side limbs win: they are in front), else head, else torso;
  * child parts keep a round joint cap around their proximal joint (overlap, so rotation never opens a gap);
  * the torso is filled behind the near limbs (LaMa on Kaggle when available, OpenCV Telea otherwise);
  * one PNG per part + rig.json, loaded by engine2.rig.load_partsheet with the same bone names as vector rigs.
CPU-only and deterministic; the GPU steps (generation, BiRefNet, RTMPose, LaMa) run on Kaggle.
"""
import json, math, os
import numpy as np
from PIL import Image

COCO = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
        "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]


def _ang_down(a, b):
    """Angle (deg, clockwise) that rotates 'down' onto a->b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    return math.degrees(math.atan2(-dx, dy))


def _ang_up(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    return math.degrees(math.atan2(dx, -dy))


def _seg_dist(px, py, a, b):
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy or 1e-9
    u = np.clip(((px - ax) * vx + (py - ay) * vy) / L2, 0, 1)
    return np.hypot(px - (ax + u * vx), py - (ay + u * vy))


def _limb_radius(alpha, a, b, default):
    """Half the opaque run across the limb's midpoint, perpendicular to it."""
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy) or 1
    nx, ny = -dy / L, dx / L
    H, W = alpha.shape
    run = 0
    for sgn in (1, -1):
        for k in range(1, int(default * 4)):
            x, y = int(round(mx + sgn * nx * k)), int(round(my + sgn * ny * k))
            if not (0 <= x < W and 0 <= y < H) or alpha[y, x] < 128:
                break
            run += 1
    return float(np.clip(run / 2 + 2, default * 0.5, default * 2.2))


def autorig(rgba, kp, out_dir, name="char", inpaint=None):
    """rgba: HxWx4 uint8 (cut-out, facing right). kp: {coco name: (x, y)}. Writes out_dir/rig.json + part PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    a = rgba[..., 3]
    H, W = a.shape
    P = {k: np.array(v, float) for k, v in kp.items()}
    near = "r" if P["r_shoulder"][0] >= P["l_shoulder"][0] else "l"  # the front shoulder is the near side
    far = "l" if near == "r" else "r"
    side = {"n": near, "f": far}
    pelvis = (P["l_hip"] + P["r_hip"]) / 2
    neck_base = (P["l_shoulder"] + P["r_shoulder"]) / 2
    face_pts = [P[k] for k in ("nose", "l_eye", "r_eye", "l_ear", "r_ear") if k in P]
    head_c = np.mean(face_pts, 0)
    chest_piv = pelvis + 0.5 * (neck_base - pelvis)
    head_piv = neck_base + 0.45 * (head_c - neck_base)
    ys = np.where(a > 128)[0]
    fig_h = (ys.max() - ys.min()) if len(ys) else H
    k = 1000.0 / fig_h  # rig units per px
    J = {"pelvis": pelvis, "spine": pelvis, "chest": chest_piv, "neck": neck_base, "head": head_piv}
    rest = {"pelvis": 0.0, "spine": _ang_up(pelvis, chest_piv), "chest": _ang_up(chest_piv, neck_base),
            "neck": _ang_up(neck_base, head_piv), "head": _ang_up(neck_base, head_piv)}
    seg = {}
    for s, c in side.items():
        sh, el, wr = P[f"{c}_shoulder"], P[f"{c}_elbow"], P[f"{c}_wrist"]
        hp, kn, an = P[f"{c}_hip"], P[f"{c}_knee"], P[f"{c}_ankle"]
        J.update({f"uarm_{s}": sh, f"farm_{s}": el, f"hand_{s}": wr, f"thigh_{s}": hp, f"shin_{s}": kn, f"foot_{s}": an})
        rest.update({f"uarm_{s}": _ang_down(sh, el), f"farm_{s}": _ang_down(el, wr), f"hand_{s}": _ang_down(el, wr),
                     f"thigh_{s}": _ang_down(hp, kn), f"shin_{s}": _ang_down(kn, an), f"foot_{s}": _ang_down(kn, an)})
        hand_end = wr + 0.45 * (wr - el)
        foot_end = an + np.array([0.18 * fig_h * 0.35, 0.05 * fig_h])
        seg.update({f"uarm_{s}": (sh, el), f"farm_{s}": (el, wr), f"hand_{s}": (wr, hand_end),
                    f"thigh_{s}": (hp, kn), f"shin_{s}": (kn, an), f"foot_{s}": (an, foot_end)})
    parent = {"pelvis": None, "spine": "pelvis", "chest": "spine", "neck": "chest", "head": "neck",
              "uarm_f": "chest", "farm_f": "uarm_f", "hand_f": "farm_f", "uarm_n": "chest", "farm_n": "uarm_n", "hand_n": "farm_n",
              "thigh_f": "pelvis", "shin_f": "thigh_f", "foot_f": "shin_f", "thigh_n": "pelvis", "shin_n": "thigh_n", "foot_n": "shin_n"}
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    solid = a > 20
    base_r = fig_h * 0.035
    rad = {b: _limb_radius(a, *seg[b], base_r * (0.8 if b.startswith(("farm", "hand", "shin")) else 1.0)) for b in seg}
    owner = np.full((H, W), "", dtype=object)
    order = ["hand_n", "farm_n", "uarm_n", "foot_n", "shin_n", "thigh_n"]  # in front first
    head_r = max(np.linalg.norm(head_c - neck_base) * 0.95, fig_h * 0.06)
    head_mask = solid & (np.hypot(xx - head_c[0], yy - head_c[1]) < head_r) & (yy < neck_base[1])
    claimed = np.zeros((H, W), bool)
    masks = {}
    for b in order:
        m = solid & ~claimed & (_seg_dist(xx, yy, *seg[b]) < rad[b])
        masks[b] = m
        claimed |= m
    masks["head"] = head_mask & ~claimed
    claimed |= masks["head"]
    for b in ["hand_f", "farm_f", "uarm_f", "foot_f", "shin_f", "thigh_f"]:
        m = solid & ~claimed & (_seg_dist(xx, yy, *seg[b]) < rad[b])
        masks[b] = m
        claimed |= m
    torso = solid & ~claimed
    masks["chest"] = torso & (yy < chest_piv[1])
    masks["spine"] = torso & (yy >= chest_piv[1])
    # joint caps: child parts also carry a disc of their parent's pixels around the joint (overlap on rotation)
    for b in list(seg):
        cap = solid & (np.hypot(xx - J[b][0], yy - J[b][1]) < rad[b] * 1.05)
        masks[b] = masks[b] | cap
    masks["head"] = masks["head"] | (solid & (np.hypot(xx - head_piv[0], yy - head_piv[1]) < base_r * 0.9))
    # fill the torso behind the near arm (and near thigh), where the arm used to cover it
    rgb = rgba[..., :3].copy()
    tor = masks["chest"] | masks["spine"]
    hole = np.zeros((H, W), bool)
    if tor.any():
        tx0, tx1 = np.percentile(xx[tor], [2, 98])
        ty0, ty1 = yy[tor].min(), yy[tor].max()
        for b in ("uarm_n", "farm_n", "hand_n"):
            hole |= masks[b] & (xx >= tx0) & (xx <= tx1) & (yy >= ty0) & (yy <= ty1)
    if hole.any():
        rgb = (inpaint or _telea)(rgb, hole)
        ty = yy < chest_piv[1]
        masks["chest"] |= hole & ty
        masks["spine"] |= hole & ~ty
    z = {"thigh_f": 10, "shin_f": 11, "foot_f": 12, "uarm_f": 13, "farm_f": 14, "hand_f": 15, "thigh_n": 16, "shin_n": 17, "foot_n": 18,
         "spine": 30, "chest": 31, "head": 34, "uarm_n": 40, "farm_n": 41, "hand_n": 42}
    parts = []
    for b, m in masks.items():
        if not m.any():
            continue
        ys_, xs_ = np.where(m)
        x0, x1, y0, y1 = xs_.min(), xs_.max() + 1, ys_.min(), ys_.max() + 1
        crop = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
        crop[..., :3] = rgb[y0:y1, x0:x1]
        crop[..., 3] = np.where(m[y0:y1, x0:x1], rgba[y0:y1, x0:x1, 3], 0)
        if b in ("chest", "spine"):
            crop[..., 3] = np.where(hole[y0:y1, x0:x1] & m[y0:y1, x0:x1], 255, crop[..., 3])
        Image.fromarray(crop, "RGBA").save(os.path.join(out_dir, f"{b}.png"))
        jx, jy = J[b]
        parts.append({"bone": b, "image": f"{b}.png", "anchor_px": [float(jx - x0), float(jy - y0)], "z": z[b]})
    bones = [{"name": b, "parent": parent[b], "pivot_px": [float(J[b][0]), float(J[b][1])], "rest_deg": round(float(rest[b]), 3)}
             for b in ["pelvis", "spine", "chest", "neck", "head", "uarm_f", "farm_f", "hand_f", "uarm_n", "farm_n", "hand_n",
                       "thigh_f", "shin_f", "foot_f", "thigh_n", "shin_n", "foot_n"]]
    contacts = []
    for s in ("n", "f"):
        L = float(np.linalg.norm(seg[f"foot_{s}"][1] - seg[f"foot_{s}"][0]))
        contacts += [[f"foot_{s}", [0.0, L * 0.5]], [f"shin_{s}", [0.0, 0.0]]]
    pel_h = (max(P["l_ankle"][1], P["r_ankle"][1]) - pelvis[1]) * k
    rig = {"name": name, "kind": "humanoid", "units": 1000.0, "units_per_px": k, "pelvis_h": pel_h, "bones": bones,
           "parts": sorted(parts, key=lambda p: p["z"]), "contacts": contacts, "near_side": near,
           "source_size": [W, H]}
    json.dump(rig, open(os.path.join(out_dir, "rig.json"), "w"), indent=1)
    return rig


def _telea(rgb, hole):
    import cv2
    m = (hole * 255).astype(np.uint8)
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    return cv2.inpaint(np.ascontiguousarray(rgb), m, 7, cv2.INPAINT_TELEA)
