"""Feasibility test: turn ONE AI-generated character cut-out into a jointed 2D puppet on CPU, automatically.

1. RTMPose (rtmlib, Apache-2.0, ONNX on CPU) finds shoulders/elbows/wrists/hips/head.
2. Capsule masks around upper arm / forearm cut those parts into separate RGBA layers; the head becomes its own layer.
   Each child layer keeps a disk at its pivot so the joint stays covered when it rotates (the reference's
   'ball-joint mannequin' look does the same thing on purpose).
3. The torso pixels hidden behind the arms are filled with OpenCV inpainting (a stand-in for LaMa / SD inpainting).
4. Forward kinematics + eased pose-to-pose keys (point, hand-to-chest, nod, look) + breathing and sway.
Output: <out>.mp4 (540x960, 24 fps), <out>_strip.jpg, <out>_rig.jpg
Usage: PYTHONPATH=<dir with rtmlib> python puppet_test.py <character_cut.png> <out_prefix> [--pose rest|apose]
"""
import subprocess, sys, json
import numpy as np
import cv2

FPS, W, H, DUR = 24, 540, 960, 5.0


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def keyed(t, keys):
    """keys: list of (time, value). Piecewise smoothstep between keys (pose-to-pose with holds)."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            return v0 + (v1 - v0) * ease((t - t0) / max(1e-6, t1 - t0))
    return keys[-1][1]


def capsule(shape, a, b, r):
    m = np.zeros(shape, np.uint8)
    cv2.line(m, tuple(int(v) for v in a), tuple(int(v) for v in b), 255, int(2 * r))
    cv2.circle(m, tuple(int(v) for v in a), int(r), 255, -1)
    cv2.circle(m, tuple(int(v) for v in b), int(r), 255, -1)
    return m > 0


def rot(pivot, deg, tx=0.0, ty=0.0):
    M = cv2.getRotationMatrix2D((float(pivot[0]), float(pivot[1])), deg, 1.0)
    M[:, 2] += (tx, ty)
    return np.vstack([M, [0, 0, 1]])


def main(src, out):
    from rtmlib import Body
    im = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if im.shape[2] == 3:
        im = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
    # pad so rotated limbs never leave the layer canvas
    pad = int(0.25 * max(im.shape[:2]))
    im = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    a = im[..., 3] > 127
    rgb = im[..., :3].copy()
    flat = rgb.copy(); flat[~a] = 255
    body_model = Body(mode="balanced", backend="onnxruntime", device="cpu")
    kps, sc = body_model(flat)
    k, s = kps[0], sc[0]
    K = {n: k[i] for i, n in enumerate(["nose", "leye", "reye", "lear", "rear", "lsh", "rsh", "lel", "rel", "lwr", "rwr", "lhip", "rhip", "lkn", "rkn", "lank", "rank"])}
    conf = {n: float(s[i]) for i, n in enumerate(K)}
    sh_w = np.linalg.norm(K["lsh"] - K["rsh"])
    ru, rf = 0.22 * sh_w, 0.17 * sh_w  # upper arm / forearm half widths
    parts = {}
    used = np.zeros_like(a)
    for side in ("l", "r"):
        sh, el, wr = K[side + "sh"], K[side + "el"], K[side + "wr"]
        hand = wr + (wr - el) * 0.35
        fore = capsule(a.shape, el, hand, rf) & a
        upper = (capsule(a.shape, sh, el, ru) & a) & ~fore
        parts[side + "fore"] = dict(mask=fore, pivot=el)
        parts[side + "upper"] = dict(mask=upper | (capsule(a.shape, el, el, rf * 1.05) & a), pivot=sh)
        used |= fore | upper
    neck = (K["lsh"] + K["rsh"]) / 2
    head_c = (K["leye"] + K["reye"]) / 2
    neck_pt = neck + (head_c - neck) * 0.35
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    # head = everything above a line through the neck point, perpendicular to the neck->head axis
    ax = (head_c - neck) / (np.linalg.norm(head_c - neck) + 1e-6)
    head = (((xx - neck_pt[0]) * ax[0] + (yy - neck_pt[1]) * ax[1]) > 0) & a & ~used
    parts["head"] = dict(mask=head | (capsule(a.shape, neck_pt, neck_pt, 0.18 * sh_w) & a), pivot=neck_pt)
    # torso = rest; fill pixels the arms covered inside the torso hull
    torso_poly = np.array([K["lsh"], K["rsh"], K["rhip"], K["lhip"]], np.float32)
    hull = np.zeros(a.shape, np.uint8); cv2.fillConvexPoly(hull, cv2.convexHull(torso_poly.astype(np.int32)), 255)
    hull = cv2.dilate(hull, np.ones((int(0.15 * sh_w),) * 2, np.uint8)) > 0
    hole = used & hull
    torso_mask = (a & ~used & ~head) | hole | (capsule(a.shape, neck_pt, neck_pt, 0.2 * sh_w) & a)
    torso_rgb = cv2.inpaint(rgb, (hole * 255).astype(np.uint8), 9, cv2.INPAINT_TELEA)
    parts["torso"] = dict(mask=torso_mask, pivot=(K["lhip"] + K["rhip"]) / 2, rgb=torso_rgb)

    def layer(p):
        rgba = np.dstack([p.get("rgb", rgb), (p["mask"] * 255).astype(np.uint8)])
        return cv2.GaussianBlur(rgba.astype(np.float32), (0, 0), 0.6)  # soften cut edges

    L = {n: layer(p) for n, p in parts.items()}
    # front/back arm: the arm whose shoulder is closer to the viewer is drawn last (higher x of nose side heuristic)
    front = "r" if conf["rsh"] >= conf["lsh"] else "l"
    back = "l" if front == "r" else "r"
    order = [back + "upper", back + "fore", "torso", "head", front + "upper", front + "fore"]

    # canvas mapping: fit character height to 80% of frame
    ys, xs = np.where(a)
    ch = ys.max() - ys.min()
    S = 0.82 * H / ch
    base = np.array([[S, 0, W / 2 - S * (xs.min() + xs.max()) / 2], [0, S, H * 0.95 - S * ys.max()], [0, 0, 1]])

    # gesture keys (degrees, image coords: +deg = counter-clockwise on screen)
    sgn = 1 if front == "r" else -1
    keys = {
        front + "upper": [(0, 0), (0.6, 0), (1.1, sgn * 35), (2.6, sgn * 35), (3.1, sgn * 10), (5, sgn * 10)],
        front + "fore": [(0, 0), (0.6, 0), (1.1, sgn * 70), (1.6, sgn * 60), (2.0, sgn * 70), (2.6, sgn * 70), (3.1, sgn * 15), (5, sgn * 15)],
        back + "upper": [(0, 0), (2.9, 0), (3.4, -sgn * 20), (4.4, -sgn * 20), (4.9, -sgn * 5)],
        back + "fore": [(0, 0), (2.9, 0), (3.4, -sgn * 45), (4.4, -sgn * 40), (4.9, -sgn * 10)],
        "head": [(0, 0), (1.2, 0), (1.5, -7), (1.8, 2), (2.1, -4), (3.3, -4), (3.7, 6), (5, 6)],
    }
    frames = []
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
                             "-i", "-", "-threads", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out + ".mp4"], stdin=subprocess.PIPE)
    bgc = np.dstack(np.meshgrid(np.zeros(W), np.linspace(0, 1, H))[1:] * 3)
    bg = (np.array([60, 35, 25]) * (1 - bgc) + np.array([150, 120, 90]) * bgc).astype(np.float32)
    for f in range(int(DUR * FPS)):
        t = f / FPS
        breath = 1 + 0.010 * np.sin(2 * np.pi * 0.28 * t)
        sway = 1.2 * np.sin(2 * np.pi * 0.17 * t)
        root_p = parts["torso"]["pivot"]
        R = rot(root_p, sway)
        R = R @ np.array([[1, 0, 0], [0, breath, root_p[1] * (1 - breath)], [0, 0, 1]])
        M = {"torso": R}
        M["head"] = R @ rot(parts["head"]["pivot"], keyed(t, keys["head"]) + 0.6 * np.sin(2 * np.pi * 0.21 * t + 1))
        for side in ("l", "r"):
            up = keyed(t, keys[side + "upper"]) + 1.5 * np.sin(2 * np.pi * 0.3 * t + (0 if side == "l" else 2))
            M[side + "upper"] = R @ rot(parts[side + "upper"]["pivot"], up)
            M[side + "fore"] = M[side + "upper"] @ rot(parts[side + "fore"]["pivot"], keyed(t, keys[side + "fore"]))
        canvas = bg.copy()
        for n in order:
            A = (base @ M[n])[:2]
            warped = cv2.warpAffine(L[n], A, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
            al = warped[..., 3:4] / 255.0
            canvas = canvas * (1 - al) + warped[..., :3] * al
        fr8 = np.clip(canvas, 0, 255).astype(np.uint8)
        proc.stdin.write(fr8.tobytes())
        if f % 12 == 0:
            frames.append(fr8)
    proc.stdin.close(); proc.wait()
    strip = np.concatenate([cv2.resize(x, (W // 2, H // 2)) for x in frames[:10]], 1)
    cv2.imwrite(out + "_strip.jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, 85])
    # rig debug: parts coloured + joints
    dbg = flat.copy()
    col = {"lupper": (255, 80, 80), "lfore": (255, 160, 80), "rupper": (80, 80, 255), "rfore": (80, 160, 255), "head": (80, 255, 80)}
    for n, c in col.items():
        dbg[parts[n]["mask"]] = (0.5 * dbg[parts[n]["mask"]] + 0.5 * np.array(c)).astype(np.uint8)
    for n, p in K.items():
        cv2.circle(dbg, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    side = np.concatenate([cv2.resize(dbg[y0:y1, x0:x1], (300, int(300 * (y1 - y0) / (x1 - x0)))),
                           cv2.resize(np.dstack([torso_rgb[y0:y1, x0:x1]]), (300, int(300 * (y1 - y0) / (x1 - x0))))], 1)
    cv2.imwrite(out + "_rig.jpg", side, [cv2.IMWRITE_JPEG_QUALITY, 85])
    json.dump(dict(conf=conf, front=front), open(out + "_rig.json", "w"), indent=1)
    print("done", out, "front arm", front, {k_: round(v, 2) for k_, v in conf.items()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
