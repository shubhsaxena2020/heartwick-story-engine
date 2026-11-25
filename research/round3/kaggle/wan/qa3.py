# Heartwick round 3 QA on Kaggle (no local models): runs on every clip / key of the permutation grid.
#   qa_consistency (round 2): OWLv2 boxes + DINOv2 identity of Nell's body / head vs the master key K0, Pip's size
#     relative to Nell's head (pip_scale; the clip's growth = last/first and max/min), headroom, cut-off edges.
#   pose (new): rtmlib RTMPose body keypoints (Apache-2.0) on Nell's crop -> rule label kneel_one / crouch / stand,
#     and the distance of the detected pose to the hand-blocked one-knee "kneel_hold" skeleton (the pose INPUT used for
#     the end keys), in torso lengths. Long skirts hide knees, so the keypoint confidence is reported with it.
#   contact sheets: 8 evenly spaced frames per clip, for the vision critic and for eyes.
# usage: python qa3.py --refs K0.png --inputs a.mp4 b.png ... --out qa.json --sheets dir
import argparse, json, os, sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qa_consistency as qa

# hand-blocked one-knee kneel holding the squirrel (round2 make_poses.py "kneel_hold"), fractions of the frame
KNEEL_HOLD = {"lsho": (0.48, 0.435), "rsho": (0.40, 0.435), "lelb": (0.52, 0.56), "relb": (0.42, 0.55), "lwri": (0.55, 0.52),
              "rwri": (0.51, 0.52), "lhip": (0.45, 0.62), "rhip": (0.40, 0.62), "lkne": (0.58, 0.68), "rkne": (0.38, 0.845),
              "lank": (0.57, 0.86), "rank": (0.21, 0.86)}
COCO = {"lsho": 5, "rsho": 6, "lelb": 7, "relb": 8, "lwri": 9, "rwri": 10, "lhip": 11, "rhip": 12, "lkne": 13, "rkne": 14,
        "lank": 15, "rank": 16}
_body = None


def body():
    global _body
    if _body is None:
        from rtmlib import Body
        _body = Body(mode="balanced", backend="onnxruntime", device="cpu")
    return _body


def pose(im, box):
    """COCO-17 keypoints of the most confident person inside Nell's (padded) box; None if nothing."""
    W, H = im.size
    if box:
        x0, y0, x1, y1 = box; pw, ph = (x1 - x0) * 0.15, (y1 - y0) * 0.1
        x0, y0, x1, y1 = max(0, x0 - pw), max(0, y0 - ph), min(W, x1 + pw), min(H, y1 + ph)
    else:
        x0, y0, x1, y1 = 0, 0, W, H
    crop = im.crop((int(x0), int(y0), int(x1), int(y1)))
    arr = np.asarray(crop)[:, :, ::-1].copy()
    kp, sc = body()(arr)
    if kp is None or len(kp) == 0:
        return None
    i = int(np.argmax(sc.mean(1)))
    k = kp[i].astype(float); k[:, 0] += x0; k[:, 1] += y0
    return k, sc[i].astype(float)


def classify(k, s):
    g = lambda n: k[COCO[n]]
    ground = max(g("lank")[1], g("rank")[1])
    leg = np.mean([np.linalg.norm(g("lhip") - g("lank")), np.linalg.norm(g("rhip") - g("rank")),
                   np.linalg.norm(g("lhip") - g("lkne")) + np.linalg.norm(g("lkne") - g("lank"))]) + 1e-6
    hk = sorted([(ground - g("lkne")[1]) / leg, (ground - g("rkne")[1]) / leg])
    hip = (ground - (g("lhip")[1] + g("rhip")[1]) / 2) / leg
    conf = float(min(s[COCO[n]] for n in ("lhip", "rhip", "lkne", "rkne", "lank", "rank")))
    if hip > 0.72:
        lab = "stand"
    elif hk[0] < 0.2 and hk[1] > 0.3:
        lab = "kneel_one"
    elif hk[0] < 0.2 and hk[1] <= 0.3:
        lab = "kneel_both_or_sit"
    else:
        lab = "crouch"
    return {"label": lab, "knee_heights": [round(v, 3) for v in hk], "hip_height": round(hip, 3), "leg_conf": round(conf, 3)}


def dist_to_kneel(k, W, H):
    """mean joint distance (torso lengths) to KNEEL_HOLD after centring on the mid-hip; min over a left/right mirror"""
    def norm(P):
        hip = (P["lhip"] + P["rhip"]) / 2; sho = (P["lsho"] + P["rsho"]) / 2
        t = np.linalg.norm(sho - hip) + 1e-6
        return {n: (v - hip) / t for n, v in P.items()}
    ref = norm({n: np.array([x * W, y * H]) for n, (x, y) in KNEEL_HOLD.items()})
    det = {n: k[i] for n, i in COCO.items()}
    best = 9e9
    for mirror in (False, True):
        D = det
        if mirror:  # flip horizontally around the hips and swap sides
            cx = (det["lhip"][0] + det["rhip"][0]) / 2
            D = {(n[0].translate(str.maketrans("lr", "rl")) + n[1:]): np.array([2 * cx - v[0], v[1]]) for n, v in det.items()}
        d = norm(D)
        best = min(best, float(np.mean([np.linalg.norm(d[n] - ref[n]) for n in ref])))
    return round(best, 3)


def sheet(path, out, n=8, w=270):
    fr = qa.frames_of(path, n)
    tiles = [im.resize((w, int(w * im.height / im.width)), Image.LANCZOS) for _, im in fr]
    h = max(t.height for t in tiles)
    S = Image.new("RGB", (w * len(tiles), h), "white")
    for i, t in enumerate(tiles):
        S.paste(t, (i * w, 0))
    S.save(out, quality=90)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--refs", nargs="+", required=True); a.add_argument("--inputs", nargs="+", required=True)
    a.add_argument("--out", required=True); a.add_argument("--sheets"); a.add_argument("--frames", type=int, default=9)
    A = a.parse_args()
    bn, bp = qa.ref_bank(A.refs, "nell"), qa.ref_bank(A.refs, "pip")
    res = {}
    if os.path.exists(A.out):
        res = json.load(open(A.out))
    for p in A.inputs:
        key = os.path.basename(p)
        if key in res:
            continue
        try:
            rows = qa.analyse(p, bn, bp, A.frames)
            summ = qa.summary(rows)
            ps = [r.get("pip_scale") for r in rows]
            ps_valid = [v for v in ps if v]
            if len(ps_valid) >= 2:
                summ["pip_growth_last_over_first"] = round(ps_valid[-1] / ps_valid[0], 2)
            poses = []
            for r in rows[-3:] if len(rows) > 1 else rows:   # the end of the action
                im = r["_im"]; d = r["_d"]
                pr = pose(im, d["woman"][1] if d["woman"] else None)
                if pr is None:
                    poses.append({"frame": r["frame"], "label": "none"}); continue
                c = classify(*pr); c["frame"] = r["frame"]; c["dist_kneel_hold"] = dist_to_kneel(pr[0], *im.size)
                poses.append(c)
            summ["end_pose"] = poses
            res[key] = {"summary": summ, "frames": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]}
            if A.sheets and not p.lower().endswith((".png", ".jpg")):
                os.makedirs(A.sheets, exist_ok=True)
                sheet(p, os.path.join(A.sheets, os.path.splitext(key)[0] + "_sheet.jpg"))
        except Exception as e:
            res[key] = {"error": repr(e)[:400]}
        print(key, json.dumps(res[key].get("summary", res[key]))[:600], flush=True)
        json.dump(res, open(A.out, "w"), indent=1)
