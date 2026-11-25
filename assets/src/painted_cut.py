"""Cut a painted animal (generated on Kaggle against its vector rig's edge map, see painted_prep.py) into a part sheet
with the vector rig's own masks: every pixel goes to the top-most rig part covering it, hidden areas of lower parts
are filled (OpenCV Telea), pivots and rest angles come straight from the rig pose. CPU, a few seconds.
python assets/src/painted_cut.py <name> <painted_cut.png> <out_dir>"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cv2
import numpy as np
from PIL import Image
import painted_prep as PP


def main(name, painted, out_dir):
    rig, ang, root, w, masks = PP.part_masks(name)
    rgba = np.asarray(Image.open(painted).convert("RGBA"))
    f = rgba.shape[0] / PP.SIZE  # hires factor (1.25)
    H, Wd = rgba.shape[:2]
    up = [(n, b, cv2.resize(m.astype(np.uint8), (Wd, H), interpolation=cv2.INTER_NEAREST) > 0) for n, b, m in masks]
    alpha = rgba[..., 3]
    solid = alpha > 24
    # owner = top-most part (later in z order wins); uncovered silhouette pixels -> nearest part
    owner = np.full((H, Wd), -1, np.int32)
    for i, (_, _, m) in enumerate(up):
        owner[m] = i
    if (solid & (owner < 0)).any():
        dist = np.full((H, Wd), np.inf)
        best = np.full((H, Wd), -1, np.int32)
        for i, (_, _, m) in enumerate(up):
            d = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)
            sel = d < dist
            dist[sel], best[sel] = d[sel], i
        fill = solid & (owner < 0) & (dist < 40 * f)
        owner[fill] = best[fill]
    rgb = rgba[..., :3].copy()
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(out_dir):
        if fn.endswith(".png"):
            os.remove(os.path.join(out_dir, fn))
    k = 1.0 / (root.getScaleX() * f)  # rig units per px
    rest = {b.name: rig.skel.world_angle(ang, b.name) for b in rig.skel.bones}
    piv = {}
    for b in rig.skel.bones:
        p = w[b.name].mapXY(0, 0)
        piv[b.name] = (p.x() * f, p.y() * f)
    parts = []
    for i, (pname, bone, m) in enumerate(up):
        own = (owner == i) & solid
        hidden = m & ~own & (owner > i)  # under a part drawn in front: fill so rotation never shows a hole
        mask = own | hidden
        if not mask.any():
            continue
        img = rgb.copy()
        if hidden.any():
            img = cv2.inpaint(np.ascontiguousarray(rgb), (cv2.dilate(hidden.astype(np.uint8), np.ones((5, 5), np.uint8)) * 255), 7, cv2.INPAINT_TELEA)
        a = np.where(own, alpha, 0).astype(np.uint8)
        a[hidden] = np.maximum(a[hidden], 255)
        a = np.where(mask, a, 0)
        ys, xs = np.where(a > 8)
        x0, x1, y0, y1 = xs.min() - 4, xs.max() + 5, ys.min() - 4, ys.max() + 5
        x0, y0 = max(0, x0), max(0, y0)
        crop = np.dstack([img[y0:y1, x0:x1], a[y0:y1, x0:x1]])
        fn = f"{pname}.png"
        Image.fromarray(crop, "RGBA").save(os.path.join(out_dir, fn), optimize=True)
        px, py = piv[bone]
        z = next(p.z for p in rig.parts if p.name == pname)
        parts.append({"bone": bone, "name": pname, "image": fn, "anchor_px": [round(px - x0, 2), round(py - y0, 2)], "z": z})
    bones = [{"name": b.name, "parent": b.parent, "pivot_px": [round(piv[b.name][0], 2), round(piv[b.name][1], 2)],
              "rest_deg": round(rest[b.name], 3), "length": round(b.length / k, 2)} for b in rig.skel.bones]

    def rot(x, y, deg):
        r = math.radians(deg)
        return x * math.cos(r) - y * math.sin(r), x * math.sin(r) + y * math.cos(r)
    vec_rig = json.load(open(os.path.join(PP.VEC, name, "rig", "rig.json")))
    kv = vec_rig["units_per_px"]
    contacts = []
    for b, (cx, cy) in vec_rig["contacts"]:
        ux, uy = cx * kv, cy * kv  # canonical units
        ix, iy = rot(ux, uy, rest[b])
        contacts.append([b, [round(ix / k, 2), round(iy / k, 2)]])
    J = {"name": name, "kind": vec_rig["kind"], "units": vec_rig["units"], "units_per_px": k, "pelvis_h": vec_rig["pelvis_h"],
         "source": "SDXL + IP-Adapter painted (V1 storybook style), cut with the vector rig's part masks (assets/src/painted_cut.py)",
         "bones": bones, "parts": sorted(parts, key=lambda p: p["z"]), "contacts": contacts, "springs": vec_rig.get("springs", []),
         "variants": {}, "source_size": [Wd, H]}
    json.dump(J, open(os.path.join(out_dir, "rig.json"), "w"), indent=1)
    return J


if __name__ == "__main__":
    J = main(*sys.argv[1:4])
    print(J["name"], len(J["parts"]), "parts")
