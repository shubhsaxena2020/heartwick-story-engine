"""Inputs for the painted (V1 storybook) asset job on Kaggle: V1 master images, the brand style reference, and
control images for the animals (their vector rig drawn in a fixed pose, flat-filled, as edge maps). The painted
animal then lines up with the rig, so parts can be cut with the rig's own masks (assets/src/painted_cut.py).
Writes kaggle/build/painted_inputs/ (git-ignored) and its dataset-metadata.json."""
import json, os, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
import numpy as np
import skia
from PIL import Image
from engine2.rig import load_partsheet, mcopy

OUT = os.path.join(REPO, "kaggle", "build", "painted_inputs")
VEC = os.path.join(REPO, "assets", "characters")
SIZE = 1024
# a natural crouch for the control (and therefore the rest pose of the painted rig)
ANIMAL_POSE = {"body": -14.0, "head": 10.0, "paw_n": 8.0, "paw_f": -6.0, "leg_n": 4.0, "leg_f": -6.0, "tail1": 58.0, "tail2": 28.0, "tail3": 24.0}
HARE_POSE = {"body": -20.0, "head": 14.0, "paw_n": 6.0, "paw_f": -4.0, "leg_n": 4.0, "leg_f": -6.0, "tail1": 30.0}


def animal_layout(name):
    """-> (rig, angles, root matrix) placing the animal in a SIZE x SIZE frame."""
    rig = load_partsheet(os.path.join(VEC, name, "rig"))
    ang = dict(HARE_POSE if name == "hare" else ANIMAL_POSE)
    m0 = skia.Matrix()
    w = rig.skel.world(ang, m0)
    # bounds from every part's corners
    xs, ys = [], []
    for p in rig.parts:
        img = p.image
        for cx, cy in ((0, 0), (img.width(), 0), (0, img.height()), (img.width(), img.height())):
            q = mcopy(w[p.bone])
            q.preRotate(p.rot)
            q.preScale(p.scale, p.scale)
            pt = q.mapXY(cx - p.anchor[0], cy - p.anchor[1])
            xs.append(pt.x()); ys.append(pt.y())
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    s = SIZE * 0.8 / max(x1 - x0, y1 - y0)
    root = skia.Matrix()
    root.setTranslate(SIZE / 2 - (x0 + x1) / 2 * s, SIZE / 2 - (y0 + y1) / 2 * s + SIZE * 0.04)
    root.preScale(s, s)
    return rig, ang, root


def part_masks(name):
    """{part name: HxW bool} drawn with the same transform, in z order (index = draw order)."""
    rig, ang, root = animal_layout(name)
    w = rig.skel.world(ang, root)
    out = []
    for p in rig.parts:
        s = skia.Surface(SIZE, SIZE)
        c = s.getCanvas()
        c.clear(skia.ColorTRANSPARENT)
        c.concat(w[p.bone])
        c.rotate(p.rot)
        c.scale(p.scale, p.scale)
        c.drawImage(p.image, -p.anchor[0], -p.anchor[1])
        a = s.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType)[..., 3] > 40
        out.append((p.name, p.bone, a))
    return rig, ang, root, w, out


def control(name):
    import cv2
    rig, ang, root, w, masks = part_masks(name)
    lab = np.zeros((SIZE, SIZE), np.uint8)
    for i, (_, _, a) in enumerate(masks):
        lab[a] = 40 + i * 22
    edges = cv2.Canny(lab, 10, 30)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    # eye + nose hints so the painter puts the face where the head part is
    hp = w["head"]
    for (x, y, r) in ((50, -16, 11), (86, -2, 5)):
        pt = hp.mapXY(x, y)
        cv2.circle(edges, (int(pt.x()), int(pt.y())), int(r * hp.getScaleX() if hp.getScaleX() else 8), 255, 2)
    return edges


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for n in ("nell", "pip", "pip_kit", "dee", "eli"):
        Image.open(os.path.join(REPO, "characters", n, "master.png")).convert("RGB").save(os.path.join(OUT, f"master_{n}.png"))
    Image.open(os.path.join(REPO, "brand", "style_ref.png")).convert("RGB").resize((1024, 1024)).save(os.path.join(OUT, "style_ref.png"))
    for n in ("pip", "pip_kit", "hare"):
        Image.fromarray(control(n)).save(os.path.join(OUT, f"ctrl_{n}.png"))
    for f in ("__init__.py", "autorig.py", "rig.py", "poses.py", "style.py", "mathx.py"):
        shutil.copy(os.path.join(REPO, "engine2", f), os.path.join(OUT, f"engine2__{f}"))
    json.dump({"title": "heartwick-painted-inputs", "id": "shubhsaxena2008/heartwick-painted-inputs", "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(OUT, "dataset-metadata.json"), "w"))
    print(sorted(os.listdir(OUT)))
