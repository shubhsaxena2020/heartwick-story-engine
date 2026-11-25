"""Heartwick QA gate: identity, scale and 9:16 framing checks on keyframes or clips (CPU, open weights).

Detector : OWLv2 base (google/owlv2-base-patch16-ensemble, Apache-2.0), open-vocabulary boxes for
           "an elderly woman", "a squirrel" and "a woman's head with grey hair".
Embedder : DINOv2 base (facebook/dinov2-base, Apache-2.0), CLS + mean patch token, L2-normalised.

Per sampled frame it reports
  body_sim / head_sim  cosine similarity of Nell's body / face crop to the closest reference crop
  pip_sim              same for the squirrel
  headroom             gap between the top of Nell's box and the top of the frame, as a fraction of frame height
  cut                  which frame edges Nell's box touches (a cut-off character)
  pip_scale            sqrt(squirrel box area / Nell head box area): camera-zoom invariant size of Pip
Clip summary: min/median sims, pip_scale max/min ratio (1.0 = constant size), framing failures.

usage: python qa_consistency.py --refs-nell a.png b.png --refs-pip p.png --inputs clip.mp4 key.png ... --out qa.json
       [--frames 9] [--sheet sheet.jpg] [--label name]
"""
import argparse, json, os, sys
import numpy as np, torch
from PIL import Image, ImageDraw

torch.set_num_threads(int(os.environ.get("QA_THREADS", "6")))  # keep the laptop CPU under 75 %
DEV = os.environ.get("QA_DEVICE", "cpu")
from transformers import Owlv2Processor, Owlv2ForObjectDetection, AutoImageProcessor, AutoModel

Q = ["an elderly woman", "a squirrel", "a woman's head with grey hair"]  # "a human face" misses small anime faces
_det = _emb = None


def models():
    global _det, _emb
    if _det is None:
        p = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        m = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").eval().to(DEV)
        ep = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        em = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(DEV)
        _det, _emb = (p, m), (ep, em)
    return _det, _emb


@torch.no_grad()
def detect(im):
    (p, m), _ = models()
    W, H = im.size
    inp = p(text=[Q], images=im, return_tensors="pt").to(DEV)
    out = m(**inp)
    # OWLv2 pads to a square: boxes are relative to max(W, H)
    S = max(W, H)
    r = p.post_process_grounded_object_detection(out, threshold=0.08, target_sizes=[(S, S)])[0]
    r = {k: (v.cpu() if hasattr(v, "cpu") else v) for k, v in r.items()}
    boxes = {}
    for s, l, b in zip(r["scores"].tolist(), r["labels"].tolist(), r["boxes"].tolist()):
        b = [max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])]
        boxes.setdefault(Q[l], []).append((s, b))
    for k in boxes:
        boxes[k].sort(key=lambda x: -x[0])
    woman = boxes.get(Q[0], [None])[0]
    pip = boxes.get(Q[1], [None])[0]
    face = None
    if woman:
        wb = woman[1]
        for s, b in boxes.get(Q[2], []):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if wb[0] <= cx <= wb[2] and wb[1] <= cy <= wb[1] + 0.55 * (wb[3] - wb[1]):
                face = (s, b); break
    return {"woman": woman, "pip": pip, "face": face}


def crop_sq(im, b, pad=0.08):
    x0, y0, x1, y1 = b
    w, h = x1 - x0, y1 - y0; s = max(w, h) * (1 + pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    c = im.crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)))
    bg = Image.new("RGB", c.size, (128, 128, 128)); bg.paste(c); return bg


@torch.no_grad()
def embed(ims):
    _, (ep, em) = models()
    o = em(**ep(images=ims, return_tensors="pt").to(DEV)).last_hidden_state.float().cpu()
    v = torch.cat([o[:, 0], o[:, 1:].mean(1)], 1)
    return torch.nn.functional.normalize(v, dim=1).numpy()


def frames_of(path, n):
    if path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        return [(0, Image.open(path).convert("RGB"))]
    import imageio.v3 as iio
    fr = list(iio.imiter(path))
    idx = np.linspace(0, len(fr) - 1, n).round().astype(int)
    return [(int(i), Image.fromarray(fr[i]).convert("RGB")) for i in idx]


def ref_bank(paths, key):
    crops = {"body": [], "face": [], "pip": []}
    for p in paths:
        im = Image.open(p).convert("RGB"); d = detect(im)
        if key == "nell":
            if d["woman"]: crops["body"].append(crop_sq(im, d["woman"][1]))
            if d["face"]: crops["face"].append(crop_sq(im, d["face"][1], 0.3))
        else:
            b = d["pip"][1] if d["pip"] else [0, 0, im.width, im.height]
            crops["pip"].append(crop_sq(im, b))
    return {k: embed(v) for k, v in crops.items() if v}


def analyse(path, bank_n, bank_p, n):
    rows = []
    for fi, im in frames_of(path, n):
        W, H = im.size; d = detect(im); r = {"frame": fi}
        if d["woman"]:
            b = d["woman"][1]
            r["woman_box"] = [round(v / s, 4) for v, s in zip(b, (W, H, W, H))]
            r["body_sim"] = float((embed([crop_sq(im, b)]) @ bank_n["body"].T).max())
            r["headroom"] = round(b[1] / H, 4)
            e = 0.006
            r["cut"] = [k for k, c in (("left", b[0] <= e * W), ("right", b[2] >= (1 - e) * W),
                                        ("top", b[1] <= e * H), ("bottom", b[3] >= (1 - e) * H)) if c]
        if d["face"] and "face" in bank_n:
            fb = d["face"][1]
            r["head_sim"] = float((embed([crop_sq(im, fb, 0.3)]) @ bank_n["face"].T).max())
            r["head_h"] = round((fb[3] - fb[1]) / H, 4)
        if d["pip"]:
            pb = d["pip"][1]
            r["pip_box"] = [round(v / s, 4) for v, s in zip(pb, (W, H, W, H))]
            r["pip_sim"] = float((embed([crop_sq(im, pb)]) @ bank_p["pip"].T).max())
            if d["face"]:
                fb = d["face"][1]  # sqrt-area ratio: robust to the squirrel turning or curling its tail
                r["pip_scale"] = round(float(np.sqrt((pb[2] - pb[0]) * (pb[3] - pb[1]) / ((fb[2] - fb[0]) * (fb[3] - fb[1])))), 3)
        r["_im"] = im; r["_d"] = d
        rows.append(r)
    return rows


def summary(rows):
    def col(k): return [r[k] for r in rows if k in r]
    s = {}
    for k in ("body_sim", "head_sim", "pip_sim"):
        v = col(k)
        if v: s[k] = {"min": round(min(v), 3), "median": round(float(np.median(v)), 3)}
    v = col("pip_scale")
    if len(v) >= 2:
        s["pip_scale"] = {"min": min(v), "max": max(v), "max_over_min": round(max(v) / min(v), 2)}
    s["headroom_min"] = min(col("headroom")) if col("headroom") else None
    s["frames_cut"] = sum(1 for r in rows if r.get("cut"))
    s["frames"] = len(rows)
    return s


def sheet(all_rows, out):
    tiles = []
    for name, rows in all_rows:
        for r in rows:
            im = r["_im"].copy(); dr = ImageDraw.Draw(im)
            for k, c in (("woman", "lime"), ("face", "yellow"), ("pip", "cyan")):
                if r["_d"][k]: dr.rectangle(r["_d"][k][1], outline=c, width=max(2, im.width // 160))
            t = f"{os.path.basename(name)[:18]} f{r['frame']}\nbody {r.get('body_sim', 0):.2f} face {r.get('head_sim', 0):.2f}\n" \
                f"pip x{r.get('pip_scale', 0):.2f} head {r.get('headroom', 0):.2f} {','.join(r.get('cut', []))}"
            im = im.resize((240, int(240 * im.height / im.width)))
            dr = ImageDraw.Draw(im); dr.rectangle([0, 0, 240, 40], fill=(0, 0, 0)); dr.text((3, 2), t, fill="white")
            tiles.append(im)
    if not tiles: return
    h = max(t.height for t in tiles); cols = min(9, len(tiles)); rows_n = (len(tiles) + cols - 1) // cols
    S = Image.new("RGB", (cols * 240, rows_n * h), "white")
    for i, t in enumerate(tiles): S.paste(t, ((i % cols) * 240, (i // cols) * h))
    S.save(out, quality=88)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--refs-nell", nargs="+", required=True); a.add_argument("--refs-pip", nargs="+", required=True)
    a.add_argument("--inputs", nargs="+", required=True); a.add_argument("--frames", type=int, default=9)
    a.add_argument("--out", required=True); a.add_argument("--sheet")
    A = a.parse_args()
    bn, bp = ref_bank(A.refs_nell, "nell"), ref_bank(A.refs_pip, "pip")
    res, allr = {}, []
    for p in A.inputs:
        rows = analyse(p, bn, bp, A.frames); allr.append((p, rows))
        res[p] = {"summary": summary(rows), "frames": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]}
        print(os.path.basename(p), json.dumps(res[p]["summary"]), flush=True)
    json.dump(res, open(A.out, "w"), indent=1)
    if A.sheet: sheet(allr, A.sheet)
