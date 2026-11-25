"""Heartwick engine v4: "living still" (limited animation) renderer, CPU-only, built to run on a Kaggle CPU kernel.

Derived from research/round4/kaggle/bench/la.py, with no ESRGAN: v4 keys are already 1088x1920 FLUX.2-klein stills, so the
key (or a crop of it) is just Lanczos-resized to the working canvas (a mild unsharp mask is added when a crop is upscaled
by more than 1.6x).

Per shot: key -> Depth-Anything-V2-Small depth + isnet-anime matte (rembg) + Telea clean plate -> per-frame render:
  * camera on ones (24 fps): push/pull/pan/tilt/drift, with depth parallax on the background (lateral parallax for pans,
    "dolly" parallax for push/pull: far background scales less than the character);
  * character on twos (12 drawings/s): breathing (+ optional sway) warp of the matted figure, anchored at the hips/seat,
    applied as a smooth near-rigid shift of the head, so the mouth never animates;
  * FX presets (world-anchored, they move with the camera);
  * light breathing.
Every output is 1080x1920, 24 fps, H.264 yuv420p, and has EXACTLY round(dur*24) frames (checked with ffprobe).

usage:  python la_render.py jobs.json OUTDIR
env:    LA_SEARCH  colon-separated dirs searched (recursively) for job file names that are not existing paths
                   (default "/tmp/la_in:/kaggle/input").

JOB SCHEMA (one dict per shot; a JSON list of them):
  id          str    required. Output files are OUTDIR/<id>.mp4, <id>_mid.jpg, <id>_sheet.jpg, <id>_matte.jpg, <id>.json
  image       str    key image path or bare file name (resolved via LA_SEARCH). Required unless tail_of is given.
  dur         float  seconds; the mp4 has exactly round(dur*24) frames.
  crop        [x0,y0,x1,y1] normalised to the key, or null (= full key). Expanded/shrunk around its centre to 9:16.
  cam         {"type": push_in|pull_out|pan_left|pan_right|tilt_up|tilt_down|drift|static, "amount": 0.04-0.10}
              amount = zoom delta for push/pull, fraction of the view width travelled for pans/tilts. Default
              push_in 0.08. Pans/tilts also add a 0.25*amount push so the frame never feels locked off.
  fx          list of preset strings (default []). Coordinates are NORMALISED TO THE KEY (not the crop):
                "dust" | "dust@x0,y0,x1,y1"      warm drifting, twinkling motes (in front)
                "steam@x,y"                       steam rising from point x,y (e.g. a mug), drawn on twos
                "snow" | "rain"                   behind the character, only in the bright+far plate area (sky)
                "snow@x0,y0,x1,y1" | "rain@..."   box around a WINDOW: drawn only on its glass (k-means colour
                                                  clusters denser in the box centre than on its border = not the
                                                  frame/wall). Give the box tight around the window.
                "leaves" | "leaves@x0,y0,x1,y1"  falling, fluttering leaves; colours sampled from the key; larger
                                                  ones pass in front of the character, smaller ones behind
                "light"                           highlights breathe (+-5% on bright areas, 4.7 s period)
  char_motion "breath" | "breath_sway" | "none" (default "breath"). On twos, via the isnet-anime matte.
  blink       bool   NOT IMPLEMENTED: recorded as skipped (see record["blink"]).
  tail_of     null | clip path/glob-able name. The output is then the WHOLE shot: the acting clip (expected 1080x1920,
              24 fps; anything else is scaled/cropped and fps-converted by duplication with a warning, never slowed)
              trimmed to dur if longer, followed by a living still rendered from the clip's LAST frame. Tail frame 0
              is that last frame, pixel-identical; the tail camera starts from rest (pull_out becomes push_in).
              cam/fx/char_motion apply to the tail; FX fade in over 0.5 s; the camera amount is scaled by
              tail_len/5 s (min 0.3) so the tail moves at the speed of a normal shot instead of cramming the move in.
  optional tuning: hip (0-1 of the figure's height where the warp anchor sits, default 0.62), breath_amp / sway_amp
              (fractions of figure height, defaults 0.0065 / 0.006 for breath_sway, 0.0015 for breath), seed (int).

RECORD (<id>.json, also gathered in la_rec.json): ok, frames (requested), frames_probe (ffprobe count), n_clip, n_tail,
  cam (final), base, cam_move, cam_move_per_5s (tails: move scaled to a 5 s shot; used by the gate), prep {load_key_s, depth_s, matte_s, plate_s}, render_s, ms_per_frame, audit
  {mad_mean, flow_med_px, subject_p97_med_px, longest_static_s, ...  (sampled at 12 fps: px per 1/12 s = per drawing)},
  gate {pass, reason}, attempts [...], warnings [...], blink.
MOTION GATE: longest static run <= 0.5 s AND (subject motion >= 1 px/sample OR camera move >= 3%). A failing shot is
  re-rendered once with the camera amount x1.6 (cap 0.16) and the body motion x1.3; both attempts are recorded.
"""
import glob, hashlib, json, math, os, subprocess, sys, time, traceback

import cv2
import numpy as np
from PIL import Image

OW, OH, FPS = 1080, 1920, 24
CW, CH = 1350, 2400                  # working canvas for keys (1.25 x output)
ASPECT = OW / OH
cv2.setNumThreads(4)
_M = {}
_LAYERS = {}


# ----------------------------------------------------------------------------------------------------------- models / io
def models():
    if not _M:
        from transformers import pipeline
        _M["depth"] = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=-1)
        from rembg import new_session
        _M["seg"] = new_session("isnet-anime")
    return _M


def resolve(p):
    if p and os.path.exists(p):
        return p
    name = os.path.basename(p)
    for root in os.environ.get("LA_SEARCH", "/tmp/la_in:/kaggle/input").split(":"):
        hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(p)


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", path],
                         capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    a, b = s["r_frame_rate"].split("/")
    return {"w": int(s["width"]), "h": int(s["height"]), "fps": float(a) / float(b), "frames": int(s.get("nb_read_frames", 0))}


def read_clip(path, warnings):
    """All frames of the acting clip as RGB uint8 (OH, OW, 3). Never retimed: a non-24 fps clip is converted by frame
    duplication/drop (constant speed), a non-1080x1920 clip is scaled to cover and centre-cropped."""
    p = probe(path)
    vf = []
    if (p["w"], p["h"]) != (OW, OH):
        vf.append(f"scale={OW}:{OH}:force_original_aspect_ratio=increase:flags=lanczos,crop={OW}:{OH}")
        warnings.append(f"clip is {p['w']}x{p['h']}, scaled to {OW}x{OH}")
    if abs(p["fps"] - FPS) > 0.01:
        vf.append(f"fps={FPS}")
        warnings.append(f"clip is {p['fps']:.3g} fps, converted to 24 by frame duplication (speed unchanged)")
    cmd = ["ffmpeg", "-v", "error", "-i", path] + (["-vf", ",".join(vf)] if vf else []) + \
          ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, OH, OW, 3)


# ------------------------------------------------------------------------------------------------------------ layers
def fit_crop(crop, iw, ih):
    x0, y0, x1, y1 = crop if crop else (0, 0, 1, 1)
    cx, cy = (x0 + x1) / 2 * iw, (y0 + y1) / 2 * ih
    w, h = max(8.0, (x1 - x0) * iw), max(8.0, (y1 - y0) * ih)
    if w / h > ASPECT:
        h = w / ASPECT
    else:
        w = h * ASPECT
    if w > iw:
        w = iw; h = w / ASPECT
    if h > ih:
        h = ih; w = h * ASPECT
    cx = min(max(cx, w / 2), iw - w / 2); cy = min(max(cy, h / 2), ih - h / 2)
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def build_layers(rgb, W, H, src_box=None, src_size=None):
    """rgb: canvas image (H, W, 3) uint8. Returns the layer dict used by the renderer."""
    M = models(); t = time.time(); r = {}
    pil = Image.fromarray(rgb)
    d = M["depth"](pil)["predicted_depth"]
    d = np.asarray(d.squeeze().float().cpu().numpy() if hasattr(d, "cpu") else d, np.float32)
    d = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)
    d = np.clip((d - np.percentile(d, 2)) / (np.percentile(d, 98) - np.percentile(d, 2) + 1e-6), 0, 1)   # 1 = near
    r["depth_s"] = round(time.time() - t, 2); t = time.time()
    from rembg import remove
    m = np.asarray(remove(pil, session=M["seg"], only_mask=True)).astype(np.float32) / 255.0
    r["matte_s"] = round(time.time() - t, 2); t = time.time()
    core = (m > 0.5).astype(np.uint8)
    # drop matte specks smaller than 0.15% of the frame (table corners, leaves...)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(core, 8)
    keep = np.zeros(nl, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] > 0.0015 * W * H
    core = keep[lab].astype(np.uint8)
    m = m * cv2.dilate(core, np.ones((15, 15), np.uint8)).astype(np.float32)
    # foreground alpha = matte + a thin feathered ring of original pixels, so a still frame equals the key exactly
    ring = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(np.float32)
    a_fg = np.maximum(m, cv2.GaussianBlur(ring, (0, 0), 2.0)).clip(0, 1)
    hole = cv2.dilate((a_fg > 0.02).astype(np.uint8), np.ones((5, 5), np.uint8))
    sw, sh = W // 2, H // 2
    small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    hs = cv2.resize(hole, (sw, sh), interpolation=cv2.INTER_NEAREST)
    hs = cv2.dilate(hs, np.ones((3, 3), np.uint8))
    fill = cv2.inpaint(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), hs * 255, 9, cv2.INPAINT_TELEA)
    fill = cv2.cvtColor(cv2.resize(fill, (W, H), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    plate = np.where(hole[..., None] > 0, fill, rgb)
    dfill = cv2.inpaint((d * 255).astype(np.uint8), hole * 255, 9, cv2.INPAINT_TELEA).astype(np.float32) / 255.0
    sig = 0.012 * W                                       # smooth depth => parallax stretches instead of tearing
    dbg = cv2.GaussianBlur(dfill, (0, 0), sig)
    r["plate_s"] = round(time.time() - t, 2)
    ys, xs = np.nonzero(core)
    L = {"W": W, "H": H, "rgb": rgb, "plate": plate, "a_fg": a_fg.astype(np.float32), "d": d, "dbg": dbg.astype(np.float32),
         "prep": r, "src_box": src_box, "src_size": src_size}
    if len(ys) > 0.002 * W * H:
        y_top, y_bot = float(np.percentile(ys, 0.5)), float(np.percentile(ys, 99.5))
        up = ys < y_top + 0.6 * (y_bot - y_top)
        xl, xr = float(np.percentile(xs[up], 2)), float(np.percentile(xs[up], 98))
        head = ys < y_top + 0.35 * (y_bot - y_top)
        L.update(char=True, y_top=y_top, y_bot=y_bot, xl=xl, xr=xr, d_ref=float(np.median(d[core > 0])),
                 target=(float(np.median(xs[head])), y_top + 0.22 * (y_bot - y_top)))
    else:
        L.update(char=False, y_top=0.0, y_bot=float(H), xl=0.0, xr=float(W), d_ref=float(np.median(d)), target=(W / 2, H / 2))
    # sky / window mask for snow + rain: bright AND far (luminance + depth), cleaned
    lum = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
    thr = max(150.0, float(np.percentile(lum, 80)))
    sky = ((lum > thr) & (d < 0.4) & (a_fg < 0.05)).astype(np.uint8)
    if sky.mean() < 0.005:
        sky = ((lum > 184) & (a_fg < 0.05)).astype(np.uint8)
    sky = cv2.erode(cv2.morphologyEx(sky, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), np.ones((7, 7), np.uint8))
    L["sky"] = sky; L["lum"] = lum
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    warm = ((hsv[..., 0] < 24) | (hsv[..., 0] > 168)) & (hsv[..., 1] > 130) & (hsv[..., 2] > 150)
    pix = rgb[warm]
    L["leaf_cols"] = pix[np.random.default_rng(0).choice(len(pix), 64)] if len(pix) > 500 else \
        np.array([[205, 92, 38], [226, 140, 46], [178, 58, 40], [232, 170, 70]], np.uint8)
    return L


def key_layers(image, crop):
    ck = (image, tuple(crop) if crop else None)
    if ck in _LAYERS:
        L = _LAYERS[ck]; return dict(L, prep=dict(L["prep"], cached=True))
    t = time.time()
    im = Image.open(resolve(image)).convert("RGB")
    iw, ih = im.size
    box = fit_crop(crop, iw, ih)
    can = im.resize((CW, CH), Image.LANCZOS, box=box)
    rgb = np.asarray(can).copy()
    up = CW / (box[2] - box[0])
    if up > 1.6:   # soft upscaled crop: mild unsharp mask
        b = cv2.GaussianBlur(rgb, (0, 0), 1.0)
        rgb = cv2.addWeighted(rgb, 1.4, b, -0.4, 0)
    lk = round(time.time() - t, 2)
    L = build_layers(rgb, CW, CH, src_box=box, src_size=(iw, ih))
    L["prep"]["load_key_s"] = lk; L["prep"]["upscale"] = round(up, 2); L["prep"]["crop_px"] = [round(v, 1) for v in box]
    _LAYERS[ck] = L
    return L


def key_to_canvas(L, x, y):
    """key-normalised point -> canvas pixels (tail canvases are the clip frame itself: identity)."""
    if not L["src_box"]:
        return x * L["W"], y * L["H"]
    x0, y0, x1, y1 = L["src_box"]; iw, ih = L["src_size"]
    return (x * iw - x0) / (x1 - x0) * L["W"], (y * ih - y0) / (y1 - y0) * L["H"]


def fx_scale(L):
    """FX sizes were tuned on full-frame keys at canvas 1350 wide; scale them with crop zoom and canvas width."""
    s = L["W"] / CW
    if L["src_box"]:
        s *= L["src_size"][0] / (L["src_box"][2] - L["src_box"][0])
    return s


# ----------------------------------------------------------------------------------------------------------- camera
def cam_path(cam, m, L, tail):
    typ = cam.get("type", "push_in"); a = float(np.clip(cam.get("amount", 0.08), 0.0, 0.2))
    if tail:   # same camera SPEED as a ~5 s shot: a short tail gets a proportionally shorter move
        a *= float(np.clip(m / 120.0, 0.3, 1.0))
    W, H = L["W"], L["H"]
    u = np.linspace(0, 1, m) if m > 1 else np.zeros(1)
    e = 0.5 * u + 0.5 * u * u if tail else 0.75 * u + 0.25 * u * u * (3 - 2 * u)   # never ease to a standstill
    ec = e if tail else e - 0.5                                                      # centred travel (tails start at 0)
    z = np.ones(m); dx = np.zeros(m); dy = np.zeros(m); sx = np.zeros(m); sy = np.zeros(m); lat = 0.0; dol = 0.6
    tx, ty = L["target"][0] - W / 2, L["target"][1] - H / 2
    if typ == "pull_out" and tail:
        typ = "push_in"
    if typ == "push_in":
        z = 1 + a * e; sx, sy = 0.35 * e * tx, 0.35 * e * ty
    elif typ == "pull_out":
        z = 1 + a * (1 - e); sx, sy = 0.35 * (1 - e) * tx, 0.35 * (1 - e) * ty
    elif typ in ("pan_left", "pan_right"):
        z = 1 + 0.25 * a * e; dx = (-1 if typ == "pan_left" else 1) * a * ec; lat = 0.6; dol = 0.3
    elif typ in ("tilt_up", "tilt_down"):
        z = 1 + 0.25 * a * e; dy = (-1 if typ == "tilt_up" else 1) * a * ec; lat = 0.6; dol = 0.3
    elif typ == "drift":
        sgn = 1 if int(hashlib.sha1(str(L.get("seed_key", "")).encode()).hexdigest(), 16) % 2 else -1
        z = 1 + 0.45 * a * e; dx = sgn * 0.5 * a * ec; dy = -0.3 * a * ec; lat = 0.6; dol = 0.4
    elif typ != "static":
        raise ValueError(f"unknown cam type {typ}")
    dd_max = max(L["d_ref"], 1 - L["d_ref"])

    def frames(B):
        ww0 = W / B
        ww, hh = W / B / z, H / B / z
        cx = W / 2 + dx * ww0 + sx; cy = H / 2 + dy * ww0 + sy
        parx, pary = lat * dx * ww0, lat * dy * ww0
        dolly = dol * (z - 1)
        pmx = np.abs(parx) * dd_max + dolly * dd_max * ww / 2
        pmy = np.abs(pary) * dd_max + dolly * dd_max * hh / 2
        return ww, hh, cx, cy, parx, pary, dolly, pmx, pmy

    if tail:
        B = 1.0
    else:
        B = 1.30
        for b in np.arange(1.06, 1.301, 0.01):
            ww, hh, cx, cy, parx, pary, dolly, pmx, pmy = frames(b)
            if (cx - ww / 2 - pmx >= 0).all() and (cx + ww / 2 + pmx <= W).all() and \
               (cy - hh / 2 - pmy >= 0).all() and (cy + hh / 2 + pmy <= H).all():
                B = float(b); break
    ww, hh, cx, cy, parx, pary, dolly, _, _ = frames(B)
    cx = np.clip(cx, ww / 2, W - ww / 2); cy = np.clip(cy, hh / 2, H - hh / 2)
    ww0 = W / B
    move = max(float(z.max() / z.min() - 1), float((cx.max() - cx.min()) / ww0), float((cy.max() - cy.min()) / ww0))
    return dict(B=round(B, 3), z=z, ww=ww, hh=hh, cx=cx, cy=cy, parx=parx, pary=pary, dolly=dolly, move=round(move, 4))


# --------------------------------------------------------------------------------------------------------------- FX
def blob_layer(pts, shape, sigma):
    """pts: (x, y, radius, value) in output pixels -> blurred float layer drawn only inside its bbox."""
    pts = [p for p in pts if -60 < p[0] < shape[1] + 60 and -60 < p[1] < shape[0] + 60]
    if not pts:
        return None, None
    pad = int(3 * sigma + max(p[2] for p in pts)) + 2
    x0 = max(0, int(min(p[0] for p in pts)) - pad); x1 = min(shape[1], int(max(p[0] for p in pts)) + pad)
    y0 = max(0, int(min(p[1] for p in pts)) - pad); y1 = min(shape[0], int(max(p[1] for p in pts)) + pad)
    if x1 <= x0 or y1 <= y0:
        return None, None
    roi = np.zeros((y1 - y0, x1 - x0), np.float32)
    for x, y, rad, v in pts:
        cv2.circle(roi, (int(x) - x0, int(y) - y0), max(1, int(round(rad))), float(v), -1, cv2.LINE_AA)
    return cv2.GaussianBlur(roi, (0, 0), sigma), (y0, y1, x0, x1)


def line_layer(segs, shape, sigma):
    segs = [s for s in segs if -80 < s[0] < shape[1] + 80 and -80 < s[1] < shape[0] + 80]
    if not segs:
        return None, None
    xs = [v for s in segs for v in (s[0], s[2])]; ys = [v for s in segs for v in (s[1], s[3])]
    pad = int(3 * sigma) + 3
    x0 = max(0, int(min(xs)) - pad); x1 = min(shape[1], int(max(xs)) + pad)
    y0 = max(0, int(min(ys)) - pad); y1 = min(shape[0], int(max(ys)) + pad)
    if x1 <= x0 or y1 <= y0:
        return None, None
    roi = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for ax, ay, bx, by, v in segs:
        cv2.line(roi, (int(ax) - x0, int(ay) - y0), (int(bx) - x0, int(by) - y0), int(255 * v), 2, cv2.LINE_AA)
    return cv2.GaussianBlur(roi.astype(np.float32) / 255, (0, 0), sigma), (y0, y1, x0, x1)


def parse_fx(fx, L):
    W, H = L["W"], L["H"]
    out = []
    for s in fx or []:
        name, _, arg = s.partition("@")
        vals = [float(v) for v in arg.split(",")] if arg else []
        if len(vals) == 4:
            (ax, ay), (bx, by) = key_to_canvas(L, vals[0], vals[1]), key_to_canvas(L, vals[2], vals[3])
            box = (max(0, ax), max(0, ay), min(W, bx), min(H, by))
        else:
            box = None
        pt = key_to_canvas(L, *vals) if len(vals) == 2 else None
        out.append((name.strip(), box, pt))
    return out


def fx_setup(fxl, L, rng):
    W, H = L["W"], L["H"]; st = {}
    sky = L["sky"]
    ys, xs = np.nonzero(sky)
    sky_box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) > 200 else None
    for name, box, pt in fxl:
        if name in ("snow", "rain"):
            b = box or sky_box
            if b is None:
                st.setdefault("_skipped", []).append(f"{name}: no bright sky/window region in the plate")
                continue
            if box:   # explicit box around a window: the glass = colour clusters denser in its centre than on its border
                msk = glass_mask(L, box)
            else:
                msk = sky
            n = 420 if name == "snow" else 260
            st[name] = {"box": b, "mask": msk, "maskf": cv2.GaussianBlur(msk.astype(np.float32), (0, 0), 1.5), "x": rng.uniform(b[0], b[2], n), "y": rng.uniform(b[1], b[3], n),
                        "s": rng.uniform(0.6, 1.4, n), "p": rng.uniform(0, 2 * np.pi, n)}
        elif name == "dust":
            b = box or (0, 0, W, H); n = 90
            st["dust"] = {"box": b, "x": rng.uniform(b[0], b[2], n), "y": rng.uniform(b[1], b[3], n),
                          "s": rng.uniform(0.6, 1.4, n), "p": rng.uniform(0, 2 * np.pi, n)}
        elif name == "leaves":
            b = box or (0, 0, W, H); n = 16
            cols = L["leaf_cols"]
            st["leaves"] = {"box": b, "x": rng.uniform(b[0], b[2], n), "y": rng.uniform(b[1] - 100, b[3] + 100, n),
                            "s": rng.uniform(0.7, 1.3, n), "p": rng.uniform(0, 2 * np.pi, n),
                            "v": rng.uniform(110, 190, n), "col": cols[rng.integers(0, len(cols), n)].astype(np.float32)}
        elif name == "steam":
            if pt is None:
                st.setdefault("_skipped", []).append("steam needs @x,y"); continue
            st.setdefault("steam", []).append(pt)
        elif name == "light":
            st["light"] = True
        else:
            st.setdefault("_skipped", []).append(f"unknown fx {name}")
    return st


def glass_mask(L, box, k=4):
    """Window box -> mask of the glass (view outside). k-means (Lab, on a 4x-downsampled box) and keep the clusters that
    are >1.2x denser in the central 50% than on the outer 8% ring (frame, wall); never on the character."""
    x0, y0, x1, y1 = (int(v) for v in box)
    sub = L["rgb"][y0:y1, x0:x1]
    h, w = sub.shape[:2]
    sm = cv2.resize(sub, (max(8, w // 4), max(8, h // 4)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(sm, cv2.COLOR_RGB2Lab).reshape(-1, 3).astype(np.float32)
    _, lb, _ = cv2.kmeans(lab, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3,
                          cv2.KMEANS_PP_CENTERS)
    lb = lb.reshape(sm.shape[:2]); hh, ww = lb.shape
    cen = np.zeros((hh, ww), bool); cen[int(hh * .25):int(hh * .75), int(ww * .25):int(ww * .75)] = True
    bd = max(2, int(0.08 * min(hh, ww))); ring = np.ones((hh, ww), bool); ring[bd:-bd, bd:-bd] = False
    keep = [c for c in range(k) if (lb[cen] == c).mean() > 1.2 * (lb[ring] == c).mean()]
    m = np.isin(lb, keep).astype(np.uint8)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    out = np.zeros(L["rgb"].shape[:2], np.uint8)
    out[y0:y1, x0:x1] = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    out &= (L["a_fg"] < 0.05).astype(np.uint8)
    return cv2.erode(out, np.ones((5, 5), np.uint8))


def wrap(v, lo, hi):
    return lo + (v - lo) % max(1.0, hi - lo)


def draw_leaf(comp, x, y, length, ang, flat, col, alpha):
    r = int(length) + 3
    xi, yi = int(x), int(y)
    if xi + r < 0 or yi + r < 0 or xi - r >= comp.shape[1] or yi - r >= comp.shape[0]:
        return
    patch = np.zeros((2 * r + 1, 2 * r + 1), np.uint8)
    rib = np.zeros_like(patch)
    uu = np.linspace(0, 1, 13)
    lx = length / 2 * (1 - 2 * uu); ly = length * 0.42 * flat * np.sin(np.pi * uu) ** 0.85   # pointed lens
    xs_ = np.concatenate([lx, lx[::-1]]); ys_ = np.concatenate([ly, -ly[::-1]])
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    poly = np.stack([r + xs_ * ca - ys_ * sa, r + xs_ * sa + ys_ * ca], 1)
    cv2.fillPoly(patch, [np.round(poly * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
    cv2.line(rib, (int(r + lx[0] * ca), int(r + lx[0] * sa)), (int(r + lx[-1] * 1.25 * ca), int(r + lx[-1] * 1.25 * sa)),
             255, 1, cv2.LINE_AA)
    a = patch.astype(np.float32) / 255 * alpha
    ya, yb, xa, xb = yi - r, yi + r + 1, xi - r, xi + r + 1
    py0, px0 = max(0, -ya), max(0, -xa)
    ya, xa = max(0, ya), max(0, xa); yb, xb = min(comp.shape[0], yb), min(comp.shape[1], xb)
    a = a[py0:py0 + yb - ya, px0:px0 + xb - xa][..., None]
    rb = (rib.astype(np.float32) / 255 * alpha * 0.45)[py0:py0 + yb - ya, px0:px0 + xb - xa][..., None]
    shade = 0.85 + 0.25 * flat
    c = comp[ya:yb, xa:xb]
    c += (np.minimum(255, col * shade) - c) * a
    c += (col * 0.55 - c) * rb


# ------------------------------------------------------------------------------------------------------------ render
def render(L, job, cam, path, clip=None, tail=False, boost=1.0, seed=0, mid_path=None):
    n_total = int(round(job["dur"] * FPS))
    n_clip = 0 if clip is None else min(len(clip), n_total)
    m = n_total - n_clip
    W, H = L["W"], L["H"]
    rng = np.random.default_rng(seed)
    P = cam_path(cam, max(m, 1), L, tail)
    fxl = parse_fx(job.get("fx"), L); FX = fx_setup(fxl, L, rng); fxs = fx_scale(L)
    cm = job.get("char_motion", "breath")
    ch = L["y_bot"] - L["y_top"]
    A_b = A_s = 0.0
    if L["char"] and cm != "none":
        A_b = job.get("breath_amp", 0.0065) * ch * boost
        A_s = job.get("sway_amp", 0.006 if cm == "breath_sway" else 0.0015) * ch * boost
    y_hip = L["y_top"] + job.get("hip", 0.62) * ch
    fw = max(20.0, 0.15 * (L["xr"] - L["xl"]))
    rgb, plate, a_fg, dbg, d_ref = L["rgb"], L["plate"], L["a_fg"], L["dbg"], L["d_ref"]
    ox, oy = np.arange(OW, dtype=np.float32), np.arange(OH, dtype=np.float32)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                            "-framerate", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "16", "-preset", "fast",
                            "-pix_fmt", "yuv420p", "-r", str(FPS), "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    mid = n_total // 2
    for k in range(n_clip):
        enc.stdin.write(np.ascontiguousarray(clip[k]).tobytes())
        if k == mid and mid_path:
            Image.fromarray(clip[k]).save(mid_path, quality=92)
    t_r = time.time(); steam_pts = []
    for k in range(m):
        tc = k / FPS
        if tail and k == 0:
            out = clip[n_clip - 1]                      # pixel-identical seam into the tail
        else:
            fin = min(1.0, k / 12.0) if tail else 1.0  # tail FX fade in over 0.5 s
            ww, hh, cx, cy = float(P["ww"][k]), float(P["hh"][k]), float(P["cx"][k]), float(P["cy"][k])
            sxk, syk = ww / OW, hh / OH; x0c, y0c = cx - ww / 2, cy - hh / 2
            xr = x0c + ox * sxk; yr = y0c + oy * syk
            Xc = np.ascontiguousarray(np.broadcast_to(xr, (OH, OW))); Yc = np.ascontiguousarray(np.broadcast_to(yr[:, None], (OH, OW)))
            to_out = lambda x, y: ((x - x0c) / sxk, (y - y0c) / syk)
            dd = cv2.remap(dbg, Xc, Yc, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) - d_ref
            dol = float(P["dolly"][k]); pxk, pyk = float(P["parx"][k]), float(P["pary"][k])
            Xb = (Xc + dd * (pxk - dol * (Xc - cx))).astype(np.float32); Yb = (Yc + dd * (pyk - dol * (Yc - cy))).astype(np.float32)
            bg = cv2.remap(plate, Xb, Yb, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101).astype(np.float32)
            # ---- background FX (behind glass / in the sky): drawn on the plate before the character goes on
            for kind in ("snow", "rain"):
                if kind not in FX:
                    continue
                S = FX[kind]; b = S["box"]
                if kind == "snow":
                    px = wrap(S["x"] + 18 * np.sin(tc * 0.9 + S["p"]), b[0], b[2])
                    py = wrap(S["y"] + (80 + 70 * S["s"]) * tc, b[1], b[3])
                else:
                    sp = 1500 + 400 * S["s"]
                    py = wrap(S["y"] + sp * tc, b[1], b[3]); px = wrap(S["x"] + 0.16 * sp * tc, b[0], b[2])
                keep = S["mask"][py.astype(int).clip(0, H - 1), px.astype(int).clip(0, W - 1)] > 0
                qx, qy = to_out(px[keep], py[keep])
                if kind == "snow":
                    lay, bb = blob_layer([(a_, b_, (3.0 + 3.4 * s) * fxs / sxk, 1.0) for a_, b_, s in zip(qx, qy, S["s"][keep])],
                                         (OH, OW), 1.1)
                    gain = 0.9
                else:
                    ln = (38 + 30 * S["s"][keep]) * fxs / syk
                    lay, bb = line_layer([(a_, b_, a_ - 0.16 * l_, b_ - l_, 0.9) for a_, b_, l_ in zip(qx, qy, ln)], (OH, OW), 0.7)
                    gain = 0.55
                if bb:
                    y0, y1, x0, x1 = bb; c = bg[y0:y1, x0:x1]
                    # clip by the sky/glass mask sampled through the SAME map as the plate: pixel-accurate, so streak
                    # tails and parallax never spill onto the frame, wall or character
                    clipm = cv2.remap(S["maskf"], np.ascontiguousarray(Xb[y0:y1, x0:x1]), np.ascontiguousarray(Yb[y0:y1, x0:x1]),
                                      cv2.INTER_LINEAR)
                    c += (255 - c) * (gain * fin) * (lay * clipm)[..., None]
            def leaves(target, front):
                S = FX["leaves"]; b = S["box"]
                for i in range(len(S["x"])):
                    if (S["s"][i] >= 0.95) != front:     # smaller = farther: behind the character
                        continue
                    lx = S["x"][i] + 60 * S["s"][i] * math.sin(tc * 1.3 / S["s"][i] + S["p"][i])
                    ly = wrap(S["y"][i] + S["v"][i] * tc, b[1] - 120, b[3] + 120)
                    qx, qy = to_out(lx, ly)
                    ang = math.degrees(S["p"][i] + tc * 2.2 * S["s"][i]) % 360
                    flat = 0.2 + 0.8 * abs(math.cos(tc * 2.6 * S["s"][i] + S["p"][i]))   # flutter: turns edge-on and back
                    draw_leaf(target, qx, qy, fxs * 40 * S["s"][i] / sxk, ang, flat, S["col"][i], 0.97 * fin)
            if "leaves" in FX:
                leaves(bg, False)
            # ---- character on twos
            if A_b or A_s:
                tm = (k // 2) * 2 / FPS
                br = math.sin(2 * math.pi * tm / 3.6); sw = math.sin(2 * math.pi * tm / 5.3 + 0.7)
                wrow = np.clip((y_hip - yr) / max(1.0, y_hip - L["y_top"]), 0, 1)[:, None]
                wcol = np.clip(1 - (np.maximum(L["xl"] - xr, xr - L["xr"])) / fw, 0, 1)[None, :]
                wt = wrow * wcol
                my = (Yc + (A_b * br) * wt).astype(np.float32); mx = (Xc - (A_s * sw) * (wrow * wrow * wcol)).astype(np.float32)
            else:
                mx, my = Xc, Yc
            fg = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
            al = cv2.remap(a_fg, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            comp = bg + (fg - bg) * al[..., None]
            # ---- foreground FX
            if "steam" in FX:
                if k % 2 == 0 or not steam_pts:
                    steam_pts = []
                    tm2 = (k // 2) * 2 / FPS
                    for (sx0, sy0) in FX["steam"]:
                        for j in range(30):
                            life = 2.8; age = (tm2 + j * life / 30) % life; f = age / life
                            qx, qy = to_out(sx0 + fxs * (26 * math.sin(2.1 * age + j) * f + 10 * math.sin(j * 1.7)),
                                            sy0 - fxs * 270 * f)
                            steam_pts.append((qx, qy, fxs * (12 + 30 * f) / sxk, math.sin(math.pi * f) ** 1.4))
                lay, bb = blob_layer(steam_pts, (OH, OW), fxs * 13 / sxk)
                if bb:
                    y0, y1, x0, x1 = bb; c = comp[y0:y1, x0:x1]; c += (255 - c) * (0.42 * fin) * lay[..., None]
            if "dust" in FX:
                S = FX["dust"]; b = S["box"]
                px = wrap(S["x"] + 30 * np.sin(tc * 0.4 + S["p"]), b[0], b[2])
                py = wrap(S["y"] - 16 * tc + 12 * np.cos(tc * 0.5 + S["p"]), b[1], b[3])
                lit = (L["lum"][py.astype(int).clip(0, H - 1), px.astype(int).clip(0, W - 1)] / 255.0) ** 1.5
                tw = (0.3 + 0.7 * np.sin(tc * 2.4 + 3 * S["p"]) ** 2) * (0.2 + 0.8 * lit)
                qx, qy = to_out(px, py)
                lay, bb = blob_layer([(a_, b_, (1.6 + 2.6 * s) * fxs / sxk, w_) for a_, b_, s, w_ in zip(qx, qy, S["s"], tw)],
                                     (OH, OW), 2.2)
                if bb:
                    y0, y1, x0, x1 = bb; c = comp[y0:y1, x0:x1]
                    c += (np.array([255, 236, 190], np.float32) - c) * (0.6 * fin) * lay[..., None]
            if "leaves" in FX:
                leaves(comp, True)
            if "light" in FX:
                g = cv2.cvtColor(np.clip(comp, 0, 255), cv2.COLOR_RGB2GRAY) / 255.0
                comp *= (1 + 0.05 * fin * math.sin(2 * math.pi * tc / 4.7) * g * g)[..., None]
            out = np.clip(comp, 0, 255).astype(np.uint8)
        enc.stdin.write(np.ascontiguousarray(out).tobytes())
        if n_clip + k == mid and mid_path:
            Image.fromarray(out).save(mid_path, quality=92)
    enc.stdin.close(); enc.wait()
    rs = time.time() - t_r
    return {"n_clip": n_clip, "n_tail": m if clip is not None else 0, "frames": n_total, "base": P["B"], "cam_move": P["move"],
            "render_s": round(rs, 1), "ms_per_frame": round(1000 * rs / max(1, m), 1), "fx_skipped": FX.get("_skipped", []),
            "amp_px": {"breath": round(A_b, 1), "sway": round(A_s, 1)}}


# ------------------------------------------------------------------------------------------------------------- audit
def audit(path, fps=12.0, w=270, static_thr=0.35, t0=0.0):
    """Port of research/round4/tools/motion_audit.py (whole clip, optionally from t0 seconds)."""
    p = probe(path)
    W0, H0 = p["w"], p["h"]
    h = int(round(H0 * w / W0 / 2) * 2)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, h, w)
    n = len(fr)
    mad = np.zeros(n); flow = np.zeros(n); res = np.zeros(n)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    X = np.stack([xx.ravel(), yy.ravel(), np.ones(h * w, np.float32)], 1)
    for i in range(1, n):
        a0, a1 = fr[i - 1].astype(np.float32), fr[i].astype(np.float32)
        mad[i] = np.abs(a1 - a0).mean()
        f = cv2.calcOpticalFlowFarneback(fr[i - 1], fr[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow[i] = np.median(np.linalg.norm(f, axis=2)) * (W0 / w)
        F2 = f.reshape(-1, 2)
        A, *_ = np.linalg.lstsq(X[::7], F2[::7], rcond=None)
        res[i] = np.percentile(np.linalg.norm(F2 - X @ A, axis=1), 97) * (W0 / w)
    i0 = max(1, int(round(t0 * fps)) + 1)
    m_, fl, rs = mad[i0:], flow[i0:], res[i0:]
    if len(m_) == 0:
        return {"n": 0}
    static = m_ < static_thr
    run = best = 0
    for s in static:
        run = run + 1 if s else 0
        best = max(best, run)
    return {"n": int(len(m_)), "fps_sampled": fps, "mad_mean": round(float(m_.mean()), 3), "mad_p10": round(float(np.percentile(m_, 10)), 3),
            "flow_med_px": round(float(np.median(fl)), 2), "flow_p95_px": round(float(np.percentile(fl, 95)), 2),
            "subject_p97_med_px": round(float(np.median(rs)), 2), "subject_still_frac": round(float((rs < 1.0).mean()), 3),
            "static_frac": round(float(static.mean()), 3), "longest_static_s": round(best / fps, 2)}


def gate(a, cam_move):
    if not a.get("n"):
        return {"pass": True, "reason": "too short to audit"}
    ok_static = a["longest_static_s"] <= 0.5
    ok_move = a["subject_p97_med_px"] >= 1.0 or cam_move >= 0.03
    why = []
    if not ok_static:
        why.append(f"static run {a['longest_static_s']} s > 0.5")
    if not ok_move:
        why.append(f"subject {a['subject_p97_med_px']} px < 1 and camera {cam_move:.3f} < 0.03")
    return {"pass": ok_static and ok_move, "reason": "; ".join(why) or "ok"}


def sheet(path, out, n):
    step = max(1, n // 8)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-vf", f"select=not(mod(n\\,{step})),scale=240:-2,tile=8x1",
                    "-frames:v", "1", "-vsync", "vfr", out])


# --------------------------------------------------------------------------------------------------------------- jobs
def render_job(job, outdir):
    sid = job["id"]; rec = {"id": sid, "ok": False, "dur": job["dur"], "warnings": []}
    t_all = time.time()
    seed = int(job.get("seed", int(hashlib.sha1(sid.encode()).hexdigest()[:6], 16)))
    if job.get("blink"):
        rec["blink"] = "skipped: no robust procedural eyelid blink; FLUX blink edits are not pixel-aligned with the key"
    clip = None
    if job.get("tail_of"):
        cp = resolve(job["tail_of"])
        clip = read_clip(cp, rec["warnings"])
        n_total = int(round(job["dur"] * FPS))
        if len(clip) >= n_total:
            rec["warnings"].append(f"clip has {len(clip)} frames >= {n_total}: trimmed, no tail")
        last = np.ascontiguousarray(clip[min(len(clip), n_total) - 1])
        L = build_layers(last, OW, OH)
        L["prep"]["clip_frames"] = int(len(clip))
        tail = True
    else:
        L = key_layers(job["image"], job.get("crop"))
        tail = False
    L["seed_key"] = sid
    rec["prep"] = {k: v for k, v in L["prep"].items()}
    rec["char_found"] = bool(L["char"])
    cam = dict(job.get("cam") or {"type": "push_in", "amount": 0.08})
    cam.setdefault("amount", 0.08)
    path = os.path.join(outdir, sid + ".mp4"); mid = os.path.join(outdir, sid + "_mid.jpg")
    rec["attempts"] = []
    boost = 1.0
    for attempt in range(2):
        r = render(L, job, cam, path, clip=clip, tail=tail, boost=boost, seed=seed, mid_path=mid)
        a = audit(path, t0=(r["n_clip"] / FPS) if tail else 0.0)
        n_r = r["frames"] - r["n_clip"]
        r["cam_move_per_5s"] = round(r["cam_move"] * 120 / max(1, n_r), 4) if tail else r["cam_move"]
        g = gate(a, r["cam_move_per_5s"])
        rec["attempts"].append(dict(r, cam=dict(cam), audit=a, gate=g))
        if g["pass"] or attempt == 1:
            break
        cam["amount"] = min(0.16, cam["amount"] * 1.6); boost = 1.3
    rec.update({k: v for k, v in rec["attempts"][-1].items()})
    pr = probe(path)
    rec["frames_probe"] = pr["frames"]; rec["fps_probe"] = pr["fps"]; rec["size"] = [pr["w"], pr["h"]]
    if pr["frames"] != rec["frames"]:
        raise RuntimeError(f"{sid}: {pr['frames']} frames encoded, {rec['frames']} requested")
    sheet(path, os.path.join(outdir, sid + "_sheet.jpg"), rec["frames"])
    small = lambda x: cv2.resize((x * 255).astype(np.uint8), (360, int(360 * L["H"] / L["W"])))
    cv2.imwrite(os.path.join(outdir, sid + "_matte.jpg"), np.concatenate([small(L["a_fg"]), small(L["d"])], 1))  # matte | depth
    rec["ok"] = True
    rec["total_s"] = round(time.time() - t_all, 1)
    return rec


def main(jobs_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    jobs = json.load(open(jobs_path))
    allrec = {"n_jobs": len(jobs), "shots": {}}
    t = time.time(); models(); allrec["load_s"] = round(time.time() - t, 1)
    print("models loaded", allrec["load_s"], "s", flush=True)
    for job in jobs:
        try:
            r = render_job(job, outdir)
        except Exception:
            r = {"id": job.get("id"), "ok": False, "error": traceback.format_exc()[-2000:]}
        json.dump(r, open(os.path.join(outdir, f"{job.get('id')}.json"), "w"), indent=1)
        allrec["shots"][job.get("id")] = r
        print(job.get("id"), json.dumps({k: r.get(k) for k in ("ok", "frames", "frames_probe", "render_s", "ms_per_frame",
                                                                "cam_move", "gate", "error")})[:1500], flush=True)
        json.dump(allrec, open(os.path.join(outdir, "la_rec.json"), "w"), indent=1)
    allrec["wall_s"] = round(time.time() - t, 1)
    json.dump(allrec, open(os.path.join(outdir, "la_rec.json"), "w"), indent=1)
    return allrec


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
