# Heartwick round 4: "limited animation" shot renderer prototype (anime-style motion on a high-quality still).
# One key image -> Real-ESRGAN animevideov3 x4 (GPU) -> Depth-Anything-V2-Small depth (GPU) + isnet-anime character
# matte (rembg, CPU) -> inpainted background plate -> per-frame render on CPU:
#   camera on ones (push-in + lateral drift) with depth parallax on the background layer,
#   character on twos (12 drawings/s): breathing warp of the upper body, nothing else on the face (mouth stays shut),
#   cinemagraph FX layers: steam, snow behind the glass, dust motes, light breathing.
# usage: python la.py shots.json outdir
import json, math, os, subprocess, sys, time
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

S = json.load(open(sys.argv[1])); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
dev = "cuda"; rec = {}


class SRVGGNetCompact(nn.Module):   # Real-ESRGAN realesr-animevideov3 (BSD-3)
    def __init__(s, c=3, f=64, n=16, up=4):
        super().__init__(); s.up = up
        L = [nn.Conv2d(c, f, 3, 1, 1), nn.PReLU(f)]
        for _ in range(n): L += [nn.Conv2d(f, f, 3, 1, 1), nn.PReLU(f)]
        L += [nn.Conv2d(f, c * up * up, 3, 1, 1)]; s.body = nn.ModuleList(L); s.ps = nn.PixelShuffle(up)

    def forward(s, x):
        o = x
        for l in s.body: o = l(o)
        return s.ps(o) + F.interpolate(x, scale_factor=s.up, mode="nearest")


t0 = time.time()
wp = "/tmp/realesr-animevideov3.pth"
if not os.path.exists(wp):
    subprocess.run(f"wget -q -O {wp} https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth", shell=True)
sd = torch.load(wp, map_location="cpu"); esr = SRVGGNetCompact(); esr.load_state_dict(sd.get("params", sd)); esr = esr.to(dev).half().eval()
from transformers import pipeline
depth_pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0)
from rembg import new_session, remove
seg = new_session("isnet-anime")
rec["load_s"] = round(time.time() - t0, 1)

CW, CH = 1350, 2400            # working canvas (1.25 x 1080x1920)
BASE = 1.12                    # crop window at zoom 1 = canvas / BASE, always downscaled to 1080x1920
OW, OH, FPS = 1080, 1920, 24


def prep(path):
    t = time.time(); r = {}
    im = np.asarray(Image.open(path).convert("RGB"))
    x = torch.from_numpy(im).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev).half()
    with torch.no_grad():
        y = esr(x).clamp(0, 1)[0].float().permute(1, 2, 0).cpu().numpy()
    torch.cuda.synchronize(); r["esrgan_s"] = round(time.time() - t, 2)
    up = Image.fromarray((y * 255).round().astype(np.uint8)).resize((CW, CH), Image.LANCZOS)
    t = time.time()
    d = depth_pipe(up)["predicted_depth"]
    d = d.squeeze().float().cpu().numpy(); d = cv2.resize(d, (CW, CH), interpolation=cv2.INTER_CUBIC)
    d = (d - np.percentile(d, 2)) / (np.percentile(d, 98) - np.percentile(d, 2) + 1e-6); d = np.clip(d, 0, 1)   # 1 = near
    r["depth_s"] = round(time.time() - t, 2)
    t = time.time()
    m = np.asarray(remove(up, session=seg, only_mask=True)).astype(np.float32) / 255.0
    r["matte_s"] = round(time.time() - t, 2)
    t = time.time()
    rgb = np.asarray(up)
    hole = (cv2.dilate((m > 0.2).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0).astype(np.uint8)
    small = cv2.resize(rgb, (CW // 2, CH // 2), interpolation=cv2.INTER_AREA)
    hs = cv2.resize(hole, (CW // 2, CH // 2), interpolation=cv2.INTER_NEAREST)
    fill = cv2.inpaint(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), hs * 255, 9, cv2.INPAINT_TELEA)
    fill = cv2.cvtColor(cv2.resize(fill, (CW, CH), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    plate = np.where(hole[..., None] > 0, fill, rgb)
    dfill = cv2.inpaint((d * 255).astype(np.uint8), hole * 255, 9, cv2.INPAINT_TELEA).astype(np.float32) / 255.0
    r["plate_s"] = round(time.time() - t, 2)
    return rgb, m, plate, d, dfill, r


def particles(kind, n, box, rng):
    x0, y0, x1, y1 = box
    return {"x": rng.uniform(x0, x1, n) * CW, "y": rng.uniform(y0, y1, n) * CH, "s": rng.uniform(0.6, 1.4, n),
            "p": rng.uniform(0, 2 * np.pi, n), "kind": kind}


def blob_layer(pts, shape, sigma):
    """pts: list of (x, y, radius, value) in output pixels -> blurred float layer, drawn and blurred only in its bbox."""
    pts = [p for p in pts if -40 < p[0] < shape[1] + 40 and -40 < p[1] < shape[0] + 40]
    if not pts:
        return None, None
    pad = int(3 * sigma + max(p[2] for p in pts)) + 2
    x0 = max(0, int(min(p[0] for p in pts)) - pad); x1 = min(shape[1], int(max(p[0] for p in pts)) + pad)
    y0 = max(0, int(min(p[1] for p in pts)) - pad); y1 = min(shape[0], int(max(p[1] for p in pts)) + pad)
    if x1 <= x0 or y1 <= y0:
        return None, None
    roi = np.zeros((y1 - y0, x1 - x0), np.float32)
    for x, y, rad, v in pts:
        cv2.circle(roi, (int(x) - x0, int(y) - y0), max(1, int(rad)), float(v), -1, cv2.LINE_AA)
    return cv2.GaussianBlur(roi, (0, 0), sigma), (y0, y1, x0, x1)


def render(shot):
    """Everything is sampled straight into the 1080x1920 output grid: camera, parallax and the character warp are one
    remap per layer, FX are drawn in output pixels inside their bounding box only."""
    rgb, m, plate, d, dbg, r = prep(shot["image"])
    rng = np.random.default_rng(3)
    ys, xs = np.nonzero(m > 0.5)
    y_top = ys.min() if len(ys) else 0
    y_hip = shot.get("hip_y", 0.6) * CH
    d_ref = float(np.median(d[m > 0.5])) if len(ys) else 0.5
    lum = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
    n = int(shot.get("dur", 6) * FPS)
    cam = shot.get("cam", {"zoom": 0.06, "pan": -40})
    snow = particles("snow", 160, shot["snow_box"], rng) if shot.get("snow_box") else None
    dust = particles("dust", 70, shot["dust_box"], rng) if shot.get("dust_box") else None
    steam = shot.get("steam")
    ox, oy = np.arange(OW, dtype=np.float32), np.arange(OH, dtype=np.float32)
    path = os.path.join(OUT, shot["id"] + ".mp4")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", str(FPS),
                            "-i", "-", "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    t_r = time.time(); steam_pts = []
    for k in range(n):
        u = k / max(1, n - 1); e = u * u * (3 - 2 * u); tc = k / FPS
        # camera on ones: output pixel -> canvas coordinates
        z = 1 + cam["zoom"] * e
        ww, hh = CW / BASE / z, CH / BASE / z
        cx = CW / 2 + cam["pan"] * (e - 0.5); cy = CH / 2 + cam.get("tilt", 0) * (e - 0.5)
        sx, sy = ww / OW, hh / OH; x0c, y0c = cx - ww / 2, cy - hh / 2
        Xc = np.ascontiguousarray(np.broadcast_to(x0c + ox * sx, (OH, OW))); Yc = np.ascontiguousarray(np.broadcast_to((y0c + oy * sy)[:, None], (OH, OW)))
        to_out = lambda x, y: ((x - x0c) / sx, (y - y0c) / sy)
        # background parallax on ones
        par = cam["pan"] * 0.45 * (e - 0.5)
        Dc = cv2.remap(dbg, Xc, Yc, cv2.INTER_LINEAR)
        bg = cv2.remap(plate, (Xc + par * (Dc - d_ref)).astype(np.float32), Yc, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        # character on twos: the deformation time only advances every 2nd frame (camera still moves on ones)
        tm = (k // 2) * 2 / FPS
        br = math.sin(2 * math.pi * tm / shot.get("breath_s", 3.6)); sw = math.sin(2 * math.pi * tm / 5.3)
        wgt = np.clip((y_hip - Yc[:, :1]) / max(1.0, y_hip - y_top), 0, 1)
        my = (Yc + shot.get("breath_px", 7) * br * wgt).astype(np.float32)
        mx = (Xc - shot.get("sway_px", 2.5) * sw * wgt * wgt).astype(np.float32)
        fg = np.float32(cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        al = cv2.remap(m, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if snow:   # behind the glass: only where the plate is bright sky
            px = (snow["x"] + 18 * np.sin(tc * 0.9 + snow["p"])) % CW
            py = snow["y"] + (90 + 60 * snow["s"]) * tc
            y0, y1 = shot["snow_box"][1] * CH, shot["snow_box"][3] * CH
            py = y0 + (py - y0) % (y1 - y0)
            keep = lum[py.astype(int).clip(0, CH - 1), px.astype(int).clip(0, CW - 1)] > 184
            qx, qy = to_out(px[keep], py[keep])
            lay, bb = blob_layer([(a, b, (2 + 2 * s) / sx, 1.0) for a, b, s in zip(qx, qy, snow["s"][keep])], (OH, OW), 1.2)
            if bb:
                y0, y1, x0, x1 = bb; c = bg[y0:y1, x0:x1]; c += (255 - c) * 0.85 * lay[..., None]
        comp = cv2.add(bg, cv2.multiply(cv2.subtract(fg, bg), cv2.merge([al, al, al])))
        if steam:
            if k % 2 == 0:   # FX on twos too
                steam_pts = []
                for j in range(26):
                    life = 2.6; age = (tm + j * life / 26) % life; f = age / life
                    qx, qy = to_out(steam[0] * CW + 22 * math.sin(2.1 * age + j) * f + 10 * math.sin(j * 1.7), steam[1] * CH - 230 * f)
                    steam_pts.append((qx, qy, (10 + 26 * f) / sx, math.sin(math.pi * f) ** 1.5))
            lay, bb = blob_layer(steam_pts, (OH, OW), 14 / sx)
            if bb:
                y0, y1, x0, x1 = bb; c = comp[y0:y1, x0:x1]; c += (255 - c) * 0.32 * lay[..., None]
        if dust:
            px = (dust["x"] + 25 * np.sin(tc * 0.4 + dust["p"])) % CW
            py = dust["y"] - 12 * tc + 10 * np.cos(tc * 0.5 + dust["p"])
            tw = 0.5 + 0.5 * np.sin(tc * 2.2 + 3 * dust["p"])
            qx, qy = to_out(px, py)
            lay, bb = blob_layer([(a, b, (3 * s + 1) / sx, w_) for a, b, s, w_ in zip(qx, qy, dust["s"], tw)], (OH, OW), 2.0)
            if bb:
                y0, y1, x0, x1 = bb
                comp[y0:y1, x0:x1] += lay[..., None] * np.array([255, 225, 150], np.float32) * 0.8
        out = cv2.convertScaleAbs(comp, alpha=1 + shot.get("light", 0.012) * math.sin(2 * math.pi * tc / 4.7))
        enc.stdin.write(out.tobytes())
        if k == n // 2:
            Image.fromarray(out).save(os.path.join(OUT, shot["id"] + "_mid.jpg"), quality=92)
    enc.stdin.close(); enc.wait()
    r["render_s"] = round(time.time() - t_r, 1); r["frames"] = n; r["ms_per_frame"] = round(1000 * r["render_s"] / n, 1)
    cv2.imwrite(os.path.join(OUT, shot["id"] + "_matte.jpg"), (m * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(OUT, shot["id"] + "_depth.jpg"), (d * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(OUT, shot["id"] + "_plate.jpg"), cv2.cvtColor(plate, cv2.COLOR_RGB2BGR))
    return r


for shot in S:
    try:
        rec[shot["id"]] = render(shot)
    except Exception:
        import traceback; rec[shot["id"]] = {"error": traceback.format_exc()[-1500:]}
    print(shot["id"], rec[shot["id"]], flush=True)
    json.dump(rec, open(os.path.join(OUT, "la_rec.json"), "w"), indent=1)


def bench_torch(n=144):
    """GPU cost of the same per-frame work (2 layers + depth parallax + composite + download), for the production renderer."""
    g = torch.Generator().manual_seed(0)
    plate = torch.rand(1, 3, CH, CW, generator=g).to(dev).half(); fgi = torch.rand(1, 4, CH, CW, generator=g).to(dev).half()
    dep = torch.rand(1, 1, CH, CW, generator=g).to(dev).half()
    ys = torch.linspace(-1, 1, OH, device=dev).view(OH, 1).expand(OH, OW); xs = torch.linspace(-1, 1, OW, device=dev).view(1, OW).expand(OH, OW)
    torch.cuda.synchronize(); t = time.time(); buf = np.empty((OH, OW, 3), np.uint8)
    for k in range(n):
        z = 1 + 0.07 * k / n; gx = xs / z / BASE + 0.02 * k / n; gy = ys / z / BASE
        grid = torch.stack([gx, gy], -1)[None].half()
        dd = F.grid_sample(dep, grid, align_corners=False)[0, 0]
        bg = F.grid_sample(plate, torch.stack([gx + 0.01 * (dd - 0.5), gy], -1)[None].half(), align_corners=False)
        fy = gy + 0.003 * math.sin(k / 9) * ((ys + 1) / 2)
        fg = F.grid_sample(fgi, torch.stack([gx, fy], -1)[None].half(), align_corners=False)
        a = fg[:, 3:4]; comp = fg[:, :3] * a + bg * (1 - a)
        buf[:] = (comp[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
    torch.cuda.synchronize()
    return {"torch_ms_per_frame": round(1000 * (time.time() - t) / n, 2)}


try:
    rec["torch_bench"] = bench_torch()
except Exception as e:
    rec["torch_bench"] = {"error": repr(e)[:300]}
print("torch bench", rec["torch_bench"], flush=True)
json.dump(rec, open(os.path.join(OUT, "la_rec.json"), "w"), indent=1)
