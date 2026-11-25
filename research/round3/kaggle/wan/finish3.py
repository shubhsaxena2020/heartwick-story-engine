# Heartwick round 3 finishing on a Kaggle T4 (replaces the laptop iGPU finish.sh; same steps, same look):
#   RIFE 4.25 (Practical-RIFE, MIT) 16 fps -> 30 or 60 fps (arbitrary-timestep interpolation, exact frame count)
#   Real-ESRGAN x4 (BSD-3): "anime" = realesr-animevideov3 (SRVGG compact), "general" = RealESRGAN_x4plus (RRDB)
#   cover-crop to 1080x1920 with the round-1 slow push-in (zoom 1.00 -> 1.05, 12 px drift right), x264 crf 17.
# usage: python finish3.py jobs.json   (jobs: [{"in": clip.mp4, "out": out.mp4, "fps": 30, "esr": "anime"}, ...])
import json, os, subprocess, sys, time, types
import numpy as np
import torch
import torch.nn.functional as F

jobs = json.load(open(sys.argv[1]))
D = torch.device("cuda")
MD = "/kaggle/working/fin_models"
os.makedirs(MD, exist_ok=True)


def fetch(url, name):
    p = os.path.join(MD, name)
    if not os.path.exists(p):
        subprocess.run(["wget", "-q", "-O", p, url], check=True)
    return p


# ---- RIFE 4.25 ------------------------------------------------------------------------------------------------------
from huggingface_hub import hf_hub_download


def warp(tenInput, tenFlow):
    B, _, H, W = tenFlow.shape
    hor = torch.linspace(-1.0, 1.0, W, device=tenFlow.device, dtype=tenFlow.dtype).view(1, 1, 1, W).expand(B, -1, H, -1)
    ver = torch.linspace(-1.0, 1.0, H, device=tenFlow.device, dtype=tenFlow.dtype).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([hor, ver], 1)
    tenFlow = torch.cat([tenFlow[:, 0:1] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
    g = (grid + tenFlow).permute(0, 2, 3, 1)
    return F.grid_sample(input=tenInput, grid=g, mode="bilinear", padding_mode="border", align_corners=True)


mod = types.ModuleType("model"); wl = types.ModuleType("model.warplayer"); wl.warp = warp
sys.modules["model"] = mod; sys.modules["model.warplayer"] = wl
src = open(hf_hub_download("MonsterMMORPG/Practical-RIFE", "train_log/IFNet_HDv3.py")).read()
ns = {}
exec(compile(src, "IFNet_HDv3.py", "exec"), ns)
rife = ns["IFNet"]()
sd = torch.load(hf_hub_download("MonsterMMORPG/Practical-RIFE", "train_log/flownet.pkl"), map_location="cpu")
sd = {k.replace("module.", ""): v for k, v in sd.items()}
missing, unexpected = rife.load_state_dict(sd, strict=False)
rife = rife.to(D).eval()
INFO = {"rife_missing": len(missing), "rife_unexpected": len(unexpected)}


@torch.no_grad()
def rife_mid(a, b, t):
    """a, b: (1,3,H,W) in [0,1] fp32 on D; H, W padded to /128"""
    flow, mask, merged = rife(torch.cat([a, b], 1), t, [16, 8, 4, 2, 1])
    return merged[-1]


# ---- Real-ESRGAN via spandrel ---------------------------------------------------------------------------------------
from spandrel import ModelLoader
ESR_URL = {"anime": ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth", "realesr-animevideov3.pth"),
           "general": ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", "RealESRGAN_x4plus.pth")}
_esr = {}


def esr(name):
    if name not in _esr:
        m = ModelLoader().load_from_file(fetch(*ESR_URL[name])).model
        _esr[name] = m.to(D).half().eval()
    return _esr[name]


def read_frames(p):
    import imageio.v3 as iio
    return np.stack(list(iio.imiter(p)))


def pad128(x):
    H, W = x.shape[2:]
    ph, pw = (128 - H % 128) % 128, (128 - W % 128) % 128
    return F.pad(x, (0, pw, 0, ph), mode="replicate"), H, W


res = {}
for j in jobs:
    t0 = time.time()
    fr = read_frames(j["in"])
    n = len(fr); src_fps = j.get("src_fps", 16); dur = n / src_fps
    M = int(round(dur * j["fps"]))
    T = torch.from_numpy(fr).to(D).permute(0, 3, 1, 2).float() / 255.0
    # ---- 1. RIFE to exactly M frames spanning first..last source frame
    t1 = time.time()
    out = []
    for k in range(M):
        p = k * (n - 1) / (M - 1)
        i0 = int(np.floor(p)); f = p - i0
        if f < 1e-3 or i0 >= n - 1:
            out.append(T[min(i0, n - 1)][None])
        else:
            a, H, W = pad128(T[i0][None]); b, _, _ = pad128(T[i0 + 1][None])
            out.append(rife_mid(a, b, float(f))[:, :, :H, :W].clamp(0, 1))
    rife_s = time.time() - t1
    del T
    # ---- 2. ESRGAN x4 + push-in + 1080x1920, piped to x264
    t2 = time.time()
    m = esr(j["esr"])
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1080x1920",
                            "-r", str(j["fps"]), "-i", "-", "-c:v", "libx264", "-crf", "17", "-preset", "medium",
                            "-pix_fmt", "yuv420p", j["out"]], stdin=subprocess.PIPE)
    for k, x in enumerate(out):
        with torch.no_grad():
            y = m(x.half()).float().clamp(0, 1)                      # (1,3,4H,4W)
        H4, W4 = y.shape[2:]
        s = max(1080 / W4, 1920 / H4)                                  # cover-crop scale
        z = 1.0 + 0.05 * k / max(1, M - 1)
        cw, ch = 1080 / (s * z), 1920 / (s * z)
        cx = W4 / 2 + 12 * (k / max(1, M - 1)) / s; cy = H4 / 2
        x0 = int(round(min(max(cx - cw / 2, 0), W4 - cw))); y0 = int(round(min(max(cy - ch / 2, 0), H4 - ch)))
        crop = y[:, :, y0:y0 + int(round(ch)), x0:x0 + int(round(cw))]
        o = F.interpolate(crop, size=(1920, 1080), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
        enc.stdin.write((o[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy().tobytes())
    enc.stdin.close(); enc.wait()
    torch.cuda.synchronize()
    res[os.path.basename(j["out"])] = {"in": os.path.basename(j["in"]), "fps": j["fps"], "esr": j["esr"], "src_frames": n,
                                       "out_frames": M, "rife_s": round(rife_s, 1), "esr_encode_s": round(time.time() - t2, 1),
                                       "total_s": round(time.time() - t0, 1)}
    print(res[os.path.basename(j["out"])], flush=True)
    del out; torch.cuda.empty_cache()
res["_info"] = INFO
json.dump(res, open(os.path.splitext(sys.argv[1])[0] + "_rec.json", "w"), indent=1)
