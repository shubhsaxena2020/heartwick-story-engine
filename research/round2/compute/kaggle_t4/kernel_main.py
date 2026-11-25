# Heartwick round 2 compute test on Kaggle T4 x2. Measures Wan2.2-I2V-A14B (lightx2v 4-step distill 260412, GGUF)
# seconds of GPU per clip, with BOTH T4s busy at once (one gen.py process per GPU, phase-split by expert).
# GEN_SRC and KEY_B64 are prepended by build.py.
import base64, json, os, subprocess, sys, time, textwrap, threading
from pathlib import Path

W = Path("/kaggle/working"); OUT = W / "out"; OUT.mkdir(parents=True, exist_ok=True)
LOG = W / "logs"; LOG.mkdir(exist_ok=True)
(W / "key.jpg").write_bytes(base64.b64decode(KEY_B64))
(W / "gen.py").write_text(GEN_SRC)
status = {"t_start": time.time()}


def save():
    (OUT / "status.json").write_text(json.dumps(status, indent=1))


def sh(cmd, log, timeout=None, env=None):
    t = time.time()
    with open(LOG / log, "a") as f:
        f.write(f"\n$ {cmd}\n"); f.flush()
        try:
            rc = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, env=env).returncode
        except subprocess.TimeoutExpired:
            rc = -999
    return rc, round(time.time() - t, 1)


rc, s = sh("pip install -q -U 'diffusers>=0.36' 'transformers>=4.51' accelerate gguf ftfy imageio imageio-ffmpeg 2>&1 | tail -3", "setup.log", 900)
status["setup"] = {"rc": rc, "s": s}
sh("nvidia-smi; free -g; nproc; df -h /root /kaggle/working | tail -2; python -c 'import diffusers,transformers,torch;print(diffusers.__version__,transformers.__version__,torch.__version__)'", "setup.log")
save()

# ---- downloads (parallel with the microbench + text-encoder step) --------------------------------------------------
REPO = "jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF"
FILES = {q: {"high": f"high_noise_260412/wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412-{q}.gguf",
             "low": f"low_noise_260412/wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412-{q}.gguf"} for q in ("Q4_K_M", "Q5_K_M")}
PATHS, DL = {}, {}


def dl(q, which):
    from huggingface_hub import hf_hub_download
    t = time.time()
    PATHS.setdefault(q, {})[which] = hf_hub_download(REPO, FILES[q][which])
    DL[f"{q}_{which}_s"] = round(time.time() - t, 1)


threads = [threading.Thread(target=dl, args=(q, w)) for q in FILES for w in ("high", "low")]
[t.start() for t in threads]

# ---- microbench: attention backends + fp16 GEMM on one T4 ----------------------------------------------------------
BENCH = textwrap.dedent('''
import torch, time, json, torch.nn.functional as F
r = {}
def tm(fn, n=3):
    fn(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n
a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16); b = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
s = tm(lambda: a @ b, 5); r["gemm_fp16_tflops"] = round(2 * 8192**3 / s / 1e12, 1)
del a, b
for L in (21168, 32760, 42840):
    q = torch.randn(1, 40, L, 128, device="cuda", dtype=torch.float16); k = torch.randn_like(q); v = torch.randn_like(q)
    s = tm(lambda: F.scaled_dot_product_attention(q, k, v), 2); r[f"sdpa_L{L}_s"] = round(s, 2); r[f"sdpa_L{L}_tflops"] = round(4 * L * L * 128 * 40 / s / 1e12, 1)
    try:
        from sageattention import sageattn
        s = tm(lambda: sageattn(q, k, v, tensor_layout="HND"), 2); r[f"sage_L{L}_s"] = round(s, 2)
    except Exception as e:
        r["sage_err"] = repr(e)[:300]
    del q, k, v; torch.cuda.empty_cache()
print(json.dumps(r)); open("/kaggle/working/out/bench.json", "w").write(json.dumps(r, indent=1))
''')
(W / "bench.py").write_text(BENCH)
rc, s = sh("pip install -q sageattention==1.0.6 2>&1 | tail -2", "bench.log", 300)
rc, s = sh(f"CUDA_VISIBLE_DEVICES=0 python {W/'bench.py'}", "bench.log", 900)
status["bench"] = {"rc": rc, "s": s}; save()

# ---- text embeddings (umt5-xxl fp16 on GPU 0, NaN check, CPU bf16 fallback) ----------------------------------------
PROMPT = ("2D hand-drawn anime animation. The older woman in the green cardigan takes two slow steps to the maple tree, "
          "kneels down on the leaves, and gently lifts the small grey squirrel in both hands, smiling. Golden maple leaves "
          "keep falling. The camera slowly pushes in and drifts right. Smooth natural motion, consistent character, no morphing.")
T0 = textwrap.dedent(f'''
import torch, time, json, gc
from transformers import AutoTokenizer, UMT5EncoderModel
repo = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"; P = {PROMPT!r}
tok = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")
t = tok([P], padding="max_length", max_length=512, truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
n = int(t.attention_mask.sum()); r = {{"tokens": n}}
def run(dev, dt):
    t0 = time.time(); te = UMT5EncoderModel.from_pretrained(repo, subfolder="text_encoder", torch_dtype=dt).to(dev)
    with torch.no_grad():
        e = te(t.input_ids.to(dev), t.attention_mask.to(dev)).last_hidden_state[:, :n].float().cpu()
    del te; gc.collect(); torch.cuda.empty_cache(); return e, round(time.time() - t0, 1)
e, s = run("cuda", torch.float16); r["gpu_fp16_s"] = s; r["finite"] = bool(torch.isfinite(e).all())
if not r["finite"]:
    e, s = run("cpu", torch.bfloat16); r["cpu_bf16_s"] = s; r["finite_cpu"] = bool(torch.isfinite(e).all())
e = torch.cat([e, e.new_zeros(1, 512 - n, e.shape[-1])], 1)
torch.save({{"pos": e}}, "/kaggle/working/emb.pt"); print(r); open("/kaggle/working/out/t0.json", "w").write(json.dumps(r))
''')
(W / "t0.py").write_text(T0)
rc, s = sh(f"CUDA_VISIBLE_DEVICES=0 python {W/'t0.py'}", "t0.log", 1800)
status["t0"] = {"rc": rc, "s": s}
sh("free -g", "t0.log")
[t.join() for t in threads]
status["downloads"] = DL; save()

EMB = str(W / "emb.pt"); KEY = str(W / "key.jpg")


def clip(cid, w, h, frames, seed, image=KEY):
    return {"id": cid, "image": image, "w": w, "h": h, "frames": frames, "seed": seed, "emb": EMB}


def plan(name, q, clips):
    p = {"name": name, "high": PATHS[q]["high"], "low": PATHS[q]["low"], "quant": q, "clips": clips}
    f = W / f"plan_{name}.json"; f.write_text(json.dumps(p)); return f


def run_pair(jobs, timeout=5400):
    """jobs: list of (gpu, plan_file, name). Runs them concurrently; returns {name: rc}."""
    procs = {}
    for gpu, pf, name in jobs:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        f = open(LOG / f"gen_{name}.log", "w")
        procs[name] = (subprocess.Popen([sys.executable, str(W / "gen.py"), "--plan", str(pf), "--out", str(OUT)],
                                        stdout=f, stderr=subprocess.STDOUT, env=env), time.time())
    rcs = {}
    for name, (p, t) in procs.items():
        try:
            rcs[name] = {"rc": p.wait(timeout=max(60, timeout - (time.time() - t))), "wall_s": round(time.time() - t, 1)}
        except subprocess.TimeoutExpired:
            p.kill(); rcs[name] = {"rc": -999, "wall_s": round(time.time() - t, 1)}
    return rcs


# round 1: both T4s at once. GPU0 = Q4_K_M (3 clips: base, low-res, half-frames), GPU1 = Q5_K_M (2 clips: same seed + new seed)
p0 = plan("g0_q4", "Q4_K_M", [clip("q4_480x832_81f", 480, 832, 81, 7), clip("q4_384x672_81f", 384, 672, 81, 7),
                              clip("q4_480x832_41f", 480, 832, 41, 7)])
p1 = plan("g1_q5", "Q5_K_M", [clip("q5_480x832_81f", 480, 832, 81, 7), clip("q5_480x832_81f_s11", 480, 832, 81, 11)])
t = time.time()
status["round1"] = run_pair([(0, p0, "g0_q4"), (1, p1, "g1_q5")]); status["round1"]["wall_s"] = round(time.time() - t, 1); save()
if status["round1"]["g1_q5"]["rc"] != 0:   # Q5 OOM on a 15 GB T4? fall back to Q4 so the parallel number still exists
    p1 = plan("g1_q4", "Q4_K_M", [clip("g1q4_480x832_81f_s11", 480, 832, 81, 11)])
    status["round1_g1_retry"] = run_pair([(1, p1, "g1_q4")]); save()

# round 2: GPU0 = last-frame continuation of the base clip; GPU1 = a higher generation resolution (544x960)
last = OUT / "q4_480x832_81f_last.png"
jobs = []
if last.exists():
    jobs.append((0, plan("g0_cont", "Q4_K_M", [clip("q4_cont_480x832_81f", 480, 832, 81, 8, str(last))]), "g0_cont"))
jobs.append((1, plan("g1_544", "Q4_K_M", [clip("q4_544x960_81f", 544, 960, 81, 7)]), "g1_544"))
t = time.time()
status["round2"] = run_pair(jobs); status["round2"]["wall_s"] = round(time.time() - t, 1); save()

# contact sheets: 5 frames per clip
SHEET = textwrap.dedent('''
import imageio.v3 as iio, glob, os
from PIL import Image
for p in sorted(glob.glob("/kaggle/working/out/*.mp4")):
    fr = iio.imread(p); n = len(fr); idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    ims = [Image.fromarray(fr[i]) for i in idx]; w, h = ims[0].size
    s = Image.new("RGB", (w * 5, h)); [s.paste(im, (i * w, 0)) for i, im in enumerate(ims)]
    s.thumbnail((2400, 2400)); s.save(p[:-4] + "_sheet.jpg", quality=85)
''')
(W / "sheet.py").write_text(SHEET)
sh(f"python {W/'sheet.py'}", "sheet.log", 600)
status["total_s"] = round(time.time() - status["t_start"], 1); save()
sh(f"cp {LOG}/*.log {OUT}/ ; ls -la {OUT}", "final.log")
print(json.dumps(status, indent=1))
