# Heartwick round 3, session 1 (Kaggle T4 x2): Wan2.2-I2V-A14B permutations on ONE fixed beat (Nell walks to the
# maple, kneels, lifts Pip, smiles), same key K0, prompt and seed 7 unless the axis itself changes them.
#   phase 1 (both T4s, one gen3.py per GPU): Q5 480x832 at 4 / 6 / 8 steps, Q5 544x960 4 steps
#   phase 2 (one expert across BOTH T4s):    Q8_0 480x832 4 steps, Q5 720x1280 4 steps
#   phase 3: finishing grid on the base clip (RIFE 30/60 fps x ESRGAN anime/general) + default finish of every clip
#   phase 4: QA (identity, Pip scale, pose, headroom, cut-offs) + contact sheets for every raw clip, incl. round 1 / 2
# Sources GEN3, FIN3, QA3, QAC are prepended by build.py.
import glob, json, os, subprocess, sys, textwrap, threading, time
from pathlib import Path

W = Path("/kaggle/working"); OUT = W / "out"; OUT.mkdir(parents=True, exist_ok=True)
LOG = W / "logs"; LOG.mkdir(exist_ok=True)
for n, s in (("gen3.py", GEN3), ("finish3.py", FIN3), ("qa3.py", QA3), ("qa_consistency.py", QAC)):
    (W / n).write_text(s)
status = {"t_start": time.time()}
IN = Path(sorted(glob.glob("/kaggle/input/**/K0.png", recursive=True))[0]).parent
status["inputs"] = str(IN)


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


rc, s = sh("pip install -q -U 'diffusers>=0.36' 'transformers>=4.51' accelerate gguf ftfy imageio imageio-ffmpeg spandrel rtmlib onnxruntime 2>&1 | tail -3", "setup.log", 1200)
status["setup"] = {"rc": rc, "s": s}
sh("nvidia-smi; free -g; nproc; df -h /root /kaggle/working | tail -2; python -c 'import diffusers,transformers,torch;print(diffusers.__version__,transformers.__version__,torch.__version__)'", "setup.log")
save()

REPO = "jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF"
FILES = {q: {"high": f"high_noise_260412/wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412-{q}.gguf",
             "low": f"low_noise_260412/wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412-{q}.gguf"} for q in ("Q5_K_M", "Q8_0")}
PATHS, DL = {}, {}


def dl(q, which):
    from huggingface_hub import hf_hub_download
    t = time.time()
    for attempt in range(3):
        try:
            PATHS.setdefault(q, {})[which] = hf_hub_download(REPO, FILES[q][which]); break
        except Exception as e:
            DL[f"{q}_{which}_err{attempt}"] = repr(e)[:200]
    DL[f"{q}_{which}_s"] = round(time.time() - t, 1)


q5 = [threading.Thread(target=dl, args=("Q5_K_M", w)) for w in ("high", "low")]
[t.start() for t in q5]

# ---- umt5 prompt embeddings (GPU0 fp16, NaN check, CPU fallback). P3 / PJ are for session 2 (keys + FLF) -----------
PROMPTS = {
    "p_base": ("2D hand-drawn anime animation. The older woman in the green cardigan takes two slow steps to the maple tree, "
               "kneels down on the leaves, and gently lifts the small grey squirrel in both hands, smiling. Golden maple leaves "
               "keep falling. The camera slowly pushes in and drifts right. Smooth natural motion, consistent character, no morphing."),
    "p_v3": ("2D hand-drawn anime animation. The older woman in the green cardigan walks slowly to the maple tree, kneels down "
             "on one knee in the fallen leaves, and gently picks up the small grey squirrel in both hands, holding it at her chest "
             "and smiling at it. The squirrel stays small, the same size the whole time. Golden maple leaves keep falling. The camera "
             "slowly pushes in. Smooth natural motion, consistent character, no morphing."),
    "p_jewel": ("2D flat-shaded storybook animation with rich jewel-tone colours and clean outlines. The older woman in the green "
                "cardigan walks slowly to the maple tree, kneels down on one knee in the fallen leaves, and gently picks up the small "
                "grey squirrel in both hands, holding it at her chest and smiling at it. The squirrel stays small, the same size the "
                "whole time. Golden maple leaves keep falling. The camera slowly pushes in. Smooth natural motion, consistent "
                "character, no morphing."),
}
T0 = textwrap.dedent(f'''
import torch, time, json, gc
from transformers import AutoTokenizer, UMT5EncoderModel
repo = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"; PS = {PROMPTS!r}
tok = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")
r = {{}}
t0 = time.time(); te = UMT5EncoderModel.from_pretrained(repo, subfolder="text_encoder", torch_dtype=torch.float16).to("cuda")
for k, P in PS.items():
    t = tok([P], padding="max_length", max_length=512, truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
    n = int(t.attention_mask.sum())
    with torch.no_grad():
        e = te(t.input_ids.cuda(), t.attention_mask.cuda()).last_hidden_state[:, :n].float().cpu()
    r[k] = {{"tokens": n, "finite": bool(torch.isfinite(e).all())}}
    e = torch.cat([e, e.new_zeros(1, 512 - n, e.shape[-1])], 1)
    torch.save({{"pos": e}}, f"/kaggle/working/out/emb_{{k}}.pt")
r["s"] = round(time.time() - t0, 1); print(r); open("/kaggle/working/out/emb.json", "w").write(json.dumps(r))
''')
(W / "t0.py").write_text(T0)
rc, s = sh(f"CUDA_VISIBLE_DEVICES=0 python {W/'t0.py'}", "t0.log", 1800)
status["emb"] = {"rc": rc, "s": s}
[t.join() for t in q5]
q8 = [threading.Thread(target=dl, args=("Q8_0", w)) for w in ("high", "low")]   # downloads during phase 1
[t.start() for t in q8]
status["downloads"] = DL; save()

K0 = str(IN / "K0.png")


def emb(k):
    return str(OUT / f"emb_{k}.pt")


def clip(cid, w, h, seed=7, steps=4, frames=81, image=K0, last=None, prompt="p_base"):
    c = {"id": cid, "image": image, "w": w, "h": h, "frames": frames, "seed": seed, "steps": steps, "emb": emb(prompt)}
    if last:
        c["last"] = last
    return c


def plan(name, q, clips, split=False):
    p = {"name": name, "high": PATHS[q]["high"], "low": PATHS[q]["low"], "quant": q, "clips": clips, "split": split}
    f = W / f"plan_{name}.json"; f.write_text(json.dumps(p)); return f


def run(jobs, timeout=5400):
    """jobs: list of (gpus, plan_file, name). Runs them concurrently; returns {name: rc}."""
    procs = {}
    for gpus, pf, name in jobs:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        f = open(LOG / f"gen_{name}.log", "w")
        procs[name] = (subprocess.Popen([sys.executable, str(W / "gen3.py"), "--plan", str(pf), "--out", str(OUT)],
                                        stdout=f, stderr=subprocess.STDOUT, env=env), time.time())
    rcs = {}
    for name, (p, t) in procs.items():
        try:
            rcs[name] = {"rc": p.wait(timeout=max(60, timeout - (time.time() - t))), "wall_s": round(time.time() - t, 1)}
        except subprocess.TimeoutExpired:
            p.kill(); rcs[name] = {"rc": -999, "wall_s": round(time.time() - t, 1)}
    return rcs


# ---- phase 1: steps + 544 on both T4s ---------------------------------------------------------------------------------
t = time.time()
status["p1"] = run([("0", plan("p1_g0", "Q5_K_M", [clip("q5_480_4s", 480, 832), clip("q5_480_8s", 480, 832, steps=8)]), "p1_g0"),
                    ("1", plan("p1_g1", "Q5_K_M", [clip("q5_480_6s", 480, 832, steps=6), clip("q5_544_4s", 544, 960)]), "p1_g1")])
status["p1"]["wall_s"] = round(time.time() - t, 1); save()
[t.join() for t in q8]
status["downloads"] = DL; save()

# ---- phase 2a: Q8_0 sharded across both T4s ------------------------------------------------------------------------------
t = time.time()
if "Q8_0" in PATHS and len(PATHS["Q8_0"]) == 2:
    status["p2_q8"] = run([("0,1", plan("p2_q8", "Q8_0", [clip("q8_480_4s", 480, 832)], split=True), "p2_q8")], timeout=3000)
status["p2_q8_wall_s"] = round(time.time() - t, 1); save()
for f in PATHS.get("Q8_0", {}).values():   # free ~31 GB of disk
    try:
        os.remove(os.path.realpath(f))
    except OSError:
        pass

# ---- phase 3: finishing grid + default finish of every clip ---------------------------------------------------------------
FIN = OUT / "finished"; FIN.mkdir(exist_ok=True)
base = OUT / "q5_480_4s.mp4"
fj = []
if base.exists():
    for fps in (30, 60):
        for e in ("anime", "general"):
            fj.append({"in": str(base), "out": str(FIN / f"q5_480_4s__rife{fps}_{e}.mp4"), "fps": fps, "esr": e})
for c in ("q5_480_6s", "q5_480_8s", "q5_544_4s", "q8_480_4s"):
    if (OUT / f"{c}.mp4").exists():
        fj.append({"in": str(OUT / f"{c}.mp4"), "out": str(FIN / f"{c}__rife30_anime.mp4"), "fps": 30, "esr": "anime"})
(W / "fin1.json").write_text(json.dumps(fj))
rc, s = sh(f"CUDA_VISIBLE_DEVICES=0 python {W/'finish3.py'} {W/'fin1.json'}", "finish.log", 3600)
status["finish1"] = {"rc": rc, "s": s}; sh(f"cp {W}/fin1_rec.json {OUT}/ 2>/dev/null", "final.log"); save()

# ---- phase 4: QA + contact sheets ------------------------------------------------------------------------------------------
def qa_all():
    raws = sorted(str(p) for p in OUT.glob("*.mp4")) + [str(IN / n) for n in ("r1_wan22_i2v.mp4", "r2_q4_480.mp4", "r2_q5_480.mp4", "r2_q5_480_s11.mp4", "r2_q4_544.mp4")]
    env = dict(os.environ, QA_DEVICE="cuda:0", QA_THREADS="4")
    return sh(f"cd {W} && python qa3.py --refs {K0} --inputs {' '.join(raws)} --out {OUT/'qa.json'} --sheets {OUT/'sheets'}",
              "qa.log", 2400, env=env)


rc, s = qa_all(); status["qa1"] = {"rc": rc, "s": s}; save()

# ---- phase 2b: 720x1280 sharded (last: the most expensive and most likely to fail) ------------------------------------------
t = time.time()
status["p2_720"] = run([("0,1", plan("p2_720", "Q5_K_M", [clip("q5_720_4s", 720, 1280)], split=True), "p2_720")], timeout=4200)
status["p2_720_wall_s"] = round(time.time() - t, 1); save()
if (OUT / "q5_720_4s.mp4").exists():
    (W / "fin2.json").write_text(json.dumps([{"in": str(OUT / "q5_720_4s.mp4"), "out": str(FIN / "q5_720_4s__rife30_anime.mp4"), "fps": 30, "esr": "anime"}]))
    rc, s = sh(f"CUDA_VISIBLE_DEVICES=0 python {W/'finish3.py'} {W/'fin2.json'}", "finish.log", 1800)
    status["finish2"] = {"rc": rc, "s": s}; sh(f"cp {W}/fin2_rec.json {OUT}/ 2>/dev/null", "final.log")
    rc, s = qa_all(); status["qa2"] = {"rc": rc, "s": s}
status["total_s"] = round(time.time() - status["t_start"], 1); save()
sh(f"cp {LOG}/*.log {OUT}/ ; ls -la {OUT} {FIN}", "final.log")
print(json.dumps(status, indent=1))
