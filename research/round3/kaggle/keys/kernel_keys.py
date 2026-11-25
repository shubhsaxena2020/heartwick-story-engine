# Heartwick round 3, session 2 (Kaggle T4 x2): keyframe axis + first-last-frame (FLF2V) + style, on the same beat.
#   1. keys3.py (both T4s): end keys by Z-Image alone / Qwen-Edit chain / Qwen-Edit chain + OpenPose / direct edit,
#      jewel-tone restyle pair; QA picks seeds.
#   2. Wan2.2-I2V-A14B Q5 480x832 4-step (the session-1 baseline settings), seed 7:
#      FLF K0 -> each end key, jewel FLF, plain I2V with the improved prompt (prompt effect alone), FLF seed 11.
#   3. default finish (RIFE 30 + ESRGAN anime -> 1080x1920) + QA + contact sheets.
# Sources GEN3, FIN3, QA3, QAC, KEYS3 are prepended by build.py.
import glob, json, os, subprocess, sys, textwrap, threading, time
from pathlib import Path

W = Path("/kaggle/working"); OUT = W / "out"; OUT.mkdir(parents=True, exist_ok=True)
LOG = W / "logs"; LOG.mkdir(exist_ok=True); KEYS = OUT / "keys"; KEYS.mkdir(exist_ok=True)
for n, s in (("gen3.py", GEN3), ("finish3.py", FIN3), ("qa3.py", QA3), ("qa_consistency.py", QAC), ("keys3.py", KEYS3)):
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


rc, s = sh("pip install -q -U diffusers 'transformers>=4.51' accelerate gguf ftfy imageio imageio-ffmpeg spandrel rtmlib onnxruntime bitsandbytes 2>&1 | tail -3", "setup.log", 1200)
status["setup"] = {"rc": rc, "s": s}
sh("nvidia-smi; free -g; nproc; df -h /root /kaggle/working | tail -2; python -c 'import diffusers,transformers,torch,bitsandbytes;print(diffusers.__version__,transformers.__version__,torch.__version__,bitsandbytes.__version__)'", "setup.log")
save()

# ---- downloads ---------------------------------------------------------------------------------------------------------
REPO = "jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF"
WAN = {w: f"{w}_noise_260412/wan2.2_i2v_A14b_{w}_noise_lightx2v_4step_720p_260412-Q5_K_M.gguf" for w in ("high", "low")}
P, DL = {}, {}


def dl(key, fn):
    t = time.time()
    for attempt in range(3):
        try:
            P[key] = fn(); break
        except Exception as e:
            DL[f"{key}_err{attempt}"] = repr(e)[:300]
    DL[f"{key}_s"] = round(time.time() - t, 1)


from huggingface_hub import hf_hub_download, snapshot_download
jobs_dl = {
    "zimage": lambda: snapshot_download("Tongyi-MAI/Z-Image-Turbo"),
    "qwen_repo": lambda: snapshot_download("Qwen/Qwen-Image-Edit-2511", allow_patterns=["model_index.json", "processor/*", "scheduler/*", "text_encoder/*",
                                                                                          "tokenizer/*", "vae/*", "transformer/config.json"]),
    "qwen_gguf": lambda: hf_hub_download("unsloth/Qwen-Image-Edit-2511-GGUF", "qwen-image-edit-2511-Q4_K_M.gguf"),
    "lora": lambda: hf_hub_download("lightx2v/Qwen-Image-Edit-2511-Lightning", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"),
    "wan_high": lambda: hf_hub_download(REPO, WAN["high"]), "wan_low": lambda: hf_hub_download(REPO, WAN["low"]),
}
th = {k: threading.Thread(target=dl, args=(k, f)) for k, f in jobs_dl.items()}
[t.start() for t in th.values()]

# ---- Wan prompt embeddings: reuse session 1's (kernel source), else compute -----------------------------------------------
EMB = {}
for k in ("p_base", "p_v3", "p_jewel"):
    hits = sorted(glob.glob(f"/kaggle/input/**/emb_{k}.pt", recursive=True))
    if hits:
        EMB[k] = hits[0]
status["emb_from_session1"] = sorted(EMB)
if len(EMB) < 3:
    PROMPTS = json.loads(PROMPTS_JSON)
    T0 = textwrap.dedent(f'''
    import torch, json
    from transformers import AutoTokenizer, UMT5EncoderModel
    repo = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"; PS = {PROMPTS!r}
    tok = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")
    te = UMT5EncoderModel.from_pretrained(repo, subfolder="text_encoder", torch_dtype=torch.float16).to("cuda")
    for k, P in PS.items():
        t = tok([P], padding="max_length", max_length=512, truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        n = int(t.attention_mask.sum())
        with torch.no_grad():
            e = te(t.input_ids.cuda(), t.attention_mask.cuda()).last_hidden_state[:, :n].float().cpu()
        e = torch.cat([e, e.new_zeros(1, 512 - n, e.shape[-1])], 1)
        torch.save({{"pos": e}}, f"/kaggle/working/out/emb_{{k}}.pt"); print(k, n, bool(torch.isfinite(e).all()))
    ''')
    (W / "t0.py").write_text(T0)
    status["emb"] = sh(f"CUDA_VISIBLE_DEVICES=0 python {W/'t0.py'}", "t0.log", 1800)
    EMB = {k: str(OUT / f"emb_{k}.pt") for k in ("p_base", "p_v3", "p_jewel")}
save()
for k in ("zimage", "qwen_repo", "qwen_gguf", "lora"):
    th[k].join()
status["downloads"] = DL; save()

# ---- 1. keys ------------------------------------------------------------------------------------------------------------
K0 = str(IN / "K0.png"); PK_R = str(IN / "pose_kneel_reach.png"); PK_H = str(IN / "pose_kneel_hold.png")
FRAME = ("Vertical 9:16 composition: her whole body from the top of her hair to her shoes is inside the frame, her head in the upper "
         "third with clear space above her hair, nothing cut off by the frame edges.")
KEEP = ("Keep her face, glasses, hair, clothes and colours identical, keep the same small grey squirrel, and keep the exact same "
        "hand-drawn anime art style, line art, colours and lighting. No text.")
SAME = ("Keep everything identical: the same autumn yard, maple tree, house, fence, camera angle and framing, lighting and art style, "
        "the same character and the same small grey squirrel.")
KNEEL = ("her right knee and shin rest on the ground behind her, her left foot is planted flat in front with the left knee bent up, "
         "her back is straight. She is kneeling on one knee, not crouching or squatting.")
HOLD = ("she has gently lifted the small grey squirrel and holds it in both cupped hands at her chest, smiling warmly at it. The "
        "squirrel is exactly the same size as before: its body fits inside her two hands, it is not bigger.")
SKEL = " Image 2 is a pose skeleton: match her body pose to it exactly. No skeleton lines in the result."
JEWEL = ("Redraw image 1 as the same frame in a flatter, richer storybook style: flat vector-like colour shapes with soft airbrushed "
         "gradients and gentle glows, clean confident dark outlines, deep saturated jewel-tone colours (emerald green, ruby red, "
         "sapphire blue sky, marigold and gold leaves), no watercolour texture. Keep the exact same composition, camera, character pose, "
         "face, glasses, hair, clothes, and the same small grey squirrel at the same size and place. No text.")
ZI = ("Hand-drawn 2D anime film still, Studio-Ghibli-like warm watercolor backgrounds, clean line art, vertical 9:16 frame. Autumn New "
      "England yard: a big red-orange maple tree on the right, golden leaves falling, white farmhouse with picket fence far left, soft "
      "late-afternoon sun rays. Nell, a kind older woman about 70, silver hair in a low bun, round glasses, dark green knitted cardigan "
      "over a cream blouse, long rust-brown skirt, brown shoes, kneels on one knee in the fallen leaves beside the maple trunk, her other "
      "foot flat on the ground, holding a small grey squirrel gently in both cupped hands at her chest and smiling warmly at it. The "
      "squirrel is small: its body fits inside her two hands. Her whole body is inside the frame, her head in the upper third with clear "
      "space above her hair. No text, no watermark.")
S = [3, 21]
jobs = {
    "k0": K0, "qwen_gguf": P.get("qwen_gguf"), "qwen_split": 36, "qwen_repo": P.get("qwen_repo"), "zimage_repo": P.get("zimage"),
    "zimage": [{"id": "ZI_K2", "prompt": ZI, "W": 720, "H": 1280, "seeds": [5, 9]}],
    "qwen_layers": [
        [{"id": "QE_K1", "method": "edit_chain", "refs": [K0], "W": 576, "H": 1024, "seeds": S,
          "prompt": f"{SAME} Change only this: Nell has walked to the maple and now kneels on one knee in the leaves right beside the squirrel: {KNEEL} Both of her open hands reach gently toward the squirrel. The squirrel stays small, about the size of her head. {FRAME} {KEEP}"},
         {"id": "QP_K1", "method": "edit_chain_pose", "refs": [K0, PK_R], "W": 576, "H": 1024, "seeds": S,
          "prompt": f"{SAME} Change only this: Nell has walked to the maple and now kneels on one knee in the leaves right beside the squirrel: {KNEEL} Both of her open hands reach gently toward the squirrel. The squirrel stays small, about the size of her head.{SKEL} {FRAME} {KEEP}"},
         {"id": "QD_K2", "method": "edit_direct_pose", "refs": [K0, PK_H], "W": 576, "H": 1024, "seeds": S,
          "prompt": f"{SAME} Change only this: Nell has walked to the maple and kneels on one knee beside its trunk: {KNEEL} And {HOLD}{SKEL} {FRAME} {KEEP}"},
         {"id": "JT_K0", "method": "restyle", "refs": [K0], "W": 576, "H": 1024, "seeds": [3], "pip": True, "prompt": JEWEL}],
        [{"id": "QE_K2", "method": "edit_chain", "refs": ["@QE_K1"], "W": 576, "H": 1024, "seeds": S,
          "prompt": f"{SAME} Change only this: still kneeling on one knee in the same place, {HOLD} {FRAME} {KEEP}"},
         {"id": "QP_K2", "method": "edit_chain_pose", "refs": ["@QP_K1", PK_H], "W": 576, "H": 1024, "seeds": S,
          "prompt": f"{SAME} Change only this: still kneeling on one knee in the same place, {HOLD}{SKEL} {FRAME} {KEEP}"}],
        [{"id": "JT_K2", "method": "restyle", "refs": ["@QP_K2"], "W": 576, "H": 1024, "seeds": [3], "prompt": JEWEL}],
    ],
}
(W / "keys_jobs.json").write_text(json.dumps(jobs, indent=1))
env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", QA_DEVICE="cpu", QA_THREADS="4")
status["keys"] = sh(f"cd {W} && python keys3.py {W/'keys_jobs.json'} {KEYS}", "keys.log", 4200, env=env); save()
chosen = json.load(open(KEYS / "chosen.json")) if (KEYS / "chosen.json").exists() else {}
status["chosen"] = {k: os.path.basename(v) for k, v in chosen.items()}; save()
sh("rm -rf ~/.cache/huggingface/hub/models--Tongyi-MAI* ~/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511 ~/.cache/huggingface/hub/models--unsloth*; df -h /root | tail -1", "final.log")
for k in ("wan_high", "wan_low"):
    th[k].join()
status["downloads"] = DL; save()

# ---- 2. Wan FLF / I2V ---------------------------------------------------------------------------------------------------
def clip(cid, seed=7, image=K0, last=None, prompt="p_v3"):
    c = {"id": cid, "image": image, "w": 480, "h": 832, "frames": 81, "seed": seed, "steps": 4, "emb": EMB[prompt]}
    if last:
        c["last"] = last
    return c


g0, g1 = [], []
if chosen.get("QP_K2"):
    g0 += [clip("flf_qp", last=chosen["QP_K2"]), clip("flf_qp_s11", seed=11, last=chosen["QP_K2"])]
if chosen.get("ZI_K2"):
    g0.append(clip("flf_zi", last=chosen["ZI_K2"]))
if chosen.get("QE_K2"):
    g1.append(clip("flf_qe", last=chosen["QE_K2"]))
if chosen.get("JT_K0") and chosen.get("JT_K2"):
    g1.append(clip("flf_jt", image=chosen["JT_K0"], last=chosen["JT_K2"], prompt="p_jewel"))
g1.append(clip("i2v_v3"))
if chosen.get("QD_K2") and len(g0) + len(g1) < 7:
    (g0 if len(g0) <= len(g1) else g1).append(clip("flf_qd", last=chosen["QD_K2"]))
while len(g0) > len(g1) + 1:
    g1.append(g0.pop())


def plan(name, clips):
    p = {"name": name, "high": P["wan_high"], "low": P["wan_low"], "quant": "Q5_K_M", "clips": clips, "split": False}
    f = W / f"plan_{name}.json"; f.write_text(json.dumps(p)); return f


procs = {}
t = time.time()
for gpu, clips, name in (("0", g0, "s2_g0"), ("1", g1, "s2_g1")):
    if clips:
        e = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        procs[name] = subprocess.Popen([sys.executable, str(W / "gen3.py"), "--plan", str(plan(name, clips)), "--out", str(OUT)],
                                       stdout=open(LOG / f"gen_{name}.log", "w"), stderr=subprocess.STDOUT, env=e)
status["wan"] = {n: {"rc": p.wait(timeout=5400)} for n, p in procs.items()}
status["wan"]["wall_s"] = round(time.time() - t, 1); save()

# ---- 3. finish + QA -----------------------------------------------------------------------------------------------------
FIN = OUT / "finished"; FIN.mkdir(exist_ok=True)
fj = [{"in": str(p), "out": str(FIN / f"{p.stem}__rife30_anime.mp4"), "fps": 30, "esr": "anime"} for p in sorted(OUT.glob("*.mp4"))]
(W / "fin.json").write_text(json.dumps(fj))
status["finish"] = sh(f"CUDA_VISIBLE_DEVICES=0 python {W/'finish3.py'} {W/'fin.json'}", "finish.log", 3600)
sh(f"cp {W}/fin_rec.json {OUT}/ 2>/dev/null", "final.log"); save()
keys = sorted(str(p) for p in KEYS.glob("*.png"))
raws = sorted(str(p) for p in OUT.glob("*.mp4"))
env = dict(os.environ, QA_DEVICE="cuda:0", QA_THREADS="4")
status["qa"] = sh(f"cd {W} && python qa3.py --refs {K0} --inputs {' '.join(raws + keys)} --out {OUT/'qa.json'} --sheets {OUT/'sheets'}", "qa.log", 2400, env=env)
status["total_s"] = round(time.time() - status["t_start"], 1); save()
sh(f"cp {LOG}/*.log {OUT}/ ; ls -la {OUT} {FIN} {KEYS}", "final.log")
print(json.dumps(status, indent=1))
