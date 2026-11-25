# Heartwick round 4: FLUX.2-klein-4B (Apache-2.0, 4-step distilled, multi-reference edit) keyframe speed/quality on Kaggle T4 x2.
# IMGS (base64 refs) and JOBS are prepended by build.py. One process per T4, each takes half of the jobs.
import base64, json, os, subprocess, sys, time
from pathlib import Path

W = Path("/kaggle/working"); OUT = W / "out"; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "logs"; LOG.mkdir(exist_ok=True); IN = W / "in"; IN.mkdir(exist_ok=True)
for k, b in IMGS.items():
    (IN / k).write_bytes(base64.b64decode(b))
status = {"t_start": time.time()}


def save():
    (OUT / "status.json").write_text(json.dumps(status, indent=1))


def sh(cmd, log, env=None, timeout=None):
    t = time.time()
    with open(LOG / log, "a") as f:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
        for line in p.stdout:
            f.write(line); f.flush()
            if "it/s]" not in line and "s/it]" not in line:
                print(f"[{log}] {line}", end="", flush=True)
        rc = p.wait(timeout=timeout)
    return {"rc": rc, "s": round(time.time() - t, 1)}


status["setup"] = sh("pip install -q -U 'diffusers>=0.37' 'transformers>=4.57' accelerate 2>&1 | tail -2; "
                     "python -c 'import diffusers,transformers,torch;print(diffusers.__version__,transformers.__version__,torch.__version__)'",
                     "setup.log", timeout=1500)
# weight-cache read test: FastWan transformer + VAE saved as the output of CPU kernel heartwick-r4-wcache, mounted here
WC = sorted(Path("/kaggle/input").glob("**/fastwan5b/transformer"))
if WC:
    root = WC[0].parent
    status["wcache_ls"] = sh(f"ls -la {root}/transformer; df -h {root} | tail -1", "wcache.log")
    status["wcache_read"] = sh(f"cat {root}/transformer/*.safetensors > /dev/null", "wcache.log")
    status["wcache_load_gpu"] = sh(f"CUDA_VISIBLE_DEVICES=0 python -c \"import time,torch;t=time.time();from diffusers import WanTransformer3DModel as M;"
                                   f"m=M.from_pretrained('{root}',subfolder='transformer',torch_dtype=torch.float16).to('cuda');"
                                   f"torch.cuda.synchronize();print('load_to_gpu_s',round(time.time()-t,1))\"", "wcache.log")
else:
    status["wcache"] = "not mounted"
save()
t = time.time()
from huggingface_hub import snapshot_download
snapshot_download("black-forest-labs/FLUX.2-klein-4B", allow_patterns=["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*",
                                                                      "transformer/*", "vae/*"])
status["download_s"] = round(time.time() - t, 1); save()

GEN = r'''
import json, sys, time, torch
from PIL import Image
from diffusers import Flux2KleinPipeline
jobs = json.load(open(sys.argv[1])); out = sys.argv[2]; rec = {}
t = time.time()
pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
rec["load_s"] = round(time.time() - t, 1)
for j in jobs:
    refs = [Image.open(p).convert("RGB") for p in j["refs"]]
    for seed in j["seeds"]:
        torch.cuda.synchronize(); t = time.time()
        try:
            im = pipe(prompt=j["prompt"], image=refs or None, height=j["h"], width=j["w"], guidance_scale=1.0,
                      num_inference_steps=4, generator=torch.Generator("cuda").manual_seed(seed)).images[0]
            torch.cuda.synchronize(); dt = round(time.time() - t, 1)
            import numpy as np
            a = np.asarray(im); ok = bool(a.std() > 3)
            im.save(f"{out}/{j['id']}_s{seed}.png")
            rec[f"{j['id']}_s{seed}"] = {"s": dt, "nonblank": ok, "w": j["w"], "h": j["h"], "n_refs": len(refs),
                                         "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        except Exception as e:
            rec[f"{j['id']}_s{seed}"] = {"error": repr(e)[-600:]}
        print(j["id"], seed, rec[f"{j['id']}_s{seed}"], flush=True)
        json.dump(rec, open(f"{out}/rec_{sys.argv[3]}.json", "w"), indent=1)
'''
(W / "gen.py").write_text(GEN)
KO = OUT / "keys"; KO.mkdir(exist_ok=True)
J = [dict(j, refs=[str(IN / r) for r in j["refs"]]) for j in JOBS]
procs = []
for g in (0, 1):
    pf = W / f"jobs_{g}.json"; pf.write_text(json.dumps(J[g::2]))
    e = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    procs.append(subprocess.Popen(f"python {W/'gen.py'} {pf} {KO} g{g} > {LOG}/gen_g{g}.log 2>&1", shell=True, env=e))
t = time.time()
for p in procs:
    p.wait()
status["gen_wall_s"] = round(time.time() - t, 1); status["rc"] = [p.returncode for p in procs]
sh(f"tail -n 25 {LOG}/gen_g0.log {LOG}/gen_g1.log", "final.log")
status["total_s"] = round(time.time() - status["t_start"], 1); save()
print(json.dumps(status, indent=1))
