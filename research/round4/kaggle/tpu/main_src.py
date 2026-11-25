# Heartwick round 4: Wan2.2-I2V-A14B (bf16) + lightx2v 4-step distill LoRA on a Kaggle TPU v5e-8 via MaxDiffusion (JAX).
# IMGS (base64 keys) and PROMPT are prepended by build.py. Measures: install, download, load, XLA compile, per-clip time.
import base64, json, os, subprocess, sys, threading, time
from pathlib import Path

W = Path("/kaggle/working"); OUT = W / "out"; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "logs"; LOG.mkdir(exist_ok=True); IN = W / "in"; IN.mkdir(exist_ok=True)
for k, b in IMGS.items():
    (IN / k).write_bytes(base64.b64decode(b))
status = {"t_start": time.time()}
TMP = Path("/tmp/r4"); TMP.mkdir(exist_ok=True)


def save():
    (OUT / "status.json").write_text(json.dumps(status, indent=1))


def sh(cmd, log, env=None, timeout=None):
    t = time.time()
    with open(LOG / log, "a") as f:
        f.write(f"\n$ {cmd}\n"); f.flush()
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1,
                             executable="/bin/bash")
        for line in p.stdout:
            f.write(line); f.flush()
            if "it/s]" not in line and "s/it]" not in line and "Downloading" not in line:
                print(f"[{log}] {line}", end="", flush=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill(); rc = -999
    print(f"[{log}] rc={rc} {round(time.time() - t, 1)}s", flush=True)
    return {"rc": rc, "s": round(time.time() - t, 1)}


status["env"] = sh("python -V; nproc; free -g; df -h / /kaggle/working /tmp; ls /dev | grep -i accel | head; "
                   "python -c 'import jax; print(jax.__version__, jax.devices())' 2>&1 | tail -3", "env.log", timeout=600)
save()
du = subprocess.run("df -BG --output=avail /tmp | tail -1", shell=True, capture_output=True, text=True).stdout.strip().rstrip("G")
status["disk_free_gb"] = int(du) if du.isdigit() else None; save()
if status["disk_free_gb"] is not None and status["disk_free_gb"] < 85:
    print("NOT ENOUGH DISK for 57 GB transformers + 12 GB encoder:", status["disk_free_gb"], flush=True)

# ---- weights download (background) ---------------------------------------------------------------------------------------
HF = str(TMP / "hf"); os.makedirs(HF, exist_ok=True)


def dl():
    t = time.time()
    r = subprocess.run(
        f"pip install -q -U huggingface_hub hf_transfer >/dev/null 2>&1; HF_HOME={HF} HF_HUB_ENABLE_HF_TRANSFER=1 python -c \""
        "from huggingface_hub import snapshot_download as s;"
        "s('ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16');"
        "s('Wan-AI/Wan2.2-I2V-A14B-Diffusers', allow_patterns=['model_index.json','scheduler/*','tokenizer/*','text_encoder/*','vae/*']);"
        "s('lightx2v/Wan2.2-Distill-Loras', allow_patterns=['wan2.2_i2v_A14b_*_4step_1022.safetensors'])\"",
        shell=True, capture_output=True, text=True)
    status["download"] = {"rc": r.returncode, "s": round(time.time() - t, 1), "err": r.stderr[-800:]}
    save()


th = threading.Thread(target=dl); th.start()

# ---- MaxDiffusion in its own Python 3.12 venv ------------------------------------------------------------------------------
MD = TMP / "maxdiffusion"; VENV = TMP / "venv"
status["install"] = sh(
    f"pip install -q uv && git clone -q --depth 1 https://github.com/AI-Hypercomputer/maxdiffusion.git {MD} && cd {MD} && "
    f"git log -1 --format=%H && uv venv -q -p 3.12 {VENV} && source {VENV}/bin/activate && "
    f"uv pip install -q -r dependencies/requirements/generated_requirements/requirements.txt && uv pip install -q -e . --no-deps && "
    f"python -c 'import jax; print(jax.__version__, jax.devices())'", "install.log", timeout=3600)
save()
th.join(); save()
sh(f"du -sh {HF}/hub/* ; df -h /tmp | tail -1; free -g | head -2", "env.log")

# ---- config: base_wan_i2v_27b.yml + overrides --------------------------------------------------------------------------------
import yaml
base = yaml.safe_load(open(MD / "src/maxdiffusion/configs/base_wan_i2v_27b.yml"))
base["lora_config"] = {"rank": [64], "lora_model_name_or_path": ["lightx2v/Wan2.2-Distill-Loras"],
                       "high_noise_weight_name": ["wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors"],
                       "low_noise_weight_name": ["wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"],
                       "adapter_name": ["lightx2v"], "scale": [1.0], "from_pt": []}
CFG = TMP / "r4_wan22_i2v.yml"; yaml.safe_dump(base, open(CFG, "w"), sort_keys=False)

COMMON = (f"pretrained_model_name_or_path=Wan-AI/Wan2.2-I2V-A14B-Diffusers "
          f"wan_transformer_pretrained_model_name_or_path=ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16 "
          f"enable_lora=True num_inference_steps=4 guidance_scale_low=1.0 guidance_scale_high=1.0 flow_shift=5.0 "
          f"attention=flash per_device_batch_size=0.125 ici_data_parallelism=1 ici_context_parallelism=8 "
          f"jax_cache_dir={W}/jax_cache fps=16 seed=7 write_metrics=False enable_profiler=False "
          f"converted_weights_dir={TMP}/converted")
RUNS = [{"id": "tpu_a14b_480_k2", "h": 832, "w": 480, "f": 81, "img": "K2.jpg"},
        {"id": "tpu_a14b_720_k2", "h": 1280, "w": 720, "f": 81, "img": "K2.jpg"}]
env = dict(os.environ, HF_HOME=HF, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1",
           LIBTPU_INIT_ARGS="--xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true "
                            "--xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true "
                            "--xla_enable_async_all_reduce=true")
(TMP / "prompt.txt").write_text(PROMPT)
for r in RUNS:
    od = OUT / r["id"]; od.mkdir(exist_ok=True)
    cmd = (f"source {VENV}/bin/activate && cd {MD} && python src/maxdiffusion/generate_wan.py {CFG} {COMMON} "
           f"run_name={r['id']} output_dir={od} height={r['h']} width={r['w']} num_frames={r['f']} image_url={IN/r['img']} "
           f"prompt=\"$(cat {TMP/'prompt.txt'})\"")
    status[r["id"]] = sh(cmd, f"{r['id']}.log", env, 5400); save()
    sh(f"grep -iE 'load_time|compile_time|generation_time|generation time|Compile:|Inference:|Load:|Error|error' {LOG}/{r['id']}.log | tail -25; ls -la {od}", "summary.log")
sh(f"du -sh {W}/jax_cache; rm -rf {W}/jax_cache", "summary.log")   # keep the kernel output small
status["total_s"] = round(time.time() - status["t_start"], 1); save()
print(json.dumps(status, indent=1))
