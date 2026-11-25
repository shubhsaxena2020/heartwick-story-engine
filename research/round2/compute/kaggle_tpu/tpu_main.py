# Heartwick round 2: is Wan2.2-I2V-A14B (+ lightx2v 4-step distill LoRA) usable on Kaggle's free TPU v5e-8?
# Uses Google's MaxDiffusion (JAX), which lists Wan2.2 I2V + Wan LoRA inference on TPU. Measures load, compile and
# per-video generation time at 480x832x81 (1 video, then 2 videos in one batch) and 720x1280x81.
# KEY_B64 is prepended by build.py.
import base64, json, os, subprocess, sys, time
from pathlib import Path

W = Path("/kaggle/working"); OUT = W / "out"; OUT.mkdir(parents=True, exist_ok=True)
(W / "key.jpg").write_bytes(base64.b64decode(KEY_B64))
st = {"t0": time.time()}


def save():
    (OUT / "status.json").write_text(json.dumps(st, indent=1))


def sh(cmd, log, timeout=None):
    t = time.time()
    with open(OUT / log, "a") as f:
        f.write(f"\n$ {cmd}\n"); f.flush()
        try:
            rc = subprocess.run(["bash", "-lc", cmd], stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -999
    return {"rc": rc, "s": round(time.time() - t, 1)}


st["env"] = sh("python3 --version; nproc; free -g; df -h; ls /dev/accel* /dev/vfio 2>/dev/null | head; "
               "python3 -c 'import jax;print(jax.__version__, jax.devices())' 2>&1 | tail -3", "env.log", 600)
save()
# biggest writable disk for the ~130 GB of weights
cand = []
for p in ("/kaggle/tmp", "/tmp", "/root", "/kaggle/working"):
    try:
        os.makedirs(p, exist_ok=True); s = os.statvfs(p); cand.append((s.f_bavail * s.f_frsize, p))
    except Exception:
        pass
cand.sort(reverse=True); BIG = cand[0][1]; st["disk"] = {p: round(b / 1e9) for b, p in cand}; save()
HF = f"{BIG}/hf"; MD = f"{BIG}/maxdiffusion"; VENV = f"{BIG}/mdv"

st["install"] = sh(f"set -e; pip install -q uv; git clone -q https://github.com/AI-Hypercomputer/maxdiffusion.git {MD}; "
                   f"python3 -m uv venv --python 3.12 {VENV} --seed -q; source {VENV}/bin/activate; cd {MD}; "
                   f"bash setup.sh MODE=stable DEVICE=tpu 2>&1 | tail -25; python -c 'import jax;print(jax.__version__, jax.devices())'",
                   "install.log", 3600)
save()

import yaml  # noqa: E402  (system python has PyYAML)
cfg = yaml.safe_load(open(f"{MD}/src/maxdiffusion/configs/base_wan_i2v_27b.yml"))
cfg.update({
    "prompt": ("2D hand-drawn anime animation. The older woman in the green cardigan takes two slow steps to the maple tree, "
               "kneels down on the leaves, and gently lifts the small grey squirrel in both hands, smiling. Golden maple leaves "
               "keep falling. The camera slowly pushes in and drifts right. Smooth natural motion, consistent character, no morphing."),
    "negative_prompt": "static, morphing, extra limbs, blurry, text, watermark, low quality",
    "image_url": str(W / "key.jpg"), "height": 832, "width": 480, "num_frames": 81, "fps": 16, "flow_shift": 5.0,
    "num_inference_steps": 4, "do_classifier_free_guidance": False, "guidance_scale_low": 1.0, "guidance_scale_high": 1.0,
    "enable_lora": True, "seed": 7, "output_dir": str(OUT), "per_device_batch_size": 0.125,
    "lora_config": {"rank": [64], "lora_model_name_or_path": ["lightx2v/Wan2.2-Distill-Loras"],
                    "high_noise_weight_name": ["wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors"],
                    "low_noise_weight_name": ["wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"],
                    "adapter_name": ["wan22-distill-lora"], "scale": [1.0], "from_pt": []},
})
runs = [("r480_b1", {}), ("r480_b2", {"per_device_batch_size": 0.25}), ("r720_b1", {"height": 1280, "width": 720})]
for name, over in runs:
    c = dict(cfg, **over, run_name=f"hw-{name}")
    yml = W / f"{name}.yml"; yaml.safe_dump(c, open(yml, "w"), sort_keys=False)
    st[name] = sh(f"source {VENV}/bin/activate; cd {MD}; set -o pipefail; export HF_HUB_CACHE={HF}; "
                  f"export LIBTPU_INIT_ARGS='--xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true "
                  f"--xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_reduce=true'; "
                  f"python src/maxdiffusion/generate_wan.py {yml} 2>&1 | grep -v '^\\s*$' | tail -400", f"{name}.log", 7200)
    sh(f"grep -E 'load_time|compile_time|generation_time|generation time|Error|error' {OUT}/{name}.log | tail -20; free -g; df -h {BIG}", f"{name}.log")
    save()
    if st[name]["rc"] != 0 and name == "r480_b1":
        break
st["total_s"] = round(time.time() - st["t0"], 1); save()
