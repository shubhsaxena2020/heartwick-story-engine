# Heartwick engine v3 kernel 1 (Kaggle T4 x2): character sheets + identity-locked keys + QA, then the music cue.
# ASSETS (base64), KEYS_SRC, QA_SRC, JOBS, MUSIC are prepended by engine3/stages.py.
import base64, json, os, subprocess, sys, time, textwrap
from pathlib import Path

W = Path("/kaggle/working"); OUT = W / "out"; OUT.mkdir(parents=True, exist_ok=True)
LOG = W / "logs"; LOG.mkdir(exist_ok=True); IN = W / "in"; IN.mkdir(exist_ok=True)
for name, b in ASSETS.items():
    (IN / name).write_bytes(base64.b64decode(b))
for name, src in (("keys_gen.py", KEYS_SRC), ("qa.py", QA_SRC)):
    (W / name).write_text(src)
status = {"t_start": time.time()}


def save():
    (OUT / "status.json").write_text(json.dumps(status, indent=1))


def sh(cmd, log, timeout=None, env=None, tee=False):
    t = time.time()
    with open(LOG / log, "a") as f:
        f.write(f"\n$ {cmd}\n"); f.flush()
        try:
            if tee:   # stream to the kernel log too, so `kaggle kernels logs` shows progress live
                p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
                for line in p.stdout:
                    f.write(line); f.flush()
                    if not line.startswith(("Loading", "Fetching", "\r")) and "it/s]" not in line:
                        print(line, end="", flush=True)
                rc = p.wait(timeout=timeout)
            else:
                rc = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, env=env).returncode
        except subprocess.TimeoutExpired:
            rc = -999
    print(f"[{log}] rc={rc} {round(time.time() - t, 1)}s", flush=True)
    return rc, round(time.time() - t, 1)


rc, s = sh("pip install -q -U 'diffusers>=0.36' 'transformers>=4.51' accelerate gguf bitsandbytes peft ftfy imageio imageio-ffmpeg 2>&1 | tail -3", "setup.log", 1200)
status["setup"] = {"rc": rc, "s": s}
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; free -g | head -2; python -c 'import diffusers,transformers,torch;print(diffusers.__version__,transformers.__version__,torch.__version__)'", "setup.log", tee=True)
save()

J = json.loads(JOBS.replace("@IN@", str(IN)))
(W / "jobs.json").write_text(json.dumps(J))
KO = OUT / "keys"; KO.mkdir(exist_ok=True)
for f in IN.glob("*.png"):   # the master + any already-approved keys travel with the outputs (the clips kernel reads them here)
    if not f.name.startswith("pose_"):
        (KO / f.name).write_bytes(f.read_bytes())
env =dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONUNBUFFERED="1")
rc, s = sh(f"cd {W} && python keys_gen.py {W/'jobs.json'} {KO}", "keys.log", 5 * 3600, env, tee=True)
status["keys"] = {"rc": rc, "s": s}; save()
sh("rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511 ~/.cache/huggingface/hub/models--unsloth* "
   "~/.cache/huggingface/hub/models--lightx2v*; df -h /root | tail -1", "keys.log")

# ---- music: ACE-Step 1.5 turbo (MIT), soft storybook pad, 2 takes -------------------------------------------------
if MUSIC:
    ACE = "/tmp/ace"; (OUT / "music").mkdir(exist_ok=True)
    rc, s = sh(f"pip install -q uv && git clone -q --depth 1 https://github.com/ace-step/ACE-Step-1.5 {ACE} && cd {ACE} && uv sync -q --python 3.11", "music.log", 2400)
    MUS = textwrap.dedent(f'''
    import os, sys, glob, shutil, json, time
    os.chdir("/tmp/ace"); sys.path.insert(0, "/tmp/ace")
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music
    dit = AceStepHandler(); dit.initialize_service(project_root="/tmp/ace", config_path="acestep-v15-turbo", device="cuda")
    llm = LLMHandler(); t = time.time()
    p = GenerationParams(caption={MUSIC["caption"]!r}, lyrics="[Instrumental]", instrumental=True, bpm={MUSIC["bpm"]},
                         duration={MUSIC["duration"]}, inference_steps=8, thinking=False)
    try:
        r = generate_music(dit, llm, p, GenerationConfig(batch_size=2, audio_format="wav"), save_dir="/tmp/mus"); ok, err = r.success, None if r.success else str(r.error)[:300]
    except Exception as e:
        ok, err = False, repr(e)[:300]
    if not ok:
        llm.initialize(checkpoint_dir="/tmp/ace/checkpoints", lm_model_path="acestep-5Hz-lm-1.7B", backend="pt", device="cuda")
        r = generate_music(dit, llm, p, GenerationConfig(batch_size=2, audio_format="wav"), save_dir="/tmp/mus"); ok = r.success
    fs = sorted(glob.glob("/tmp/mus/**/*.wav", recursive=True))
    for k, f in enumerate(fs[:2]): shutil.copy(f, f"/kaggle/working/out/music/music_{{k + 1}}.wav")
    json.dump({{"ok": ok, "err": err, "files": len(fs), "s": round(time.time() - t, 1), "params": {MUSIC!r}}}, open("/kaggle/working/out/music/music.json", "w"))
    print("music", ok, err, len(fs))
    ''')
    (W / "music.py").write_text(MUS)
    if rc == 0:
        rc, s = sh(f"cd {ACE} && CUDA_VISIBLE_DEVICES=0 {ACE}/.venv/bin/python {W/'music.py'}", "music.log", 2400, tee=True)
    status["music"] = {"rc": rc, "s": s}
status["total_s"] = round(time.time() - status["t_start"], 1); save()
sh(f"cp {LOG}/*.log {OUT}/ ; ls -la {OUT} {KO}", "final.log", tee=True)
print(json.dumps(status, indent=1))
