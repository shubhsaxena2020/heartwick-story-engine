# Heartwick pipeline prototype B on Kaggle T4: can free Kaggle GPUs run Wan2.2 image-to-video fast enough?
# Same keyframe as the ZeroGPU run (Z-Image-Turbo, Apache-2.0), same beat prompt.
# Tests (each in its own subprocess with a timeout so one failure doesn't kill the rest):
#   t1  FastWan2.2-TI2V-5B (Apache-2.0, DMD 3-step distill) fp16, 480x832, 121 frames @24 fps
#   t2  Wan2.2-I2V-A14B GGUF Q4_K_S (Apache-2.0) + lightx2v Lightning 4-step LoRA (Apache-2.0), 480x832, 81 frames @16 fps
import base64, json, os, subprocess, sys, time, textwrap
from pathlib import Path

W = Path("/kaggle/working"); W.mkdir(exist_ok=True, parents=True)
LOG = W / "logs"; LOG.mkdir(exist_ok=True)
KEY_B64 = "__KEY_B64__"
(W / "key.jpg").write_bytes(base64.b64decode(KEY_B64))

PROMPT = ("2D hand-drawn anime animation. The older woman in the green cardigan takes two slow steps to the maple tree, "
          "kneels down on the leaves, and gently lifts the small grey squirrel in both hands, smiling. Golden maple leaves "
          "keep falling. The camera slowly pushes in and drifts right. Smooth natural motion, consistent character, no morphing.")
NEG = "static, morphing, extra limbs, blurry, text, watermark, low quality"

def sh(cmd, log, timeout=None):
    t = time.time()
    with open(LOG / log, "a") as f:
        f.write(f"\n$ {cmd}\n"); f.flush()
        try:
            rc = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -999
    return rc, round(time.time() - t, 1)

status = {}
rc, s = sh("pip install -q -U 'diffusers>=0.35' 'transformers>=4.51' accelerate gguf ftfy imageio imageio-ffmpeg peft 2>&1 | tail -5", "setup.log", 900)
status["setup"] = {"rc": rc, "s": s}
sh("nvidia-smi; free -g; python -c 'import diffusers,transformers,torch;print(diffusers.__version__,transformers.__version__,torch.__version__)'", "setup.log")

COMMON = textwrap.dedent(f'''
import time, json, torch, gc, numpy as np
from PIL import Image
from diffusers.utils import export_to_video
PROMPT = {PROMPT!r}; NEG = {NEG!r}
img = Image.open("/kaggle/working/key.jpg").convert("RGB").resize((480, 832))
def stamp(d, name):
    d["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    open(f"/kaggle/working/{{name}}.json", "w").write(json.dumps(d, indent=1)); print(d, flush=True)
''')

T0 = COMMON + textwrap.dedent('''
from transformers import AutoTokenizer, UMT5EncoderModel
def enc(repo, tag):
    tok = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")
    te = UMT5EncoderModel.from_pretrained(repo, subfolder="text_encoder", torch_dtype=torch.float16).to("cuda")
    out = {}
    for k, p in (("pos", PROMPT), ("neg", NEG)):
        t = tok([p], padding="max_length", max_length=512, truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        n = int(t.attention_mask.sum())
        with torch.no_grad():
            e = te(t.input_ids.cuda(), t.attention_mask.cuda()).last_hidden_state[:, :n]
        e = torch.cat([e, e.new_zeros(1, 512 - n, e.shape[-1])], 1)
        out[k] = e.cpu()
    torch.save(out, f"/kaggle/working/emb_{tag}.pt"); del te; gc.collect(); torch.cuda.empty_cache()
t0 = time.time(); enc("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "5b"); print("emb s", time.time() - t0)
''')

T1 = COMMON + textwrap.dedent('''
from diffusers import WanImageToVideoPipeline, WanTransformer3DModel, AutoencoderKLWan
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
t0 = time.time()
class DMD3(SchedulerMixin, ConfigMixin):
    """x0-predict then re-noise sampler for the DMD-distilled FastWan (timesteps 1000,757,522)."""
    order = 1
    @register_to_config
    def __init__(self, num_train_timesteps=1000, steps=(1000, 757, 522)):
        self.timesteps = torch.tensor(steps, dtype=torch.float32); self._i = 0
    def set_timesteps(self, n=None, device=None, **kw):
        self.timesteps = torch.tensor(self.config.steps, dtype=torch.float32, device=device); self._i = 0
    def scale_model_input(self, x, t=None): return x
    def step(self, v, t, x, return_dict=False, **kw):
        s = float(t) / 1000.0
        x0 = x.float() - s * v.float()
        self._i += 1
        if self._i < len(self.timesteps):
            sn = float(self.timesteps[self._i]) / 1000.0
            x = (1 - sn) * x0 + sn * torch.randn_like(x0)
        else:
            x = x0
        return (x.to(v.dtype),)
base = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
tr = WanTransformer3DModel.from_pretrained("FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers", subfolder="transformer", torch_dtype=torch.float16)
vae = AutoencoderKLWan.from_pretrained(base, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(base, transformer=tr, vae=vae, text_encoder=None, torch_dtype=torch.float16)
pipe.scheduler = DMD3()
pipe.to("cuda"); pipe.vae.enable_tiling()
E = torch.load("/kaggle/working/emb_5b.pt")
load_s = time.time() - t0
t1 = time.time()
g = torch.Generator("cpu").manual_seed(7)
frames = pipe(image=img, prompt_embeds=E["pos"].cuda().half(), negative_prompt_embeds=E["neg"].cuda().half(), height=832, width=480, num_frames=121,
              num_inference_steps=3, guidance_scale=1.0, generator=g).frames[0]
gen_s = time.time() - t1
export_to_video(frames, "/kaggle/working/t1_fastwan5b.mp4", fps=24)
stamp({"test": "fastwan5b_dmd3", "load_s": round(load_s, 1), "gen_s": round(gen_s, 1), "frames": 121, "fps": 24, "res": "480x832"}, "t1")
''')

T2 = COMMON + textwrap.dedent('''
from diffusers import WanImageToVideoPipeline, WanTransformer3DModel, GGUFQuantizationConfig
t0 = time.time()
q = GGUFQuantizationConfig(compute_dtype=torch.float16)
rep = "https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/blob/main/"
base = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
hi = WanTransformer3DModel.from_single_file(rep + "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf", quantization_config=q,
                                            config=base, subfolder="transformer", torch_dtype=torch.float16)
lo = WanTransformer3DModel.from_single_file(rep + "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf", quantization_config=q,
                                            config=base, subfolder="transformer_2", torch_dtype=torch.float16)
pipe = WanImageToVideoPipeline.from_pretrained(base, transformer=hi, transformer_2=lo, text_encoder=None, torch_dtype=torch.float16)
E = torch.load("/kaggle/working/emb_5b.pt")   # Wan2.2 A14B and TI2V-5B share the same umt5-xxl encoder
lr = "lightx2v/Wan2.2-Lightning"; sub = "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/"
pipe.load_lora_weights(lr, weight_name=sub + "high_noise_model.safetensors", adapter_name="hi")
pipe.load_lora_weights(lr, weight_name=sub + "low_noise_model.safetensors", adapter_name="lo", load_into_transformer_2=True)
pipe.enable_model_cpu_offload()
load_s = time.time() - t0
t1 = time.time()
g = torch.Generator("cpu").manual_seed(7)
pipe.vae.enable_tiling()
frames = pipe(image=img, prompt_embeds=E["pos"].cuda().half(), negative_prompt_embeds=E["neg"].cuda().half(), height=832, width=480, num_frames=81,
              num_inference_steps=4, guidance_scale=1.0, guidance_scale_2=1.0, generator=g).frames[0]
gen_s = time.time() - t1
export_to_video(frames, "/kaggle/working/t2_wan14b_q4_lightning.mp4", fps=16)
stamp({"test": "wan22_a14b_q4ks_lightning4", "load_s": round(load_s, 1), "gen_s": round(gen_s, 1), "frames": 81, "fps": 16, "res": "480x832"}, "t2")
''')

for name, code, tmo in [("t0", T0, 1200), ("t1", T1, 3000), ("t2", T2, 4800)]:
    p = W / f"{name}.py"; p.write_text(code)
    rc, s = sh(f"python {p}", f"{name}.log", tmo)
    status[name] = {"rc": rc, "wall_s": s}
    sh("rm -rf ~/.cache/huggingface/hub/models--FastVideo* ; df -h /root | tail -1; free -g", f"{name}.log")
    (W / "status.json").write_text(json.dumps(status, indent=1))
print(json.dumps(status, indent=1))
