# umt5-xxl prompt embedding on the Arc iGPU (XPU, fp16 with a NaN check, bf16 fallback). Same padding as diffusers' Wan pipeline.
import json, sys, time, torch, gc
from transformers import AutoTokenizer, UMT5EncoderModel
repo = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
prompt, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")
t = tok([prompt], padding="max_length", max_length=512, truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
n = int(t.attention_mask.sum()); r = {"tokens": n}
def load_te(dt):
    # manual load: transformers 5.x from_pretrained segfaulted on this Windows box mid-load; plain safetensors reads fine
    import glob, os
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import UMT5Config
    d = os.path.join(snapshot_download(repo, allow_patterns=["text_encoder/*"]), "text_encoder")
    with torch.device("meta"):
        m = UMT5EncoderModel(UMT5Config.from_pretrained(d))
    sd = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        sd.update({k: v.to("xpu", dt) for k, v in load_file(f).items()})
    if "encoder.embed_tokens.weight" not in sd and "shared.weight" in sd:
        sd["encoder.embed_tokens.weight"] = sd["shared.weight"]
    m.load_state_dict(sd, strict=False, assign=True)
    return m.to("xpu", dt).eval()


for dt in (torch.float16, torch.bfloat16):
    t0 = time.time()
    te = load_te(dt)
    with torch.no_grad():
        e = te(t.input_ids.to("xpu"), t.attention_mask.to("xpu")).last_hidden_state[:, :n].float().cpu()
    del te; gc.collect(); torch.xpu.empty_cache()
    r[str(dt)] = {"s": round(time.time() - t0, 1), "finite": bool(torch.isfinite(e).all())}
    if r[str(dt)]["finite"]:
        break
e = torch.cat([e, e.new_zeros(1, 512 - n, e.shape[-1])], 1)
torch.save({"pos": e}, out); print(json.dumps(r))
