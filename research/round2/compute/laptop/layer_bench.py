# Arc 130T: time Wan2.2-A14B transformer blocks at real token counts without holding a whole 10 GB expert
# (the shared laptop had no commit headroom for it). 4 real-size blocks in fp16 + the measured cost of dequantising
# one real Q4_K_M block from the GGUF on the XPU. Step time ~= 40 * (block forward + block dequant) + small extras.
import json, sys, time, torch, gguf
from diffusers import WanTransformer3DModel
from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor

d = "xpu"; r = {}
sync = torch.xpu.synchronize
m = WanTransformer3DModel(num_attention_heads=40, attention_head_dim=128, in_channels=36, out_channels=16, text_dim=4096,
                          ffn_dim=13824, num_layers=4, image_dim=None).to(d, torch.float16).eval()
emb = torch.randn(1, 512, 4096, device=d, dtype=torch.float16)
for (h, w) in ((832, 480), (672, 384)):
    lat = torch.randn(1, 36, 21, h // 8, w // 8, device=d, dtype=torch.float16)
    ts = torch.tensor([937.5], device=d)
    with torch.no_grad():
        m(hidden_states=lat, timestep=ts, encoder_hidden_states=emb, return_dict=False); sync()
        t = time.time()
        for _ in range(2):
            m(hidden_states=lat, timestep=ts, encoder_hidden_states=emb, return_dict=False)
        sync()
    r[f"{w}x{h}_s_per_block"] = round((time.time() - t) / 2 / 4, 2)
    r[f"{w}x{h}_tokens"] = 21 * (h // 16) * (w // 16)
del m; torch.xpu.empty_cache()
# dequant cost of one real block (Q4_K_M + Q6_K mix as in the file)
rd = gguf.GGUFReader(sys.argv[1])
blk = [t for t in rd.tensors if t.name.startswith("blocks.0.") and t.tensor_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)]
ps = [GGUFParameter(torch.from_numpy(t.data.copy()).to(d), quant_type=t.tensor_type) for t in blk]
for p in ps:
    dequantize_gguf_tensor(p)
sync(); t = time.time()
for _ in range(3):
    for p in ps:
        dequantize_gguf_tensor(p).to(torch.float16)
sync()
r["dequant_s_per_block"] = round((time.time() - t) / 3, 3)
r["block_weight_mb"] = round(sum(p.numel() * p.element_size() for p in ps) / 1e6)
for k in ("480x832", "384x672"):
    r[f"{k}_est_step_s"] = round(40 * (r[f"{k}_s_per_block"] + r["dequant_s_per_block"]), 0)
print(json.dumps(r, indent=1))
