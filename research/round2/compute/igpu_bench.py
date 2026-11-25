# Arc 130T raw throughput through PyTorch XPU: fp16 GEMM and SDPA at Wan2.2 A14B 480p token counts.
import json, time, torch, torch.nn.functional as F
d = "xpu"; r = {"device": torch.xpu.get_device_name(0), "torch": torch.__version__,
               "mem_gb": round(torch.xpu.get_device_properties(0).total_memory / 1e9, 1)}
def tm(fn, n=3):
    fn(); torch.xpu.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.xpu.synchronize(); return (time.time() - t) / n
for N in (4096, 8192):
    a = torch.randn(N, N, device=d, dtype=torch.float16); b = torch.randn(N, N, device=d, dtype=torch.float16)
    s = tm(lambda: a @ b, 5); r[f"gemm_fp16_{N}_tflops"] = round(2 * N**3 / s / 1e12, 2)
    del a, b
# one Wan FFN-shaped GEMM: 32760 tokens x 5120 -> 13824
x = torch.randn(32760, 5120, device=d, dtype=torch.float16); w = torch.randn(5120, 13824, device=d, dtype=torch.float16)
s = tm(lambda: x @ w, 3); r["gemm_wan_ffn_tflops"] = round(2 * 32760 * 5120 * 13824 / s / 1e12, 2); del x, w
for L in (8190, 32760):
    q = torch.randn(1, 40, L, 128, device=d, dtype=torch.float16); k = torch.randn_like(q); v = torch.randn_like(q)
    s = tm(lambda: F.scaled_dot_product_attention(q, k, v), 2); r[f"sdpa_L{L}_s"] = round(s, 3)
    r[f"sdpa_L{L}_tflops"] = round(4 * L * L * 128 * 40 / s / 1e12, 2); del q, k, v
print(json.dumps(r, indent=1))
