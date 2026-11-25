# umt5-xxl prompt embeddings (padded to 512 like the diffusers Wan pipelines), one .pt per prompt id.
# usage: CUDA_VISIBLE_DEVICES=1 python te.py prompts.json outdir
import gc, json, sys, time
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

P = json.load(open(sys.argv[1])); OUT = sys.argv[2]
repo = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
t0 = time.time()
tok = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")


def load(dev, dt):
    return UMT5EncoderModel.from_pretrained(repo, subfolder="text_encoder", torch_dtype=dt).to(dev).eval()


te, dev = load("cuda", torch.float16), "cuda"
r = {"load_s": round(time.time() - t0, 1)}
for pid, text in P.items():
    t = tok([text], padding="max_length", max_length=512, truncation=True, add_special_tokens=True,
            return_attention_mask=True, return_tensors="pt")
    n = int(t.attention_mask.sum())
    with torch.no_grad():
        e = te(t.input_ids.to(dev), t.attention_mask.to(dev)).last_hidden_state[:, :n].float().cpu()
    if not torch.isfinite(e).all() and dev == "cuda":   # fp16 umt5 can overflow: redo on CPU in bf16
        del te; gc.collect(); torch.cuda.empty_cache(); te, dev = load("cpu", torch.bfloat16)
        with torch.no_grad():
            e = te(t.input_ids, t.attention_mask).last_hidden_state[:, :n].float()
    r[pid] = {"tokens": n, "finite": bool(torch.isfinite(e).all()), "dev": dev}
    torch.save({"pos": torch.cat([e, e.new_zeros(1, 512 - n, e.shape[-1])], 1)}, f"{OUT}/emb_{pid}.pt")
r["total_s"] = round(time.time() - t0, 1)
print(json.dumps(r)); json.dump(r, open(f"{OUT}/te_rec.json", "w"))
