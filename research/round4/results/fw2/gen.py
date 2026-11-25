
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
