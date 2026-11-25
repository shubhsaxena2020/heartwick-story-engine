# builds push/r2_compute_t4.py = GEN_SRC + KEY_B64 + kernel_main.py
import json, os, re
H = os.path.dirname(os.path.abspath(__file__))
v2 = open(os.path.join(H, "../../../pipeline_proto/B_open_i2v/kaggle_t4/push/i2v_t4.py"), encoding="utf-8").read()
key = re.search(r'^KEY_B64 = "([^"]+)"', v2, re.M).group(1)
gen = open(os.path.join(H, "gen.py"), encoding="utf-8").read()
main = open(os.path.join(H, "kernel_main.py"), encoding="utf-8").read()
os.makedirs(os.path.join(H, "push"), exist_ok=True)
src = f"GEN_SRC = {gen!r}\nKEY_B64 = {key!r}\n" + main
open(os.path.join(H, "push", "r2_compute_t4.py"), "w", encoding="utf-8").write(src)
json.dump({"id": "shubhsaxena2008/heartwick-r2-compute-t4", "title": "Heartwick R2 Compute T4", "code_file": "r2_compute_t4.py",
           "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
           "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []},
          open(os.path.join(H, "push", "kernel-metadata.json"), "w"), indent=1)
print(len(src))
