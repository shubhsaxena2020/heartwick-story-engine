import json, os, re
H = os.path.dirname(os.path.abspath(__file__))
v2 = open(os.path.join(H, "../../../pipeline_proto/B_open_i2v/kaggle_t4/push/i2v_t4.py"), encoding="utf-8").read()
key = re.search(r'^KEY_B64 = "([^"]+)"', v2, re.M).group(1)
main = open(os.path.join(H, "tpu_main.py"), encoding="utf-8").read()
os.makedirs(os.path.join(H, "push"), exist_ok=True)
open(os.path.join(H, "push", "r2_tpu.py"), "w", encoding="utf-8").write(f"KEY_B64 = {key!r}\n" + main)
json.dump({"id": "shubhsaxena2008/heartwick-r2-tpu-wan", "title": "Heartwick R2 TPU Wan", "code_file": "r2_tpu.py",
           "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": True,
           "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []},
          open(os.path.join(H, "push", "kernel-metadata.json"), "w"), indent=1)
