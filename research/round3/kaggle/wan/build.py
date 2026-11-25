# builds push/r3_wan.py = sources + kernel_wan.py
import json, os
H = os.path.dirname(os.path.abspath(__file__))
rd = lambda p: open(os.path.join(H, p), encoding="utf-8").read()
qac = open(os.path.join(H, "../../../round2/quality/tools/qa_consistency.py"), encoding="utf-8").read()
src = (f"GEN3 = {rd('gen3.py')!r}\nFIN3 = {rd('finish3.py')!r}\nQA3 = {rd('qa3.py')!r}\nQAC = {qac!r}\n" + rd("kernel_wan.py"))
os.makedirs(os.path.join(H, "push"), exist_ok=True)
open(os.path.join(H, "push", "r3_wan.py"), "w", encoding="utf-8").write(src)
json.dump({"id": "shubhsaxena2008/heartwick-r3-wan", "title": "Heartwick R3 Wan", "code_file": "r3_wan.py",
           "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
           "enable_internet": True, "dataset_sources": ["shubhsaxena2008/heartwick-r3-inputs"], "competition_sources": [],
           "kernel_sources": [], "model_sources": []},
          open(os.path.join(H, "push", "kernel-metadata.json"), "w"), indent=1)
print(len(src))
