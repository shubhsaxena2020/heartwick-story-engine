# builds push/r3_keys.py = sources + kernel_keys.py (session 2: keys + FLF + style)
import ast, json, os, re
H = os.path.dirname(os.path.abspath(__file__))
rd = lambda p: open(os.path.join(H, p), encoding="utf-8").read()
wan = rd("../wan/kernel_wan.py")
m = re.search(r"^PROMPTS = (\{.*?^\})", wan, re.S | re.M)
prompts = ast.literal_eval(m.group(1))
qac = open(os.path.join(H, "../../../round2/quality/tools/qa_consistency.py"), encoding="utf-8").read()
src = (f"GEN3 = {rd('../wan/gen3.py')!r}\nFIN3 = {rd('../wan/finish3.py')!r}\nQA3 = {rd('../wan/qa3.py')!r}\nQAC = {qac!r}\n"
       f"KEYS3 = {rd('keys3.py')!r}\nPROMPTS_JSON = {json.dumps(prompts)!r}\n" + rd("kernel_keys.py"))
os.makedirs(os.path.join(H, "push"), exist_ok=True)
open(os.path.join(H, "push", "r3_keys.py"), "w", encoding="utf-8").write(src)
json.dump({"id": "shubhsaxena2008/heartwick-r3-keys", "title": "Heartwick R3 Keys", "code_file": "r3_keys.py",
           "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
           "enable_internet": True, "dataset_sources": ["shubhsaxena2008/heartwick-r3-inputs"], "competition_sources": [],
           "kernel_sources": [], "model_sources": []},
          open(os.path.join(H, "push", "kernel-metadata.json"), "w"), indent=1)
print(len(src), sorted(prompts))
