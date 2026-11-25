"""Round 3 push helper. GPU kernels: push only if (a) no other heartwick-r3-* GPU kernel is RUNNING/QUEUED (this agent
owns ONE GPU session) and (b) fewer than 2 GPU sessions are in use account-wide (every gpu=true slug in kaggle/usage.jsonl
plus the most recently run kernels). CPU kernels push straight away. Every push is appended to kaggle/usage.jsonl.
usage: python r3_push.py <push_dir> <slug> "<what>" [--cpu] [--wait-min 0] [--est-h 2]"""
import json, subprocess, sys, time
from datetime import datetime

K = r"C:\Users\shubh\.kaggle-venv\Scripts\kaggle.exe"
USAGE = r"C:\Users\shubh\Projects\heartwick\kaggle\usage.jsonl"
push_dir, slug, what = sys.argv[1], sys.argv[2], sys.argv[3]
cpu = "--cpu" in sys.argv
wait_min = float(sys.argv[sys.argv.index("--wait-min") + 1]) if "--wait-min" in sys.argv else 0
est_h = float(sys.argv[sys.argv.index("--est-h") + 1]) if "--est-h" in sys.argv else None


def k(*a):
    r = subprocess.run([K, *a], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    return (r.stdout or "") + (r.stderr or "")


def gpu_slugs():
    s, cpus = set(), set()
    for line in open(USAGE, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("slug"):
            (s if d.get("gpu") else cpus).add(d["slug"])
    for line in k("kernels", "list", "--mine", "--page-size", "15", "--sort-by", "dateRun").splitlines():
        tok = line.split()
        if tok and tok[0].startswith("shubhsaxena2008/") and tok[0] not in cpus:
            s.add(tok[0])
    return s


def busy():
    n, mine, detail = 0, 0, {}
    for s in sorted(gpu_slugs()):
        st = k("kernels", "status", s)
        if "RUNNING" in st or "QUEUED" in st:
            n += 1; detail[s] = "RUNNING" if "RUNNING" in st else "QUEUED"
            if "heartwick-r3-" in s:
                mine += 1
    return n, mine, detail


t0 = time.time()
while True:
    if cpu:
        break
    n, mine, d = busy()
    print(datetime.now().strftime("%H:%M:%S"), "gpu sessions in use:", n, "mine:", mine, d, flush=True)
    if n < 2 and mine == 0:
        break
    if time.time() - t0 > wait_min * 60:
        print("no free GPU slot; not pushing", flush=True); sys.exit(2)
    time.sleep(120)
out = k("kernels", "push", "-p", push_dir, *([] if cpu else ["--accelerator", "NvidiaTeslaT4"]))
print(out, flush=True)
ok = "successfully" in out.lower() or "kernel version" in out.lower()
rec = {"slug": slug, "gpu": not cpu, "pushed": datetime.now().isoformat(timespec="seconds"), "renders": [what],
       "engine": "round3-permutations", "state": "pushed" if ok else "push-failed", "by": "round3-permutation agent"}
if est_h:
    rec["est_gpu_h"] = est_h
open(USAGE, "a", encoding="utf-8").write(json.dumps(rec) + "\n")
sys.exit(0 if ok else 1)
