"""Push a round-4 kernel dir (with kernel-metadata.json) only if the account has a free GPU slot and no other
heartwick-r4 kernel is running/queued; then poll to completion and download the output.
usage: python r4push.py <push_dir> <slug> <download_dir> [--no-push]   (slug must start with heartwick-r4-)"""
import json, os, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from engine3 import kaggle3 as K  # noqa: E402

d, slug, dest = sys.argv[1], sys.argv[2], sys.argv[3]
assert slug.startswith("heartwick-r4-"), slug
TPU = "--tpu" in sys.argv
if "--no-push" not in sys.argv and TPU:   # TPU v5e-8: separate quota, not a GPU slot
    out = K._k("kernels", "push", "-p", d, "--accelerator", "TpuV5E8", check=False, timeout=600)
    print(out[-400:], flush=True)
    if "successfully" not in out.lower() and "kernel version" not in out.lower():
        sys.exit("push failed")
    K.log_usage({"slug": f"{K.USER}/{slug}", "gpu": False, "tpu": True, "pushed": K.now(), "renders": ["round-4 tpu"], "by": "round4"})
elif "--no-push" not in sys.argv:
    mine, total, active = K.gpu_busy()
    r4 = [a for a in active if a[0].startswith("heartwick-r4-")]
    print(K.now(), "active:", active, flush=True)
    if r4 or total >= 2:
        sys.exit(f"not pushing: r4 active {r4}, account GPU kernels {total}")
    out = K._k("kernels", "push", "-p", d, "--accelerator", "NvidiaTeslaT4", check=False, timeout=600)
    print(out[-400:], flush=True)
    if "successfully" not in out.lower() and "kernel version" not in out.lower():
        sys.exit("push failed")
    K.log_usage({"slug": f"{K.USER}/{slug}", "gpu": True, "pushed": K.now(), "renders": ["round-4 bench"], "by": "round4"})
t0 = time.time(); last = None; t_run = None
while True:
    st = K.status(slug)
    if st != last:
        print(K.now(), slug, st, flush=True); last = st
    if st == "running" and t_run is None:
        t_run = time.time()
    if st in ("complete", "error", "cancel"):
        break
    time.sleep(60)
fin = {"slug": f"{K.USER}/{slug}", "gpu": not TPU, "tpu": TPU, "state": st, "wall_min": round((time.time() - t0) / 60, 1),
       "run_min_est": round((time.time() - (t_run or t0)) / 60, 1), "by": "round4"}
K.log_usage(fin); print(json.dumps(fin), flush=True)
if "--no-download" in sys.argv:   # big weight-cache outputs stay on Kaggle; only the small json/log is fetched
    os.makedirs(dest, exist_ok=True)
    print(K._k("kernels", "output", f"{K.USER}/{slug}", "-p", dest, "--file-pattern", r".*\.(json|log)$", check=False, timeout=900)[-300:])
else:
    K.download(slug, dest)
print("downloaded to", dest, flush=True)
