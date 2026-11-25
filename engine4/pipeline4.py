"""Engine v4 orchestrator: `python -m engine4.pipeline4 stories/pip.md [--name pip_v4]` (resumable).

  shots     shots4.json: the story cut into narration chunks + the LLM's picture plan (laptop, text only)
  gpu_a     Kaggle GPU: narration (Qwen3-TTS on GPU1) + FLUX.2-klein master keys (both T4s)
  clips     Kaggle GPU: Wan2.2-A14B acting clips for the class-A shots only, natural length, 24 fps
  la        Kaggle CPU kernels (free, parallel): living-still renders for every L/E shot and the tails of A shots
  assemble  Kaggle CPU: v3 assembly at 24 fps (captions, music bed, -14 LUFS, srt, thumbnail)
State in out/<name>/state4.json; every stage resumes, a finished stage is never pushed again."""
import json, os, sys, time

from engine3 import kaggle3 as K
from engine3.pipeline import _od
from . import shots4 as SH, stages4 as ST

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFIX4 = "heartwick-v4-"


class State:
    def __init__(self, out):
        self.p = os.path.join(out, "state4.json")
        self.d = json.load(open(self.p)) if os.path.exists(self.p) else {"engine": "v4", "stages": {}}

    def set(self, stage, **kw):
        s = self.d["stages"].setdefault(stage, {}); s.update(kw); s["updated"] = K.now()
        json.dump(self.d, open(self.p, "w"), indent=1)

    def get(self, stage, k, default=None):
        return self.d["stages"].get(stage, {}).get(k, default)

    def done(self, stage):
        return self.get(stage, "status") == "done"


def _kernel(st, stage, slug, script, gpu, sources=(), note=""):
    """Attach to a run that is already queued/running (never re-push a queued kernel: phantom sessions); else push."""
    if st.get(stage, "kernel") == slug and st.get(stage, "status") in ("queued", "running", "pushing"):
        cur = K.status(slug)
        if cur in ("queued", "running", "complete"):
            print(K.now(), f"attaching to {slug} ({cur})", flush=True)
            rec = {"slug": f"{K.USER}/{slug}", "gpu": gpu, "pushed": st.get(stage, "pushed"), "renders": [note], "engine": "v4",
                   "by": "engine-v4"}
            fin = K.wait(rec, on_poll=lambda s: st.set(stage, status=s if s in ("queued", "running") else st.get(stage, "status")))
            return fin
    st.set(stage, status="pushing", kernel=slug)
    while gpu and K.gpu_busy()[1] >= 2:          # never a 3rd GPU kernel on the account
        print(K.now(), "2 GPU kernels already active; waiting", flush=True); time.sleep(120)
    rec = K.push(slug, script, gpu, kernel_sources=list(sources), note=note, wait_slot=False)
    st.set(stage, status="queued", pushed=rec["pushed"])
    fin = K.wait(rec, on_poll=lambda s: st.set(stage, status=s if s in ("queued", "running") else st.get(stage, "status")))
    return fin


def make(story, name=None, voice="narrator_baritone", stop_after=None):
    meta = {}
    try:
        import yaml
        raw = open(story, encoding="utf-8").read()
        meta = yaml.safe_load(raw.split("---")[1]) if raw.startswith("---") else {}
    except Exception:
        pass
    slug = meta.get("slug") or os.path.splitext(os.path.basename(story))[0]
    name = name or f"{slug}_v4"; ks = name.replace("_", "-")
    out = os.path.join(ROOT, "out", name); os.makedirs(out, exist_ok=True)
    st = State(out); t_all = time.time()

    # 1. shots
    sp = os.path.join(out, "shots4.json")
    if not os.path.exists(sp):
        st.set("shots", status="running"); SH.write_shots(story, sp)
    shots4 = json.load(open(sp, encoding="utf-8"))
    st.set("shots", status="done", n_shots=len(shots4["shots"]), n_masters=len(shots4["masters"]),
           n_a=sum(1 for s in shots4["shots"] if s["class"] == "A"))
    if stop_after == "shots":
        return

    # 2. narration + master keys (GPU session A)
    a_slug = f"{PREFIX4}a-{ks}"
    if not st.done("gpu_a"):
        script, jobs = ST.build_gpu_a(shots4, voice)
        fin = _kernel(st, "gpu_a", a_slug, script, True, note=f"narration + {len(jobs)} FLUX masters")
        dl = _od(K.download(a_slug, os.path.join(out, "kaggle", "gpu_a")))
        stt = json.load(open(os.path.join(dl, "status.json"))) if os.path.exists(os.path.join(dl, "status.json")) else {}
        missing = sorted({j["id"] for j in jobs} - set(stt.get("keys_done", [])))
        ok = os.path.exists(os.path.join(dl, "narration.json")) and not missing
        st.set("gpu_a", status="done" if ok else "failed", missing=missing, kaggle_state=fin.get("state"), run_min=fin.get("run_min_est"))
        if not ok:
            sys.exit(f"gpu_a incomplete: narration={os.path.exists(os.path.join(dl, 'narration.json'))} missing masters={missing}")
    adl = _od(os.path.join(out, "kaggle", "gpu_a"))
    narr = json.load(open(os.path.join(adl, "narration.json")))
    tl = ST.timeline(shots4, narr); json.dump(tl, open(os.path.join(out, "timeline.json"), "w"), indent=1)
    if stop_after == "gpu_a":
        return

    # 3. acting clips (GPU session B) and 4. living stills (CPU kernels) - the stills of L/E shots start at once
    from . import la_jobs as LA
    a_ids = [s["id"] for s in shots4["shots"] if s["class"] == "A"]
    la_jobs = LA.jobs_for(shots4, tl, os.path.join(adl, "keys"), clips_dir=None)
    b_slug = f"{PREFIX4}clips-{ks}"
    import threading
    res = {}

    def stills():
        if st.done("la"):
            return
        st.set("la", status="running")
        r = LA.render(la_jobs, out_dir=os.path.join(out, "kaggle", "la"), tag=ks + "-s", n_kernels=3)
        res["la"] = r
        st.set("la", status="done" if not r.get("missing") else "failed", missing=r.get("missing", []), kernels=r.get("kernels"))

    th = threading.Thread(target=stills); th.start()
    if a_ids and not st.done("clips"):
        plan, script, srcs = ST.build_clips4(shots4, tl, a_slug, K.USER)
        fin = _kernel(st, "clips", b_slug, script, True, sources=srcs, note=f"{len(a_ids)} A14B acting clips")
        dl = _od(K.download(b_slug, os.path.join(out, "kaggle", "clips")))
        stt = json.load(open(os.path.join(dl, "status.json"))) if os.path.exists(os.path.join(dl, "status.json")) else {}
        made = [f[:-4] for f in (stt.get("finish", {}).get("done") or [])]
        st.set("clips", status="done", made=made, missing=sorted(set(a_ids) - set(made)), run_min=fin.get("run_min_est"))
    th.join()
    # A shots: clip + living-still tail (an A shot whose clip failed is rendered as a plain living still)
    if a_ids and not st.done("la_a"):
        cdl = _od(os.path.join(out, "kaggle", "clips"))
        made = set(st.get("clips", "made", []))
        jobs = LA.jobs_for(shots4, tl, os.path.join(adl, "keys"), clips_dir=os.path.join(cdl, "finished"), only=a_ids, made=made)
        r = LA.render(jobs, out_dir=os.path.join(out, "kaggle", "la_a"), tag=ks + "-a", n_kernels=2)
        st.set("la_a", status="done" if not r.get("missing") else "failed", missing=r.get("missing", []), kernels=r.get("kernels"))
    if stop_after == "la":
        return

    # 5. assembly (CPU) - v3 assembler at 24 fps; every shot is one <id>.mp4 in some LA kernel's output
    from engine3.stages import build_assemble
    srcs = {}          # each shot is found by name (<id>.mp4) across the mounted living-still kernels
    la_kernels = [k for stage in ("la", "la_a") for k in (st.get(stage, "kernels") or [])]
    shots_like = {"shots": [{"id": s["id"], "sfx": "", "hero": False} for s in shots4["shots"]]}
    plan, script = build_assemble(shots_like, tl, meta, name, srcs=srcs, fps=24)
    ksrc = [f"{K.USER}/{k}" for k in la_kernels] + [f"{K.USER}/{a_slug}", f"{K.USER}/heartwick-v3-keys-pip-short"]
    fin = _kernel(st, "assemble", f"{PREFIX4}assemble-{ks}", script, False, sources=ksrc, note="v4 assembly 24 fps")
    dl = _od(K.download(f"{PREFIX4}assemble-{ks}", os.path.join(out, "kaggle", "assemble")))
    final = os.path.join(dl, f"{name}.mp4")
    st.set("assemble", status="done" if os.path.exists(final) else "failed", wall_min_total=round((time.time() - t_all) / 60, 1))
    print("final:", final if os.path.exists(final) else "MISSING")


if __name__ == "__main__":
    a = sys.argv[1:]
    make(a[0], name=a[a.index("--name") + 1] if "--name" in a else None,
         stop_after=a[a.index("--stop-after") + 1] if "--stop-after" in a else None)
