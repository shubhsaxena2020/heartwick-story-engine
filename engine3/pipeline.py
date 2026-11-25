"""Engine v3 orchestrator: `python heartwick.py make stories/x.md --engine v3`.

Stages (each resumable; state in out/<slug>_v3/state.json, which the Heartwick Studio dashboard reads):
  shots     LLM shot list (local 9Router)                                   laptop, text only
  audio     narration (Qwen3-TTS clone) + Whisper word times                Kaggle CPU kernel   } in parallel
  keys      character sheets + keys + QA gate, then the ACE-Step music cue  Kaggle GPU (1 slot) }
  clips     Wan2.2 FLF/I2V clips, QA, best take, RIFE + ESRGAN finish       Kaggle GPU (1 slot)
  assemble  captions, series tag, end card, mix, -14 LUFS, srt, thumbnail   Kaggle CPU kernel
  post      copies deliverables into out/<slug>_v3/ + SEO pack + review sheets (light, laptop)
A failed or rejected shot re-runs alone: `--redo S05,S07` pushes a clips kernel for just those shots and the assembly
takes each shot from the newest kernel that made it."""
import json, os, shutil, sys, time

from . import kaggle3 as K, shots as SH, stages as ST

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["shots", "audio", "keys", "clips", "assemble", "post"]


class State:
    def __init__(self, out):
        self.p = os.path.join(out, "state.json")
        self.d = json.load(open(self.p)) if os.path.exists(self.p) else {"engine": "v3", "stages": {}}

    def set(self, stage, **kw):
        s = self.d["stages"].setdefault(stage, {})
        s.update(kw); s["updated"] = K.now()
        self.d["updated"] = K.now()
        json.dump(self.d, open(self.p, "w"), indent=1)

    def done(self, stage):
        return self.d["stages"].get(stage, {}).get("status") == "done"

    def get(self, stage, k, default=None):
        return self.d["stages"].get(stage, {}).get(k, default)


def _meta(story):
    import yaml
    raw = open(story, encoding="utf-8").read()
    return yaml.safe_load(raw.split("---")[1]) if raw.startswith("---") else {}


def _od(dl):
    """Kernel outputs download with their /kaggle/working layout: the files sit under out/."""
    return os.path.join(dl, "out") if os.path.isdir(os.path.join(dl, "out")) else dl


def _run_kernel(st, stage, slug, script, gpu, kernel_sources=(), note=""):
    rec = None
    if st.get(stage, "kernel") == slug and st.get(stage, "status") in ("queued", "running", "pushing", "failed"):
        cur = K.status(slug)   # resume: attach to the run already queued/running (or finished) instead of pushing again
        if cur in ("queued", "running") or (cur == "complete" and st.get(stage, "status") != "failed"):
            print(K.now(), f"attaching to {slug} ({cur})", flush=True)
            rec = {"slug": f"{K.USER}/{slug}", "gpu": gpu, "pushed": st.get(stage, "pushed") or K.now(), "renders": [note or slug],
                   "engine": "v3", "by": "engine-v3"}
    if rec is None:
        st.set(stage, status="pushing", kernel=slug, gpu=gpu)
        rec = K.push(slug, script, gpu, kernel_sources=kernel_sources, note=note)
        st.set(stage, status="queued", pushed=rec["pushed"])
    fin = K.wait(rec, on_poll=lambda s: st.set(stage, status=("running" if s == "running" else st.get(stage, "status"))))
    st.set(stage, kaggle_state=fin["state"], wall_min=fin.get("wall_min"), run_min=fin.get("run_min_est"))
    return fin


def make(story, name=None, redo=None, stop_after=None, voice=None, key_overrides=None):
    meta = _meta(story)
    slug = meta.get("slug") or os.path.splitext(os.path.basename(story))[0]
    name = name or f"{slug}_v3"
    out = os.path.join(ROOT, "out", name); os.makedirs(out, exist_ok=True)
    st = State(out); st.d["story"] = story; st.d["name"] = name
    ks = slug.replace("_", "-")
    t_all = time.time()

    # plan.json in the dashboard's shape (no v1/v2 sections, so it lists the video without asset checks)
    json.dump({"slug": slug, "title": meta.get("title"), "series_tag": meta.get("series_tag"), "voice": voice or "narrator_baritone",
               "speed": 1.0, "engine": "v3", "sections": []}, open(os.path.join(out, "plan.json"), "w"), indent=1)

    # 1. shots
    sp = os.path.join(out, "shots.json")
    if not st.done("shots") or not os.path.exists(sp):
        st.set("shots", status="running")
        d = SH.write_shots(story, sp)
        st.set("shots", status="done", n_shots=len(d["shots"]), n_keys=len(d["keys"]))
    shots = json.load(open(sp, encoding="utf-8"))
    if stop_after == "shots":
        return

    # 2. audio (CPU) and keys (GPU) in parallel
    import threading
    res = {}

    def audio():
        if st.done("audio"):
            return
        v, script = ST.build_audio(shots, voice)
        fin = _run_kernel(st, "audio", f"{K.PREFIX}audio-{ks}", script, False, note=f"narration {v} + whisper words")
        dl = _od(K.download(f"{K.PREFIX}audio-{ks}", os.path.join(out, "kaggle", "audio")))
        ok = os.path.exists(os.path.join(dl, "narration.json"))
        st.set("audio", status="done" if ok else "failed", voice=v)
        res["audio"] = ok

    def keys():
        if st.done("keys"):
            return
        est = sum(float(s["duration_s"]) for s in shots["shots"]) + 10
        need = {k["id"] for k in shots["keys"]} | {"sheet_nell", "sheet_pip"}
        prev, srcs = dict(st.get("keys", "chosen", {}) or {}), list(st.get("keys", "sources", []) or [])
        kdir = lambda s: os.path.join(out, "kaggle", "keys" if s == f"{K.PREFIX}keys-{ks}" else s.replace(K.PREFIX, ""))
        if prev and srcs and st.get("keys", "missing"):          # failed, or a partial run still going (resume)
            # partial re-run (24 Nov): only the missing keys, in their own kernel; the approved keys ride along as
            # inputs (local copies of their files) and the music cue from the first run is kept
            kslug = f"{K.PREFIX}keys-{ks}-r{len(srcs)}"
            files = {}
            for kid, f in prev.items():
                hit = [os.path.join(r, "keys", f) for r in (_od(kdir(s)) for s in srcs)
                       if os.path.exists(os.path.join(r, "keys", f))]
                if hit:
                    files[kid] = hit[0]
            only = need - set(files)
            # only ship the approved keys the missing ones reference ("@sheet_nell", "@K9"...): all of them
            # made the script 1.8 MB, over Kaggle's ~1 MB limit (24 Nov run 5)
            used = {r[1:] for j in ST.keys_jobs(shots, only=only)["keys"] for r in j["refs"] if r.startswith("@") and not r.startswith("@IN@")}
            files = {k: v for k, v in files.items() if k in used}
            script = ST.build_keys(shots, music_s=None, given=files, only=only)
            note = f"missing keys only ({len(only)}) + QA"
        else:
            kslug, prev, srcs = f"{K.PREFIX}keys-{ks}", {}, []
            script = ST.build_keys(shots, music_s=est)
            note = "sheets + keys (Qwen-Image-Edit-2511) + QA + ACE-Step"
        fin = _run_kernel(st, "keys", kslug, script, True, note=note)
        dl = _od(K.download(kslug, kdir(kslug)))
        rp = os.path.join(dl, "keys", "keys_rec.json")
        rec = json.load(open(rp)) if os.path.exists(rp) else {}
        chosen = dict(prev, **rec.get("chosen", {}))
        srcs = srcs + [kslug]
        missing = sorted(need - set(chosen))
        st.set("keys", status="done" if not missing else "failed", missing=missing, chosen=chosen, sources=srcs,
               gpu_min=(st.get("keys", "gpu_min", 0) or 0) + (fin.get("run_min_est") or 0),
               config=rec.get("phases", {}).get("config"))
        res["keys"] = not missing

    th = [threading.Thread(target=audio), threading.Thread(target=keys)]
    [t.start() for t in th]; [t.join() for t in th]
    if not (st.done("audio") and st.done("keys")):
        sys.exit(f"stopped: audio={st.get('audio', 'status')} keys={st.get('keys', 'status')} (see {out}/state.json)")
    if stop_after == "keys":
        return

    narr = json.load(open(os.path.join(_od(os.path.join(out, "kaggle", "audio")), "narration.json")))
    tl = ST.timeline(shots, narr)
    json.dump(tl, open(os.path.join(out, "timeline.json"), "w"), indent=1)
    chosen = dict(st.get("keys", "chosen", {}))
    chosen.update(key_overrides or {})

    # 3. clips (GPU); --redo pushes just those shots to a separate kernel
    srcs = st.get("clips", "sources", {})
    todo = [s["id"] for s in shots["shots"]] if not st.done("clips") else []
    kslug = f"{K.PREFIX}clips-{ks}"
    if redo:
        todo = list(redo); kslug = f"{K.PREFIX}clips-{ks}-r{int(time.time()) % 100000}"
    if todo:
        plan, script = ST.build_clips(shots, tl, chosen, only=set(todo))
        fin = _run_kernel(st, "clips", kslug, script, True, kernel_sources=[f"{K.USER}/{x}" for x in (st.get("keys", "sources") or [f"{K.PREFIX}keys-{ks}"])],
                          note=f"wan2.2 FLF/I2V {len(plan['clips'])} shots + QA + RIFE/ESRGAN")
        dl = _od(K.download(kslug, os.path.join(out, "kaggle", kslug.replace(K.PREFIX, ""))))
        stt = json.load(open(os.path.join(dl, "status.json"))) if os.path.exists(os.path.join(dl, "status.json")) else {}
        made = [f[:-4] for f in (stt.get("finish", {}).get("done") or [])]
        for sid in made:
            srcs[sid] = kslug
        missing = [s["id"] for s in shots["shots"] if s["id"] not in srcs]
        st.set("clips", status="done" if not missing else "failed", sources=srcs, missing=missing,
               gpu_min=(st.get("clips", "gpu_min", 0) or 0) + (fin.get("run_min_est") or 0), best=stt.get("best"))
        if missing:
            sys.exit(f"clips missing for {missing}; re-run with --redo {','.join(missing)}")
    if stop_after == "clips":
        return

    # 4. assemble (CPU)
    if not st.done("assemble") or redo:
        plan, script = ST.build_assemble(shots, tl, meta, name, srcs=srcs)
        ksrc = sorted({f"{K.USER}/{v}" for v in srcs.values()}) + [f"{K.USER}/{K.PREFIX}audio-{ks}"] + [f"{K.USER}/{x}" for x in (st.get("keys", "sources") or [f"{K.PREFIX}keys-{ks}"])]
        fin = _run_kernel(st, "assemble", f"{K.PREFIX}assemble-{ks}", script, False, kernel_sources=ksrc, note="assembly + master")
        dl = _od(K.download(f"{K.PREFIX}assemble-{ks}", os.path.join(out, "kaggle", "assemble")))
        ok = os.path.exists(os.path.join(dl, f"{name}.mp4"))
        st.set("assemble", status="done" if ok else "failed")
        if not ok:
            sys.exit("assembly failed: see " + dl)

    # 5. post: deliverables where the dashboard looks
    dl = _od(os.path.join(out, "kaggle", "assemble"))
    for f in os.listdir(dl):
        if f.endswith((".mp4", ".srt", ".jpg")) or f in ("qa_report.json",):
            shutil.copy(os.path.join(dl, f), os.path.join(out, f))
    from . import post
    post.finish(out, name, meta, shots, tl, st)
    st.set("post", status="done", wall_min_total=round((time.time() - t_all) / 60, 1))
    print("final:", os.path.join(out, f"{name}.mp4"))
