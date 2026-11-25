"""One-command v2 pipeline behind `python heartwick.py make <story.md>`:

  1. parse the story, work out every narration line for the chosen voice (content-addressed, cached in cache/<slug>/tts/)
  2. missing lines -> one Kaggle narration job (Kokoro: CPU kernel, no queue; cloning voices: GPU kernel)
  3. render on a Kaggle CPU kernel at 1080x1920/30 fps (narration rides in the script, or the private cache dataset)
  4. download to out/<name>/: <name>.mp4, contact sheet, QA report (motion density, pace, loudness), .srt, timeline
The laptop only orchestrates.
"""
import json, os, shutil, subprocess, sys, time
from engine import kaggle_run as K, plan as P1, story as S1
from . import story2, kaggle2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VOICE = "kokoro:af_heart"


def voice_key(voice):
    """Utterance-id key: Kokoro voices keep v1's bare name so v1 narration caches are reused."""
    return voice.split(":", 1)[1] if voice.startswith("kokoro:") else voice


def utterances(story, voice, speed, sections=None):
    meta, secs = story2.parse(story)
    secs = S1.select_sections(secs, sections)
    plan = P1.build_plan(meta, secs, voice=voice_key(voice), speed=speed)
    seen, out = set(), []
    for s in plan["sections"]:
        for sh in s["shots"]:
            for u in sh["utts"]:
                if u["id"] not in seen:
                    seen.add(u["id"])
                    out.append({"id": u["id"], "text": u["text"]})
    return meta, out


def tts_dirs(slug):
    """Narration caches to look in: v2's own, then v1's asset cache (same ids for Kokoro)."""
    d = os.path.join(ROOT, "cache", slug, "tts")
    os.makedirs(d, exist_ok=True)
    return [d] + [os.path.join(ROOT, "cache", s, "assets") for s in (slug, "pip") if os.path.isdir(os.path.join(ROOT, "cache", s, "assets"))]


def collect(slug, utts):
    """Copy every cached line into cache/<slug>/tts/. Returns the ids still missing."""
    dirs = tts_dirs(slug)
    missing = []
    for u in utts:
        if os.path.exists(os.path.join(dirs[0], u["id"] + ".json")):
            continue
        src = next((d for d in dirs[1:] if os.path.exists(os.path.join(d, u["id"] + ".json"))), None)
        if src:
            for ext in (".json", ".wav"):
                shutil.copy(os.path.join(src, u["id"] + ext), dirs[0])
        else:
            missing.append(u)
    return dirs[0], missing


def narrate(slug, voice, speed, missing):
    engine = voice.split(":", 1)[0]
    gpu = engine != "kokoro"
    d = os.path.join(ROOT, "kaggle", "build", f"tts-{slug}")
    os.makedirs(d, exist_ok=True)
    src = open(os.path.join(ROOT, "kaggle", "tts_v2.py"), encoding="utf-8").read()
    job = {"voice": voice, "speed": speed, "utts": missing}
    open(os.path.join(d, "tts_v2.py"), "w", encoding="utf-8").write(f"JOB = {json.dumps(job)!r}\nJOB = __import__('json').loads(JOB)\n" + src)
    kslug = f"heartwick-tts-{slug.replace('_', '-')}"
    meta = {"id": f"{K.USER}/{kslug}", "title": kslug.replace("-", " ").title(), "code_file": "tts_v2.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False, "enable_internet": True,
            "dataset_sources": [f"{K.USER}/heartwick-research-voices1"], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    args = ["kernels", "push", "-p", d] + (["--accelerator", "NvidiaTeslaT4"] if gpu else [])
    for attempt in range(240):
        out = K._k(*args, check=False)
        if "successfully pushed" in out.lower():
            break
        if "session" in out.lower() or "maximum" in out.lower():
            time.sleep(120)
            continue
        raise RuntimeError(out[-500:])
    ref = f"{K.USER}/{kslug}"
    rec = {"slug": ref, "gpu": gpu, "pushed": time.strftime("%Y-%m-%dT%H:%M:%S"), "utts": len(missing), "engine": "v2-tts"}
    state = K.wait(ref, rec, time.time(), poll=45)
    dl = os.path.join(ROOT, "cache", slug, "kaggle_out", kslug)
    K.download(ref, dl)
    n = 0
    for f in os.listdir(os.path.join(dl, "tts")) if os.path.isdir(os.path.join(dl, "tts")) else []:
        shutil.copy(os.path.join(dl, "tts", f), os.path.join(ROOT, "cache", slug, "tts", f))
        n += 1
    print(f"narration: {state}, {n // 2} lines")
    return state


def srt(timeline, path):
    def ts(t):
        h, r = divmod(t, 3600)
        m, s = divmod(r, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"
    lines = []
    for i, c in enumerate(timeline["captions"], 1):
        lines.append(f"{i}\n{ts(c['s'])} --> {ts(c['e'])}\n{' '.join(c['words'])}\n")
    open(path, "w", encoding="utf-8").write("\n".join(lines))


def _upload_once(stage_dir, files, key):
    """Upload a private dataset only when its content changed since the last upload (refine passes reuse it)."""
    import hashlib
    h = hashlib.sha1()
    for f in sorted(files):
        p = os.path.join(stage_dir, f)
        h.update(f.encode())
        h.update(str(os.path.getsize(p)).encode())
        h.update(open(p, "rb").read(65536))
    stamp = os.path.join(ROOT, "kaggle", "build", f"{key}.sha")
    ref = f"{K.USER}/heartwick-cache-{key.replace('_', '-')}"
    if os.path.exists(stamp) and open(stamp).read().strip() == h.hexdigest():
        return ref
    ref = K.upload_cache(stage_dir, files, story=key)
    if ref:
        open(stamp, "w").write(h.hexdigest())
    return ref


def prepare(story, style="C", art="c", voice=DEFAULT_VOICE, speed=1.0, sections=None, short=True, cta=None, vector=False):
    """Narration (synthesised once, cached) + every asset the render kernel needs, as datasets. -> context dict."""
    story_rel = os.path.relpath(os.path.abspath(story), ROOT).replace("\\", "/")
    meta, utts = utterances(story, voice, speed, sections)
    slug = meta.get("slug", os.path.splitext(os.path.basename(story))[0])
    tdir, missing = collect(slug, utts)
    if missing:
        if narrate(slug, voice, speed, missing) != "complete":
            sys.exit("narration job did not complete")
        tdir, missing = collect(slug, utts)
        if missing:
            sys.exit(f"{len(missing)} narration lines still missing")
    cfg = {"style": style, "art": art, "W": 1080, "H": 1920, "fps": 60, "voice": voice_key(voice), "speed": speed,
           "pause_scale": 0.45 if short else 0.8, "max_shot_s": 4.6 if short else 6.5, "ss": 1.25,
           "end_card_s": 3.6, "cta": cta or ("Follow for the next Heartwick story" if short else "Subscribe for a new story every week")}
    if sections:
        cfg["sections"] = sections
    from . import assets
    parsed = story2.parse(story)[1]
    names = sorted({c["char"] for sec in parsed for sh in sec["shots"] for c in sh["chars"]})
    locs = sorted({sh["bg"] for sec in parsed for sh in sec["shots"]})
    extra = {}
    if art == "v1":  # A: V1 painted plates + V1-style painted rigs (squirrels: V1 master cut-outs)
        from . import v1art
        for loc in locs:
            pth, dp = v1art.plate_paths(loc)
            if pth:
                extra[f"v1plates/bg_{loc}.png"] = pth
                if dp:
                    extra[f"v1plates/bg_{loc}_depth.png"] = dp
        cfg["plate_dir"] = "v1plates"
        for n in names:
            d = v1art.find_v1_rig(n)
            if d:
                for f in os.listdir(d):
                    extra[f"rigs/{n}/{f}"] = os.path.join(d, f)
            mc = os.path.join(ROOT, "characters", n, "master_cut.png")
            if n in ("pip", "pip_kit") and os.path.exists(mc):
                extra[f"rigs/{n}/master_cut.png"] = mc
        cfg["raster_dir"] = "rigs"
        cfg["scene_dir"] = ""
    else:  # B: style C, the art agent's parts and plates
        if not vector:
            extra.update(assets.rig_files(names))
        cfg["raster_dir"] = "rigs" if not vector else ""
        plates = assets.scene_files(locs)
        extra.update(plates)
        cfg["scene_dir"] = "scenes" if plates else ""
    music_dir = os.path.join(ROOT, "assets", "music")
    if os.path.isdir(music_dir):
        for f in os.listdir(music_dir):
            if f.endswith((".wav", ".flac")):
                extra[f"music/{f}"] = os.path.join(music_dir, f)
        cfg["music_dir"] = "music"
    datasets = []
    if extra:
        stage = os.path.join(ROOT, "kaggle", "build", f"assets_{slug}_{art}")
        shutil.rmtree(stage, ignore_errors=True)
        os.makedirs(stage)
        flat = []
        for arc, src in extra.items():
            fn = "A__" + arc.replace("/", "-D-")
            shutil.copy(src, os.path.join(stage, fn))
            flat.append(fn)
        datasets.append(_upload_once(stage, flat, f"{slug}-{art}-assets"))
    ids = [u["id"] for u in utts]
    tstage = os.path.join(ROOT, "kaggle", "build", f"tts_{slug}")
    shutil.rmtree(tstage, ignore_errors=True)
    os.makedirs(tstage)
    tfiles = []
    for i in ids:
        for ext in (".wav", ".json"):
            shutil.copy(os.path.join(tdir, i + ext), tstage)
            tfiles.append(i + ext)
    datasets.append(_upload_once(tstage, tfiles, f"{slug}-tts"))
    return {"story": story_rel, "slug": slug, "cfg": cfg, "datasets": [d for d in datasets if d], "tdir": tdir, "art": art, "style": style}


def render_job(ctx, name, overrides=None, preview=False, parts=1, sheet_every=None, wait=True):
    """Render on Kaggle CPU kernels. preview: 540x960 @ 30 fps for refine passes. parts: frame ranges rendered by
    parallel kernels, concatenated losslessly here (copy) and muxed with part 0's mastered mix."""
    cfg = dict(ctx["cfg"])
    if preview:
        cfg.update(W=540, H=960, fps=30, ss=1.0, crf=20, preset="veryfast")
    for k, v in (overrides or {}).items():
        if k == "tune":
            cfg["tune"] = dict(cfg.get("tune") or {}, **v)
        else:
            cfg[k] = v
    dest = os.path.join(ROOT, "out", name)
    os.makedirs(dest, exist_ok=True)
    sheet = sheet_every or (0.25 if preview else 2.0)
    if parts <= 1:
        job = {"renders": [{"story": ctx["story"], "out": name, "cfg": cfg, "workers": 4, "sheet_every": sheet}]}
        t0 = time.time()
        state = kaggle2.run(job, [ctx["story"]], f"heartwick-v2-{name.replace('_', '-')}".lower()[:50], dest, dataset=ctx["datasets"])
        _finish(dest, name)
        return {"state": state, "wall_s": round(time.time() - t0, 1), "dest": dest}
    # split: estimate frames from the story length (narration) + end card
    from . import render as R
    F = R.Film(os.path.join(ROOT, ctx["story"]), ctx["tdir"], dict(cfg, W=270, H=480))
    n = F.n_frames()
    refs = []
    t0 = time.time()
    for k in range(parts):
        a, b = n * k // parts, n * (k + 1) // parts
        job = {"renders": [{"story": ctx["story"], "out": f"{name}_part{k}", "cfg": cfg, "workers": 4, "sheet_every": sheet,
                            "frames": [a, b], "part": k}]}
        ref, rec, _ = kaggle2.push(job, [ctx["story"]], f"heartwick-v2-{name.replace('_', '-')}-p{k}".lower()[:50], dataset=ctx["datasets"])
        refs.append((ref, rec))
    states = []
    for k, (ref, rec) in enumerate(refs):
        states.append(K.wait(ref, rec, t0, poll=45))
        K.download(ref, os.path.join(dest, f"part{k}"))
    lst = os.path.join(dest, "parts.txt")
    open(lst, "w").write("".join(f"file '{os.path.join(dest, f'part{k}', f'{name}_part{k}.mp4')}'\n" for k in range(parts)))
    video = os.path.join(dest, f"{name}_video.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", video], check=True)
    mix = os.path.join(dest, "part0", f"{name}_part0_mix.wav")
    out = os.path.join(dest, f"{name}.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-i", mix, "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
                    "-shortest", "-movflags", "+faststart", out], check=True)
    os.remove(video)
    rep = json.load(open(os.path.join(dest, "part0", f"{name}_part0_report.json")))
    per_s, rs = [], []
    for k in range(parts):
        r = json.load(open(os.path.join(dest, f"part{k}", f"{name}_part{k}_report.json")))
        per_s += r.get("motion_per_second", [])
        rs.append(r.get("render_s", 0))
        shutil.copy(os.path.join(dest, f"part{k}", f"{name}_part{k}_sheet.jpg"), os.path.join(dest, f"{name}_sheet_part{k}.jpg"))
    secs = [p for p in per_s if p["t"] < rep.get("story_s", 1e9)]
    rep["motion"] = {"motion_seconds_pct": round(100 * sum(p["local_frac"] > 0.01 for p in secs) / max(1, len(secs)), 1), "seconds_measured": len(secs)}
    rep["render_s_per_part"] = rs
    rep["parts"] = parts
    json.dump(rep, open(os.path.join(dest, f"{name}_report.json"), "w"), indent=1)
    shutil.copy(os.path.join(dest, "part0", f"{name}_part0_timeline.json"), os.path.join(dest, f"{name}_timeline.json"))
    _finish(dest, name)
    return {"state": states, "wall_s": round(time.time() - t0, 1), "dest": dest}


def _finish(dest, name):
    tl = os.path.join(dest, f"{name}_timeline.json")
    if os.path.exists(tl):
        srt(json.load(open(tl)), os.path.join(dest, f"{name}.srt"))


def make(story, style="C", voice=DEFAULT_VOICE, speed=1.0, name=None, sections=None, short=True, cta=None, dry=False, vector=False,
         art="c", parts=1, overrides=None):
    meta, utts = utterances(story, voice, speed, sections)
    slug = meta.get("slug", os.path.splitext(os.path.basename(story))[0])
    name = name or f"{slug}_{'A' if art == 'v1' else 'B'}"
    tdir, missing = collect(slug, utts)
    print(f"{name}: {len(utts)} narration lines, {len(missing)} to synthesise with {voice}; art {art}, style {style}")
    if dry:
        return None
    ctx = prepare(story, style, art, voice, speed, sections, short, cta, vector)
    res = render_job(ctx, name, overrides=overrides, parts=parts)
    rp = os.path.join(res["dest"], f"{name}_report.json")
    if os.path.exists(rp):
        r = json.load(open(rp))
        print(json.dumps({k: r.get(k) for k in ("duration_s", "shots_per_min", "cuts_in_pauses", "character_motion_seconds_pct", "audio", "motion_qa")}, indent=1))
        print("motion beyond the camera (reference method):", r.get("motion", {}).get("motion_seconds_pct"), "% of seconds")
    print("kaggle:", res["state"], "->", res["dest"])
    return res["dest"]
