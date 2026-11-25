"""Heartwick Studio snapshot (schema 1): everything the site shows, read from the engine's real files.

Sources: stories/*.md, out/<video>/ (plan.json, qa_report.json, worker_log.json, SEO_PACK.json, *.srt, media),
cache/<slug>/assets/, kaggle/usage.jsonl, kaggle/queue_runner.log, kaggle/queue.json, research/,
dashboard/data/approvals.json + actions_applied.json, and dashboard/data/kaggle_live.json (Kaggle status/quota,
polled by the exporter). build() is pure file reading (no network); self_check() recomputes key numbers a
second, independent way (ffprobe/ffmpeg on the mp4s, direct file counts, Kaggle's own quota) and lists mismatches.
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from studio_common import (APPLIED, APPROVALS, DATA, KAGGLE_LIVE, QUEUE, QUEUE_LOG, ROOT, USAGE, iso,  # noqa: E402
                           local_iso_to_epoch, read_json, write_json)
import media as M  # noqa: E402

SCHEMA = 1
STAGES = ["script", "narration", "images", "render", "review", "approved", "published"]
QUOTA_H = 30.0
RESEARCH = os.path.join(ROOT, "research")
TRUTH_CACHE = os.path.join(DATA, "truth_cache.json")
AUDIO_EXT = (".wav", ".mp3", ".m4a", ".ogg", ".flac")
LOG_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (.*)$")
TRANSITION = re.compile(r"^(heartwick-[a-z0-9-]+) -> ([A-Z_.]+)")
LUFS_RANGE = (-16.0, -14.0)
MAX_MD = 200_000


# --------------------------------------------------------------------------- change signature
def _sig_paths():
    pats = ["stories/*.md", "out/*/*.json", "out/*/*.mp4", "out/*/*.jpg", "out/*/*.srt", "out/*/*.md", "out/*/samples/*",
            "kaggle/usage.jsonl", "kaggle/queue_runner.log", "kaggle/queue.json", "research/**/*",
            "dashboard/data/approvals.json", "dashboard/data/actions_applied.json", "dashboard/data/kaggle_live.json",
            "dashboard/data/previews/*.mp4", "cache/*/assets"]
    out = []
    for p in pats:
        out += glob.glob(os.path.join(ROOT, p), recursive=True)
    return out


def signature():
    """Cheap fingerprint of every input (stat only, ~100 files): changes whenever build() could change."""
    h = hashlib.sha1()
    for p in sorted(_sig_paths()):
        try:
            st = os.stat(p)
            h.update(f"{p}|{st.st_mtime_ns}|{st.st_size}\n".encode())
        except OSError:
            pass
    return h.hexdigest()


# --------------------------------------------------------------------------- stories
def _stories():
    sys.path.insert(0, ROOT)
    from engine import story as S
    res = []
    for p in sorted(glob.glob(os.path.join(ROOT, "stories", "*.md"))):
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            meta, secs = S.parse_story(p)
            utts = [u for s in secs for sh in s["shots"] for u in sh["utts"]]
            res.append({"slug": meta.get("slug") or base, "title": meta.get("title"), "file": os.path.relpath(p, ROOT).replace("\\", "/"),
                        "chapters": len(secs), "shots": sum(len(s["shots"]) for s in secs), "lines": len(utts),
                        "words": sum(len(u["text"].split()) for u in utts), "characters": sorted((meta.get("characters") or {}).keys()),
                        "mtime": iso(os.path.getmtime(p)), "error": None})
        except Exception as e:
            res.append({"slug": base, "title": None, "file": os.path.relpath(p, ROOT).replace("\\", "/"), "error": f"{type(e).__name__}: {e}"[:300],
                        "mtime": iso(os.path.getmtime(p))})
    return res


# --------------------------------------------------------------------------- videos
def _mtime(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return None


def _ctime(p):
    try:
        return os.path.getctime(p)
    except OSError:
        return None


def _plan_needs(plan):
    utts = [u["id"] for s in plan.get("sections", []) for sh in s.get("shots", []) for u in sh.get("utts", [])]
    locs = {sh["bg"] for s in plan.get("sections", []) for sh in s.get("shots", []) if sh.get("bg")}
    poses = {f"pose_{c['char']}_{c['pose']}_cut" for s in plan.get("sections", []) for sh in s.get("shots", []) for c in sh.get("chars", [])}
    imgs = {f"bg_{l}" for l in locs} | {f"bg_{l}_depth" for l in locs} | poses
    return utts, sorted(imgs)


def _assets(slug):
    d = os.path.join(ROOT, "cache", slug, "assets")
    have = {}
    if os.path.isdir(d):
        for f in os.listdir(d):
            have.setdefault(os.path.splitext(f)[0], os.path.join(d, f))
    return have


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _video(name, index, approvals, previews):
    out = os.path.join(ROOT, "out", name)
    plan = read_json(os.path.join(out, "plan.json"))
    if not plan or not plan.get("slug"):
        return None
    slug = plan["slug"]
    have = _assets(slug)
    utts, imgs = _plan_needs(plan)
    lines_done = sum(u in have for u in utts)
    imgs_done = sum(i in have for i in imgs)
    mp4 = os.path.join(out, f"{name}.mp4")
    qa_p = os.path.join(out, "qa_report.json")
    qa = read_json(qa_p)
    has_mp4 = os.path.isfile(mp4)
    reached = {"script": _ctime(os.path.join(out, "plan.json"))}
    if utts and lines_done == len(utts):
        reached["narration"] = max(_mtime(have[u]) or 0 for u in utts)
    if utts and lines_done == len(utts) and imgs_done == len(imgs):
        reached["images"] = max([reached["narration"]] + [_mtime(have[i]) or 0 for i in imgs])
    if has_mp4:
        reached["render"] = _mtime(mp4)
    if has_mp4 and qa:
        reached["review"] = _mtime(qa_p)
    # stage times must be ordered: a stage is entered no earlier than the plan existed, and no later than the
    # next stage it led to (cached narration re-downloaded after the render keeps a newer mtime)
    order = [s for s in ("script", "narration", "images", "render", "review") if reached.get(s)]
    for a, b in zip(order, order[1:]):
        reached[b] = max(reached[b], reached[a]) if b in ("narration", "images") else reached[b]
    for a, b in reversed(list(zip(order, order[1:]))):
        reached[a] = min(reached[a], reached[b])
    ap = approvals.get(name) or None
    stage = "script"
    for s in ("narration", "images", "render", "review"):
        if s in reached:
            stage = s
    if ap and stage == "review":
        if ap.get("status") in ("approved", "published"):
            stage = "approved"
            reached["approved"] = local_iso_to_epoch(ap.get("at")) or reached["review"]
        if ap.get("status") == "published":
            stage = "published"
            reached["published"] = local_iso_to_epoch(ap.get("at")) or reached["approved"]
    audio = (qa or {}).get("audio") or {}
    sync = (qa or {}).get("caption_sync") or {}
    asr = (qa or {}).get("asr_check") or {}
    wl = read_json(os.path.join(out, "worker_log.json")) or {}
    stages_s = wl.get("stages") or {}
    seo = read_json(os.path.join(out, "SEO_PACK.json")) or {}
    srt_p = os.path.join(out, f"{name}.srt")
    seo_md_p = os.path.join(out, "SEO_PACK.md")
    errors, warnings = [], []
    for e in wl.get("errors") or []:
        errors.append(f"Kaggle worker: {e}")
    lufs = _num(audio.get("integrated_lufs"))
    peak = _num(audio.get("true_peak_dbfs"))
    if lufs is not None and not (LUFS_RANGE[0] <= lufs <= LUFS_RANGE[1]):
        errors.append(f"Loudness {lufs} LUFS is outside {LUFS_RANGE[0]:g}..{LUFS_RANGE[1]:g}")
    if peak is not None and peak > -1.0:
        warnings.append(f"True peak {peak} dBFS is above -1.0")
    if _num(asr.get("wer")) is not None and asr["wer"] > 0.05:
        warnings.append(f"Whisper word error rate {asr['wer']:.1%} (captions may not match the voice)")
    p90 = _num(asr.get("caption_vs_speech_p90_abs_s"))
    if p90 is not None and p90 > 0.25:
        warnings.append(f"Caption sync p90 {p90:.2f} s is above 0.25 s")
    if (qa or {}).get("thumbnail_error"):
        errors.append("Thumbnail: " + str(qa["thumbnail_error"]))
    if has_mp4 and not qa:
        warnings.append("Rendered but no qa_report.json yet")

    med = {}
    for key, fname in (("thumb", "thumbnail.jpg"), ("sheet", "contact_sheet.jpg"), ("vsref", "vs_reference.jpg"),
                       ("masters", "master_candidates.jpg")):
        p = os.path.join(out, fname)
        if os.path.isfile(p):
            med[key] = index.ref(p)
    preview_pending = False
    if has_mp4:
        sha = index.sha(mp4)
        pv = M.preview_path(name, sha)
        if os.path.isfile(pv):
            med["preview"] = index.ref(pv, f"{name}_480p.mp4")
        else:
            preview_pending = True
        previews.append((name, mp4, pv))
    samples = []
    for p in sorted(glob.glob(os.path.join(out, "samples", "*"))):
        if p.lower().endswith(AUDIO_EXT):
            samples.append({"name": os.path.splitext(os.path.basename(p))[0], **(index.ref(p) or {})})

    def text(p, cap=60_000):
        try:
            return open(p, encoding="utf-8").read()[:cap]
        except OSError:
            return None

    dur = _num((qa or {}).get("duration_s"))
    return {
        "name": name, "story": slug, "title": plan.get("title"), "series_tag": plan.get("series_tag"),
        "kind": "short" if plan.get("speed") == 1.0 else "long", "voice": plan.get("voice"),
        "stage": stage, "reached": {k: iso(v) for k, v in reached.items() if v},
        "progress": {"lines_done": lines_done, "lines_total": len(utts), "images_done": imgs_done, "images_total": len(imgs)},
        "chapters": len(plan.get("sections", [])),
        "duration_s": dur, "shots": _num((qa or {}).get("shots")), "shots_per_min": _num((qa or {}).get("shots_per_min")),
        "caption_chunks": _num((qa or {}).get("caption_chunks")), "srt_cues": _num((qa or {}).get("srt_cues")),
        "lufs": lufs, "true_peak": peak, "lra": _num(audio.get("lra")),
        "sync": {k: _num(sync.get(k)) for k in ("chunks", "speech_present", "onset_chunks", "onset_median_s", "onset_p90_abs_s")} if sync else None,
        "asr": {k: _num(asr.get(k)) for k in ("wer", "ref_words", "asr_words", "matched_chunk_starts",
                                                "caption_vs_speech_median_s", "caption_vs_speech_p90_abs_s")} if asr else None,
        "render_s": _num(stages_s.get("render")), "verify_s": _num(stages_s.get("verify_asr")),
        "worker_total_s": _num(wl.get("total_s")), "worker_stages": {k: v for k, v in stages_s.items() if _num(v)},
        "mp4_bytes": os.path.getsize(mp4) if has_mp4 else None,
        "media": med, "preview_pending": preview_pending, "samples": samples,
        "srt": text(srt_p), "seo_md": text(seo_md_p),
        "seo": {"title": (seo.get("titles") or [None])[0], "titles": seo.get("titles") or [], "filename": seo.get("filename"),
                "tags": len(seo.get("tags") or []), "hashtags": seo.get("hashtags") or []} if seo else None,
        "approval": ap, "errors": errors, "warnings": warnings,
        "updated_at": iso(max(t for t in reached.values() if t)),
        # engine v3 writes per-stage progress (shots, audio, keys, clips, assemble, post) to out/<video>/state.json
        "engine": plan.get("engine") or "v2", "v3_stages": (read_json(os.path.join(out, "state.json")) or {}).get("stages"),
    }


# --------------------------------------------------------------------------- Kaggle + queue
def _kernel(slug):
    return (slug or "").split("/")[-1]


def parse_queue_log(path=QUEUE_LOG, keep=400):
    """[{at (epoch), text, body: [continuation lines]}] from kaggle/queue_runner.log, newest last."""
    entries = []
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()[-4000:]
    except OSError:
        return []
    for line in lines:
        m = LOG_LINE.match(line)
        if m:
            entries.append({"at": local_iso_to_epoch(m.group(1)), "text": m.group(2), "body": []})
        elif entries:
            entries[-1]["body"].append(line)
    return entries[-keep:]


def _state_word(s):
    s = (s or "").lower()
    for w, n in (("complete", "complete"), ("error", "error"), ("fail", "error"), ("cancel", "cancelled"), ("gone", "cancelled"),
                 ("running", "running"), ("queued", "queued")):
        if w in s:
            return n
    return None


def _jobs(qlog, klive, video_names, story_slugs, now):
    rows = [json.loads(l) for l in open(USAGE, encoding="utf-8") if l.strip()] if os.path.exists(USAGE) else []
    pushes, finals, workers = [], {}, []
    for r in rows:
        if "worker_log" in r:
            workers.append(r)
        elif "state" in r and r.get("pushed"):
            key = (_kernel(r["slug"]), r["pushed"][:16])
            if r["state"] == "pushed":
                pushes.append(r)
            else:
                finals[key] = r
    seen = {(_kernel(p["slug"]), p["pushed"][:16]) for p in pushes}
    for k, r in finals.items():   # early runs logged only their final record
        if k not in seen:
            pushes.append(r)
    # observed transitions per kernel: queue_runner.log + the exporter's Kaggle polls
    trans = {}
    for e in qlog:
        m = TRANSITION.match(e["text"])
        if m and e["at"]:
            trans.setdefault(m.group(1), []).append((e["at"], _state_word(m.group(2))))
    for k, lst in ((klive or {}).get("transitions") or {}).items():
        for t in lst:
            at = local_iso_to_epoch(t.get("at"))
            if at:
                trans.setdefault(k, []).append((at, _state_word(t.get("state"))))
    pushes.sort(key=lambda r: r["pushed"])
    jobs = []
    for i, p in enumerate(pushes):
        k = _kernel(p["slug"])
        t_push = local_iso_to_epoch(p["pushed"])
        later = [q for q in pushes[i + 1:] if _kernel(q["slug"]) == k]
        t_next = local_iso_to_epoch(later[0]["pushed"]) if later else None
        fin = finals.get((k, p["pushed"][:16]))
        obs = sorted((a, s) for a, s in trans.get(k, []) if s and a >= t_push - 60 and (t_next is None or a < t_next))
        first = {}
        for a, s in obs:
            first.setdefault(s, a)
        state, run_min, wait_min, wall_min, state_at = "unknown", None, None, None, None
        if fin:
            state = _state_word(fin["state"]) or "cancelled"
            run_min, wall_min = _num(fin.get("run_min_est")), _num(fin.get("wall_min"))
            if run_min is not None and wall_min is not None:
                wait_min = round(max(0.0, wall_min - run_min), 1)
            if state == "cancelled":
                run_min = 0.0
            state_at = (t_push + wall_min * 60) if wall_min is not None else None
        elif obs:
            state, state_at = obs[-1][1], obs[-1][0]
            if "running" in first:
                wait_min = round((first["running"] - t_push) / 60, 1)
                end = first.get("complete") or first.get("error") or (now if state == "running" else None)
                if end:
                    run_min = round((end - first["running"]) / 60, 1)
            elif state == "queued":
                wait_min = round((now - t_push) / 60, 1)
            if state in ("complete", "error"):
                wall_min = round((state_at - t_push) / 60, 1)
        elif t_next is not None:
            state = "superseded"
        w = [r for r in workers if _kernel(r["slug"]) == k and (local_iso_to_epoch(r.get("at")) or 0) >= t_push
             and (t_next is None or (local_iso_to_epoch(r.get("at")) or 0) < t_next)]
        wl = (w[0]["worker_log"] if w else None) or {}
        rest = k.replace("heartwick-gpu", "").replace("heartwick-render", "").strip("-").replace("-", "_")
        video = rest if rest in video_names else None
        if not video and wl:
            for v in video_names:   # pre-naming runs: the out/ dir whose worker_log.json is this job's
                if (read_json(os.path.join(ROOT, "out", v, "worker_log.json")) or {}).get("stages") == wl.get("stages"):
                    video = v
                    break
        story = rest if rest in story_slugs else next((s for s in story_slugs if (video or "").startswith(s)), None)
        jobs.append({
            "kernel": k, "gpu": bool(p.get("gpu")), "pushed_at": iso(t_push), "state": state,
            "state_at": iso(state_at) if state_at else None, "wait_min": wait_min, "run_min": run_min, "wall_min": wall_min,
            "images": p.get("images"), "utts": p.get("utts"), "render": p.get("render"),
            "video": video, "story": story, "worker_stages": wl.get("stages"), "worker_errors": wl.get("errors") or [],
            "worker_total_s": wl.get("total_s"),
            "note": fin.get("state") if fin and state == "cancelled" else
                    ("cancelled or deleted on Kaggle (no final record in usage.jsonl)" if obs and str(obs[-1][1]) == "cancelled" and not fin else None),
            "live_status": (((klive or {}).get("kernels") or {}).get(k) or {}).get("status") if not fin else None,
        })
    return jobs


def _kaggle(qlog, klive, video_names, story_slugs, now):
    jobs = _jobs(qlog, klive, video_names, story_slugs, now)
    quota = (klive or {}).get("quota") or None
    refresh = local_iso_to_epoch((quota or {}).get("refresh_at"))
    week_start = (refresh - 7 * 86400) if refresh else now - 7 * 86400
    by_day, by_video = {}, {}
    gpu_week = gpu_all = cpu_all = 0.0
    for j in jobs:
        m = j["run_min"] or 0.0
        if not m:
            continue
        t = local_iso_to_epoch(j["pushed_at"])
        day = iso(t)[:10]
        d = by_day.setdefault(day, {"date": day, "gpu_min": 0.0, "cpu_min": 0.0})
        v = by_video.setdefault(j["video"] or (f"{j['story']} (story)" if j["story"] else j["kernel"]),
                                {"video": j["video"] or (f"{j['story']} (story)" if j["story"] else j["kernel"]), "gpu_min": 0.0, "cpu_min": 0.0})
        if j["gpu"]:
            d["gpu_min"] += m
            v["gpu_min"] += m
            gpu_all += m
            if t >= week_start:
                gpu_week += m
        else:
            d["cpu_min"] += m
            v["cpu_min"] += m
            cpu_all += m
    rnd = lambda x: round(x, 1)  # noqa: E731
    return {
        "quota": quota, "week_start": iso(week_start), "quota_total_h": (quota or {}).get("total_h") or QUOTA_H,
        "gpu_min_week_local": rnd(gpu_week), "gpu_min_total_local": rnd(gpu_all), "cpu_min_total_local": rnd(cpu_all),
        "jobs": list(reversed(jobs)),
        "by_day": [{**d, "gpu_min": rnd(d["gpu_min"]), "cpu_min": rnd(d["cpu_min"])} for d in sorted(by_day.values(), key=lambda d: d["date"])],
        "by_video": sorted(({**v, "gpu_min": rnd(v["gpu_min"]), "cpu_min": rnd(v["cpu_min"])} for v in by_video.values()),
                           key=lambda v: -(v["gpu_min"] + v["cpu_min"])),
        "poll": {k: (klive or {}).get(k) for k in ("checked_at", "error")},
    }


def _queue(qlog, now):
    from actions import load_queue, runner_pids
    pids = runner_pids()
    current = None
    for e in qlog:
        if e["text"].startswith("RUN "):
            current = e
        elif e["text"].startswith("rc=") and current:
            current = None
    started = [e for e in qlog if e["text"].startswith("queue runner started")]
    tail = []
    for e in qlog[-60:]:
        txt, level = e["text"], "info"
        if txt.startswith("rc="):
            rc = txt[3:].split()[0]
            if rc != "0":
                level = "err"
                last = next((l.strip() for l in reversed(e["body"]) if l.strip() and not l.startswith(" ")), "")
                txt = f"rc={rc} · {last[:240]}" if last else txt
        elif "error" in txt.lower() or "did not complete" in txt:
            level = "warn"
        tail.append({"at": iso(e["at"]) if e["at"] else None, "text": txt[:300], "level": level})
    q = load_queue()
    return {
        "runner": {"running": bool(pids), "pids": pids,
                   "started_at": iso(started[-1]["at"]) if started and started[-1]["at"] else None},
        "current": {"text": current["text"][4:][:300], "since": iso(current["at"])} if current and pids else None,
        "log_tail": tail,
        "jobs": [{k: j.get(k) for k in ("id", "label", "args", "status", "added_at", "started_at", "finished_at", "rc", "source", "note")}
                 for j in q["jobs"][-40:]],
    }


# --------------------------------------------------------------------------- research
NOT_SAMPLES = ("parabrahma/", "kaggle_jobs/", "reference/")   # third-party reference media / job inputs: never uploaded


def is_voice_sample(rel):
    """Audio in research/ that is a voice sample (its path says voice/sample), not third-party reference audio."""
    r = rel.replace("\\", "/").lower()
    return r.endswith(AUDIO_EXT) and ("voice" in r or "sample" in r) and not any(x in r for x in NOT_SAMPLES)


def _research(index):
    rep = os.path.join(RESEARCH, "RESEARCH_REPORT.md")
    present = os.path.isdir(RESEARCH)
    md = None
    if os.path.isfile(rep):
        try:
            md = open(rep, encoding="utf-8", errors="replace").read()
        except OSError:
            md = None
    others = []
    samples = []
    if present:
        for p in sorted(glob.glob(os.path.join(RESEARCH, "**", "*"), recursive=True)):
            if not os.path.isfile(p):
                continue
            rel = os.path.relpath(p, RESEARCH).replace("\\", "/")
            if is_voice_sample(rel) and os.path.getsize(p) <= 15 * 1024 * 1024:
                ref = index.ref(p, os.path.basename(p))
                if ref:
                    samples.append({"name": rel, **ref})
            elif p.lower().endswith(".md") and rel != "RESEARCH_REPORT.md":
                others.append({"name": rel, "bytes": os.path.getsize(p), "mtime": iso(os.path.getmtime(p))})
    return {"present": present, "report_md": md[:MAX_MD] if md else None, "report_truncated": bool(md and len(md) > MAX_MD),
            "report_mtime": iso(os.path.getmtime(rep)) if md is not None else None, "samples": samples, "other_docs": others[:50]}


# --------------------------------------------------------------------------- events feed
def _events(videos, kaggle, queue, stories, extra):
    ev = []
    add = lambda at, level, source, text, video=None: ev.append({"at": at, "level": level, "source": source, "text": text[:400], "video": video})  # noqa: E731
    for v in videos:
        for e in v["errors"]:
            if not e.startswith("Kaggle worker:"):          # shown once, on its Kaggle job below
                add(v["updated_at"], "err", "qa", e, v["name"])
        for w in v["warnings"]:
            add(v["updated_at"], "warn", "qa", w, v["name"])
        if v["reached"].get("render"):
            add(v["reached"]["render"], "info", "render", f"Rendered {v['name']} ({v['duration_s'] or '?'} s)", v["name"])
        if v.get("approval"):
            a = v["approval"]
            add(a.get("applied_at") or a.get("at"), "info", "review", f"{v['name']} marked {a.get('status')}" + (f": {a['note']}" if a.get("note") else ""), v["name"])
    for j in kaggle["jobs"]:
        add(j["pushed_at"], "info", "kaggle", f"Pushed {j['kernel']} ({'GPU' if j['gpu'] else 'CPU'})", j["video"])
        if j["state"] == "error":
            add(j["state_at"] or j["pushed_at"], "err", "kaggle", f"Kaggle job {j['kernel']} ended in error", j["video"])
        elif j["state"] == "cancelled":
            add(j["state_at"] or j["pushed_at"], "warn", "kaggle", f"Kaggle job {j['kernel']} " + (j.get("note") or "cancelled"), j["video"])
        elif j["state"] == "complete" and j["state_at"]:
            add(j["state_at"], "info", "kaggle", f"{j['kernel']} complete in {j['run_min']} min" + (f" after {j['wait_min']} min queued" if j["wait_min"] else ""), j["video"])
        for e in j["worker_errors"]:
            add(j["state_at"] or j["pushed_at"], "err", "kaggle", f"{j['kernel']} worker: {e}", j["video"])
    for t in queue["log_tail"]:
        if t["level"] != "info":
            add(t["at"], t["level"], "queue", "Queue runner: " + t["text"])
    for s in stories:
        if s.get("error"):
            add(s["mtime"], "err", "story", f"{s['file']} does not parse: {s['error']}")
    for at, level, source, text in extra:
        add(at, level, source, text)
    ev = [e for e in ev if e["at"]]
    ev.sort(key=lambda e: e["at"], reverse=True)
    return ev[:250]


# --------------------------------------------------------------------------- build
def build(index=None, now=None, extra_events=()):
    now = time.time() if now is None else now
    index = index or M.Index()
    approvals = read_json(APPROVALS, {}) or {}
    stories = _stories()
    story_slugs = [s["slug"] for s in stories]
    previews = []
    videos = []
    outd = os.path.join(ROOT, "out")
    for n in sorted(os.listdir(outd)) if os.path.isdir(outd) else []:
        if os.path.isdir(os.path.join(outd, n)):
            v = _video(n, index, approvals, previews)
            if v:
                videos.append(v)
    videos.sort(key=lambda v: v["updated_at"], reverse=True)
    for s in stories:
        s["outputs"] = [v["name"] for v in videos if v["story"] == s["slug"]]
    qlog = parse_queue_log()
    klive = read_json(KAGGLE_LIVE, {}) or {}
    kaggle = _kaggle(qlog, klive, [v["name"] for v in videos], story_slugs, now)
    queue = _queue(qlog, now)
    research = _research(index)
    applied = (read_json(APPLIED, []) or [])[-150:]
    snap = {
        "schema": SCHEMA, "generated_at": iso(now), "stages": STAGES,
        "counts": {s: sum(v["stage"] == s for v in videos) for s in STAGES},
        "stories": stories, "videos": videos, "kaggle": kaggle, "queue": queue, "research": research,
        "channel": {"exists": False, "youtube": None, "instagram": None,
                    "note": "Nothing is uploaded by the engine; the channel owner uploads every video. Subscriber and view numbers appear here once the channel exists."},
        "actions": {"applied": list(reversed(applied))},
    }
    snap["events"] = _events(videos, kaggle, queue, stories, extra_events)
    index.save()
    return snap, previews


# --------------------------------------------------------------------------- self-check
def _probe(mp4, cache):
    """Duration (ffprobe) and integrated loudness (ffmpeg ebur128, audio only), cached by mtime+size."""
    st = os.stat(mp4)
    k = f"{mp4}|{st.st_mtime_ns}|{st.st_size}"
    if k in cache:
        return cache[k]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    dur = lufs = None
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", mp4],
                           capture_output=True, text=True, timeout=60, creationflags=flags)
        dur = float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        pass
    try:
        r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-i", mp4, "-vn", "-af", "ebur128", "-threads", "1",
                            "-f", "null", "-"], capture_output=True, text=True, timeout=300, creationflags=flags,
                           encoding="utf-8", errors="replace")
        m = re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)
        lufs = float(m[-1]) if m else None
    except Exception:
        pass
    cache[k] = {"duration_s": dur, "lufs": lufs}
    return cache[k]


def self_check(snap, now=None):
    now = time.time() if now is None else now
    cache = read_json(TRUTH_CACHE, {}) or {}
    checks = []

    def chk(key, label, expected, shown, page, source, tol=0, note=None):
        if expected is None or shown is None:
            ok = None
        elif isinstance(expected, (int, float)) and isinstance(shown, (int, float)):
            ok = abs(expected - shown) <= tol
        else:
            ok = expected == shown
        checks.append({"key": key, "label": label, "expected": expected, "shown": shown, "ok": ok, "tol": tol or None,
                       "page": page, "source": source, "note": note})

    vids = {v["name"]: v for v in snap["videos"]}
    chk("stories", "Stories", len(glob.glob(os.path.join(ROOT, "stories", "*.md"))), len(snap["stories"]), "Overview", "stories/*.md")
    mp4s = [d for d in glob.glob(os.path.join(ROOT, "out", "*")) if os.path.isfile(os.path.join(d, os.path.basename(d) + ".mp4"))]
    chk("rendered", "Videos rendered", len(mp4s), sum(STAGES.index(v["stage"]) >= 3 for v in vids.values()), "Overview",
        "out/<name>/<name>.mp4 files")
    reviewed = [d for d in mp4s if os.path.isfile(os.path.join(d, "qa_report.json"))]
    chk("reviewable", "Videos with QA (review or later)", len(reviewed), sum(STAGES.index(v["stage"]) >= 4 for v in vids.values()),
        "Overview", "mp4 + qa_report.json")
    ap = read_json(APPROVALS, {}) or {}
    want_ok = sum(1 for n, a in ap.items() if a.get("status") in ("approved", "published") and os.path.join(ROOT, "out", n) in reviewed)
    chk("approved", "Approved or published", want_ok, sum(v["stage"] in ("approved", "published") for v in vids.values()),
        "Overview", "dashboard/data/approvals.json")
    for d in mp4s:
        n = os.path.basename(d)
        v = vids.get(n)
        if not v:
            continue
        pr = _probe(os.path.join(d, n + ".mp4"), cache)
        chk(f"dur:{n}", f"{n} duration (s)", round(pr["duration_s"], 2) if pr["duration_s"] else None, v["duration_s"], "Videos",
            "ffprobe of the mp4 vs qa_report.json", tol=0.15)
        chk(f"lufs:{n}", f"{n} loudness (LUFS)", round(pr["lufs"], 1) if pr["lufs"] is not None else None, v["lufs"], "Videos",
            "ffmpeg ebur128 of the mp4 vs qa_report.json", tol=0.6)
        plan = read_json(os.path.join(d, "plan.json")) or {}
        utts, _ = _plan_needs(plan)
        adir = os.path.join(ROOT, "cache", plan.get("slug", ""), "assets")
        n_wav = sum(os.path.isfile(os.path.join(adir, u + ".wav")) for u in utts)
        chk(f"lines:{n}", f"{n} narration lines on disk", n_wav, v["progress"]["lines_done"], "Overview", "cache/<slug>/assets/<line id>.wav")
        chk(f"preview:{n}", f"{n} 480p preview", 1, 1 if v["media"].get("preview") else 0, "Videos",
            "dashboard/data/previews (encoded from the current mp4)", note="encoded and uploaded by the exporter when the mp4 changes")
    k = snap["kaggle"]
    q = k.get("quota") or {}
    if q.get("used_h") is not None:
        used = round(q["used_h"] * 60, 1)
        chk("gpu_week", "GPU minutes this quota week", used, k["gpu_min_week_local"], "Kaggle", "`kaggle quota` vs kaggle/usage.jsonl + observed runs",
            tol=6.0, note="one-sided: Kaggle also bills other kernels on the account (research jobs) and whole sessions, "
                          "so only a local total above Kaggle's is a mismatch")
        c = checks[-1]
        c["ok"] = c["shown"] <= c["expected"] + c["tol"]
    live = (read_json(os.path.join(DATA, "kaggle_live.json"), {}) or {}).get("kernels") or {}
    fresh = {kk: s for kk, s in live.items() if (local_iso_to_epoch(s.get("checked_at")) or 0) > now - 900 and _state_word(s.get("status"))}
    if fresh:
        shown = {j["kernel"]: j["state"] for j in k["jobs"] if j["kernel"] in fresh and j["state"] not in ("superseded",)}
        for kk, s in fresh.items():
            if kk in shown:
                chk(f"kstate:{kk}", f"{kk} state", _state_word(s["status"]), shown[kk], "Kaggle", "`kaggle kernels status` (polled by the exporter)")
    write_json(TRUTH_CACHE, {kk: vv for kk, vv in cache.items() if os.path.exists(kk.split("|")[0])})
    n_fail = sum(c["ok"] is False for c in checks)
    return {"at": iso(now), "n_checks": len(checks), "n_fail": n_fail, "n_skip": sum(c["ok"] is None for c in checks), "checks": checks}


if __name__ == "__main__":
    s, _ = build()
    s["truth"] = self_check(s)
    print(json.dumps(s, indent=1, ensure_ascii=False)[:20000])
