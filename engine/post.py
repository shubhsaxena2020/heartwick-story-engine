"""After the render: .srt, SEO pack, thumbnail, contact sheet, loudness + caption-sync QA report."""
import json, math, os, re, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from . import brand, render
from .style import PALETTE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ts(t, srt=True):
    h, m = int(t // 3600), int(t % 3600 // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int(round((s % 1) * 1000)) % 1000:03d}" if srt else (f"{h}:{m:02d}:{int(s):02d}" if h else f"{m:02d}:{int(s):02d}")


def write_srt(tl, path, max_chars=42):
    cues, cur = [], None
    for c in tl["captions"]:
        txt = " ".join(c["words"])
        if cur and len(cur["t"]) + 1 + len(txt) <= max_chars and c["s"] - cur["e"] < 0.3 and not re.search(r"[.!?]\"?$", cur["t"]):
            cur["t"] += " " + txt; cur["e"] = c["e"]
        else:
            if cur:
                cues.append(cur)
            cur = {"s": c["s"], "e": c["e"], "t": txt}
    if cur:
        cues.append(cur)
    with open(path, "w", encoding="utf-8") as f:
        for i, q in enumerate(cues, 1):
            f.write(f"{i}\n{_ts(q['s'])} --> {_ts(q['e'])}\n{q['t']}\n\n")
    return len(cues)


def chapters(tl, min_len=10.0):
    ch = [dict(c) for c in tl["chapters"]]
    if not ch:
        return []
    ch[0]["t"] = 0.0
    out = []
    for c in ch:  # YouTube: >=3 chapters, each >=10 s
        if out and c["t"] - out[-1]["t"] < min_len:
            continue
        out.append(c)
    if out and tl["total"] - out[-1]["t"] < min_len and len(out) > 1:
        out.pop()
    return out if len(out) >= 3 else []


def seo_pack(meta, tl, name, short, path):
    seo = meta.get("seo", {})
    titles = seo.get("titles", [meta.get("title", name)])
    chs = [] if short else chapters(tl)
    tags = seo.get("tags", [])
    tag_str = ", ".join(tags)
    while len(tag_str) > 480:
        tags = tags[:-1]; tag_str = ", ".join(tags)
    hashtags = seo.get("hashtags", [])[:3]
    desc = [seo.get("summary", ""), "",
            "This is a fictional story. Visuals and voice are AI-generated (illustration and narration).", ""]
    desc += seo.get("extra_description", [])
    if chs:
        desc += ["", "Chapters:"] + [f"{_ts(c['t'], False)} {c['title']}" for c in chs]
    desc += ["", seo.get("comment_prompt", "Tell us in the comments: what would you have done?"),
             "New stories every week on Heartwick Stories.", "", " ".join(hashtags)]
    filename = seo.get("filename", name) + (("-short" if short else "") + ".mp4")
    lines = [f"# SEO pack: {meta.get('title', name)}" + (" (Short)" if short else ""), "",
             "Ready to paste into YouTube Studio. Nothing is uploaded automatically: the channel owner reviews the video and approves every upload.", "",
             "## Upload file name", f"`{filename}`", "",
             "## Title"]
    if short:
        lines += [f"**Recommended (Shorts):** {seo.get('shorts_title', titles[0])}", ""]
    lines += [f"{i + 1}. {t} ({len(t)} chars)" for i, t in enumerate(titles)]
    lines += ["", "## Description", "```", *desc, "```", "",
              "## Tags", f"`{tag_str}`", f"({len(tag_str)} characters, limit 500)", "",
              "## Hashtags (first 3 show above the title)", " ".join(hashtags), "",
              "## Pinned comment", seo.get("pinned_comment", ""), "",
              "## Thumbnail", f"Text: **{' '.join(seo.get('thumbnail_text', []))}** (file `thumbnail.jpg`, 1280x720)", "",
              "## Settings checklist",
              "- Audience: **No, it's not made for kids**",
              "- Altered or synthetic content: **No** for clearly animated, non-realistic visuals. Say **Yes** if any shot could be mistaken for real footage or a real person's voice. The narration is a synthetic voice that does not imitate a real person.",
              "- Language: English. Caption language: English (upload `" + name + ".srt`)",
              f"- Playlist: {seo.get('playlist', 'Heartwick Stories')}",
              "- End screen: last 15-20 s (long video): next story plus subscribe. Shorts: use the related-video link to the full story.",
              "- Category: Film & Animation",
              "- Publish slot to test: Fri 6 PM ET or Sun 8 PM ET",
              "- Music: if a YouTube Audio Library track was added, credit it in the description when its licence requires it."]
    if chs:
        lines += ["", "## Chapters (from the real narration timing)"] + [f"- {_ts(c['t'], False)} {c['title']}" for c in chs]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump({"filename": filename, "titles": titles, "description": "\n".join(desc), "tags": tags, "hashtags": hashtags,
               "pinned_comment": seo.get("pinned_comment", ""), "chapters": chs},
              open(path.replace(".md", ".json"), "w", encoding="utf-8"), indent=1)


def thumbnail(meta, tl, assets, path):
    seo = meta.get("seo", {})
    want = seo.get("thumbnail_shot")
    shots = tl["shots"]
    sh = next((s for s in shots if s.get("id") == want), None) or next((s for s in shots if s.get("chars")), shots[0])
    W, H = 1280, 720
    sh2 = dict(sh, cam="push_in", framing="wide", fx=[f for f in sh.get("fx", []) if f == "leaves"], trans_in=0)
    # landscape frame: keep feet on the ground (same y), make figures read at thumbnail size, push them right of the text
    sh2["chars"] = [dict(c, y=min(0.99, max(0.9, c["y"] + 0.04)), h=min(0.8, c["h"] * 1.6), x=0.58 + c["x"] * 0.4, anim="idle")
                    for c in sh.get("chars", [])]
    sr = render.ShotRenderer(sh2, tl, render.Assets(assets, 1.0), W, H)
    img = sr.render(sh2["s"] + 0.8 * (sh2["e"] - sh2["s"]))
    # darken the left for the text
    xx = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
    img = img * (0.55 + 0.45 * np.clip(xx * 1.8, 0, 1))
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).convert("RGBA")
    words = seo.get("thumbnail_text", ["A STORY"])
    f = brand.font("caption", 118, 1000)
    y = 150
    lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    for i, line in enumerate(words):
        col = PALETTE["ember"] if i == len(words) - 1 else PALETTE["cream"]
        d.text((60, y), line, font=f, fill=col + (255,), stroke_width=6, stroke_fill=PALETTE["night_pine"] + (255,))
        y += 132
    sh_ = lay.split()[3].filter(ImageFilter.GaussianBlur(10))
    shadow = Image.new("RGBA", im.size, (0, 0, 0, 0)); shadow.putalpha(sh_.point(lambda v: int(v * 0.7)))
    im = Image.alpha_composite(Image.alpha_composite(im, shadow), lay)
    tag = brand.series_tag(meta.get("series_tag", ""), 1.1)
    im.alpha_composite(tag, (60, 40))
    im.convert("RGB").save(path, quality=92)
    return path


def contact_sheet(video, path, every=None, cols=8, w=200):
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video],
                               capture_output=True, text=True).stdout.strip())
    n = cols * 4
    every = every or dur / n
    h = int(w * 16 / 9)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-threads", "4", "-i", video, "-vf", f"fps=1/{every:.3f},scale={w}:{h}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    rows = math.ceil(len(fr) / cols)
    sheet = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    d = ImageDraw.Draw(sheet)
    for i, f in enumerate(fr):
        sheet.paste(Image.fromarray(f), ((i % cols) * w, (i // cols) * h))
        d.text(((i % cols) * w + 4, (i // cols) * h + 4), f"{i * every:.1f}s", fill=(255, 255, 0))
    sheet.save(path, quality=88)
    return path


def side_by_side(video, ref_video, path, times_ours, times_ref, w=270):
    h = int(w * 16 / 9)
    def grab(v, t):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", v, "-frames:v", "1", "-vf", f"scale={w}:{h}",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        return Image.fromarray(np.frombuffer(raw, np.uint8).reshape(h, w, 3))
    n = len(times_ours)
    sheet = Image.new("RGB", (n * w, 2 * h + 30), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    for i, (a, b) in enumerate(zip(times_ref, times_ours)):
        sheet.paste(grab(ref_video, a), (i * w, 0)); sheet.paste(grab(video, b), (i * w, h + 30))
    d.text((6, h + 8), "top: reference (Parabrahma World)   bottom: Heartwick pilot", fill=(255, 255, 0))
    sheet.save(path, quality=88)


def loudness(video):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-threads", "4", "-i", video, "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    tail = r[r.rfind("Summary:"):]
    I = re.search(r"I:\s+(-?[\d.]+) LUFS", tail); tp = re.search(r"Peak:\s+(-?[\d.]+) dBFS", tail); lra = re.search(r"LRA:\s+([\d.]+) LU", tail)
    return {"integrated_lufs": float(I[1]) if I else None, "true_peak_dbfs": float(tp[1]) if tp else None, "lra": float(lra[1]) if lra else None}


def caption_sync(tl, video):
    """Independent check against the rendered audio (not the timeline):
    (a) speech is really present during the first 200 ms of every caption chunk;
    (b) for chunks that start after a pause, the measured speech onset vs the caption start."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True).stdout
    a = np.frombuffer(raw, np.float32)
    hop = 160  # 10 ms frames
    n = len(a) // hop
    rms = np.sqrt((a[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms)
    thr = np.percentile(db, 90) - 26
    act = db > thr
    present, onsets = 0, []
    for c in tl["captions"]:
        i = int(round(c["s"] * 100))
        if act[i:i + 20].mean() > 0.4:
            present += 1
        if i >= 30 and act[i - 28:i - 6].mean() < 0.1:  # preceded by a pause -> a measurable onset
            w = np.where(act[i - 15:i + 30])[0]
            if len(w):
                onsets.append((w[0] - 15) / 100)
    on = np.array(onsets) if onsets else np.zeros(1)
    return {"chunks": len(tl["captions"]), "speech_present": present, "onset_chunks": len(onsets),
            "onset_median_s": round(float(np.median(on)), 3), "onset_p90_abs_s": round(float(np.percentile(np.abs(on), 90)), 3)}


def _norm(w):
    return re.sub(r"[^a-z0-9']", "", w.lower().replace("’", "'"))


def asr_check(tl, asr):
    """Whisper transcript of the final mix vs the script: word error rate + caption-start vs spoken-word offsets."""
    import difflib
    ref = [(_norm(w), c["s"] if i == 0 else None) for c in tl["captions"] for i, w in enumerate(c["words"])]
    ref = [(w, t) for w, t in ref if w]
    hyp = [(_norm(c["w"]), c["t"][0]) for c in asr["chunks"] if _norm(c["w"])]
    sm = difflib.SequenceMatcher(a=[w for w, _ in ref], b=[w for w, _ in hyp], autojunk=False)
    errors = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
    offs = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                rt, ht = ref[i1 + k][1], hyp[j1 + k][1]
                if rt is not None and ht is not None:
                    offs.append(ht - rt)
    o = np.array(offs) if offs else np.zeros(1)
    return {"wer": round(errors / max(1, len(ref)), 3), "ref_words": len(ref), "asr_words": len(hyp), "matched_chunk_starts": len(offs),
            "caption_vs_speech_median_s": round(float(np.median(o)), 3), "caption_vs_speech_p90_abs_s": round(float(np.percentile(np.abs(o), 90)), 3)}


def finish(plan, meta, name, out, assets, short=None):
    tl = json.load(open(os.path.join(out, "timeline.json")))
    video = os.path.join(out, f"{name}.mp4")
    short = (tl["total"] < 180) if short is None else short
    rep = {"name": name, "duration_s": tl["total"], "shots": len(tl["shots"]),
           "shots_per_min": round(len(tl["shots"]) / tl["total"] * 60, 1), "caption_chunks": len(tl["captions"])}
    rep["srt_cues"] = write_srt(tl, os.path.join(out, f"{name}.srt"))
    seo_pack(meta, tl, name, short, os.path.join(out, "SEO_PACK.md"))
    try:
        thumbnail(meta, tl, assets, os.path.join(out, "thumbnail.jpg"))
    except Exception as e:
        rep["thumbnail_error"] = repr(e)
    if os.path.exists(video):
        render._lower_priority()
        contact_sheet(video, os.path.join(out, "contact_sheet.jpg"))
        rep["audio"] = loudness(video)
        rep["caption_sync"] = caption_sync(tl, video)
        asr = os.path.join(out, "asr.json")
        if os.path.exists(asr):
            rep["asr_check"] = asr_check(tl, json.load(open(asr, encoding="utf-8")))
        ref = os.path.join(ROOT, "reference", "ref_reel.mp4")
        if os.path.exists(ref):
            n = 6
            side_by_side(video, ref, os.path.join(out, "vs_reference.jpg"),
                         [tl["total"] * (k + 0.5) / n * 0.92 for k in range(n)], [20, 51, 66, 100, 142, 170][:n])
    json.dump(rep, open(os.path.join(out, "qa_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
    return rep
