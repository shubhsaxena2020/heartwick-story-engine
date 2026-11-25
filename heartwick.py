"""Heartwick story-video engine CLI.

  python heartwick.py make stories/pip_short.md                 # engine v2 puppet animation -> out/pip_short_<style>/
  python heartwick.py make stories/pip_short.md --style C --voice voxcpm2:female
  python heartwick.py pick-style C                              # make C the default art direction
  python heartwick.py make stories/pip.md --engine v1 ...       # the old 2.5D slideshow engine (options below)
  python heartwick.py make stories/pip.md --engine v1 --sections 1-2 --name pip_pilot --short
  python heartwick.py pick nell 2                               # choose master candidate c2 for a character
  python heartwick.py status                                     # Kaggle GPU usage so far

All GPU work (TTS, SDXL, cut-outs, depth) and the final render run on Kaggle. The laptop only orchestrates;
`--local-preview` renders a 540x960 preview locally at BelowNormal priority.
"""
import argparse, json, os, shutil, sys, time
from engine import story as S, plan as P, kaggle_run as K

ROOT = os.path.dirname(os.path.abspath(__file__))
CHARS = os.path.join(ROOT, "characters")


def cache_dir(slug):
    d = os.path.join(ROOT, "cache", slug, "assets")
    os.makedirs(d, exist_ok=True)
    return d


def have_set(d):
    return {os.path.splitext(f)[0] for f in os.listdir(d)}


def masters_have():
    if not os.path.isdir(CHARS):
        return set()
    return {n for n in os.listdir(CHARS) if os.path.exists(os.path.join(CHARS, n, "master.png"))}


def sync_masters(assets):
    """Copy chosen series masters into the story's asset folder (master_<c>.png and _cut)."""
    for n in masters_have():
        for src, dst in (("master.png", f"master_{n}.png"), ("master_cut.png", f"master_{n}_cut.png")):
            p = os.path.join(CHARS, n, src)
            if os.path.exists(p):
                shutil.copy(p, os.path.join(assets, dst))


def ingest(out_dir, assets):
    """Move downloaded kernel outputs into the asset cache."""
    a = os.path.join(out_dir, "assets")
    n = 0
    if os.path.isdir(a):
        for f in os.listdir(a):
            shutil.copy(os.path.join(a, f), os.path.join(assets, f)); n += 1
    print("ingested", n, "asset files")


def contact_masters(assets, names, out):
    from PIL import Image, ImageDraw
    rows = []
    for n in names:
        ims = []
        for k in range(8):
            p = os.path.join(assets, f"master_{n}_c{k}.png")
            if os.path.exists(p):
                im = Image.open(p).convert("RGB"); im.thumbnail((300, 360)); ims.append(im)
        if ims:
            rows.append((n, ims))
    if not rows:
        return
    W = max(len(r[1]) for r in rows) * 310 + 10
    sheet = Image.new("RGB", (W, len(rows) * 380), (30, 30, 30))
    d = ImageDraw.Draw(sheet)
    for i, (n, ims) in enumerate(rows):
        for k, im in enumerate(ims):
            sheet.paste(im, (10 + k * 310, 10 + i * 380))
            d.text((14 + k * 310, 12 + i * 380), f"{n} c{k}", fill=(255, 255, 0))
    sheet.save(out)
    print("master candidates sheet:", out)


STYLE_CHOICE = os.path.join(ROOT, "brand", "STYLE_CHOICE")  # written by `heartwick.py pick-style B|C`


def chosen_style():
    return open(STYLE_CHOICE).read().strip() if os.path.exists(STYLE_CHOICE) else "B"


def cmd_make(a):
    if a.engine == "v3":
        from engine3 import pipeline as p3
        return p3.make(a.story, name=a.name, redo=a.redo.split(",") if a.redo else None, stop_after=a.stop_after,
                       voice=a.voice)
    if a.engine == "v2":
        from engine2 import pipeline
        voice = a.voice or pipeline.DEFAULT_VOICE
        if ":" not in voice:
            voice = "kokoro:" + voice
        return pipeline.make(a.story, style=a.style or chosen_style(), voice=voice, speed=a.speed or 1.0, name=a.name,
                             sections=a.sections, short=a.short or a.sections is None, cta=a.cta, dry=a.dry, vector=a.vector, art=a.art, parts=a.parts)
    meta, sections = S.parse_story(a.story)
    sections = S.select_sections(sections, a.sections)
    slug = meta.get("slug", os.path.splitext(os.path.basename(a.story))[0])
    name = a.name or slug
    speed = a.speed or (1.0 if a.short else 0.94)
    plan = P.build_plan(meta, sections, voice=a.voice, speed=speed)
    if a.short:
        plan["series_tag"] = f"{meta.get('series_tag', meta.get('title'))} · Part 1" if a.sections else meta.get("series_tag")
    else:
        plan["series_tag"] = meta.get("series_tag")
        plan["hook"] = meta.get("hook") if a.hook else None
    assets = cache_dir(slug)
    sync_masters(assets)
    have = have_set(assets)
    items, need_masters = P.asset_items(meta, plan, have, masters_have(), inline_masters=a.auto_pick)
    if a.all_masters:
        need_masters = list(dict.fromkeys(need_masters + [n for n in meta.get("characters", {}) if n not in masters_have()]))
        items = [i for i in items if i["kind"] != "pose"]
    for n in need_masters:  # --auto-pick: one master per new character, made and used inside the same Kaggle job
        items.append(P.master_item(n, meta["characters"][n], n=1 if a.auto_pick else a.candidates))
    P.check_prompts(items)
    utts = P.utterances(plan, have)
    if a.no_gpu:  # composite + assemble from what exists (missing poses fall back to the master cut-outs) on a Kaggle CPU kernel
        if utts:
            sys.exit(f"--no-gpu needs all narration cached; {len(utts)} lines are missing")
        print(f"--no-gpu: skipping {len(items)} missing images (renderer falls back to masters/placeholders)")
        items, need_masters = [], []
    W, H, fps = (1080, 1920, 30)
    rcfg = {"name": name, "fps": fps, "W": W, "H": H, "pause_scale": 0.45 if a.short else 0.8,
            "max_shot_s": 4.6 if a.short else 7.0, "dissolve": 0.6 if a.short else 0.8, "lufs": -15.0,
            "cta": a.cta or ("Follow for Part 2" if a.short and a.sections else "Subscribe for a new story every week"),
            "end_card_s": 3.6, "workers": 4}
    job = {"story": slug, "images": items,
           "tts": {"voice": plan["voice"], "speed": speed, "utts": utts,
                   "samples": [{"id": v, "voice": v, "text": a.sample_text, "speed": speed} for v in a.samples.split(",")] if a.samples else []},
           "plan": plan, "render": None if ((need_masters and not a.auto_pick) or a.no_render) else rcfg}
    render_after = False
    if job["render"] and a.render_on == "cpu" and (items or utts) and not a.no_gpu and not a.local_preview:
        job["render"] = None   # GPU session makes assets only; the render follows on a free Kaggle CPU kernel
        render_after = True
    os.makedirs(os.path.join(ROOT, "out", name), exist_ok=True)
    json.dump(plan, open(os.path.join(ROOT, "out", name, "plan.json"), "w"), indent=1)
    print(f"plan: {sum(len(s['shots']) for s in plan['sections'])} shots, missing: {len(items)} images, {len(utts)} utterances,"
          f" masters needed: {need_masters or 'none'}, render: {bool(job['render'])}")
    if a.no_gpu:
        job["tts"]["samples"] = []
    if a.dry:
        print(json.dumps([i["id"] for i in items]))
        return
    refs = {"style_ref.png": os.path.join(ROOT, "brand", "style_ref.png")}
    if a.music:
        mb = "music_" + os.path.basename(a.music)
        refs[mb] = a.music
        rcfg["music"] = os.path.abspath(a.music) if a.local_preview else f"/kaggle/working/assets/{mb}"
    for n in masters_have():
        refs[f"master_{n}.png"] = os.path.join(CHARS, n, "master.png")
    needs_gpu = bool(items or utts or job["tts"]["samples"]) and not a.no_gpu
    use_ds = None
    if job["render"] and not a.local_preview:
        # the render needs every asset; cached ones go up through the private dataset
        cached = [f for f in os.listdir(assets) if _needed(f, plan)]
        if cached:
            use_ds = K.upload_cache(assets, cached, story=slug)
    if a.local_preview:
        job["render"] = None
        if not needs_gpu:
            return local_preview(plan, rcfg, assets, name)
    if not needs_gpu and not job["render"]:
        print("nothing to do"); return
    kslug = ("heartwick-gpu-" if needs_gpu else "heartwick-render-") + name.replace("_", "-")  # one kernel per output: jobs never collide
    ref, rec, t0 = K.push(job, refs, slug=kslug, gpu=needs_gpu,
                          use_cache_dataset=use_ds)
    state = K.wait(ref, rec, t0)
    dl = os.path.join(ROOT, "cache", slug, "kaggle_out", name)
    K.download(ref, dl)
    ingest(dl, assets)
    out = os.path.join(ROOT, "out", name)
    for f in os.listdir(dl):
        if f.endswith((".mp4", ".json", ".log")) or f == "samples":
            src = os.path.join(dl, f)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(out, f), dirs_exist_ok=True)
            else:
                shutil.copy(src, os.path.join(out, f))
    if need_masters and a.auto_pick:  # keep the inline masters as the series bible for next time
        for n in need_masters:
            d = os.path.join(CHARS, n); os.makedirs(d, exist_ok=True)
            for src, dst in ((f"master_{n}.png", "master.png"), (f"master_{n}_cut.png", "master_cut.png")):
                if os.path.exists(os.path.join(assets, src)):
                    shutil.copy(os.path.join(assets, src), os.path.join(d, dst))
    elif need_masters:
        contact_masters(assets, need_masters, os.path.join(out, "master_candidates.jpg"))
        print("Pick masters with: python heartwick.py pick <char> <k>   then re-run make")
        return
    if job["render"]:
        finish(plan, meta, name, out, assets)
    print("kaggle state:", state)
    if render_after and state == "complete":
        print("assets ready; rendering on a Kaggle CPU kernel...")
        a.no_gpu = True
        a.samples = ""
        cmd_make(a)


def _needed(f, plan):
    stem = os.path.splitext(f)[0]
    if stem.startswith("u_"):
        return any(u["id"] == stem for s in plan["sections"] for sh in s["shots"] for u in sh["utts"])
    if stem.startswith("bg_"):
        locs = {sh["bg"] for s in plan["sections"] for sh in s["shots"]}
        return stem[3:].replace("_depth", "") in locs
    if stem.startswith("master_") and stem.endswith("_cut") and "_c" not in stem[len("master_"):-4].replace("_kit", ""):
        return stem[len("master_"):-4] in {c["char"] for s in plan["sections"] for sh in s["shots"] for c in sh["chars"]}
    if stem.startswith("pose_"):
        poses = {f"pose_{c['char']}_{c['pose']}_cut" for s in plan["sections"] for sh in s["shots"] for c in sh["chars"]}
        return stem in poses
    return False


def local_preview(plan, rcfg, assets, name):
    from engine import timeline, render
    cfg = dict(rcfg, W=540, H=960, fps=15)
    tl = timeline.build(plan, assets, cfg)
    out = os.path.join(ROOT, "out", name, f"{name}_preview540.mp4")
    render._lower_priority()
    render.render_video(tl, assets, assets, out, cfg, workers=1, low=True)
    return out


def finish(plan, meta, name, out, assets):
    """Post-process: SRT, SEO pack, thumbnail, contact sheet, loudness report."""
    from engine import post
    post.finish(plan, meta, name, out, assets)


def pick(char, k, assets=None):
    d = os.path.join(CHARS, char)
    os.makedirs(d, exist_ok=True)
    if assets is None:
        for slug in os.listdir(os.path.join(ROOT, "cache")):
            p = os.path.join(ROOT, "cache", slug, "assets", f"master_{char}_c{k}.png")
            if os.path.exists(p):
                assets = os.path.dirname(p)
    shutil.copy(os.path.join(assets, f"master_{char}_c{k}.png"), os.path.join(d, "master.png"))
    cut = os.path.join(assets, f"master_{char}_c{k}_cut.png")
    if os.path.exists(cut):
        shutil.copy(cut, os.path.join(d, "master_cut.png"))
    print("master set:", char, "<- c", k)


def cmd_fetch(a):
    """Download a finished Kaggle job's outputs into the story's asset cache (for jobs whose waiter was not running)."""
    st = K._k("kernels", "status", f"{K.USER}/{a.kernel}", check=False)
    print(st.strip())
    if "complete" not in st.lower():
        sys.exit("job is not complete yet")
    assets = cache_dir(a.story)
    dl = os.path.join(ROOT, "cache", a.story, "kaggle_out", a.kernel)
    K.download(f"{K.USER}/{a.kernel}", dl)
    ingest(dl, assets)
    print("now run: python heartwick.py make <story.md> ... (cached assets are reused; add --no-gpu to render on Kaggle CPU)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    m = sub.add_parser("make")
    m.add_argument("story")
    m.add_argument("--sections", help="e.g. 1-2 (1-based chapter numbers)")
    m.add_argument("--name")
    m.add_argument("--short", action="store_true", help="Shorts pace (faster voice, tighter pauses and shots)")
    m.add_argument("--hook", action="store_true", help="show the hook title on long videos too")
    m.add_argument("--voice")
    m.add_argument("--speed", type=float)
    m.add_argument("--cta")
    m.add_argument("--music", help="music bed file (YouTube Audio Library), ducked 22 dB under the voice")
    m.add_argument("--samples", default="", help="comma list of voices to render a sample of, e.g. af_heart,af_bella")
    m.add_argument("--sample-text", default="Every morning that November, Ada hung her blue scarf on the porch hook. "
                                           "And every evening, it was gone.")
    m.add_argument("--candidates", type=int, default=4)
    m.add_argument("--auto-pick", action="store_true", help="new characters: make one master in the same job and continue (true one-command run)")
    m.add_argument("--all-masters", action="store_true", help="make master candidates for every character in the story")
    m.add_argument("--no-render", action="store_true")
    m.add_argument("--local-preview", action="store_true")
    m.add_argument("--render-on", choices=["cpu", "gpu"], default="cpu", help="where the final compositing runs (cpu saves GPU quota)")
    m.add_argument("--no-gpu", action="store_true", help="render on a Kaggle CPU kernel with the assets that exist (no GPU queue/quota)")
    m.add_argument("--dry", action="store_true")
    m.add_argument("--engine", choices=["v3", "v2", "v1"], default="v2",
                   help="v3: AI keys + Wan2.2 clips (engine3/); v2: puppet animation engine (default); v1: 2.5D slideshow")
    m.add_argument("--redo", help="v3: re-run only these shots' clips, e.g. S05,S07 (then re-assemble)")
    m.add_argument("--stop-after", choices=["shots", "keys", "clips"], help="v3: stop after this stage")
    m.add_argument("--style", choices=["B", "C"], help="v2 art direction: B vector cel, C paper theatre (default: pick-style choice, else B)")
    m.add_argument("--art", choices=["c", "v1"], default="c", help="v2 art source: c (style C puppets, default) or v1 (V1 painted storybook)")
    m.add_argument("--parts", type=int, default=1, help="v2: split the render across this many parallel Kaggle CPU kernels")
    m.add_argument("--vector", action="store_true", help="v2: ignore AI-art puppets (assets/characters, characters/<name>/rig) and use the vector cast")
    ps = sub.add_parser("pick-style", help="save the chosen v2 art direction (B or C) as the default")
    ps.add_argument("style", choices=["B", "C"])
    p = sub.add_parser("pick"); p.add_argument("char"); p.add_argument("k", type=int)
    sub.add_parser("status")
    f = sub.add_parser("fetch"); f.add_argument("kernel", help="e.g. heartwick-gpu-pip"); f.add_argument("--story", required=True)
    a = ap.parse_args()
    if a.cmd == "make":
        cmd_make(a)
    elif a.cmd == "pick":
        pick(a.char, a.k)
    elif a.cmd == "pick-style":
        open(STYLE_CHOICE, "w").write(a.style + "\n")
        print("default v2 style:", a.style)
    elif a.cmd == "fetch":
        cmd_fetch(a)
    elif a.cmd == "status":
        print(json.dumps(K.usage_summary(), indent=1))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
