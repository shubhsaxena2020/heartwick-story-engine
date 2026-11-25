"""Push the GPU job that makes the music library (ACE-Step 1.5) and the anime hero shots (Animagine XL 4.0 keyframes,
LTX-Video / Wan2.2 image-to-video, RIFE to 30 fps), wait, and install the results:
  assets/music/<mood>_<k>.wav            reusable cue library (mixed in by scene mood, engine2/audio.py)
  assets/hero/<story>/<beat>_<model>.mp4 hero clips (+ keyframes) cut into the puppet video (story `hero=` directive)

    python tools/hero_music.py [--story pip_short] [--models ltx,wan] [--no-music]
Checks `kaggle kernels list --mine` first: Kaggle allows 2 GPU sessions and they are shared with other agents.
"""
import argparse, json, os, shutil, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import kaggle_run as K  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MOODS = {
    "tender": {"caption": "gentle warm ambient underscore, soft harp arpeggios, celesta, warm muted strings, slow and intimate, cozy storybook film score, no drums, instrumental", "bpm": 66, "duration": 90},
    "storm": {"caption": "dark tense cinematic underscore, low cello tremolo, distant timpani rolls, deep brass swells, minor key, slow building tension, no drum kit, instrumental", "bpm": 60, "duration": 90},
    "wonder": {"caption": "magical wonder underscore, celesta and glockenspiel sparkles, harp glissandos, shimmering high strings, lydian, gentle and bright, storybook film score, no drums, instrumental", "bpm": 72, "duration": 90},
    "resolution": {"caption": "heartfelt resolution, warm string ensemble swelling slowly, soft piano melody, harp, hopeful major key, tender ending, film score, no drums, instrumental", "bpm": 70, "duration": 90},
}
NELL = ("1girl, old woman, elderly, grandmother, wrinkles, solo, grey hair, hair bun, round eyewear, green cardigan, blue patterned scarf, "
        "long red skirt with blue embroidered hem")
BEATS = {
    "pip_short": {
        "kneel": {"prompt": f"{NELL}, kneeling in wet red maple leaves, holding a tiny newborn baby squirrel in cupped hands, tender smile, "
                            "looking down, autumn morning after storm, big maple tree, mist, light rays, soft warm light, from side, storybook, emerald and ruby colors",
                  "motion": "Anime film scene. The old woman kneeling in the wet red leaves gently lifts the tiny baby squirrel in her cupped hands "
                            "closer to her face, she smiles softly and blinks, red maple leaves drift down, the mist drifts, slow gentle camera push-in.",
                  "seeds": [11, 23]},
        "goodbye": {"prompt": "grey squirrel, animal focus, sitting on a windowsill, open window, looking back, bushy tail, lace curtains blowing, "
                              "spring morning, budding red maple tree outside, small wooden ramp, sunlight, warm interior, cozy, detailed fur, emerald and ruby colors",
                    "motion": "Anime film scene. The grey squirrel on the open windowsill looks back over its shoulder, its bushy tail sways, "
                              "the lace curtains billow in the spring breeze, sunlight shimmers, slow gentle camera drift.",
                    "seeds": [11, 23]},
        "return": {"prompt": f"{NELL}, smiling with tears, hand on window glass, grey squirrel sitting on the windowsill outside holding a walnut, "
                             "dusk, orange and violet sky, maple branches, warm lamplight inside, cozy, emotional, from side, emerald and ruby colors",
                   "motion": "Anime film scene. The squirrel on the windowsill nibbles the walnut and looks up at the old woman, she smiles with "
                             "teary eyes and presses her hand to the glass, warm light glows, slow gentle camera push-in.",
                   "seeds": [11, 23]},
    }
}


def v1_job():
    """Rig-ready painted figures in the exact V1 look: SDXL + IP-Adapter Plus from the chosen V1 masters (characters/<name>/master.png)."""
    import base64, io, yaml
    from PIL import Image
    from engine.style import STYLE_PREFIX, STYLE_NEGATIVE, CHAR_SUFFIX
    raw = open(os.path.join(ROOT, "stories", "pip.md"), encoding="utf-8").read().split("---")[1]
    cast = yaml.safe_load(raw)["characters"]
    pose_h = "standing straight, side view facing right, both arms held out away from the body, hands open, legs apart, feet flat"
    pose_q = {"pip": "standing on all four paws, side view facing right, bushy tail held out behind",
              "pip_kit": "lying on its belly, side view facing right, tiny paws visible, thin tail behind"}
    chars = {}
    for n, c in cast.items():
        b = io.BytesIO()
        im = Image.open(os.path.join(ROOT, "characters", n, "master.png")).convert("RGB")
        im.thumbnail((512, 512))
        im.save(b, "JPEG", quality=88)
        animal = n in ("pip", "pip_kit")
        chars[n] = {"prompt": f"{STYLE_PREFIX}, full body, {pose_q[n] if animal else pose_h}, {c['desc']}, {CHAR_SUFFIX}",
                    "ref_b64": base64.b64encode(b.getvalue()).decode(), "kind": "squirrel" if animal else "humanoid",
                    "seeds": [c.get("seed", 1000) + k * 77 for k in range(2 if animal else 4)], "ip_scale": 0.5 if animal else 0.55,
                    "w": 1024 if animal else 832, "h": 832 if animal else 1216}
    return {"chars": chars, "neg": STYLE_NEGATIVE + ", sitting, arms crossed, hands in pockets, cropped, multiple characters, scenery"}


def gpu_busy():
    out = K._k("kernels", "list", "--mine", "--page-size", "20", "--sort-by", "dateRun", check=False)
    busy = []
    for line in out.splitlines()[2:]:
        ref = line.split()[0] if line.strip() else ""
        if not ref.startswith(K.USER + "/"):
            continue
        st = K._k("kernels", "status", ref, check=False).lower()
        if "running" in st or "queued" in st:
            busy.append((ref, "running" if "running" in st else "queued"))
    return busy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", default="pip_short")
    ap.add_argument("--models", default="ltx,wan")
    ap.add_argument("--no-music", action="store_true")
    ap.add_argument("--music-only", action="store_true", help="just the ACE-Step mood library (no hero shots)")
    ap.add_argument("--v1-chars", action="store_true", help="also make V1-style rig-ready painted characters (art source A)")
    ap.add_argument("--slug", default="heartwick-hero-music")
    a = ap.parse_args()
    print("kernels running/queued on this account:", gpu_busy())
    job = {"moods": {} if a.no_music else MOODS, "beats": {} if a.music_only else BEATS[a.story], "models": [] if a.music_only else a.models.split(",")}
    if a.v1_chars:
        job["v1"] = v1_job()
        job["engine_files"] = {f: open(os.path.join(ROOT, "engine2", f), encoding="utf-8").read() for f in ("__init__.py", "autorig.py", "mathx.py")}
    src = open(os.path.join(ROOT, "kaggle", "hero_music_job.py"), encoding="utf-8").read()
    d = os.path.join(ROOT, "kaggle", "build", "hero-music")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "hero_music_job.py"), "w", encoding="utf-8").write(f"JOB = {json.dumps(job)!r}\nJOB = __import__('json').loads(JOB)\n" + src)
    slug = a.slug
    json.dump({"id": f"{K.USER}/{slug}", "title": slug.replace("-", " ").title(), "code_file": "hero_music_job.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
               "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []},
              open(os.path.join(d, "kernel-metadata.json"), "w"))
    for attempt in range(300):  # 2 GPU sessions per account, shared with other agents: wait for a free one
        out = K._k("kernels", "push", "-p", d, "--accelerator", "NvidiaTeslaT4", check=False)
        if "successfully pushed" in out.lower():
            print(out.strip()[-200:], flush=True)
            break
        if "maximum" in out.lower() or "session" in out.lower():
            if attempt == 0:
                print("GPU sessions full; waiting for a free one...", flush=True)
            time.sleep(120)
            continue
        raise RuntimeError(out[-500:])
    ref = f"{K.USER}/{slug}"
    state = K.wait(ref, {"slug": ref, "gpu": True, "engine": "v2-hero-music"}, time.time(), poll=60, max_h=11)
    dl = os.path.join(ROOT, "out", "hero_music")
    K.download(ref, dl)
    md = os.path.join(ROOT, "assets", "music")
    hd = os.path.join(ROOT, "assets", "hero", a.story)
    os.makedirs(md, exist_ok=True)
    os.makedirs(hd, exist_ok=True)
    v1 = os.path.join(dl, "v1chars")
    for n in os.listdir(v1) if os.path.isdir(v1) else []:  # art source A: painted V1-style rigs under characters/<name>/
        src_n = os.path.join(v1, n)
        if os.path.isdir(src_n):
            shutil.copytree(src_n, os.path.join(ROOT, "characters", n, "v1rig"), dirs_exist_ok=True)
    for sub, dst in (("music", md), ("hero", hd), ("keyframes", hd)):
        p = os.path.join(dl, sub)
        for f in os.listdir(p) if os.path.isdir(p) else []:
            shutil.copy(os.path.join(p, f), os.path.join(dst, f))
    print("kaggle:", state, "->", dl)


if __name__ == "__main__":
    main()
