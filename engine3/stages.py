"""Builders for the engine v3 Kaggle kernels: each returns the full kernel script (sources + data prepended)."""
import base64, io, json, os

from PIL import Image

from . import cast, poses

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KDIR = os.path.join(HERE, "kaggle")
K0_SRC = os.path.join(ROOT, "research", "round2", "quality", "refs", "round1_keyframe.png")   # Nell's master key (Z-Image-Turbo, round 1)
WINNERS = os.path.join(ROOT, "research", "round3", "WINNERS.json")

DEFAULT_MUSIC = {"caption": "gentle warm autumn storybook underscore, soft fingerpicked acoustic guitar and harp, celesta, warm "
                            "muted strings, slow and tender, cozy hand-drawn animation film score, no drums, instrumental",
                 "bpm": 66}


def winners():
    """research/round3/WINNERS.json from the permutation agent (adopted per stage when present)."""
    try:
        return json.load(open(WINNERS, encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def src(name):
    return open(os.path.join(KDIR, name), encoding="utf-8").read()


def b64img(path, fmt="JPEG", size=None, quality=95):
    im = Image.open(path).convert("RGB")
    if size:
        im = im.resize(size, Image.LANCZOS)
    b = io.BytesIO(); im.save(b, fmt, **({"quality": quality} if fmt == "JPEG" else {}))
    return base64.b64encode(b.getvalue()).decode()


def key_prompt(k, keys):
    chars = k.get("chars", [])
    pose = k.get("pose", "none") if "nell" in chars else "none"
    frame = cast.FRAME.get(k.get("framing"), cast.FRAME["medium-full"])
    if k["from"] == "new":
        refs, desc = [], []
        if "nell" in chars:
            refs.append("@sheet_nell"); desc.append(f"Image {len(refs)} is the character sheet of {cast.LOOK['nell']}.")
        if {"pip", "pip_baby"} & set(chars):
            refs.append("@sheet_pip")
            extra = " In this scene he appears as a tiny baby version of that squirrel, small enough to fit in one palm." if "pip_baby" in chars else ""
            desc.append(f"Image {len(refs)} is the character sheet of {cast.LOOK['pip']}.{extra}")
        if pose != "none":
            refs.append(f"@IN@/pose_{pose}.png"); desc.append(f"Image {len(refs)} is a pose skeleton, a guide for Nell's body pose only: never draw the skeleton, a stick figure or any extra person.")
        if not refs:   # an environment still: paint it from the master key so the style and palette carry over
            refs.append("@K0"); desc.append("Image 1 shows the series' art style and colour palette (ignore its people and its place).")
        prompt = (f"{cast.STYLE} {' '.join(desc)} Paint a new, complete illustrated scene (not a character sheet, no plain "
                  f"background): {k['image']} {frame} {cast.KEEP}")
    else:
        refs = ["@" + k["from"]]
        desc = "Image 1 is the previous shot of this scene."
        if pose != "none":
            refs.append(f"@IN@/pose_{pose}.png"); desc += " Image 2 is a pose skeleton, a guide for Nell's pose only: never draw the skeleton, a stick figure or any extra person."
        prompt = (f"{desc} Keep everything else identical: the same camera framing, background, lighting, characters, and "
                  f"the squirrel at the same size. Change only this: {k.get('change') or k['image']} The result shows: "
                  f"{k['image']} {frame} {cast.KEEP}")
    if "nell" in chars:   # 24 Nov run 5: K8 painted a second Nell in the background
        prompt += " Nell is the only person in the picture; no other people or figures anywhere, inside or outside."
    pip_from = None
    if k["from"] not in ("new",) and {"pip", "pip_baby"} & set(chars):
        src_k = keys.get(k["from"]) or {"chars": ["nell", "pip"]}   # K0 has the adult squirrel
        same = [c for c in ("pip", "pip_baby") if c in chars and c in src_k.get("chars", [])]
        if same:   # only gate Pip's size against a source key with the same size class (baby vs adult is intended)
            pip_from = k["from"]
    return {"id": k["id"], "refs": refs, "prompt": prompt, "W": 480, "H": 832, "chars": chars,
            "seeds": [3, 21] if "nell" in chars else [3], "pip_from": pip_from}


def keys_jobs(shots, skip=(), only=None):
    keys = {k["id"]: k for k in shots["keys"]}
    sheets = [
        {"id": "sheet_nell", "refs": ["@K0"], "W": 832, "H": 480, "seeds": [11], "chars": ["nell"],
         "prompt": f"Character reference sheet of the woman in image 1: {cast.LOOK['nell']}. Three full-body views side by side "
                   "on a plain warm white background: front view, three-quarter view, side view, standing neutral with arms "
                   "relaxed, the whole body visible with a margin. Same exact character design, face and colours, same "
                   "hand-drawn anime art style. No text."},
        {"id": "sheet_pip", "refs": ["@K0"], "W": 832, "H": 480, "seeds": [11], "chars": ["pip"],
         "prompt": "Character reference sheet of the small grey squirrel in image 1 (ignore the woman and the place): "
                   f"{cast.LOOK['pip']}. Three views side by side on a plain warm white background: sitting facing front, "
                   "side view, three-quarter view, each large and fully visible with a margin. Same exact design and colours, "
                   "same hand-drawn anime art style. No text."},
    ]
    J = [j for j in sheets] + [key_prompt(k, keys) for k in shots["keys"]]
    if only:
        J = [j for j in J if j["id"] in only]
    return {"smoke_ref": "@IN@/K0.png", "given": {"K0": "@IN@/K0.png"}, "min_headroom": 0.03,
            "quant": winners().get("keys", {}).get("quant", "Q4_K_M"), "keys": J, "skip": list(skip)}


VOICES = os.path.join(ROOT, "research", "voices", "library")


def narration_lines(shots):
    """One TTS line per shot; pause after it: longer at scene changes and before the last (moral) line."""
    keys = {k["id"]: k for k in shots["keys"]}
    S = shots["shots"]; lines = []
    for i, s in enumerate(S):
        pause = 0.45
        if i + 1 < len(S):
            nk = keys.get(S[i + 1]["start"]) or {}
            if (nk.get("scene") or "") != (keys.get(s["start"], {}).get("scene") or ""):
                pause = 0.75
            if i + 2 == len(S):
                pause = 1.0
        else:
            pause = 0.6
        lines.append({"id": s["id"], "text": s["narration"].strip(), "pause": pause})
    return lines


def build_audio(shots, voice=None):
    voice = voice or winners().get("voice", {}).get("id") or "narrator_baritone"
    lib = json.load(open(os.path.join(VOICES, "voices.json"), encoding="utf-8"))
    lib = lib if isinstance(lib, list) else lib.get("voices", list(lib.values()))
    v = next(x for x in lib if x["id"] == voice)
    import soundfile as sf   # FLAC keeps the reference lossless at about half the bytes (the script must stay < 1 MB)
    x, sr = sf.read(os.path.join(VOICES, v["reference_file"]), dtype="float32")
    b = io.BytesIO(); sf.write(b, x, sr, format="FLAC")
    ref = base64.b64encode(b.getvalue()).decode()
    return voice, "".join([f"REF_B64 = {ref!r}\n", f"REF_TEXT = {v['reference_text']!r}\n", f"VOICE_ID = {voice!r}\n",
                           f"LINES = {narration_lines(shots)!r}\n", src("audio_main.py")])


def timeline(shots, narr, tail=1.6):
    """Shot spans from the narration: a cut sits inside the pause before the next shot's line (60 % into the gap),
    the first shot starts at 0, the last ends `tail` s after the last word. Returns [{id, t0, t1}]."""
    L = {ln["id"]: ln for ln in narr["lines"]}
    S = shots["shots"]; cuts = [0.0]
    for a, b in zip(S, S[1:]):
        ea, sb = L[a["id"]]["end"], L[b["id"]]["start"]
        if L[a["id"]].get("words"):
            ea = max(ea - 0.15, L[a["id"]]["words"][-1]["e"])
        if L[b["id"]].get("words"):
            sb = min(sb + 0.1, L[b["id"]]["words"][0]["s"])
        cuts.append(round(ea + 0.6 * max(0.0, sb - ea), 3))
    cuts.append(round(L[S[-1]["id"]]["end"] + tail, 3))
    return [{"id": s["id"], "t0": cuts[i], "t1": cuts[i + 1]} for i, s in enumerate(S)]


CONS = " Smooth natural motion, consistent characters, the squirrel keeps the same size, no morphing, no text."
VSTYLE = "2D hand-drawn anime animation, clean line art, warm watercolor background. "


def cam_of(s):
    m = (s.get("motion") or "").lower()
    if "pull back" in m or "pulls back" in m or "pull-back" in m:
        return "pull"
    return "drift" if (" pan" in m or "camera drift" in m or "camera slowly drift" in m) else "push"


def build_clips(shots, tl, keys_rec, overrides=None, only=None):
    """keys_rec: {key id: chosen file name} from the keys kernel; overrides: {key id: file} picked by a human look."""
    chosen = dict(keys_rec); chosen.update(overrides or {})
    chosen.setdefault("K0", "K0.png")
    keys = {k["id"]: k for k in shots["keys"]}
    w = winners().get("video", {})
    plan = {"quant": w.get("quant", "Q5_K_M"), "w": w.get("w", 480), "h": w.get("h", 832), "frames": 81,
            "qa_refs_nell": ["K0.png", chosen.get("sheet_nell", "")], "qa_refs_pip": [chosen.get("sheet_pip", "K0.png")],
            "clips": [], "durations": {}, "cams": {}}
    for s in shots["shots"]:
        if only and s["id"] not in only:
            continue
        ks = [keys.get(s["start"], {"chars": ["nell", "pip"]})] + ([keys[s["end"]]] if s.get("end") in keys else [])
        chars = set(c for k in ks for c in k.get("chars", []))
        plan["clips"].append({"shot": s["id"], "start_file": chosen[s["start"]], "end_file": chosen.get(s["end"]) if s.get("end") else None,
                              "prompt": VSTYLE + s["motion"] + CONS, "seeds": [7, 11] if s.get("hero") else [7],
                              "nell": "nell" in chars, "pip": bool({"pip", "pip_baby"} & chars)})
    for t in tl:
        plan["durations"][t["id"]] = round(t["t1"] - t["t0"], 3)
    for s in shots["shots"]:
        plan["cams"][s["id"]] = cam_of(s)
    return plan, "".join([f"GEN_SRC = {src('gen_flf.py')!r}\n", f"QA_SRC = {src('qa.py')!r}\n", f"PLAN = {json.dumps(plan)!r}\n", src("clips_main.py")])


def build_assemble(shots, tl, meta, name, music_file="music_1.wav", srcs=None, fps=None):
    sh = {s["id"]: dict(s, src=(srcs or {}).get(s["id"])) for s in shots["shots"]}
    hero = next((s["id"] for s in shots["shots"] if s.get("hero")), shots["shots"][0]["id"])
    hook = meta.get("hook") or {}
    plan = {"name": name, "shots": [dict(t, sfx=sh[t["id"]].get("sfx", ""), src=sh[t["id"]]["src"]) for t in tl], "music_file": music_file,
            "highlight_words": meta.get("highlight_words", []), "series_label": meta.get("series_tag", "Pip"),
            "cta": "Follow for more Heartwick stories", "end_card_s": 3.0, "caption_y": 0.71, "tag_s": 4.0,
            "pad_gap_db": winners().get("music", {}).get("gap_db", 20.0), "thumb_shot": shots["shots"][0]["id"] if hook else hero,
            "hook_lines": hook.get("lines"), "hook_highlight": hook.get("highlight"), "seo": meta.get("seo_pack", {})}
    if fps:                               # engine v4: 24 fps (v3 default stays 60)
        plan["fps"] = fps
    eng = {"brand.py": open(os.path.join(ROOT, "engine", "brand.py"), encoding="utf-8").read().replace("from .style import", "from engine.style import"),
           "style.py": open(os.path.join(ROOT, "engine", "style.py"), encoding="utf-8").read(),
           "audio2.py": open(os.path.join(ROOT, "engine2", "audio.py"), encoding="utf-8").read()}
    return plan, "".join([f"ENGINE = {eng!r}\n", f"PLAN = {json.dumps(plan)!r}\n", src("assemble_main.py")])


def build_keys(shots, music_s, given=None, skip=(), only=None):
    """given: {key id: local png} of already approved keys (re-runs); they ride along as assets."""
    jobs = keys_jobs(shots, skip=skip, only=only)
    pd = os.path.join(ROOT, "cache", "v3_poses"); poses.draw_all(pd)
    assets = {"K0.png": b64img(K0_SRC, "PNG")}
    if len(assets["K0.png"]) > 600_000:
        assets["K0.png"] = b64img(K0_SRC, "JPEG", quality=95)
    used = {r.split("/")[-1] for j in jobs["keys"] for r in j["refs"] if r.startswith("@IN@/pose_")}
    for p in used:
        assets[p] = b64img(os.path.join(pd, p), "PNG")
    for kid, path in (given or {}).items():
        assets[f"{kid}.png"] = b64img(path, "JPEG", quality=93)
        jobs["given"][kid] = f"@IN@/{kid}.png"
    w = winners().get("music", {})
    music = dict(DEFAULT_MUSIC, **{k: v for k, v in w.items() if k in ("caption", "bpm")}, duration=int(min(120, music_s))) if music_s else None
    return "".join([f"ASSETS = {assets!r}\n", f"KEYS_SRC = {src('keys_gen.py')!r}\n", f"QA_SRC = {src('qa.py')!r}\n",
                    f"JOBS = {json.dumps(jobs)!r}\n", f"MUSIC = {music!r}\n", src("keys_main.py")])
