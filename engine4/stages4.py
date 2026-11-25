"""Engine v4 kernel builders (laptop side, text only). Session A = narration + master keys (GPU)."""
import base64, io, json, os

from engine3 import cast
from engine3.stages import VOICES, src as _src3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K4 = os.path.join(ROOT, "engine4", "kaggle")
SHEETS = {"sheet_nell.jpg": os.path.join(ROOT, "out", "pip_short_v3", "kaggle", "keys", "out", "keys", "sheet_nell_s11.png"),
          "sheet_pip.jpg": os.path.join(ROOT, "out", "pip_short_v3", "kaggle", "keys", "out", "keys", "sheet_pip_s11.png")}
W, H = 1088, 1920

STYLE = ("Hand-drawn 2D anime film still, Studio-Ghibli-like warm watercolor backgrounds, clean line art, soft cinematic light, "
         "vertical 9:16 frame, full of gentle detail. No text, no watermark, no border.")
RULES = ("Every character's mouth is closed and relaxed. Each character appears exactly once; no other people, no extra "
         "animals. Keep faces, hair, glasses, clothes and colours identical to the character sheets.")


def _b64img(path, quality=88):
    from PIL import Image
    b = io.BytesIO(); Image.open(path).convert("RGB").save(b, "JPEG", quality=quality)
    return base64.b64encode(b.getvalue()).decode()


def master_jobs(shots4):
    jobs = []
    for m in shots4["masters"]:
        chars = m.get("chars") or []
        looks = " ".join(f"{cast.LOOK[c]}." for c in chars if c in cast.LOOK)
        refs, desc = [], []
        if m.get("ref") not in (None, "sheets"):
            refs.append("@" + m["ref"]); desc.append(f"Image {len(refs)} is the same place a moment earlier: keep its setting, "
                                                      "light, camera framing and art style.")
        if "nell" in chars:
            refs.append("sheet_nell.jpg"); desc.append(f"Image {len(refs)} is Nell's character sheet.")
        if {"pip", "pip_baby"} & set(chars):
            refs.append("sheet_pip.jpg"); desc.append(f"Image {len(refs)} is Pip's character sheet"
                                                      + (" (here he is a tiny baby version)." if "pip_baby" in chars else "."))
        prompt = f"{STYLE} {' '.join(desc)} {looks} {m['image']} {RULES if chars else 'No people, no animals unless described.'}"
        jobs.append({"id": m["id"], "prompt": prompt, "refs": refs, "w": W, "h": H, "seed": 7})
    return jobs


def narration_lines(shots4):
    lines = []
    S = shots4["shots"]
    for i, s in enumerate(S):
        nxt = S[i + 1] if i + 1 < len(S) else None
        pause = 0.45 if nxt and nxt.get("section") == s.get("section") else 0.9
        lines.append({"id": s["id"], "text": s["narration"], "pause": pause if nxt else 0.8})
    return lines


def voice_ref(voice):
    lib = json.load(open(os.path.join(VOICES, "voices.json"), encoding="utf-8"))
    lib = lib if isinstance(lib, list) else lib.get("voices", list(lib.values()))
    v = next(x for x in lib if x["id"] == voice)
    import soundfile as sf
    x, sr = sf.read(os.path.join(VOICES, v["reference_file"]), dtype="float32")
    b = io.BytesIO(); sf.write(b, x, sr, format="FLAC")
    return base64.b64encode(b.getvalue()).decode(), v["reference_text"]


def build_gpu_a(shots4, voice="narrator_baritone"):
    imgs = {k: _b64img(p) for k, p in SHEETS.items()}
    ref, ref_text = voice_ref(voice)
    jobs = master_jobs(shots4)
    head = "".join([f"IMGS = {imgs!r}\n", f"JOBS = {jobs!r}\n", f"LINES = {narration_lines(shots4)!r}\n",
                    f"REF_B64 = {ref!r}\n", f"REF_TEXT = {ref_text!r}\n", f"VOICE_ID = {voice!r}\n"])
    return head + open(os.path.join(K4, "gpu_a_main.py"), encoding="utf-8").read(), jobs


# --------------------------------------------------------------------------------------------- session B: acting
MOUTH = " Their mouths stay closed, lips together, silent; no talking."


def timeline(shots4, narr):
    """Shot i runs from its narration line's start to the next line's start (the last one to its end + 1.6 s)."""
    L = {l["id"]: l for l in narr["lines"]}
    out, S = [], shots4["shots"]
    for i, s in enumerate(S):
        t0 = 0.0 if i == 0 else L[s["id"]]["start"] - 0.12
        t1 = (L[S[i + 1]["id"]]["start"] - 0.12) if i + 1 < len(S) else L[s["id"]]["end"] + 1.6
        out.append({"id": s["id"], "t0": round(t0, 3), "t1": round(t1, 3)})
    return out


def build_clips4(shots4, tl, keys_kernel, user):
    from engine3.stages import VSTYLE, CONS, src as src3, winners
    dur = {t["id"]: t["t1"] - t["t0"] for t in tl}
    w = winners().get("video", {})
    plan = {"quant": w.get("quant", "Q5_K_M"), "w": 480, "h": 832, "frames": 81, "fps": 24,
            "qa_refs_nell": ["sheet_nell.jpg"], "qa_refs_pip": ["sheet_pip.jpg"], "clips": [], "durations": {}, "cams": {}}
    ms = {m["id"]: m for m in shots4["masters"]}
    for s in shots4["shots"]:
        if s["class"] != "A":
            continue
        frames = 49 if dur[s["id"]] < 4.0 else 81                     # 3.06 s or 5.06 s of motion, never stretched
        chars = set(ms[s["master"]].get("chars", [])) | set(ms.get(s.get("end_master") or "", {}).get("chars", []))
        plan["clips"].append({"shot": s["id"], "start_file": s["master"] + ".png",
                              "end_file": (s["end_master"] + ".png") if s.get("end_master") else None,
                              "prompt": VSTYLE + s["action"] + MOUTH + CONS, "seeds": [7], "frames": frames,
                              "nell": "nell" in chars, "pip": bool({"pip", "pip_baby"} & chars)})
        plan["durations"][s["id"]] = round((frames - 1) / 16, 3)       # natural length; the LA tail pads the rest
        plan["cams"][s["id"]] = "push"
    script = "".join([f"GEN_SRC = {src3('gen_flf.py')!r}\n", f"QA_SRC = {src3('qa.py')!r}\n", f"PLAN = {json.dumps(plan)!r}\n",
                      src3("clips_main.py")])
    return plan, script, [f"{user}/{keys_kernel}"]
