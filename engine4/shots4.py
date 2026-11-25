"""Engine v4 stage 1: a full story (7-8 min) -> shots4.json.

The narration is the story's own text, cut here into ~80 chunks (whole sentences, ~8-22 words), so the author's words
are never paraphrased. The LLM (local 9Router) only plans the pictures for each chunk:
  masters: ~30-45 still keys (FLUX.2-klein), each a full vertical composition of one moment;
  shots:   per chunk: class L (living still: camera + parallax + breathing + FX on a master),
           E (establishing / insert plate, no character acting) or A (AI acting clip, Wan2.2-A14B, only for an action the
           story needs: walk, kneel, lift, open a window, hop), the master it uses, a named crop, camera, FX.
Durations come later from the TTS word times (no stretching anywhere)."""
import json, os, re, time

from engine3 import cast
from engine3.shots import MODEL, _chat, _json

CROPS = {"full": None, "upper": [0.06, 0.0, 0.94, 0.62], "center": [0.12, 0.18, 0.88, 0.82], "lower": [0.06, 0.38, 0.94, 1.0]}
CAMS = ["push_in", "pull_out", "pan_left", "pan_right", "tilt_up", "tilt_down", "drift"]
FX = ["dust", "steam", "snow", "rain", "leaves", "light"]

SYSTEM = f"""You are the director of "Heartwick", a channel of warm hand-drawn-anime story videos for adult US viewers (not
for kids). A 7-8 minute vertical (9:16) episode is made like real limited-animation anime:
- Most shots are class "L" (living still): one beautiful still "master" image with a slow camera move, depth parallax,
  the character gently breathing, and ambient FX (dust motes in light, steam from a mug, snow or rain behind a window,
  falling leaves, light shimmer). Mouths never move.
- Class "E": establishing / insert stills with no character acting (the farmhouse at dawn, the maple, the window,
  a mug on the table, the empty sill) - camera move + FX.
- Class "A": a real AI-animated acting clip (3 or 5 seconds) ONLY where the story needs a visible action: walking to
  something, kneeling, lifting/holding the squirrel, opening a window, the squirrel hopping/leaving/returning. Use
  between 6 and 10 A shots in the whole episode, on the most important story beats. An A shot starts from its master
  and may end on another master ("end_master") when the end pose matters.
- Masters are made by an image model from the character sheets; a master may be "ref": an earlier master to keep the
  same room/setting/framing (only pose or small things change). Reuse a master for several consecutive shots with
  different crops and camera moves (that is how anime saves drawings): about 30-45 masters for ~80 shots.
- Only these characters may appear on screen: nell, pip (adult squirrel), pip_baby (tiny baby squirrel). Other people
  in the story (Walt, Dee, Eli...) are only mentioned by the narrator, never shown.
- Crops (a medium/close shot is a crop of a master): full, upper (head and shoulders area, top 62%), center, lower.
  Only use upper/center when the master's main subject is clearly there.
- Cameras: {", ".join(CAMS)}. FX: {", ".join(FX)} (snow/rain only when a window or sky is visible).
- Character looks (repeat in every master that shows them): nell = {cast.LOOK['nell']}; pip = {cast.LOOK['pip']};
  pip_baby = {cast.LOOK['pip_baby']}. Seasons, rooms and props must follow the story (autumn -> winter -> spring).

You get the narration already cut into numbered chunks C001..; make exactly one shot per chunk, in order.
Reply with ONLY a JSON object:
{{"title": str,
 "masters": [{{"id": "M01", "scene": str, "ref": "sheets" | "<earlier master id>", "chars": ["nell", "pip" | "pip_baby"],
             "image": "<one paragraph: everything visible: setting, season, light, time of day, where each character is,
                        their pose and expression (mouth closed), the squirrel's size vs her hands, camera distance>"}}],
 "shots": [{{"chunk": "C001", "class": "L" | "E" | "A", "master": "M01", "crop": "full" | "upper" | "center" | "lower",
            "cam": "<camera>", "fx": ["dust", ...], "char_motion": "breath" | "breath_sway" | "none",
            "action": "<A only: the motion in one sentence, mouths closed>", "end_master": "<A only, optional>"}}]}}"""


def narration_chunks(story_path, lo=8, hi=22):
    """Whole sentences of the story, merged per paragraph into chunks of ~lo..hi words."""
    raw = open(story_path, encoding="utf-8").read()
    body = raw.split("---", 2)[2] if raw.startswith("---") else raw
    chunks, section = [], ""
    for ln in body.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith(":::"):
            continue
        if ln.startswith("## "):
            section = ln[3:]
            continue
        ln = re.sub(r"\[pause [0-9.]+s\]", "", ln).replace("*", "").strip()
        sents = [s for s in re.split(r"(?<=[.!?…])\s+(?=[\"'A-Z(])", ln) if s.strip()]
        cur = []
        for s in sents:
            cur.append(s)
            n = len(" ".join(cur).split())
            if n >= lo:
                chunks.append({"text": " ".join(cur), "section": section}); cur = []
        if cur:                                   # a short tail joins the previous chunk of the same paragraph
            if chunks and chunks[-1]["section"] == section and len((chunks[-1]["text"] + " " + " ".join(cur)).split()) <= hi + 6:
                chunks[-1]["text"] += " " + " ".join(cur)
            else:
                chunks.append({"text": " ".join(cur), "section": section})
    for i, c in enumerate(chunks):
        c["id"] = f"C{i + 1:03d}"
    return chunks


def validate(d, chunks):
    errs, ms = [], {}
    for m in d.get("masters", []):
        if m.get("ref") not in ("sheets", None) and m.get("ref") not in ms:
            errs.append(f"master {m.get('id')}: ref {m.get('ref')} is not an earlier master")
        if set(m.get("chars", [])) - {"nell", "pip", "pip_baby"}:
            errs.append(f"master {m.get('id')}: unknown chars {m.get('chars')}")
        ms[m.get("id")] = m
    shots = d.get("shots", [])
    if [s.get("chunk") for s in shots] != [c["id"] for c in chunks]:
        errs.append(f"need exactly one shot per chunk in order C001..{chunks[-1]['id']} ({len(shots)} shots given)")
    n_a = 0
    for s in shots:
        if s.get("master") not in ms:
            errs.append(f"shot {s.get('chunk')}: master {s.get('master')} not defined")
        if s.get("class") not in ("L", "E", "A"):
            errs.append(f"shot {s.get('chunk')}: class {s.get('class')}")
        if s.get("crop") not in CROPS:
            errs.append(f"shot {s.get('chunk')}: crop {s.get('crop')}")
        if s.get("cam") not in CAMS:
            errs.append(f"shot {s.get('chunk')}: cam {s.get('cam')}")
        if set(s.get("fx") or []) - set(FX):
            errs.append(f"shot {s.get('chunk')}: fx {s.get('fx')}")
        if s.get("class") == "A":
            n_a += 1
            if not s.get("action"):
                errs.append(f"shot {s.get('chunk')}: A shot needs an action")
            if s.get("end_master") and s["end_master"] not in ms:
                errs.append(f"shot {s.get('chunk')}: end_master {s['end_master']} not defined")
    if not 5 <= n_a <= 12:
        errs.append(f"{n_a} A shots (want 6-10)")
    if not 20 <= len(ms) <= 50:
        errs.append(f"{len(ms)} masters (want 30-45)")
    return errs


def write_shots(story_path, out_path):
    chunks = narration_chunks(story_path)
    listing = "\n".join(f"{c['id']} [{c['section']}] {c['text']}" for c in chunks)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"NARRATION CHUNKS ({len(chunks)}):\n{listing}\n\nWrite the JSON."}]
    log, d, errs = [], None, ["no reply"]
    for attempt in range(3):
        t = time.time()
        txt = _chat(msgs, max_tokens=48000)
        log.append({"attempt": attempt, "s": round(time.time() - t, 1), "chars": len(txt)})
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        open(out_path + f".raw{attempt}.txt", "w", encoding="utf-8").write(txt)
        print("LLM reply", log[-1], flush=True)
        try:
            d = _json(txt); errs = validate(d, chunks)
        except Exception as e:
            d, errs = None, [f"parse: {e}"]
        log[-1]["errors"] = errs
        if not errs:
            break
        msgs += [{"role": "assistant", "content": txt},
                 {"role": "user", "content": "Fix these problems and reply with the full corrected JSON only:\n- " + "\n- ".join(errs)}]
    if d is None or errs:
        raise RuntimeError(f"shot list failed validation: {errs}")
    by = {c["id"]: c for c in chunks}
    for i, s in enumerate(d["shots"]):
        s["id"] = f"S{i + 1:03d}"; s["narration"] = by[s["chunk"]]["text"]; s["section"] = by[s["chunk"]]["section"]
        s["crop_box"] = CROPS[s["crop"]]
    d["llm"] = {"model": MODEL, "log": log, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(d, open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return d
