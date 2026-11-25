"""Stage 1: story -> shots.json. An LLM (local 9Router, OpenAI-compatible, stream:false) writes the shot list:
keys (start/end stills, each an edit of an earlier key or of the character sheets), shots (start/end key, action,
framing, narration line, duration). The result is validated here; a failed validation is sent back once for repair."""
import json, os, re, time, urllib.error, urllib.request

from . import cast

ROUTER = os.environ.get("NINEROUTER_URL", "http://127.0.0.1:20128/v1/chat/completions")
MODEL = os.environ.get("HEARTWICK_LLM", "cc/claude-sonnet-5-5")

SYSTEM = """You are the director of "Heartwick", a channel of warm, hand-drawn-anime animated story Shorts for adult
US viewers (not made for kids). You turn a story into a SHOT LIST for an AI pipeline:
- Every shot is one AI video clip of exactly 81 frames (5.06 s of motion). It may be played between 4.2 s and 6.8 s long.
- Each shot starts from an approved still "key" image and, for action shots, ends on another approved key
  (first-last-frame mode "flf": the model only moves between the two stills). "Hold" shots (reactions, close-ups,
  environment, small gestures) use mode "i2v": a start key only, plus a motion prompt with small, gentle movement.
- Keys are made by an image EDIT model. A key is either "new" (a new setting, painted from the character sheets) or
  an edit of an earlier key in the same scene ("from": that key id), where only the pose / small things change and the
  camera framing, background and character sizes stay identical. Consecutive action shots in one scene should share
  the boundary key (shot A ends on key X, shot B starts on key X) so they cut as one continuous action.
- The special key "K0" already exists: a full-body wide shot of Nell in her autumn New England yard, walking across
  fallen leaves toward a big red-orange maple tree, a white farmhouse and picket fence behind, with a small grey squirrel
  sitting at the foot of the maple trunk. Use it as the start of the yard sequence.
- Only these characters may appear on screen: nell, pip (adult squirrel), pip_baby (the same squirrel as a tiny baby).
  Other people in the story are only mentioned by the narrator, never shown.
- Pose skeleton ids (for Nell's full-body pose in a key; use "none" when Nell is absent or the framing is medium/close):
  none, stand_look (standing, looking down at something at her feet), stand_walk (mid-stride), kneel_reach (kneeling on
  ONE knee, both hands reaching forward-down), kneel_hold (kneeling on one knee, holding something in cupped hands at her
  chest), sit_table (seated at a table), stand_window (standing at a window, hands on the sill).
- Framing classes: wide, medium-full (whole body, head in the upper third), medium (waist up), close (face and hands),
  insert (an object or the squirrel alone).
- Keep it achievable: simple, readable actions per shot (walk to, kneel, lift, turn the head, open a window, hop onto
  a branch). No crowds, no text, no fast action, no hands doing fine work in close-up.
- The narration is spoken by one warm baritone narrator at about 2.9 words per second. Write it as the final voice-over:
  short, vivid, emotional sentences, keep the story's best lines and dialogue. Each shot covers one narration line
  (one or two sentences, 8-17 words, fitting its duration). Total narration 160-205 words.
- 12 to 14 shots, total 62-88 s. Shot 1 is a hook (the emotional end of the story in one image + the hook line).
  The last shot is the moral; it should be at least 6 s.

Reply with ONLY a JSON object, no prose, in this schema:
{"title": str, "hook": str,
 "keys": [{"id": "K1", "scene": str, "from": "new" | "K0" | "<earlier key id>",
           "chars": ["nell", "pip" | "pip_baby"], "pose": <pose id>, "framing": <framing class>,
           "image": "<one paragraph: everything visible in this still: setting, light, time of day, where each character is,
                     what they do, their size relation (the squirrel's size vs her hands/head), camera distance>",
           "change": "<only for keys with from != new: the exact change from the source key, e.g. 'she now kneels on one
                      knee...'; everything else identical>"}],
 "shots": [{"id": "S01", "narration": str, "duration_s": float, "mode": "flf" | "i2v", "start": "<key id>",
            "end": "<key id or null>", "framing": <framing class>,
            "motion": "<video prompt: the motion between start and end, camera move, ambient motion (leaves, steam,
                       curtains, light); describe the characters briefly>", "hero": bool, "sfx": "<few words or empty>"}]}
Mark 2-3 key emotional shots as hero (they get two takes)."""


def story_text(path):
    raw = open(path, encoding="utf-8").read()
    body = raw.split("---", 2)[2] if raw.startswith("---") else raw
    lines = []
    for ln in body.splitlines():
        if ln.startswith(":::"):
            continue
        ln = re.sub(r"\[pause [0-9.]+s\]", "", ln).strip()
        if ln.startswith("## "):
            ln = f"[{ln[3:]}]"
        if ln:
            lines.append(ln)
    return "\n".join(lines)


def _chat(messages, max_tokens=32000, temperature=0.4, stream=True):
    # the model thinks before answering and 9Router does not forward the thinking: at max_tokens 9000 the whole budget
    # went to thinking (finish_reason "length", empty content), so the budget is large
    """Long replies (a full shot list, ~6k tokens) outlast 9Router's non-streaming upstream timeout (it answers 503
    after ~3 min), so the default is SSE streaming; short calls may use stream=False."""
    body = json.dumps({"model": MODEL, "messages": messages, "stream": stream, "max_tokens": max_tokens,
                       "temperature": temperature}).encode()
    req = urllib.request.Request(ROUTER, data=body, headers={"Content-Type": "application/json",
                                                             "Authorization": "Bearer " + os.environ.get("NINEROUTER_API_KEY", "x")})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                if not stream:
                    return json.loads(r.read())["choices"][0]["message"]["content"]
                out = []
                for ln in r:
                    ln = ln.decode("utf-8", "replace").strip()
                    if not ln.startswith("data:") or ln == "data: [DONE]":
                        continue
                    try:
                        ch = json.loads(ln[5:])["choices"]
                    except (ValueError, KeyError):
                        continue
                    if ch and (ch[0].get("delta") or {}).get("content"):
                        out.append(ch[0]["delta"]["content"])
                return "".join(out)
        except urllib.error.HTTPError as e:   # 9Router answers 503 when its upstream is briefly unavailable
            print("LLM HTTP", e.code, e.read()[:300], flush=True)
            if attempt == 3:
                raise
            time.sleep(20 * (attempt + 1))


def _json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    return json.loads(m.group(0))


def validate(d):
    errs = []
    keys = {k["id"]: k for k in d.get("keys", [])}
    order = [k["id"] for k in d.get("keys", [])]
    for i, k in enumerate(d.get("keys", [])):
        if k.get("from") not in ("new", "K0") and k.get("from") not in order[:i]:
            errs.append(f"key {k['id']}: from={k.get('from')} is not an earlier key")
        if k.get("pose") not in cast.POSES:
            errs.append(f"key {k['id']}: unknown pose {k.get('pose')}")
        if k.get("framing") not in cast.FRAME:
            errs.append(f"key {k['id']}: unknown framing {k.get('framing')}")
        if set(k.get("chars", [])) - {"nell", "pip", "pip_baby"}:
            errs.append(f"key {k['id']}: unknown chars {k.get('chars')}")
    shots = d.get("shots", [])
    if not 11 <= len(shots) <= 15:
        errs.append(f"{len(shots)} shots (want 12-14)")
    tot = sum(float(s.get("duration_s", 0)) for s in shots)
    if not 58 <= tot <= 92:
        errs.append(f"total duration {tot:.1f} s (want 62-88)")
    words = sum(len(s.get("narration", "").split()) for s in shots)
    if not 140 <= words <= 220:
        errs.append(f"narration {words} words (want 160-205)")
    for s in shots:
        for f in ("start", "end"):
            v = s.get(f)
            if v and v != "K0" and v not in keys:
                errs.append(f"shot {s['id']}: {f} key {v} not defined")
        if s.get("mode") == "flf" and not s.get("end"):
            errs.append(f"shot {s['id']}: flf needs an end key")
        if s.get("mode") == "i2v":
            s["end"] = None
        if not 4.0 <= float(s.get("duration_s", 0)) <= 7.5:
            errs.append(f"shot {s['id']}: duration {s.get('duration_s')} outside 4.2-6.8")
    return errs


def write_shots(story_path, out_path):
    story = story_text(story_path)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "STORY:\n" + story + "\n\nWrite the shot list JSON for a 62-88 s vertical Short."}]
    log = []
    for attempt in range(3):
        t = time.time()
        txt = _chat(msgs)
        log.append({"attempt": attempt, "s": round(time.time() - t, 1), "chars": len(txt)})
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        open(out_path + f".raw{attempt}.txt", "w", encoding="utf-8").write(txt)
        print("LLM reply", log[-1], flush=True)
        try:
            d = _json(txt)
            errs = validate(d)
        except Exception as e:
            d, errs = None, [f"parse: {e}"]
        log[-1]["errors"] = errs
        if not errs:
            break
        msgs += [{"role": "assistant", "content": txt},
                 {"role": "user", "content": "Fix these problems and reply with the full corrected JSON only:\n- " + "\n- ".join(errs)}]
    if d is None or errs:
        raise RuntimeError(f"shot list failed validation: {errs}")
    d["llm"] = {"model": MODEL, "log": log, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(d, open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return d
