"""Turn plan + TTS word timings into a frame-accurate timeline: shots, sub-shots, caption chunks, chapters.
Runs both on Kaggle (inside the worker) and locally."""
import json, re

STOP = set("a an the and or but of to in on at for with from by as is was were be been it its he she his her him they them "
           "this that there then than so very just not no yes up out into over under one two who what where when how i you we "
           "had has have did do does said would could should will can".split())


def _clean(w):
    return re.sub(r"[^\w']", "", w).lower()


def merge_tokens(words):
    """Kokoro tokens -> display words (punctuation glued to the previous word, quotes kept)."""
    out = []
    for t in words:
        txt = t["w"]
        if not txt:
            continue
        if re.fullmatch(r"[^\w]+", txt) and out:
            opening = txt == "“" or (txt == '"' and out[-1].get("ws"))  # a quote after a space opens a quotation
            if opening:
                out.append({"w": txt, "s": t["s"], "e": t["e"], "ws": ""})
            else:
                out[-1]["w"] += txt
                out[-1]["e"] = max(out[-1]["e"], t["e"])
                out[-1]["ws"] = t.get("ws", " ")
            continue
        if out and out[-1]["w"] in ("“", '"') and not out[-1].get("ws"):
            out[-1] = {"w": out[-1]["w"] + txt, "s": out[-1]["s"], "e": t["e"], "ws": t.get("ws", " ")}
            continue
        out.append({"w": txt, "s": t["s"], "e": t["e"], "ws": t.get("ws", " ")})
    for o in out:  # straight quotes look cleaner in the caption font
        o["w"] = o["w"].replace("“", '"').replace("”", '"').replace("’", "'")
    return out


def chunk_words(words, hl_set, emph, max_words=4, max_chars=20):
    """Phrase-based 2-4 word caption chunks: split at punctuation, then balance each phrase."""
    import math as _m
    phrases, cur = [], []
    for w in words:
        cur.append(w)
        if re.search(r"[.,!?;:…]\"?$", w["w"]):
            phrases.append(cur); cur = []
    if cur:
        phrases.append(cur)
    merged = []  # glue 1-word phrases to a neighbour
    for ph in phrases:
        if merged and (len(ph) == 1 or len(merged[-1]) == 1) and len(merged[-1]) + len(ph) <= max_words:
            merged[-1] = merged[-1] + ph
        else:
            merged.append(ph)
    chunks = []
    for ph in merged:
        n = len(ph)
        k = max(1, _m.ceil(n / max_words))
        while True:
            size = _m.ceil(n / k)
            parts = [ph[i:i + size] for i in range(0, n, size)]
            if all(len(" ".join(x["w"] for x in p)) <= max_chars for p in parts) or size <= 2:
                break
            k += 1
        if len(parts) > 1 and len(parts[-1]) == 1:
            last = parts.pop(); parts[-1] = parts[-1] + last
        chunks += parts
    res = []
    for c in chunks:
        cl = [_clean(x["w"]) for x in c]
        hl = -1
        for j, w in enumerate(cl):
            if w in emph:
                hl = j
        if hl < 0:
            for j, w in enumerate(cl):
                if w in hl_set or w.rstrip("'s") in hl_set:
                    hl = j; break
        if hl < 0:
            cands = [(len(w), j) for j, w in enumerate(cl) if w not in STOP and len(w) >= 6]
            if cands:
                hl = max(cands)[1]
        res.append({"words": [x["w"] for x in c], "hl": hl, "s": c[0]["s"], "e": c[-1]["e"]})
    return res


CAM_CYCLE = {"push_in": ["push_in", "pan_right", "pull_back"], "pull_back": ["pull_back", "pan_left", "push_in"],
             "pan_right": ["pan_right", "push_in", "pan_left"], "pan_left": ["pan_left", "push_in", "pan_right"],
             "drift": ["drift", "push_in", "pan_left"], "tilt_down": ["tilt_down", "push_in", "pan_right"],
             "tilt_up": ["tilt_up", "pull_back", "pan_left"]}


def build(plan, tts_dir, cfg):
    tts = {}
    for sec in plan["sections"]:
        for sh in sec["shots"]:
            for u in sh["utts"]:
                tts[u["id"]] = json.load(open(f"{tts_dir}/{u['id']}.json"))
    fps = cfg.get("fps", 30)
    pause_scale = cfg.get("pause_scale", 0.5)
    lead = cfg.get("lead_in", 0.35)
    hl_set = set(w.lower() for w in plan.get("highlight_words", []))
    t = lead
    audio, captions, shots, chapters = [], [], [], []
    for sec in plan["sections"]:
        chapters.append({"title": sec["title"], "t": round(t if shots else 0.0, 2)})
        for sh in sec["shots"]:
            st = t
            for u in sh["utts"]:
                m = tts[u["id"]]
                audio.append({"id": u["id"], "t": round(t, 3), "dur": m["dur"]})
                words = merge_tokens(m["words"])
                for w in words:
                    w["s"] += t; w["e"] += t
                captions += chunk_words(words, hl_set, set(e.lower() for e in u.get("emph", [])))
                t += m["dur"] + max(0.12, u["pause"] * pause_scale)
            shots.append({**{k: sh[k] for k in ("bg", "cam", "fx", "chars", "id", "focus") if k in sh}, "s": st, "e": t})
    speech_end = t
    # shots are contiguous: each lasts until the next starts; first starts at 0
    shots[0]["s"] = 0.0
    for a, b in zip(shots, shots[1:]):
        a["e"] = b["s"]
    end_card = cfg.get("end_card_s", 3.6)
    tail = cfg.get("tail_s", 0.5)
    shots[-1]["e"] = speech_end + tail
    total = speech_end + tail + end_card
    # caption chunk ends: continuous until next chunk unless a long pause follows
    for a, b in zip(captions, captions[1:]):
        a["e"] = b["s"] if b["s"] - a["e"] < 0.7 else a["e"] + 0.45
    if captions:
        captions[-1]["e"] += 0.6
    # sub-shot split for pace (same layered scene, new camera framing)
    max_len = cfg.get("max_shot_s", 5.0)
    out = []
    for i, sh in enumerate(shots):
        d = sh["e"] - sh["s"]
        n = max(1, int(round(d / max_len + 0.15))) if d > max_len * 1.15 else 1
        cyc = CAM_CYCLE.get(sh["cam"], [sh["cam"]])
        for k in range(n):
            s = sh["s"] + d * k / n
            e = sh["s"] + d * (k + 1) / n
            sub = dict(sh, s=round(s, 3), e=round(e, 3), cam=cyc[k % len(cyc)], sub=k, nsub=n,
                       framing=(["wide", "medium", "close"][k % 3] if n > 1 else "wide"), key=f"{i}")
            out.append(sub)
    for j, sh in enumerate(out):  # transition into this shot
        if j == 0:
            sh["trans_in"] = 0.0
        elif out[j - 1]["key"] == sh["key"]:
            sh["trans_in"] = cfg.get("dissolve_inner", 0.3)
        else:
            sh["trans_in"] = cfg.get("dissolve", 0.6)
    return {"fps": fps, "W": cfg.get("W", 1080), "H": cfg.get("H", 1920), "total": round(total, 3),
            "speech_end": round(speech_end, 3), "audio": audio, "captions": captions, "shots": out,
            "chapters": chapters, "hook": plan.get("hook"), "series_tag": plan.get("series_tag"),
            "cta": cfg.get("cta", "Follow for the next story"), "locations": plan["locations"],
            "end_card_s": end_card}
