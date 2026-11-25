"""Edit decisions from narration word timings: where every cut lands (in a pause), dissolves, sub-shot reframes,
caption chunks. Targets (research/parabrahma/REPORT.md): 10-19 shots/min, 71-100 % of cuts in narration pauses,
hard cuts + 0.7-1.6 s dissolves, 2-4 word captions with one highlighted word, fading in over 0.2 s."""
from engine import timeline as T1

MIN_GAP = 0.15  # a "pause" is any silence between words at least this long


def place(sections, tts, cfg):
    """Lay utterances end to end. -> (scenes, words, audio, speech_end). scenes = one per story shot."""
    t = cfg.get("lead_in", 0.35)
    ps = cfg.get("pause_scale", 0.45)
    scenes, words, audio = [], [], []
    hl = set(w.lower() for w in cfg.get("highlight_words", []))
    caps = []
    for si, sec in enumerate(sections):
        for k, sh in enumerate(sec["shots"]):
            s0 = t
            sw = []
            for u in sh["utts"]:
                m = tts[u["id"]]
                audio.append({"id": u["id"], "t": round(t, 3), "dur": m["dur"]})
                ws = T1.merge_tokens(m["words"])
                for w in ws:
                    w["s"] += t
                    w["e"] += t
                sw += ws
                caps += T1.chunk_words(ws, hl, set(e.lower() for e in u.get("emph", [])))
                t += m["dur"] + max(0.12, u["pause"] * ps)
            scenes.append({"shot": sh, "section": si, "first_in_section": k == 0, "s": s0, "e": t, "words": sw})
            words += sw
    return scenes, words, audio, caps, t


def gaps(words):
    """Silences between consecutive words: [(start, end)] with length >= MIN_GAP."""
    return [(a["e"], b["s"]) for a, b in zip(words, words[1:]) if b["s"] - a["e"] >= MIN_GAP]


def cut_plan(scenes, words, cfg):
    """-> list of shots {scene, s, e, sub, trans, trans_d}. Cuts sit in the middle of a narration gap."""
    G = gaps(words)
    max_s, min_s = cfg.get("max_shot_s", 4.6), cfg.get("min_shot_s", 1.8)
    # scene boundaries: middle of the silence between the scenes' words
    bounds = [0.0]
    for a, b in zip(scenes, scenes[1:]):
        lw = a["words"][-1]["e"] if a["words"] else a["e"]
        fw = b["words"][0]["s"] if b["words"] else b["s"]
        bounds.append((lw + fw) / 2)
    end = scenes[-1]["e"] + cfg.get("tail_s", 0.5)
    bounds.append(end)
    shots = []
    for i, sc in enumerate(scenes):
        s, e = bounds[i], bounds[i + 1]
        cuts = [s]
        while e - cuts[-1] > max_s * 1.12:  # split at the pause closest to the ideal length
            ideal = cuts[-1] + min(max_s, (e - cuts[-1]) / 2 if e - cuts[-1] < 2 * max_s else max_s * 0.85)
            cands = [(abs((g0 + g1) / 2 - ideal), (g0 + g1) / 2) for g0, g1 in G
                     if cuts[-1] + min_s <= (g0 + g1) / 2 <= e - min_s]
            if not cands:  # no measurable silence inside this line: cut after a punctuated word (a spoken phrase break)
                import re as _re
                ph = [(abs(w["e"] - ideal), w["e"]) for w in words
                      if cuts[-1] + min_s <= w["e"] <= e - min_s and _re.search(r"[.,!?;:…]\"?$", w["w"])]
                if not ph:
                    break
                cands = [min(ph)]
            cuts.append(min(cands)[1])
        cuts.append(e)
        for k in range(len(cuts) - 1):
            shots.append({"scene": i, "s": round(cuts[k], 3), "e": round(cuts[k + 1], 3), "sub": k, "nsub": len(cuts) - 1})
    # transitions: dissolves at chapter changes or when asked; length from the pause around the cut
    for j, sh in enumerate(shots):
        sh["idx"] = j
        sh["trans"], sh["trans_d"] = "cut", 0.0
        if j == 0 or sh["sub"] > 0:
            continue
        sc = scenes[sh["scene"]]
        want = sc["shot"].get("v2", {}).get("trans") == "dissolve" or (sc["first_in_section"] and cfg.get("dissolve_chapters", True))
        if want:
            g = _gap_at(G, sh["s"])
            prev_bg = scenes[sh["scene"] - 1]["shot"].get("bg") if sh["scene"] > 0 else None
            same_place = prev_bg == sc["shot"].get("bg")
            # a true dissolve only within one location (time passing); between places it double-exposes two busy
            # pictures, so dip through a warm dark instead (still 0.7-1.6 s, centred in the pause)
            sh["trans"] = "dissolve" if same_place else "dip"
            sh["trans_d"] = round(min(1.6 if same_place else 1.1, max(0.7, (g or 0.5) * 1.6)), 2)
    return shots, end


def _gap_at(G, t):
    for g0, g1 in G:
        if g0 - 0.05 <= t <= g1 + 0.05:
            return g1 - g0
    return None


def cuts_in_pauses(shots, words):
    """Fraction of cuts (incl. dissolve midpoints) that fall inside a narration silence (QA)."""
    G = [(a["e"], b["s"]) for a, b in zip(words, words[1:])]
    cuts = [sh["s"] for sh in shots[1:]]
    if not cuts:
        return 1.0
    inside = sum(any(g0 - 0.02 <= c <= g1 + 0.02 and g1 - g0 >= 0.1 for g0, g1 in G) or c >= words[-1]["e"] for c in cuts)
    return inside / len(cuts)


def captions(caps):
    """Caption chunks end where the next begins unless a long pause follows (v1 rule); fade-in 0.2 s."""
    for a, b in zip(caps, caps[1:]):
        a["e"] = b["s"] if b["s"] - a["e"] < 0.7 else a["e"] + 0.45
    if caps:
        caps[-1]["e"] += 0.6
    for c in caps:
        c["fade"] = 0.2
    return caps


def word_time(scene, word, after=0.0):
    """Start time of the first word in this scene matching `word` (case/punctuation-insensitive)."""
    w0 = word.lower().strip(".,!?\"'")
    for w in scene["words"]:
        if w["s"] >= after and w["w"].lower().strip(".,!?\"'…").replace("“", "") == w0:
            return w["s"]
    return None
