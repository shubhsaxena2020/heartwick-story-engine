"""Story -> render plan + the list of GPU assets it needs (content-addressed so unchanged assets are reused)."""
import hashlib, os, re
from .style import STYLE_PREFIX, STYLE_NEGATIVE, BG_SUFFIX, BG_NEG_EXTRA, CHAR_SUFFIX, CHAR_NEG_EXTRA

ANIMAL_WORDS = ("squirrel", "cat", "dog", "bird", "fox", "rabbit", "owl", "bear", "deer", "mouse", "kitten", "puppy")


def _h(*parts):
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:10]


def build_plan(meta, sections, voice=None, speed=1.0):
    voice = voice or meta.get("voice", "af_heart")
    plan = {"slug": meta.get("slug", "story"), "title": meta.get("title"), "hook": meta.get("hook"),
            "series_tag": meta.get("series_tag_full") or meta.get("series_tag"),
            "highlight_words": meta.get("highlight_words", []), "locations": meta.get("locations", {}),
            "sections": [], "voice": voice, "speed": speed}
    for sec in sections:
        s2 = {"title": sec["title"], "shots": []}
        for sh in sec["shots"]:
            utts = [dict(u, id="u_" + _h(voice, speed, u["text"])) for u in sh["utts"]]
            chars = [dict(c, kind="animal" if _is_animal(meta.get("characters", {}).get(c["char"], {}).get("desc", "")) else "human")
                     for c in sh["chars"]]
            s2["shots"].append(dict(sh, utts=utts, chars=chars))
        plan["sections"].append(s2)
    return plan


def _is_animal(desc):
    return any(w in desc.lower() for w in ANIMAL_WORDS)


def master_item(name, ch, n=4):
    animal = _is_animal(ch["desc"])
    pose = "sitting calmly, side three-quarter view" if animal else "standing in a relaxed friendly pose, three-quarter view, arms relaxed"
    w, h = (1024, 1024) if animal else (832, 1216)
    return {"id": f"master_{name}", "kind": "master", "w": w, "h": h, "seed": ch.get("seed", 1234), "n": n, "steps": 28,
            "prompt": f"{STYLE_PREFIX}, {ch['desc']}, {pose}, {CHAR_SUFFIX}",
            "neg": f"{STYLE_NEGATIVE}, {CHAR_NEG_EXTRA}", "cutout": True,
            "ip_ref": "style_ref", "ip_mode": "style", "ip_scale": 0.5}


def asset_items(meta, plan, have, masters_have, inline_masters=False):
    """Return GPU items still missing. `have` = set of existing asset file stems, `masters_have` = set of char names."""
    items, need_masters = [], []
    locs = meta.get("locations", {})
    used_locs, used_poses = [], []
    for sec in plan["sections"]:
        for sh in sec["shots"]:
            if sh["bg"] not in used_locs:
                used_locs.append(sh["bg"])
            for c in sh["chars"]:
                if (c["char"], c["pose"]) not in used_poses:
                    used_poses.append((c["char"], c["pose"]))
    for loc in used_locs:
        L = locs.get(loc, {"prompt": loc.replace("_", " ")})
        if f"bg_{loc}" in have and f"bg_{loc}_depth" in have:
            continue
        tall = L.get("tall", False)
        w, h = (640, 1536) if tall else (768, 1344)
        items.append({"id": f"bg_{loc}", "kind": "bg", "w": w, "h": h, "seed": L.get("seed", int(_h(loc), 16) % 100000), "steps": 28,
                      "hires": 1.5, "hires_strength": 0.33,
                      "prompt": f"{STYLE_PREFIX}, {L['prompt']}, {BG_SUFFIX}",
                      "neg": f"{BG_NEG_EXTRA}, {STYLE_NEGATIVE}", "depth": True,
                      "ip_ref": None if loc == L.get("style_ref_self") else "style_ref", "ip_mode": "style", "ip_scale": L.get("style_scale", {"storm": 0.25, "night": 0.3, "dusk": 0.45}.get(L.get("tint"), 0.6))})
    chars = meta.get("characters", {})
    for name in dict.fromkeys(c for c, _ in used_poses):
        if name not in masters_have:
            need_masters.append(name)
    for name, pose in used_poses:
        if (name in need_masters and not inline_masters) or f"pose_{name}_{pose}_cut" in have:
            continue
        ch = chars[name]
        animal = _is_animal(ch["desc"])
        w, h = (1024, 1024) if animal else (832, 1216)
        items.append({"id": f"pose_{name}_{pose}", "kind": "pose", "w": w, "h": h,
                      "seed": (ch.get("pose_seeds") or {}).get(pose, ch.get("seed", 1234) + 7), "steps": 28,
                      "prompt": f"{STYLE_PREFIX}, {ch['desc']}, {ch['poses'][pose]}, {CHAR_SUFFIX}",
                      "neg": f"{STYLE_NEGATIVE}, {CHAR_NEG_EXTRA}", "cutout": True,
                      "ip_ref": f"master_{name}", "ip_scale": ch.get("ip_scale", 0.55)})
    return items, need_masters


def approx_tokens(text):
    return len(re.findall(r"\w+|[^\w\s]", text)) + 2


def check_prompts(items):
    """CLIP truncates at 77 tokens; the style prefix is first so overflow only loses trailing detail. Warn anyway."""
    for i in items:
        n = approx_tokens(i["prompt"])
        if n > 77:
            print(f"warning: prompt for {i['id']} is ~{n} tokens (>77); trailing words will be ignored")


def utterances(plan, have):
    out = []
    for sec in plan["sections"]:
        for sh in sec["shots"]:
            for u in sh["utts"]:
                if u["id"] not in have and u["id"] not in [o["id"] for o in out]:
                    out.append({"id": u["id"], "text": u["text"]})
    return out
