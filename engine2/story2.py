"""Story files for engine v2. Same Markdown format as v1 (engine/story.py, stories/pip.md) plus optional
shot directives that only v2 reads:

  do=nell:kneel@knelt,kneel_reach@little;pip_kit:curl_lift@mama   pose beats pinned to a word (first match)
  walk=nell:0.15>0.4@0-2.4        walk between frame-x fractions, seconds from the shot start (default: 0-2 s)
  turn=nell@whispered             puppet turn (flip) at a word
  trans=dissolve                  dissolve into this shot (default: hard cut, dissolves at chapter changes)
  frame=wide|medium|close         first framing (later sub-shots cycle)
  tilt=1                          4 s tilt reveal from the top of a tall scene
  hero=<beat>                     the first sub-shot plays the anime hero clip assets/hero/<slug>/<beat>_<model>.mp4
  music=<mood>                    music cue for this shot (tender, storm, wonder, resolution; default from the scene mood)
  sfx=wind,rain,creak             sparse sound effects under this shot

v1 pose names (made for AI stills) map to a v2 pose family + held prop (POSE_ALIAS), so v1 stories render as-is.
"""
import re
from engine import story as S1

POSE_ALIAS = {  # v1 pose -> (v2 base pose, {prop slot: prop}, actor flags)
    "stand_mug": ("hold", {"prop_n": "mug"}, {}), "kneel": ("kneel", {}, {}), "hold_bundle": ("hold", {"prop_n": "bundle"}, {}),
    "phone": ("phone", {"prop_n": "phone"}, {}), "sit_feed": ("sit_hold", {}, {}), "sit_read": ("sit_read", {"prop_n": "book"}, {}),
    "water": ("reach", {"prop_n": "can"}, {}), "smile": ("stand", {}, {}), "sit_sad": ("sit_sad", {}, {}),
    "wave": ("wave", {}, {"wave": True}), "couch": ("sit", {}, {}), "sit_smile": ("sit_hold", {"prop_n": "mug"}, {}),
    "shiver": ("curl", {}, {"tremble": 1.0}), "bundle": ("curl", {}, {}),
    "sit": ("sit", {}, {}), "walnut": ("nibble", {"prop_paw": "walnut"}, {}), "run": ("crouch", {}, {"run": True}),
    "look_back": ("look_back", {}, {}), "climb": ("alert", {}, {"climb": True}), "watch": ("sit_look", {}, {}),
    "stand": ("stand", {}, {}), "talk": ("talk", {}, {}), "curious": ("reach_down", {}, {}), "grin": ("hands_clasped", {}, {}),
}
ANIM_FLAGS = {"tremble": {"tremble": 1.0}, "hop": {"run": True}, "climb": {"climb": True}, "idle": {}}
SQUIRRELS = ("pip", "pip_kit")


def _kv(line):
    out = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def parse(path):
    meta, sections = S1.parse_story(path)
    extra = [_kv(l.strip()[len("::: shot"):]) for l in open(path, encoding="utf-8").read().split("\n") if l.strip().startswith("::: shot")]
    shots = [sh for sec in sections for sh in sec["shots"]]
    if len(extra) == len(shots):
        for sh, ex in zip(shots, extra):
            sh["v2"] = _v2(ex)
    for sh in shots:
        sh.setdefault("v2", _v2({}))
        for ch in sh["chars"]:
            base, props, flags = POSE_ALIAS.get(ch["pose"], (ch["pose"], {}, {}))
            ch["base"] = base
            ch["props"] = props
            ch["flags"] = dict(flags, **ANIM_FLAGS.get(ch.get("anim", "idle"), {}))
            ch["kind"] = "squirrel" if ch["char"] in SQUIRRELS else "humanoid"
    return meta, sections


def _v2(ex):
    v = {"do": {}, "walk": {}, "turn": {}, "trans": ex.get("trans", "cut"), "frame": ex.get("frame", "wide"), "tilt": ex.get("tilt") == "1",
         "hero": ex.get("hero"), "music": ex.get("music"), "sfx": [s for s in ex.get("sfx", "").split(",") if s]}
    for part in filter(None, ex.get("do", "").split(";")):
        name, beats = part.split(":", 1)
        v["do"][name] = [tuple(b.split("@", 1)) for b in beats.split(",") if "@" in b]
    for part in filter(None, ex.get("walk", "").split(";")):
        m = re.match(r"(\w+):(-?[\d.]+)>(-?[\d.]+)(?:@([\d.]+)-([\d.]+))?", part)
        if m:
            v["walk"][m[1]] = (float(m[2]), float(m[3]), float(m[4] or 0), float(m[5] or 2.0))
    for part in filter(None, ex.get("turn", "").split(";")):
        name, word = part.split("@", 1)
        v["turn"][name] = word
    return v
