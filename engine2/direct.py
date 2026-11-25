"""The director: for one story shot (a scene) builds the layered scene, places and animates the actors
(auto gestures every 1-1.5 s + beats pinned to words), and gives every sub-shot its own always-moving camera."""
import numpy as np
from engine import timeline as T1
from . import characters as CH, poses as PO, scenery as SC, edit as ED
from .camera import Camera, FRAMING
from .motion import Actor, schedule

FRAMES = ["wide", "medium", "close"]
SQ_FAMILY = {p: ("curl" if p.startswith("curl") else "crouch" if p in ("crouch", "look_back") else
                 "nibble" if p == "nibble" else "sit") for p in PO.SQUIRREL}
_rig_cache = {}


def rig_for(name, painter, raster_dir=None, art="c"):
    """A raster part sheet when one exists (raster_dir/<name>/rig.json on the render kernel, or engine2/assets.py
    locally), else the vector puppet from the cast. Springs only for chain bones the rig actually has."""
    import os
    from . import assets
    key = (name, painter.name, raster_dir, art)
    if key not in _rig_cache and art == "v1":  # art source A: V1 painted rigs; squirrels as V1 sprite rigs
        from . import v1art
        from .rig import load_partsheet
        base = os.path.join(raster_dir, name) if raster_dir else None
        d = base if base and os.path.exists(os.path.join(base, "rig.json")) else v1art.find_v1_rig(name)
        if d:
            r = load_partsheet(d)
            _rig_cache[key] = (r, [s for s in CH.build(name, painter)[1] if all(b in r.skel.by for b in s["bones"])])
        elif name in ("pip", "pip_kit"):
            src = os.path.join(base, "master_cut.png") if base and os.path.exists(os.path.join(base, "master_cut.png")) else None
            r = v1art.squirrel_sprite_rig(name, src=src)
            if r is not None:
                _rig_cache[key] = (r, [{"bones": ["tail2", "tail3"], "freq": 1.1, "zeta": 0.3, "wind": 5, "hang": 0.0, "rest": 0.0, "drag": 0.06}])
        if key not in _rig_cache:
            _rig_cache[key] = CH.build(name, painter)
    if key not in _rig_cache:
        d = None
        if raster_dir:
            for c in (os.path.join(raster_dir, name), os.path.join(raster_dir, name, "rig")):
                if os.path.exists(os.path.join(c, "rig.json")):
                    d = c
        elif raster_dir is None:
            d = assets.find_rig(name)
        if d:
            from .rig import load_partsheet
            r = load_partsheet(d)
            _, springs = CH.build(name, painter)
            springs = [s for s in springs if all(b in r.skel.by for b in s["bones"])]
            _rig_cache[key] = (r, springs)
        else:
            _rig_cache[key] = CH.build(name, painter)
    return _rig_cache[key]


TUNE = {"every_min": 0.7, "every_max": 1.1, "trans_min": 0.16, "trans_max": 0.3, "overshoot": 1.1, "overlay_p": 0.6, "char_scale": 1.0,
        "handheld": 4.0, "pan": 1.0, "cadence": 1.0, "key": 1.0, "rim": 1.0, "bloom": 1.0, "grain": 0.07}


def tune(cfg):
    t = dict(TUNE)
    t.update(cfg.get("tune") or {})
    return t


def setup(scene_info, shots, painter, cfg, seed=0):
    T = tune(cfg)
    sh = scene_info["shot"]
    v2 = sh.get("v2", {})
    s0, s1 = shots[0]["s"], shots[-1]["e"]
    pre, post = 1.8, 1.8  # dissolves overlap neighbouring scenes
    scene = SC.build_location(sh["bg"], painter, fx_override=sh.get("fx"), seed=seed + 1, scenes_root=cfg.get("scene_dir"),
                              art=cfg.get("art", "c"), plate_dir=cfg.get("plate_dir"))
    Ww, Wh = scene.world
    rng = np.random.default_rng(seed + 7)
    actors = []
    for k, ch in enumerate(sh["chars"]):
        rig, springs = rig_for(ch["char"], painter, cfg.get("raster_dir"), cfg.get("art", "c"))
        squirrel = ch["kind"] == "squirrel"
        x = Ww / 2 + (ch["x"] - 0.5) * 1080
        if ch["y"] >= 0.86:  # on the floor (a little depth variation)
            ground = scene.ground + (ch["y"] - 0.95) * 300
        elif ch["y"] < 0.78 and "sill" in scene.spots:  # up on the window sill
            x, ground = scene.spots["sill"][0] + (ch["x"] - 0.5) * 300, scene.spots["sill"][1]
        elif "table" in scene.spots:  # on the table top
            x, ground = scene.spots["table"][0] + (ch["x"] - 0.5) * 400, scene.spots["table"][1]
        else:
            ground = scene.ground - (0.95 - ch["y"]) * 1920
        h = ch["h"] * 1920 * (1.0 if not squirrel else 1.25 if ch["char"] != "pip_kit" else 1.2) * T["char_scale"]
        others = [c for c in sh["chars"] if c is not ch]
        facing = 1
        if others:
            facing = 1 if others[0]["x"] > ch["x"] else -1
        elif ch["x"] > 0.58:
            facing = -1
        base = ch["base"]
        pins = []
        after = s0 - 1
        for pose, word in v2.get("do", {}).get(ch["char"], []):
            wt = ED.word_time(scene_info, word, after)
            if wt is not None:
                pins.append((wt - 0.18, pose))
                after = wt
        if squirrel:
            keys = schedule(s0 - pre, s1 + post, base, rng, vocab=PO.SQ_VOCAB, family_of=SQ_FAMILY, pinned=pins,
                            every=(1.0, 1.5) if not ch["flags"].get("tremble") else (1.2, 1.6))
        else:
            keys = schedule(s0 - pre, s1 + post, base, rng, pinned=pins, every=(T["every_min"], T["every_max"]), trans=(T["trans_min"], T["trans_max"]))
        overlays = []
        for (t, p, tr) in keys[1:]:
            if rng.random() < T["overlay_p"] and not squirrel:  # a nod or sigh inside most holds: nobody freezes
                overlays.append((t + tr + rng.uniform(0.15, 0.4), "nod" if rng.random() < 0.6 else "sigh"))
        if ch["flags"].get("wave"):
            overlays += [(t, "wave_hand") for t in np.arange(s0, s1, 1.3)]
        walks, rises = [], []
        if ch["char"] in v2.get("walk", {}):
            a, b, ta, tb = v2["walk"][ch["char"]]
            walks.append((s0 + ta, s0 + tb, Ww / 2 + (a - 0.5) * 1080, Ww / 2 + (b - 0.5) * 1080))
            x = Ww / 2 + (a - 0.5) * 1080
        if ch["flags"].get("run"):
            walks.append((s0 + 0.2, s1 - 0.2, x - facing * 220, x + facing * 260))
            x -= facing * 220
        if ch["flags"].get("climb"):
            rises.append((s0 + 0.3, s1, -900))
        turns = []
        if ch["char"] in v2.get("turn", {}):
            wt = ED.word_time(scene_info, v2["turn"][ch["char"]])
            if wt is not None:
                turns.append((wt, -facing))
        if squirrel:  # always alive: nose twitches, ear and tail flicks
            for tt in np.arange(s0 - pre + rng.uniform(0.2, 0.8), s1 + post, rng.uniform(0.9, 1.4)):
                overlays.append((float(tt), ["twitch", "tail_flick", "ear_flick"][int(rng.integers(3))]))
        gait = {"nell": "elder", "eli": "child"}.get(ch["char"], "walk" if not squirrel else "bound")
        if ch["flags"].get("run") and not squirrel:
            gait = "run"
        a = Actor(rig, painter, x, ground, h, keys, facing=facing, walks=walks, overlays=overlays, turns=turns,
                  springs=springs, seed=seed * 13 + k, kind=ch["kind"], props=ch["props"], tremble=ch["flags"].get("tremble", 0.0),
                  rises=rises, gait=gait)
        a.cadence_k = T["cadence"]
        a.track.c1 = T["overshoot"]
        actors.append(a)
    for a in actors:  # two-character exchanges: the other one reacts (nod / glance) just after a gesture
        for b in actors:
            if a is b or b.kind != "humanoid":
                continue
            for (t, p, tr) in a.track.keys[1:]:
                if isinstance(p, str) and p in ("point", "reach", "talk", "arms_open", "wave", "heart", "kneel_reach", "hold_up"):
                    b.overlays.append((t + tr + 0.3, "nod" if rng.random() < 0.5 else "glance"))
    for a in actors:  # bake once, after every reaction is known (springs need the whole history)
        a.bake(s0 - pre, s1 + post, cfg.get("fps", 30))
    cams = {}
    main = next((a for a in actors if a.kind == "humanoid"), actors[0] if actors else None)
    walker = next((a for a in actors if a.walks), None)
    shakes = [float(k * 4.7) for k in range(int((s1 + post) / 4.7) + 2)] if scene.mood == "storm" else []
    cyc = T1.CAM_CYCLE.get(sh.get("cam") or "push_in", [sh.get("cam") or "push_in"])
    f0 = FRAMES.index(v2.get("frame", "wide")) if v2.get("frame", "wide") in FRAMES else 0
    for j, shot in enumerate(shots):
        kind = cyc[shot["sub"] % len(cyc)]
        framing = FRAMES[(f0 + shot["sub"]) % 3] if shot["nsub"] > 1 else FRAMES[f0]
        if kind.startswith("tilt") and shot["sub"] == 0:
            framing = "wide"
        z = FRAMING[framing]
        tm = (shot["s"] + shot["e"]) / 2
        if main is not None:
            hx, hy = main.head_pos(tm)
            if main.kind != "humanoid":  # a small animal alone: keep it near the middle, a little low
                focus = (hx, hy - 120 / z) if framing != "wide" else (0.5 * Ww / 2 + 0.5 * hx, min(scene.ground - 760, hy + 200))
            elif framing == "wide":
                focus = (0.5 * Ww / 2 + 0.5 * hx, scene.ground - 760)
            else:
                focus = (hx + main.facing(tm) * 120, hy + 520 / z)
        else:
            focus = (Ww / 2, scene.ground - 760)
        focus = (focus[0], focus[1] + (rng.uniform(-40, 40) if framing == "wide" else 0))
        hh = 960 / z
        tops = [a.head_pos(tm)[1] - a.height * 0.11 for a in actors if a.kind == "humanoid"]
        if tops:  # never crop a head: keep the highest head top inside the upper part of the frame
            fy = min(focus[1], min(tops) + hh * 0.78)
            xs = [a.head_pos(tm)[0] for a in actors if a.kind == "humanoid"]
            fx = focus[0] if max(xs) - min(xs) < 1080 / z * 0.7 else (max(xs) + min(xs)) / 2
            focus = (fx, fy)
        tilt_from = None
        if kind == "tilt_down":
            tilt_from = 960.0
        elif kind == "tilt_up":
            tilt_from = scene.ground - 300.0
            focus = (focus[0], 960.0 if scene.tall else focus[1] - 500)
        track = None
        if walker is not None and any(a0 < shot["e"] and a0 + 3.5 > shot["s"] for (a0, _, _, _) in walker.walks):
            kind = "follow"  # the camera moves with the action
            track = (lambda w: (lambda tt: (w._loco_at(tt)[0], w.facing(tt))))(walker)
        cams[shot["idx"]] = Camera(kind, shot["s"], shot["e"], scene.world, focus, z, rng=rng, tilt_from=tilt_from, track=track,
                                   shakes=shakes, seed=shot["idx"], handheld=T["handheld"])
        cams[shot["idx"]].pan *= T["pan"]
    return {"scene": scene, "actors": actors, "cams": cams}
