"""Animation: pose tracks, the gesture scheduler, procedural life (breathing, sway, blinks) and spring chains.

An Actor is baked once per scene into per-frame arrays (bake()), so every render worker draws identical frames
and spring simulation stays deterministic. Timing targets from the reference (research/parabrahma/REPORT.md):
a pose change every 1-1.5 s, each eased over 0.2-0.4 s, then held 0.6-1.5 s.
"""
import math
import numpy as np
import skia
from . import poses as PO
from .mathx import Noise1D, ease_io, ease_back_io, smooth, spring_follow
from .rig import mcopy


class PoseTrack:
    """Keys (t, pose, transition_s). Each key eases from wherever the previous motion was at its start time."""

    def __init__(self, library, keys):
        self.lib = library
        self.keys = sorted(keys, key=lambda k: k[0])
        self._from = []
        prev = self._pose(self.keys[0][1])
        for i, (t, p, tr) in enumerate(self.keys):
            self._from.append(prev if i else self._pose(p))
            prev = self._at_key(i, self.keys[i + 1][0]) if i + 1 < len(self.keys) else None

    def _pose(self, p):
        return dict(self.lib[p]) if isinstance(p, str) else dict(p)

    def _at_key(self, i, t):
        t0, p, tr = self.keys[i]
        u = 1.0 if tr <= 0 else (t - t0) / tr
        if u >= 1:
            return self._pose(p)
        return PO.blend(self._from[i], self._pose(p), ease_back_io(u, getattr(self, "c1", 1.1)) if u < 1 else 1.0)

    def __call__(self, t):
        i = 0
        for j, k in enumerate(self.keys):
            if k[0] <= t:
                i = j
        if t < self.keys[0][0]:
            return self._pose(self.keys[0][1])
        return self._at_key(i, t)

    def names(self):
        return [k[1] for k in self.keys]


def schedule(t0, t1, base, rng, vocab=None, family_of=None, pinned=(), every=(0.7, 1.1), trans=(0.16, 0.3), lead=(0.15, 0.45)):
    """Auto gesture keys between t0 and t1: a new pose every 1-1.5 s drawn from the current pose family.
    pinned: [(t, pose)] beats tied to words; they replace the nearest automatic change."""
    vocab = vocab or PO.VOCAB
    family_of = family_of or PO.FAMILY
    keys = [(t0, base, 0.0)]
    cur = base
    pins = sorted(pinned)
    t = t0 + rng.uniform(*lead)
    recent = [base]
    while t < t1 - 0.35:
        if pins and pins[0][0] <= t + every[1]:
            pt, pose = pins.pop(0)
            t = max(pt, keys[-1][0] + 0.45)
            tr = rng.uniform(0.28, 0.4)
        else:
            fam = family_of.get(cur, cur)
            opts = [p for p in vocab.get(fam, [fam]) if p not in recent[-2:]] or vocab.get(fam, [fam])
            pose = opts[int(rng.integers(len(opts)))]
            tr = rng.uniform(*trans)
        if t >= t1 - 0.35:
            break
        keys.append((t, pose, tr))
        cur = pose
        recent.append(pose)
        t += rng.uniform(*every)
    for pt, pose in pins:  # pins past the last auto key
        if pt < t1:
            keys.append((max(pt, keys[-1][0] + 0.45), pose, 0.32))
    return expand_transitions(keys, family_of)


STEPS = {  # multi-step body transitions between pose families: (intermediate pose, start offset s, duration s)
    ("stand", "kneel"): [("reach_down", 0.0, 0.2), ("half_kneel", 0.22, 0.22)],
    ("kneel", "stand"): [("half_kneel", 0.0, 0.24), ("reach_down", 0.26, 0.2)],
    ("stand", "sit"): [("sit_prep", 0.0, 0.26)],
    ("sit", "stand"): [("sit_prep", 0.0, 0.24)],
    ("hold", "kneel"): [("reach_down", 0.0, 0.2), ("half_kneel", 0.22, 0.22)],
    ("kneel", "hold"): [("half_kneel", 0.0, 0.24), ("reach_down", 0.26, 0.2)],
}


def expand_transitions(keys, family_of=None):
    """Kneeling, sitting and standing up happen in steps (bend, one knee, both knees), never as one morph."""
    family_of = family_of or PO.FAMILY
    out = [keys[0]]
    for (t, p, tr) in keys[1:]:
        a = family_of.get(out[-1][1], out[-1][1]) if isinstance(out[-1][1], str) else None
        b = family_of.get(p, p) if isinstance(p, str) else None
        steps = STEPS.get((a, b))
        if steps and isinstance(p, str):  # the steps lead INTO the key, so a pinned beat still lands on its word
            total = steps[-1][1] + steps[-1][2] + 0.02
            start = max(out[-1][0] + 0.12, t - total)
            k = (t - start) / total
            for (sp, off, d) in steps:
                out.append((start + off * k, sp, d * k))
        out.append((t, p, tr))
    return sorted(out, key=lambda k: k[0])


def overlay_value(kind, u, freq=None, dur=1.0):
    """Envelope for short overlay gestures, u in 0..1 over the overlay."""
    if kind == "double":  # nod: down-up-down-up, decaying
        return math.sin(2 * math.pi * u) ** 2 * (1 - 0.4 * u) * (1 if u < 1 else 0)
    if kind == "bump":
        return math.sin(math.pi * u)
    if kind == "osc":
        return math.sin(2 * math.pi * (freq or 2) * u * dur) * math.sin(math.pi * u)
    return 0.0


class Actor:
    """A placed, animated rig inside a scene.

    x, ground: world px of the feet; height: world px of the rig's full height; facing: +1 right / -1 left.
    keys: PoseTrack keys; walks: [(t0, t1, x0, x1)]; overlays: [(t, name)]; turns: [(t, new_facing)].
    springs: [{"bones": [...], "freq": Hz, "zeta": d, "wind": deg, "drag": deg per px/s}] (gravity-hanging chains).
    """

    def __init__(self, rig, painter, x, ground, height, keys, facing=1, walks=(), overlays=(), turns=(),
                 springs=(), seed=0, kind="humanoid", props=None, tremble=0.0, flip_turn_s=0.2, rises=(), gait="walk"):
        self.rig, self.P, self.x0, self.ground, self.height = rig, painter, x, ground, height
        self.kind = kind
        self.lib = PO.HUMAN if kind == "humanoid" else PO.SQUIRREL
        self.track = PoseTrack(self.lib, keys)
        self.facing0, self.walks, self.overlays, self.turns = facing, list(walks), list(overlays), sorted(turns)
        self.springs, self.seed, self.props, self.tremble = list(springs), seed, props or {}, tremble
        self.flip_turn_s = flip_turn_s
        self.rises = list(rises)  # [(t0, t1, dy_px)]: climbing / lifting, eased
        self.gait = gait  # walk | elder | child | run (humans), bound (squirrels)
        self.loco = None  # baked locomotion: x, phase, ramp, lift per frame
        for (a, b, x0, x1) in self.walks:  # face the way you are about to walk (a quick puppet turn just before)
            want = 1 if x1 >= x0 else -1
            cur = self.facing(a - 0.3)
            if (cur > 0) != (want > 0):
                self.turns.append((a - 0.24, want))
        self.turns.sort()
        self.noise = [Noise1D(seed * 17 + k) for k in range(8)]
        rng = np.random.default_rng(seed + 991)
        self.blinks = []
        t = rng.uniform(0.6, 2.0)
        while t < 600:
            self.blinks.append(t)
            t += rng.uniform(2.2, 4.8)
        self.baked = None

    # ---------------------------------------------------------------- per-time evaluation (before springs)
    def _x_and_walk(self, t):
        x = self.x0
        walk = None
        for (a, b, x0, x1) in self.walks:
            if t >= b:
                x = x1
            elif t >= a:
                u = (t - a) / max(1e-6, b - a)
                x = x0 + (x1 - x0) * u
                walk = (a, b, x0, x1, u)
                break
            else:
                break
        return x, walk

    def facing(self, t):
        f = self.facing0
        for (tt, nf) in self.turns:
            if t >= tt + self.flip_turn_s:
                f = nf
            elif t >= tt:
                u = (t - tt) / self.flip_turn_s
                return f * math.cos(math.pi * u) if f != nf else f  # squash through zero = puppet turn
        return f

    def to_bones(self, sem):
        """Semantic pose -> bone angles; rigs painted in a pose (sprite rigs) subtract their neutral pose."""
        a = PO.to_bones(sem) if self.kind == "humanoid" else PO.sq_to_bones(sem)
        neu = self.rig.meta.get("neutral")
        if neu:
            a = {k: v - neu.get(k, 0.0) for k, v in a.items()}
        return a

    def _loco_at(self, t):
        """(x, phase, ramp, lift) from the baked locomotion, or a linear estimate before baking."""
        L = self.loco
        if L is not None:
            i = int(min(len(L["x"]) - 1, max(0, round((t - L["t0"]) * L["fps"]))))
            return L["x"][i], L["phase"][i], L["ramp"][i], L["lift"][i]
        x, walk = self._x_and_walk(t)
        return x, 0.0, 0.0, 0.0

    def semantic(self, t, loco=None):
        sem = self.track(t)
        x, phase, ramp, lift = loco if loco is not None else self._loco_at(t)
        if ramp > 0 and self.kind == "humanoid":
            w = PO.walk(phase, run=self.gait == "run")
            sem = PO.blend(sem, dict(w, **{k: v for k, v in sem.items() if k not in w and k not in PO.SIGN}), ramp)
        elif ramp > 0:
            sem = PO.blend(sem, PO.run_cycle(phase), ramp)
        # overlays (nod, tremble, wave_hand...)
        for (to, name) in self.overlays:
            O = PO.OVERLAYS[name]
            if to <= t < to + O["dur"]:
                u = (t - to) / O["dur"]
                e = overlay_value(O["shape"], u, O.get("freq"), O["dur"])
                for j, v in O["joints"].items():
                    sem[j] = sem.get(j, 0.0) + v * e
        n = self.noise
        if self.kind == "humanoid":  # breathing + weight sway + head micro-motion
            br = math.sin(2 * math.pi * 0.27 * t + self.seed)
            sem["chest"] = sem.get("chest", 0) + 0.9 * br
            sem["sh_n"] = sem.get("sh_n", 0) + 1.3 * br
            sem["sh_f"] = sem.get("sh_f", 0) + 1.1 * br
            sem["spine"] = sem.get("spine", 0) + 1.4 * n[0].fbm(t * 0.17)
            sem["head"] = sem.get("head", 0) + 2.2 * n[1].fbm(t * 0.33) - 0.5 * br
            sem["neck"] = sem.get("neck", 0) + 1.0 * n[2](t * 0.25)
            sem["el_n"] = sem.get("el_n", 0) + 2.0 * n[3](t * 0.4)
            sem["el_f"] = sem.get("el_f", 0) + 1.5 * n[4](t * 0.37)
        else:
            br = math.sin(2 * math.pi * 0.9 * t + self.seed)
            sem["body"] = sem.get("body", 0) + 1.2 * br
            sem["head"] = sem.get("head", 0) + 3.0 * n[1].fbm(t * 0.6)
            sem["tail1"] = sem.get("tail1", 0) + 6 * n[2].fbm(t * 0.5)
        if self.tremble:
            k = self.tremble
            j = "body" if self.kind != "humanoid" else "spine"
            sem[j] = sem.get(j, 0) + k * 2.2 * math.sin(2 * math.pi * 11 * t)
            sem["head"] = sem.get("head", 0) + k * 2.6 * math.sin(2 * math.pi * 9.3 * t + 1)
        sem["_x"] = x
        sem["_breath"] = br
        return sem

    def face(self, t):
        blink = 1.0
        for tb in self.blinks:
            if tb <= t <= tb + 0.15:
                blink = abs(math.cos(math.pi * (t - tb) / 0.15))
                break
            if tb > t:
                break
        look = 0.6 * self.noise[5](t * 0.5)
        return {"blink": blink, "look": look}

    # ---------------------------------------------------------------- bake (springs need the whole history)
    def _integrate(self, ts, fps):
        """Locomotion, frame by frame. Humans: the gait phase advances at a natural cadence and the body travels by
        LOCKING the stance foot to the ground (the lowest foot contact keeps its world x), so steps match ground speed
        and nothing slides. Squirrels: bounds with a flight arc. Returns per-frame semantic poses."""
        n = len(ts)
        L = {"t0": ts[0], "fps": fps, "x": np.full(n, float(self.x0)), "phase": np.zeros(n), "ramp": np.zeros(n), "lift": np.zeros(n),
             "stance": [None] * n, "foot_x": np.full(n, np.nan)}
        self.loco = L
        x = float(self.x0)
        ph = 0.0
        lock = None
        cad = (PO.WALK_CADENCE.get(self.gait, 0.92) if self.kind == "humanoid" else 2.2) * getattr(self, "cadence_k", 1.0)
        to_b = self.to_bones
        feet = [(k, b, p) for k, (b, p) in enumerate(self.rig.contacts) if b.startswith(("foot", "leg", "paw"))]
        segs = sorted(self.walks)
        active, done_at, finished = None, None, set()
        sems = []
        for i, t in enumerate(ts):
            if active is None:
                for si, sg in enumerate(segs):
                    if si not in finished and sg[0] <= t:
                        active, done_at, act_i = sg, None, si
                        break
            ramp = 0.0
            if active is not None:
                a, b, x0, x1 = active
                ramp = smooth(min(1.0, (t - a) / 0.25))
                if done_at is not None:
                    ramp *= 1 - smooth((t - done_at) / 0.3)
                    if t - done_at > 0.3:
                        active, ramp = None, 0.0
                        finished.add(act_i)
                ph += cad / fps * (1.0 if ramp > 0 else 0.0)
            lift = 0.0
            if self.kind != "humanoid" and ramp > 0:  # bound: constant speed while airborne-ish, arc up
                d = 1 if active[3] >= active[2] else -1
                x += d * self.height * 0.95 * cad / fps * ramp
                lift = PO.bound_lift(ph % 1.0) * self.height * 0.32 * ramp
            L["phase"][i], L["ramp"][i], L["lift"][i] = ph % 1.0, ramp, lift
            sem = self.semantic(t, loco=(x, ph % 1.0, ramp, lift))
            if self.kind == "humanoid" and ramp > 0 and feet:
                angs = to_b(sem)
                sem["_x"] = x
                root = self.root_matrix(t, angs, sem)
                w = self.rig.skel.world(angs, root)
                stance_bone = "foot_n" if (ph % 1.0) < 0.5 else "foot_f"  # the stance foot follows the gait phase
                pts = [(k, w[b].mapXY(*p)) for k, b, p in feet if b == stance_bone] or [(k, w[b].mapXY(*p)) for k, b, p in feet]
                k_low, p_low = max(pts, key=lambda q: q[1].y())  # its heel, then its toe as it rolls over
                if lock is not None and lock[0] == k_low:
                    x += lock[1] - p_low.x()  # keep the planted point where it was
                    lock = (k_low, lock[1])
                else:
                    lock = (k_low, p_low.x())
                L["stance"][i], L["foot_x"][i] = k_low, lock[1]
            else:
                lock = None
            if active is not None and done_at is None and direction_reached(x, active):
                done_at = t
            L["x"][i] = x
            sem["_x"] = x
            sems.append(sem)
        return sems

    def max_still(self, thr_deg_s=10.0):
        """Longest stretch (s) in which no main joint moves faster than thr_deg_s (breathing alone does not count)."""
        b = self.baked
        if b is None or len(b["angs"]) < 2:
            return None
        fps = b["fps"]
        sec = ("scarf", "hair", "tail2", "tail3")
        run = best = 0
        for a0, a1 in zip(b["angs"], b["angs"][1:]):
            v = max([abs(a1.get(k, 0) - a0.get(k, 0)) for k in a1 if not k.startswith(sec)] or [0]) * fps
            run = run + 1 if v < thr_deg_s else 0
            best = max(best, run)
        return best / fps

    def foot_slide(self):
        """Mean |dx| per frame of the planted (lowest) foot contact while walking, in px (QA; should be ~0)."""
        L = self.loco
        if L is None:
            return None
        to_b = self.to_bones
        feet = [(k, b, p) for k, (b, p) in enumerate(self.rig.contacts) if b.startswith(("foot", "leg", "paw"))]
        prev, d = None, []
        for i in range(len(L["x"])):
            if L["ramp"][i] < 0.99 or self.kind != "humanoid":
                prev = None
                continue
            t = L["t0"] + i / L["fps"]
            sem, angs = self.baked["sems"][i], self.baked["angs"][i]
            w = self.rig.skel.world(angs, self.root_matrix(t, angs, sem))
            sb = "foot_n" if L["phase"][i] < 0.5 else "foot_f"
            k, p = max(((k, w[b].mapXY(*pp)) for k, b, pp in feet if b == sb), key=lambda q: q[1].y())
            if prev is not None and prev[0] == k:
                d.append(abs(p.x() - prev[1]))
            prev = (k, p.x())
        return float(np.mean(d)) if d else None

    def walk_speed(self):
        """Average travel speed while walking, in character heights per second (1.2-1.4 m/s ~ 0.75-0.85 h/s)."""
        L = self.loco
        if L is None or not (L["ramp"] > 0.99).any():
            return None
        m = L["ramp"] > 0.99
        xs = L["x"][m]
        return float(abs(xs[-1] - xs[0]) / (m.sum() / L["fps"]) / self.height)

    def bake(self, t0, t1, fps):
        n = max(1, int(round((t1 - t0) * fps)) + 1)
        ts = t0 + np.arange(n) / fps
        sems = self._integrate(ts, fps)
        to_b = self.to_bones
        angs = [to_b(s) for s in sems]
        xs = np.array([s["_x"] for s in sems])
        vx = np.gradient(xs) * fps if n > 1 else np.zeros(1)
        for ci, ch in enumerate(self.springs):
            prev_world = None
            for k, bone in enumerate(ch["bones"]):
                parent = self.rig.skel.by[bone].parent
                wind = Noise1D(self.seed * 31 + ci * 7 + k)
                target = []
                for i, a in enumerate(angs):
                    pw = self.rig.skel.world_angle(a, parent) if prev_world is None else prev_world[i]
                    hang = -pw * ch.get("hang", 0.85)
                    gust = ch.get("wind", 5.0) * (wind.fbm(ts[i] * ch.get("wind_hz", 0.55)) + 0.35 * math.sin(2 * math.pi * 0.6 * ts[i] + k))
                    drag = ch.get("drag", 0.02) * vx[i] * (1 if self.facing(ts[i]) > 0 else -1)
                    target.append(ch.get("rest", 0.0) + hang + gust + drag)
                resp = spring_follow(target, fps, freq=ch.get("freq", 1.0) * (0.85 ** k), zeta=ch.get("zeta", 0.28))
                for i, a in enumerate(angs):
                    a[bone] = float(resp[i])
                prev_world = [self.rig.skel.world_angle(a, bone) for a in angs]
        # follow-through: distal joints lag their key and overshoot a little, then settle (drag on the extremities)
        for bone, freq, zeta in (("farm_n", 3.2, 0.42), ("farm_f", 3.2, 0.42), ("hand_n", 3.6, 0.38), ("hand_f", 3.6, 0.38),
                                 ("head", 3.0, 0.5), ("tail1", 2.2, 0.35)):
            if bone in self.rig.skel.by and any(bone in a for a in angs):
                src = np.array([a.get(bone, 0.0) for a in angs])
                resp = spring_follow(src, fps, freq=freq, zeta=zeta)
                for i, a in enumerate(angs):
                    a[bone] = float(resp[i])
        self.baked = dict(t0=t0, fps=fps, ts=ts, sems=sems, angs=angs)
        return self

    def frame_index(self, t):
        b = self.baked
        return int(min(len(b["ts"]) - 1, max(0, round((t - b["t0"]) * b["fps"]))))

    def root_matrix(self, t, angs, sem):
        s = self.height / self.rig.units
        f = self.facing(t)
        x = sem["_x"]
        m = skia.Matrix()
        m.setTranslate(x, self.ground_at(t))
        m.preScale(f * s if abs(f) > 0.04 else 0.04 * s * (1 if f >= 0 else -1), s)
        pel = self.rig.meta.get("pelvis_h", 470.0)
        auto = mcopy(m)
        auto.preTranslate(0, -pel)
        dy = self.rig.ground_offset(angs, auto, self.ground_at(t)) / s
        y_auto = -pel + dy
        w = sem.get("seat_w", 1.0 if "root_y" in sem else 0.0)
        y = y_auto * (1 - w) + (-sem.get("root_y", pel)) * w
        m.preTranslate(0, y)
        return m

    def ground_at(self, t):
        g = self.ground
        for (a, b, dy) in self.rises:
            g += dy * smooth((t - a) / max(1e-6, b - a))
        if self.loco is not None:  # bound flight arc
            g -= self._loco_at(t)[3]
        return g

    def angles_at(self, t):
        if self.baked is None:
            sem = self.semantic(t)
            return sem, self.to_bones(sem)
        i = self.frame_index(t)
        return self.baked["sems"][i], self.baked["angs"][i]

    def draw(self, canvas, t, extra=None):
        sem, angs = self.angles_at(t)
        root = self.root_matrix(t, angs, sem)
        face = self.face(t)
        face.update({k: v for k, v in sem.items() if k.startswith("hand_") or k == "smile"})
        face["breath"] = sem.get("_breath", 0.0)
        ex = dict(self.props)
        ex.update(extra or {})
        self._contact_shadow(canvas, t, sem)
        return self.rig.draw(canvas, angs, root, self.P, t, face, ex)

    def _contact_shadow(self, c, t, sem):
        """Soft ground shadow under the feet (keeps puppets planted in the scene)."""
        g = self.ground_at(t)
        lift = g - self.ground  # climbing: shadow stays on the floor and fades
        a = max(0.0, 1 - abs(lift) / 300.0)
        if a <= 0.02:
            return
        w = self.height * (0.32 if self.kind == "humanoid" else 0.6)
        x = sem["_x"]
        p = skia.Paint(AntiAlias=True)
        p.setShader(skia.GradientShader.MakeRadial((x, self.ground), w, [skia.Color(20, 12, 10, int(110 * a)), skia.Color(20, 12, 10, 0)]))
        c.save()
        c.translate(x, self.ground)
        c.scale(1.0, 0.16)
        c.translate(-x, -self.ground)
        c.drawCircle(x, self.ground, w, p)
        c.restore()

    def head_pos(self, t):
        """World position of the head (camera framing target)."""
        sem, angs = self.angles_at(t)
        root = self.root_matrix(t, angs, sem)
        w = self.rig.skel.world(angs, root)
        p = w["head" if "head" in w else self.rig.skel.bones[-1].name].mapXY(0, -40)
        return p.x(), p.y()


def direction_reached(x, seg):
    a, b, x0, x1 = seg
    return (x1 - x0) * (x - x1) >= 0