"""Pose library. Poses are written in *semantic* degrees so they read naturally:

  spine/chest/neck/head  + = lean / nod forward (toward the way the character faces)
  sh_* (shoulder)        + = raise the arm forward          el_* (elbow) + = bend the forearm forward/up
  wr_* (wrist)           + = bend the hand forward/up
  hip_*                  + = swing the thigh forward        kn_* (knee)  + = bend the knee (shin goes back)
  an_* (ankle)           + = toes up
  _n = near side (drawn in front), _f = far side (behind the body)

to_bones() converts them to bone angles for engine2/rig.py (clockwise-positive, facing right).
Every pose may also carry: "hand_n"/"hand_f" (open, fist, point, cup), "root_y" (fixed pelvis height in
rig units above the ground, for sitting; otherwise the lowest contact is put on the ground), "face" (smile, eyes).
"""
import math

HUMAN_JOINTS = ("spine", "chest", "neck", "head", "sh_n", "el_n", "wr_n", "sh_f", "el_f", "wr_f",
                "hip_n", "kn_n", "an_n", "hip_f", "kn_f", "an_f")
BONE_OF = {"spine": "spine", "chest": "chest", "neck": "neck", "head": "head",
           "sh_n": "uarm_n", "el_n": "farm_n", "wr_n": "hand_n", "sh_f": "uarm_f", "el_f": "farm_f", "wr_f": "hand_f",
           "hip_n": "thigh_n", "kn_n": "shin_n", "an_n": "foot_n", "hip_f": "thigh_f", "kn_f": "shin_f", "an_f": "foot_f"}
SIGN = {"spine": 1, "chest": 1, "neck": 1, "head": 1, "sh_n": -1, "el_n": -1, "wr_n": -1, "sh_f": -1, "el_f": -1,
        "wr_f": -1, "hip_n": -1, "kn_n": 1, "an_n": -1, "hip_f": -1, "kn_f": 1, "an_f": -1}

_STAND = dict(sh_n=4, el_n=10, sh_f=3, el_f=12, kn_n=2, kn_f=3, hip_f=-2)
_KNEEL = dict(hip_n=82, kn_n=86, an_n=6, hip_f=-8, kn_f=100, an_f=-40, spine=10, chest=5, head=8)
_SIT = dict(hip_n=88, kn_n=88, hip_f=84, kn_f=92, root_y=300)


def _p(base, **kw):
    d = dict(base)
    d.update(kw)
    return d


HUMAN = {
    # standing family
    "stand": _p(_STAND),
    "point": _p(_STAND, sh_n=86, el_n=4, wr_n=-6, hand_n="point", chest=-3, head=-2, sh_f=12, el_f=22),
    "point_up": _p(_STAND, sh_n=135, el_n=6, hand_n="point", chest=-6, head=-14, neck=-6),
    "reach": _p(_STAND, sh_n=58, el_n=12, wr_n=8, spine=8, chest=6, head=8, sh_f=28, el_f=30),
    "reach_down": _p(_STAND, sh_n=38, el_n=8, spine=22, chest=10, head=14, sh_f=24, el_f=18, kn_n=8, kn_f=10, hip_n=6),
    "heart": _p(_STAND, sh_n=12, el_n=150, wr_n=22, head=7, chest=2, sh_f=6, el_f=16),
    "both_heart": _p(_STAND, sh_n=12, el_n=150, wr_n=20, sh_f=16, el_f=146, wr_f=20, head=9, chest=3),
    "arms_open": _p(_STAND, sh_n=48, el_n=22, wr_n=-10, sh_f=42, el_f=20, chest=-4, head=-6),
    "think": _p(_STAND, sh_n=30, el_n=142, wr_n=30, hand_n="fist", head=5, sh_f=22, el_f=96),
    "hold": _p(_STAND, sh_n=26, el_n=98, wr_n=6, sh_f=30, el_f=94, wr_f=6, head=10, chest=3, hand_n="cup", hand_f="cup"),
    "hold_up": _p(_STAND, sh_n=40, el_n=104, sh_f=44, el_f=100, head=-4, chest=-3, hand_n="cup", hand_f="cup"),
    "phone": _p(_STAND, sh_n=20, el_n=150, wr_n=20, hand_n="fist", head=4, sh_f=10, el_f=60),
    "wave": _p(_STAND, sh_n=150, el_n=36, wr_n=-10, chest=-3, head=-3),
    "look_up": _p(_STAND, head=-20, neck=-8, chest=-4),
    "look_down": _p(_STAND, head=18, neck=6, chest=3),
    "bow": _p(_STAND, spine=26, chest=12, neck=6, head=12, sh_n=14, el_n=24, sh_f=12, el_f=22, kn_n=5, kn_f=5),
    "shrug": _p(_STAND, sh_n=20, el_n=80, wr_n=-30, sh_f=18, el_f=80, wr_f=-30, head=-4, chest=-2, hand_n="open"),
    "hands_clasped": _p(_STAND, sh_n=18, el_n=70, sh_f=20, el_f=70, head=4),
    "talk": _p(_STAND, sh_n=30, el_n=70, wr_n=-14, hand_n="open", head=-2, sh_f=8, el_f=20),
    "half_kneel": _p(_STAND, hip_n=72, kn_n=78, an_n=4, hip_f=8, kn_f=62, an_f=-20, spine=18, chest=8, head=10, sh_n=30, el_n=30, sh_f=24, el_f=26),
    "sit_prep": _p(_STAND, hip_n=48, kn_n=52, hip_f=46, kn_f=54, spine=20, chest=6, head=4, sh_n=34, el_n=30, sh_f=30, el_f=28),
    "glance_back": _p(_STAND, head=-8, neck=-10, chest=-6, sh_n=8, el_n=14),
    "rig_ready": dict(sh_n=34, el_n=12, sh_f=-30, el_f=8, hip_n=12, hip_f=-12, kn_n=4, kn_f=4),  # arms clear (auto-rig source)
    # kneeling family
    "kneel": _p(_KNEEL, sh_n=30, el_n=36, sh_f=20, el_f=30),
    "kneel_reach": _p(_KNEEL, sh_n=58, el_n=10, wr_n=10, spine=24, chest=12, head=12, sh_f=48, el_f=14),
    "kneel_heart": _p(_KNEEL, sh_n=12, el_n=150, wr_n=22, sh_f=34, el_f=40, head=12),
    "kneel_look_up": _p(_KNEEL, sh_n=32, el_n=40, sh_f=24, el_f=34, head=-20, neck=-8, chest=-4, spine=4),
    "kneel_cup": _p(_KNEEL, sh_n=40, el_n=60, sh_f=44, el_f=56, spine=18, chest=8, head=14, hand_n="cup", hand_f="cup"),
    # sitting family
    "sit": _p(_SIT, sh_n=26, el_n=60, sh_f=24, el_f=58),
    "sit_hold": _p(_SIT, sh_n=30, el_n=96, sh_f=32, el_f=92, head=8, hand_n="cup", hand_f="cup"),
    "sit_sad": _p(_SIT, spine=14, chest=10, neck=8, head=24, sh_n=40, el_n=70, sh_f=42, el_f=68),
    "sit_look_up": _p(_SIT, head=-14, neck=-6, sh_n=26, el_n=60, sh_f=24, el_f=58),
    "sit_read": _p(_SIT, sh_n=40, el_n=88, sh_f=42, el_f=86, head=16, hand_n="cup", hand_f="cup"),
    "sit_heart": _p(_SIT, sh_n=12, el_n=150, wr_n=22, sh_f=24, el_f=58, head=8),
}

# which gestures make sense from each base pose (auto-scheduler draws from these; one change every 1-1.5 s)
VOCAB = {
    "stand": ["stand", "point", "reach", "heart", "look_up", "look_down", "arms_open", "think", "hands_clasped", "talk"],
    "kneel": ["kneel", "kneel_reach", "kneel_heart", "kneel_look_up", "kneel_cup"],
    "sit": ["sit", "sit_hold", "sit_look_up", "sit_heart", "sit_sad"],
    "hold": ["hold", "hold_up", "look_down", "hold"],
    "walk": ["walk"],
}
FAMILY = {k: ("kneel" if k.startswith("kneel") else "sit" if k.startswith("sit") else
              "hold" if k.startswith("hold") else "stand") for k in HUMAN}

# gestures that are short overlays (added on top of the current pose, then released)
OVERLAYS = {
    "nod": dict(dur=0.55, joints={"head": 13, "neck": 4}, shape="double"),
    "shake": dict(dur=0.7, joints={"head": 0}, shape="none"),
    "wave_hand": dict(dur=1.2, joints={"wr_n": 22, "el_n": 10}, shape="osc", freq=2.6),
    "sigh": dict(dur=1.1, joints={"chest": 5, "head": 6, "sh_n": -4, "sh_f": -4}, shape="bump"),
    "tremble": dict(dur=1.0, joints={"spine": 2.5, "head": 3}, shape="osc", freq=9.0),
    "glance": dict(dur=0.7, joints={"head": -9, "neck": -4}, shape="bump"),
    "talk_hand": dict(dur=0.8, joints={"el_n": 18, "wr_n": -16, "sh_n": 6}, shape="osc", freq=1.8),
    "shrug_small": dict(dur=0.6, joints={"sh_n": 8, "sh_f": 8, "head": -4}, shape="bump"),
    "twitch": dict(dur=0.45, joints={"head": 4}, shape="osc", freq=11.0),
    "tail_flick": dict(dur=0.6, joints={"tail1": 26}, shape="osc", freq=3.2),
    "ear_flick": dict(dur=0.3, joints={"head": 3}, shape="osc", freq=8.0),
    "lift": dict(dur=0.6, joints={"chest": -4, "head": -6}, shape="bump"),
}


def _g(x, mu, s):
    d = (x - mu + 0.5) % 1.0 - 0.5  # wrapped distance on the cycle
    return math.exp(-(d / s) ** 2)


def walk(phase, amp=24.0, arm=18.0, run=False):
    """Gait cycle (phase in cycles; near heel strike at 0). Heel strike -> loading (small knee flex) -> mid-stance ->
    push-off (toes down) -> swing (knee flexes to ~60 deg, foot clears) -> heel strike. Arms counter-swing, the head
    counters the bob. Travel speed is NOT set here: the actor locks the stance foot to the ground (motion.Actor)."""
    if run:
        amp, arm = 34.0, 30.0

    def leg(ph):
        hip = amp * math.cos(2 * math.pi * ph) - 2.0
        knee = 4 + 16 * _g(ph, 0.12, 0.08) + (62 if not run else 95) * _g(ph, 0.74, 0.11)
        ank = 10 * _g(ph, 0.0, 0.07) - 18 * _g(ph, 0.56, 0.07) + 6 * _g(ph, 0.8, 0.1)
        return hip, knee, ank

    hn, kn, an = leg(phase)
    hf, kf, af = leg(phase + 0.5)
    c = math.cos(2 * math.pi * phase)
    lean = 4 if not run else 12
    return dict(hip_n=hn, kn_n=kn, an_n=an, hip_f=hf, kn_f=kf, an_f=af,
                sh_n=-arm * c + 6, el_n=16 + 12 * max(0.0, c) + (40 if run else 0), sh_f=arm * c + 6, el_f=16 + 12 * max(0.0, -c) + (40 if run else 0),
                spine=lean + 1.5 * math.cos(4 * math.pi * phase), chest=-1.5 * c, head=-lean * 0.6 - 2.0 * math.cos(4 * math.pi * phase))


WALK_CADENCE = {"walk": 1.02, "elder": 0.92, "run": 1.45, "child": 1.1}  # gait cycles (2 steps) per second


def stride_speed(rig_units=1000.0, leg=470.0, amp=22.0, period=1.15):
    """Ground speed (rig units / s) that keeps the planted foot from sliding."""
    step = 2 * leg * math.sin(math.radians(amp)) * 0.92
    return 2 * step / period


def to_bones(sem):
    """Semantic pose dict -> {bone: deg} (clockwise positive, facing right)."""
    out = {}
    for k, v in sem.items():
        if k in SIGN and isinstance(v, (int, float)):
            out[BONE_OF[k]] = SIGN[k] * v
    return out


def blend(a, b, u):
    """Blend two semantic poses (numeric joints only); non-numeric fields switch at u > 0.5."""
    out = {}
    for k in (set(a) | set(b)) - {"root_y", "seat_w"}:
        va, vb = a.get(k), b.get(k)
        if k in SIGN or k in SQ_JOINTS:
            out[k] = (va or 0.0) + ((vb or 0.0) - (va or 0.0)) * u
        else:  # hand shapes, face: switch half way
            out[k] = vb if (u > 0.5 and vb is not None) or va is None else va
    # seated weight blends continuously; the actor mixes the fixed seat height with auto-grounding
    wa = a.get("seat_w", 1.0 if "root_y" in a else 0.0)
    wb = b.get("seat_w", 1.0 if "root_y" in b else 0.0)
    out["seat_w"] = wa + (wb - wa) * u
    ry = b.get("root_y", a.get("root_y"))
    if ry is not None:
        out["root_y"] = ry
    return out


# ---------------------------------------------------------------- squirrels (quadruped-ish rig, side view facing right)
SQ_JOINTS = ("body", "head", "tail1", "paw_n", "paw_f", "leg_n", "leg_f", "ear")
SQUIRREL = {
    "sit": dict(body=-62, head=50, paw_n=40, paw_f=36, leg_n=-20, leg_f=-24, tail1=104),
    "sit_look": dict(body=-62, head=34, paw_n=46, paw_f=40, leg_n=-20, leg_f=-24, tail1=98),
    "sit_look_up": dict(body=-66, head=22, paw_n=40, paw_f=36, leg_n=-20, leg_f=-24, tail1=108),
    "nibble": dict(body=-58, head=60, paw_n=70, paw_f=64, leg_n=-20, leg_f=-24, tail1=100),
    "alert": dict(body=-74, head=58, paw_n=10, paw_f=8, leg_n=-26, leg_f=-28, tail1=112),
    "crouch": dict(body=-8, head=8, paw_n=8, paw_f=-8, leg_n=4, leg_f=-6, tail1=48),
    "look_back": dict(body=-10, head=-40, paw_n=6, paw_f=-6, leg_n=4, leg_f=-6, tail1=56),
    "curl": dict(body=12, head=26, paw_n=60, paw_f=50, leg_n=40, leg_f=34, tail1=50),
    "curl_lift": dict(body=6, head=-6, paw_n=40, paw_f=30, leg_n=36, leg_f=30, tail1=46),
    "curl_reach": dict(body=4, head=4, paw_n=-20, paw_f=30, leg_n=36, leg_f=30, tail1=44),
}
SQ_VOCAB = {
    "sit": ["sit", "sit_look", "sit_look_up", "alert", "nibble"],
    "curl": ["curl", "curl_lift", "curl_reach", "curl"],
    "crouch": ["crouch", "look_back", "alert"],
    "nibble": ["nibble", "sit", "nibble", "sit_look"],
}


def run_cycle(phase):
    """Squirrel bound: hind legs push (phase 0-0.3), flight with the body stretched (0.3-0.65, the actor lifts it),
    front paws land and gather (0.65-1). The tail lags the body (springs) and flicks on landing."""
    s = math.sin(2 * math.pi * phase)
    push = _g(phase, 0.15, 0.12)
    land = _g(phase, 0.72, 0.1)
    return dict(body=-16 * push + 10 * land - 4, head=8 * push - 6 * land + 4, paw_n=-55 * push + 35 * land, paw_f=-45 * push + 40 * land,
                leg_n=60 * push - 40 * land, leg_f=52 * push - 34 * land, tail1=44 + 18 * s)


def bound_lift(phase):
    """Flight arc of a bound as a fraction of body height (0 on the ground)."""
    return max(0.0, math.sin(math.pi * (phase - 0.25) / 0.5)) if 0.25 < phase < 0.75 else 0.0


def sq_to_bones(sem):
    return {k: v for k, v in sem.items() if k in SQ_JOINTS and isinstance(v, (int, float))}
