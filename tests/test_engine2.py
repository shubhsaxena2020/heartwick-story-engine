"""Engine v2 core: rig/FK, both part sources, poses, scheduler, springs, camera limits, cuts on pauses,
captions, mastering, cached layers. CPU only, a few seconds."""
import math, os
import numpy as np
import pytest

skia = pytest.importorskip("skia")
from engine2 import (rig as RG, poses as PO, motion as MO, camera as CA, edit as ED, audio as AU, characters as CH,
                     style as ST, scenery as SC, autorig as AR)
from engine2.mathx import spring_follow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- rig + FK
def test_fk_chain_geometry():
    sk = RG.Skeleton([RG.Bone("a", None), RG.Bone("b", "a", (0, 100)), RG.Bone("c", "b", (0, 50))])
    root = skia.Matrix()
    root.setTranslate(10, 20)
    w = sk.world({"a": 0, "b": 90}, root)  # b rotated 90 deg clockwise: its +y axis points to -x
    assert RG.map_pt(w["b"], 0, 0) == pytest.approx((10, 120))
    assert RG.map_pt(w["c"], 0, 0) == pytest.approx((-40, 120), abs=1e-4)
    assert sk.world_angle({"a": 10, "b": 20, "c": 30}, "c") == pytest.approx(60)


def test_parents_must_come_first():
    with pytest.raises(ValueError):
        RG.Skeleton([RG.Bone("b", "a"), RG.Bone("a", None)])


def _actor(pose, name="nell", ground=1500.0, h=900.0, **kw):
    P = ST.VectorCel()
    rig, springs = CH.build(name, P)
    a = MO.Actor(rig, P, 500, ground, h, [(0, pose, 0)], springs=springs, kind="squirrel" if name.startswith("pip") else "humanoid", **kw)
    return a


@pytest.mark.parametrize("pose", ["stand", "kneel", "kneel_reach", "bow", "point", "rig_ready"])
def test_grounding_puts_lowest_contact_on_floor(pose):
    a = _actor(pose)
    sem, angs = a.angles_at(0.0)
    root = a.root_matrix(0.0, angs, sem)
    w = a.rig.skel.world(angs, root)
    low = max(RG.map_pt(w[b], *p)[1] for b, p in a.rig.contacts)
    assert low == pytest.approx(1500.0, abs=0.5)


def test_sitting_uses_seat_height():
    a = _actor("sit")
    sem, angs = a.angles_at(0.0)
    root = a.root_matrix(0.0, angs, sem)
    pel = RG.map_pt(root, 0, 0)[1]
    assert pel == pytest.approx(1500 - 300 * 900 / 1000, abs=1)


def test_pose_semantics_point_and_kneel():
    a = _actor("stand")
    def hand_and_shoulder(pose):
        sem = dict(PO.HUMAN[pose])
        angs = PO.to_bones(sem)
        w = a.rig.skel.world(angs, skia.Matrix())
        return RG.map_pt(w["hand_n"], 0, 0), RG.map_pt(w["uarm_n"], 0, 0), RG.map_pt(w["shin_n"], 0, 0), RG.map_pt(w["pelvis"], 0, 0)
    h, s, _, _ = hand_and_shoulder("point")
    assert h[0] - s[0] > 250 and abs(h[1] - s[1]) < 60  # arm straight out in front, shoulder height
    h, s, _, _ = hand_and_shoulder("heart")
    assert abs(h[0] - s[0]) < 80 and h[1] > s[1]  # hand back at the chest
    _, _, knee, pel = hand_and_shoulder("kneel")
    assert knee[0] - pel[0] > 180  # front thigh forward


def test_every_pose_is_numeric_and_bounded():
    for name, p in PO.HUMAN.items():
        for k, v in p.items():
            if k in PO.SIGN:
                assert -170 <= v <= 170, (name, k)
    for name, p in PO.SQUIRREL.items():
        assert set(p) <= set(PO.SQ_JOINTS)


# ---------------------------------------------------------------- raster source (auto-rig)
def _synthetic_figure(tmp_path):
    """Render the vector Nell in the rig-ready pose, read its joints from FK: a stand-in for BiRefNet + RTMPose."""
    P = ST.VectorCel()
    rig, _ = CH.build("nell", P)
    angs = PO.to_bones(PO.HUMAN["rig_ready"])
    root = skia.Matrix()
    root.setTranslate(300, 330)
    root.preScale(0.5, 0.5)
    s = skia.Surface.MakeRasterN32Premul(600, 640)
    s.getCanvas().clear(skia.ColorTRANSPARENT)
    w = rig.draw(s.getCanvas(), angs, root, P, 0.0, {"blink": 1.0})
    img = s.makeImageSnapshot()
    rgba = img.convert(alphaType=skia.kUnpremul_AlphaType, colorType=skia.kRGBA_8888_ColorType).toarray()
    pt = lambda b, x=0.0, y=0.0: RG.map_pt(w[b], x, y)
    kp = {"nose": pt("head", 78, -66), "r_eye": pt("head", 47, -72), "l_eye": pt("head", 44, -72), "r_ear": pt("head", -4, -64),
          "l_ear": pt("head", -6, -64)}
    for c, s_ in (("r", "n"), ("l", "f")):
        kp.update({f"{c}_shoulder": pt(f"uarm_{s_}"), f"{c}_elbow": pt(f"farm_{s_}"), f"{c}_wrist": pt(f"hand_{s_}"),
                   f"{c}_hip": pt(f"thigh_{s_}"), f"{c}_knee": pt(f"shin_{s_}"), f"{c}_ankle": pt(f"foot_{s_}")})
    return rgba, kp


def test_autorig_partsheet_reproduces_source_and_poses(tmp_path):
    rgba, kp = _synthetic_figure(tmp_path)
    d = str(tmp_path / "rig")
    J = AR.autorig(rgba, kp, d, name="nell_raster")
    assert {p["bone"] for p in J["parts"]} >= {"head", "chest", "uarm_n", "farm_n", "hand_n", "thigh_n", "shin_n"}
    r = RG.load_partsheet(d)
    pel = J["bones"][0]["pivot_px"]
    root = skia.Matrix()
    root.setTranslate(*pel)
    root.preScale(1 / J["units_per_px"], 1 / J["units_per_px"])
    s = skia.Surface.MakeRasterN32Premul(600, 640)
    s.getCanvas().clear(skia.ColorTRANSPARENT)
    r.draw(s.getCanvas(), RG.image_pose_angles(d), root, ST.VectorCel())
    out = s.makeImageSnapshot().convert(alphaType=skia.kUnpremul_AlphaType, colorType=skia.kRGBA_8888_ColorType).toarray()
    a, b = rgba[..., 3] > 128, out[..., 3] > 128
    iou = (a & b).sum() / (a | b).sum()
    assert iou > 0.9, iou
    # the same semantic pose library drives it: "point" puts the near hand out in front at shoulder height
    w = r.skel.world(PO.to_bones(PO.HUMAN["point"]), root)
    hx, hy = RG.map_pt(w["hand_n"], 0, 0)
    sx, sy = RG.map_pt(w["uarm_n"], 0, 0)
    assert hx - sx > 60 and abs(hy - sy) < 30


# ---------------------------------------------------------------- motion
def test_scheduler_timing_matches_reference():
    rng = np.random.default_rng(1)
    keys = MO.schedule(0, 60, "stand", rng)
    gaps = np.diff([k[0] for k in keys[1:]])
    assert 0.7 <= np.median(gaps) <= 1.1  # round 3: no hold longer than ~0.8 s (anime-like acting)
    assert all(0.16 <= k[2] <= 0.3 for k in keys[1:])
    assert len(set(k[1] for k in keys)) >= 6  # variety, not one gesture repeated


def test_pinned_beats_land_on_their_words():
    rng = np.random.default_rng(2)
    keys = MO.schedule(0, 10, "stand", rng, pinned=[(4.2, "kneel"), (6.0, "kneel_reach")])
    names = {round(k[0], 2): k[1] for k in keys}
    assert names.get(4.2) == "kneel" and names.get(6.0) == "kneel_reach"
    after = [k[1] for k in keys if k[0] > 4.2]
    assert all(PO.FAMILY.get(p) == "kneel" for p in after)  # once kneeling, gestures come from the kneel family


def test_pose_track_is_continuous():
    tr = MO.PoseTrack(PO.HUMAN, [(0, "stand", 0), (1.0, "point", 0.3), (1.15, "heart", 0.3)])  # interrupted transition
    prev = None
    for t in np.arange(0, 2, 1 / 240):
        v = tr(t)["sh_n"]
        if prev is not None:
            assert abs(v - prev) < 9.0  # eased 0.3 s transitions peak near 820 deg/s; a jump would be far larger
        prev = v
    assert tr(1.8)["el_n"] == pytest.approx(PO.HUMAN["heart"]["el_n"])


def test_spring_lags_and_overshoots():
    tgt = np.r_[np.zeros(30), np.full(90, 10.0)]
    r = spring_follow(tgt, 30, freq=1.0, zeta=0.3)
    assert r[31] < 2 and r.max() > 10.5 and abs(r[-1] - 10) < 1.0


def test_actor_bake_springs_and_blinks():
    a = _actor("stand")
    a.walks = [(0.5, 2.0, 500, 900)]
    a.bake(0, 3, 30)
    scarf = [x["scarf3"] for x in a.baked["angs"]]
    assert np.ptp(scarf) > 3  # sash/scarf moves on its own
    assert any(a.face(t)["blink"] < 0.5 for t in np.arange(0, 6, 1 / 60))


# ---------------------------------------------------------------- camera
@pytest.mark.parametrize("kind", ["push_in", "pull_back", "pan_left", "pan_right", "drift"])
def test_camera_speeds_within_reference(kind):
    rng = np.random.default_rng(3)
    for _ in range(5):
        c = CA.Camera(kind, 0, 4, (1400, 2100), (700, 1100), 1.0, rng=rng)
        px, zp = c.speeds()
        if kind.startswith("pan") or kind == "drift":
            assert 0.15 <= px <= 0.5, px
        if kind in ("push_in", "pull_back"):
            assert 0.6 <= abs(zp) <= 1.45, zp
        for t in np.arange(0, 4, 0.25):  # never frozen
            a, b = c.at(t), c.at(t + 1 / 30)
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]) * 1000 > 1e-3


def test_tilt_reveal_takes_four_seconds():
    c = CA.Camera("tilt_down", 0, 6, (1400, 3000), (700, 2000), 1.0, tilt_from=960.0)
    assert c.at(0)[1] == pytest.approx(960, abs=1)
    assert c.at(4.0)[1] == pytest.approx(2000, abs=2)
    assert c.at(2.0)[1] == pytest.approx(1480, abs=5)


# ---------------------------------------------------------------- edit
def _words(n=60, gap_every=4, rate=2.7):
    t, out = 0.3, []
    for i in range(n):
        d = 1 / rate
        out.append({"w": f"w{i}", "s": t, "e": t + d * 0.85})
        t += d + (0.45 if i % gap_every == gap_every - 1 else 0.0)
    return out


def test_cuts_land_in_pauses_and_pace_is_in_range():
    words = _words(84)
    # story shots end at utterance boundaries, i.e. after a pause (every 4th word here): 28 words each
    scenes = [{"shot": {"v2": {}}, "section": 0, "first_in_section": i == 0, "words": words[i * 28:(i + 1) * 28],
               "s": words[i * 28]["s"], "e": words[(i + 1) * 28 - 1]["e"] + 0.3} for i in range(3)]
    shots, end = ED.cut_plan(scenes, words, {"max_shot_s": 4.6})
    assert ED.cuts_in_pauses(shots, words) == 1.0
    spm = len(shots) / (end / 60)
    assert 10 <= spm <= 19, spm
    assert all(1.5 <= s["e"] - s["s"] <= 5.6 for s in shots)


def test_dissolve_lengths():
    words = _words(60)
    scenes = [{"shot": {"v2": {"trans": "dissolve"}}, "section": i, "first_in_section": True, "words": words[i * 20:(i + 1) * 20],
               "s": words[i * 20]["s"], "e": words[(i + 1) * 20 - 1]["e"] + 0.3} for i in range(3)]
    shots, _ = ED.cut_plan(scenes, words, {})
    ds = [s["trans_d"] for s in shots if s["trans"] == "dissolve"]
    assert ds and all(0.7 <= d <= 1.6 for d in ds)


def test_captions_two_to_four_words_with_fade():
    from engine import timeline as T1
    ws = [{"w": w, "s": i * 0.35, "e": i * 0.35 + 0.3, "ws": " "} for i, w in
          enumerate("Nell knelt down slowly in the leaves, complaining a little. Hello, little one, where is your mama?".split())]
    caps = ED.captions(T1.chunk_words(ws, {"nell", "mama"}, set()))
    assert all(1 <= len(c["words"]) <= 4 for c in caps) and sum(len(c["words"]) >= 2 for c in caps) >= len(caps) - 1
    assert all(c["fade"] == 0.2 for c in caps)
    assert any(c["hl"] >= 0 for c in caps)


# ---------------------------------------------------------------- audio
def test_master_hits_loudness_peak_and_pad_gap():
    t = np.arange(int(AU.SR * 8)) / AU.SR
    rng = np.random.default_rng(0)
    env = (np.sin(2 * np.pi * 3 * t) > -0.3) * (0.4 + 0.6 * rng.random(len(t)) ** 8)  # peaky speech-like
    voice = (np.sin(2 * np.pi * 180 * t) * env * 0.3).astype(np.float32)
    mix, rep = AU.master(AU.compress(voice), AU.pad(8.0))
    assert abs(rep["lufs"] - (-14.0)) <= 0.3
    assert rep["true_peak_db"] <= -1.5
    assert 18 <= rep["pad_under_voice_db"] <= 22


# ---------------------------------------------------------------- scenes
@pytest.mark.parametrize("style", ["B", "C"])
def test_scene_caches_static_layers_and_moves(style):
    P = ST.PaperTheatre() if style == "C" else ST.VectorCel()
    sc = SC.build_location("yard_morning", P)
    a = MO.Actor(*CH.build("nell", P)[:1], P, 600, sc.ground, 900, [(0, "stand", 0), (0.4, "point", 0.3)])
    a.bake(0, 1, 30)
    W, H = 270, 480
    s = skia.Surface.MakeRasterN32Premul(W, H)
    frames = []
    for t in (0.1, 0.8):
        sc.render(s.getCanvas(), t, (700, 1100, 1.0), [a], W, H, q=0.5)
        frames.append(s.makeImageSnapshot().toarray().astype(int))
    n_static = sum(1 for L in sc.layers if L.static is not None)
    assert len(sc._cache) == n_static  # built once, reused for the second frame
    assert np.abs(frames[0] - frames[1]).mean() > 0.5  # something moved without the camera moving


def test_every_pip_location_builds():
    P = ST.VectorCel()
    for loc in SC.LOCATIONS:
        sc = SC.build_location(loc, P)
        assert sc.world[0] >= 1080 * 1.2 and any(L.name == "actors" for L in sc.layers)
        assert 5 <= len(sc.layers) <= 8


# ---------------------------------------------------------------- pipeline helpers
def test_voice_keys_reuse_v1_narration_cache():
    from engine2 import pipeline as PL
    assert PL.voice_key("kokoro:af_heart") == "af_heart"
    assert PL.voice_key("voxcpm2:female") == "voxcpm2:female"
    _, utts = PL.utterances(os.path.join(ROOT, "stories", "style_test.md"), "kokoro:af_heart", 1.0)
    have = {f[:-5] for f in os.listdir(os.path.join(ROOT, "cache", "pip", "assets"))} if os.path.isdir(os.path.join(ROOT, "cache", "pip", "assets")) else None
    if have is not None:  # the style test reuses the pilot's cached Kokoro lines
        assert all(u["id"] in have for u in utts)


def test_srt_format(tmp_path):
    from engine2 import pipeline as PL
    p = tmp_path / "x.srt"
    PL.srt({"captions": [{"s": 0.35, "e": 1.5, "words": ["Nell", "knelt"]}, {"s": 61.25, "e": 62.0, "words": ["in", "the", "leaves"]}]}, str(p))
    t = p.read_text(encoding="utf-8")
    assert "00:00:00,350 --> 00:00:01,500\nNell knelt" in t and "00:01:01,250 --> 00:01:02,000" in t


def test_assets_contract_part_sheet_and_figure_keypoints(tmp_path):
    """assets/characters/<name>/: a ready part sheet loads as is; a figure + COCO keypoints is auto-rigged once."""
    import json as _json
    from PIL import Image
    from engine2 import assets as AS
    rgba, kp = _synthetic_figure(tmp_path)
    a = tmp_path / "assets" / "characters" / "nell"
    a.mkdir(parents=True)
    Image.fromarray(rgba, "RGBA").save(a / "nell.png")
    _json.dump({k: list(v) for k, v in kp.items()}, open(a / "keypoints.json", "w"))
    d = AS.find_rig("nell", root=str(tmp_path))
    assert d and os.path.exists(os.path.join(d, "rig.json"))
    r = RG.load_partsheet(d)
    assert "hand_n" in r.skel.by and len(r.parts) >= 12
    files = AS.rig_files(["nell", "pip"], root=str(tmp_path))
    assert "rigs/nell/rig.json" in files and not any(k.startswith("rigs/pip/") for k in files)
    s = tmp_path / "assets" / "characters" / "dee"  # route 1: a ready sheet is used directly
    s.mkdir(parents=True)
    for f in os.listdir(d):
        (s / f).write_bytes(open(os.path.join(d, f), "rb").read())
    assert AS.find_rig("dee", root=str(tmp_path)) == str(s)
    assert AS.find_rig("eli", root=str(tmp_path)) is None  # nothing there: vector puppet


# ---------------------------------------------------------------- locomotion (round 3)
def test_walk_feet_planted_and_natural_speed():
    a = _actor("stand", h=900.0)
    a.walks = [(0.3, 6.0, 200.0, 1400.0)]
    a.bake(0, 4.0, 60)
    slide = a.foot_slide()
    assert slide is not None and slide < 0.35, slide  # px per frame of the planted foot
    v = a.walk_speed()
    assert 0.55 <= v <= 1.0, v  # character heights per second (~1.2-1.4 m/s for a 1.6 m adult)
    xs = a.loco["x"]
    assert xs[-1] - xs[0] > 900  # it really travels
    step = np.diff(xs[a.loco["ramp"] > 0.99])
    assert (step >= -0.5).all()  # never moonwalks


def test_kneel_happens_in_steps():
    rng = np.random.default_rng(3)
    keys = MO.schedule(0, 4, "stand", rng, pinned=[(1.0, "kneel")])
    names = [k[1] for k in keys]
    i = names.index("kneel")
    assert "half_kneel" in names[:i] and "reach_down" in names[:i]


def test_squirrel_bounds_leave_the_ground():
    a = _actor("sit", name="pip", h=200.0)
    a.walks = [(0.2, 3.0, 300.0, 900.0)]
    a.bake(0, 3.0, 60)
    assert a.loco["lift"].max() > 30 and a.loco["x"][-1] > 800