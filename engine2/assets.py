"""Where raster (AI-art) puppets come from. Engine v2 uses one automatically when it exists for a character, else
the vector puppet from engine2/characters.py.

Contract for assets/characters/<name>/ (art made outside the engine), in priority order:
  1. rig.json + part PNGs            a ready part sheet (format: engine2.rig.load_partsheet). Bone names must be the
     (or rig/rig.json + PNGs)        engine's: humanoid pelvis, spine, chest, neck, head, uarm_n/f, farm_n/f, hand_n/f,
                                     thigh_n/f, shin_n/f, foot_n/f; squirrel body, head, paw_n/f, leg_n/f, tail1-3.
                                     "kind": "humanoid" | "squirrel".
  2. <name>.png + keypoints.json      one rig-ready full figure (RGBA cut-out, facing right, arms clear of the body)
                                     and its COCO-17 keypoints {"nose": [x, y], "l_shoulder": [x, y], ...};
                                     auto-rigged here (engine2/autorig.py, OpenCV fill) into cache/rigs/<name>/.
  3. characters/<name>/rig/           output of tools/autorig.py (BiRefNet + RTMPose + LaMa on Kaggle).
"""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets", "characters")


def find_rig(name, root=ROOT):
    """-> folder holding rig.json for this character, or None (use the vector puppet)."""
    a = os.path.join(root, "assets", "characters", name)
    for d in (a, os.path.join(a, "rig")):
        if os.path.exists(os.path.join(d, "rig.json")):
            return d
    png, kp = os.path.join(a, f"{name}.png"), os.path.join(a, "keypoints.json")
    if os.path.exists(png) and os.path.exists(kp):
        out = os.path.join(root, "cache", "rigs", name)
        stamp = os.path.join(out, "rig.json")
        if not os.path.exists(stamp) or os.path.getmtime(stamp) < max(os.path.getmtime(png), os.path.getmtime(kp)):
            from PIL import Image
            from . import autorig
            rgba = np.asarray(Image.open(png).convert("RGBA"))
            autorig.autorig(rgba, {k: tuple(v) for k, v in json.load(open(kp)).items()}, out, name=name)
        return out
    d = os.path.join(root, "characters", name, "rig")
    if os.path.exists(os.path.join(d, "rig.json")):
        return d
    return None


def rig_files(names, root=ROOT):
    """{zip path: local path} for every raster rig the story's characters have (shipped to the render kernel as rigs/<name>/)."""
    out = {}
    for n in names:
        d = find_rig(n, root)
        if d:
            for f in os.listdir(d):
                out[f"rigs/{n}/{f}"] = os.path.join(d, f)
    return out


# ---------------------------------------------------------------- painted scene plates (assets/scenes/<folder>/scene.json)
# story location -> plate folder (an exact assets/scenes/<location>/ always wins). Locations not listed keep the
# vector scene from engine2/scenery.py. The location's own mood (grade, light, music) still comes from scenery.LOCATIONS.
SCENE_ALIASES = {
    "village_dawn": "village_autumn", "house_quiet": "village_autumn",
    "storm_night": "storm", "storm_maple": "storm",
    "yard_morning": "maple_yard", "yard_tall": "maple_yard", "leaves_close": "maple_yard", "branches_up": "maple_yard",
    "maple_trunk": "maple_yard", "open_window_golden": "maple_yard",
    "kitchen_dawn": "kitchen", "kitchen_table": "kitchen", "kitchen_warm": "kitchen", "kitchen_spring": "kitchen",
    "kitchen_two_mugs": "kitchen", "kitchen_night": "kitchen", "cupboard": "kitchen",
    "window_ramp": "kitchen", "window_dusk": "kitchen", "snowy_sill": "kitchen",
}


def scene_dir(location, root=ROOT, scenes_root=None):
    base = scenes_root or os.path.join(root, "assets", "scenes")
    for name in (location, SCENE_ALIASES.get(location)):
        if name and os.path.exists(os.path.join(base, name, "scene.json")):
            return os.path.join(base, name)
    return None


def load_plates(folder, painter, seed=1, fx_override=None):
    """Use the plate loader that ships with the plates (assets/scenes/load_scene.py)."""
    import importlib.util
    path = os.path.join(os.path.dirname(folder), "load_scene.py")
    spec = importlib.util.spec_from_file_location("heartwick_load_scene", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_scene(folder, painter, seed=seed, fx_override=fx_override)


def scene_files(locations, root=ROOT):
    """{zip/dataset path: local path} for the plates a story uses (shipped to the render kernel as scenes/<folder>/)."""
    out = {}
    for loc in locations:
        d = scene_dir(loc, root)
        if d:
            for f in os.listdir(d):
                if not f.startswith("__"):
                    out[f"scenes/{os.path.basename(d)}/{f}"] = os.path.join(d, f)
    if out:
        out["scenes/load_scene.py"] = os.path.join(root, "assets", "scenes", "load_scene.py")
    return out
