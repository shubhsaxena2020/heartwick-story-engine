"""Pose-skeleton library for engine v3 keys (OpenPose-18 drawings, 480x832). The three round-2 poses are reused from
research/round2/quality/tools/make_poses.py (stand_look, kneel_reach, kneel_hold); stand_walk, sit_table and
stand_window are blocked the same way. Coordinates are fractions of the 9:16 frame; the figure faces right."""
import importlib.util, os

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "make_poses", os.path.join(_HERE, "..", "research", "round2", "quality", "tools", "make_poses.py"))
_mp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mp)

POSES = dict(_mp.POSES)
POSES.update({
    # mid-stride walking right, arms swinging gently
    "stand_walk": {"nose": (0.47, 0.215), "neck": (0.45, 0.27), "Rsho": (0.41, 0.29), "Relb": (0.37, 0.39), "Rwri": (0.36, 0.48),
                   "Lsho": (0.49, 0.29), "Lelb": (0.53, 0.39), "Lwri": (0.56, 0.47), "Rhip": (0.43, 0.52), "Rkne": (0.38, 0.68),
                   "Rank": (0.33, 0.85), "Lhip": (0.47, 0.52), "Lkne": (0.52, 0.68), "Lank": (0.55, 0.86)},
    # seated at a table (table edge ~y 0.62), forearms on the table
    "sit_table": {"nose": (0.50, 0.30), "neck": (0.47, 0.36), "Rsho": (0.42, 0.38), "Relb": (0.44, 0.52), "Rwri": (0.55, 0.58),
                  "Lsho": (0.52, 0.38), "Lelb": (0.56, 0.52), "Lwri": (0.64, 0.58), "Rhip": (0.44, 0.64), "Rkne": (0.56, 0.66),
                  "Rank": (0.56, 0.84), "Lhip": (0.49, 0.64), "Lkne": (0.61, 0.66), "Lank": (0.61, 0.85)},
    # standing at a window on the right, both hands resting on the sill (y ~0.50)
    "stand_window": {"nose": (0.46, 0.22), "neck": (0.42, 0.28), "Rsho": (0.37, 0.30), "Relb": (0.42, 0.41), "Rwri": (0.55, 0.49),
                     "Lsho": (0.47, 0.30), "Lelb": (0.52, 0.41), "Lwri": (0.62, 0.49), "Rhip": (0.39, 0.54), "Rkne": (0.40, 0.70),
                     "Rank": (0.39, 0.87), "Lhip": (0.45, 0.54), "Lkne": (0.46, 0.70), "Lank": (0.46, 0.87)},
})


def draw_all(outdir):
    os.makedirs(outdir, exist_ok=True)
    for k, p in POSES.items():
        _mp.draw(p, os.path.join(outdir, f"pose_{k}.png"))
    return sorted(POSES)
