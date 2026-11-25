"""Round-5 LA test: 5 shots (4 keys + 1 acting-clip tail) on 2 parallel Kaggle CPU kernels via engine4.la.render_la.
usage: python research/round5_la/run_test.py <round tag, e.g. r1>"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from engine4.la import render_la  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "r1"
FW = os.path.join(ROOT, "research/round4/results/fw2/out/keys")
V3 = os.path.join(ROOT, "out/pip_short_v3/kaggle")
JOBS = [
    {"id": "G1_garden_push", "image": f"{FW}/new_garden_1088_s21.png", "dur": 6.0, "crop": None,
     "cam": {"type": "push_in", "amount": 0.09}, "fx": ["leaves", "light"], "char_motion": "breath_sway"},
    {"id": "W1_window_pan", "image": f"{FW}/edit_closeup_K10_s3.png", "dur": 5.5, "crop": [0.2, 0.1, 1.0, 0.75],
     "cam": {"type": "pan_right", "amount": 0.08}, "fx": ["dust@0.0,0.55,0.65,1.0"], "char_motion": "breath"},
    {"id": "K2_cold_panleft", "image": f"{V3}/keys/out/keys/K2_s21.png", "dur": 6.5, "crop": None,
     "cam": {"type": "pan_left", "amount": 0.08}, "fx": ["snow", "steam@0.78,0.52"], "char_motion": "breath"},
    {"id": "KW_kitchen_pull", "image": f"{FW}/new_window_768_s21.png", "dur": 5.0, "crop": None,
     "cam": {"type": "pull_out", "amount": 0.08}, "fx": ["rain@0.48,0.1,0.97,0.46", "steam@0.51,0.62", "light"],
     "char_motion": "breath_sway", "blink": True},
    {"id": "T1_tail_S02", "tail_of": f"{V3}/clips-pip-short/out/clips/S02_s7.mp4", "dur": 6.5,
     "cam": {"type": "push_in", "amount": 0.05}, "fx": ["snow", "steam@0.76,0.56"], "char_motion": "breath"},
]
out = os.path.join(HERE, tag)
res = render_la(JOBS, out, n_kernels=2, tag="r1")   # same kernels/dataset every round; outputs per round dir
summ = {k: {kk: (v["record"].get(kk) if k != "_kernels" else None) for kk in ("ok", "frames", "frames_probe", "render_s",
        "ms_per_frame", "cam_move", "gate", "error")} for k, v in res.items() if k != "_kernels"}
print(json.dumps(summ, indent=1, default=str))
print(json.dumps(res["_kernels"], indent=1, default=str))
