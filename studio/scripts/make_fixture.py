"""Build studio/fixtures/*.json for local UI work from the laptop's real snapshot (dashboard/data/snapshot.json).

  python studio/scripts/make_fixture.py

sample-snapshot.json   the real snapshot plus every state the UI must draw: an approved and a needs-changes video,
                       a running and a failed Kaggle job, a pending dashboard job, a research report and sample.
empty-snapshot.json    schema 1 with nothing in it (first start, no data).
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "dashboard", "data", "snapshot.json")
OUT = os.path.join(os.path.dirname(HERE), "fixtures")

REPORT = """# Heartwick research report (fixture)

A **fixture** standing in for `research/RESEARCH_REPORT.md`, which another agent is writing.

## Voice
| Voice | Warmth | Clarity | Notes |
|---|---|---|---|
| af_heart | high | high | current default |
| af_bella | medium | high | brighter |

## Next steps
1. Compare three voices on the same 30-second passage.
2. Keep loudness at -15 LUFS.

> The kindest love leaves the window open.
"""


def main():
    s = json.load(open(SRC, encoding="utf-8"))
    os.makedirs(OUT, exist_ok=True)
    f = copy.deepcopy(s)
    vids = {v["name"]: v for v in f["videos"]}
    if "pip_pilot_v0" in vids:
        v = vids["pip_pilot_v0"]
        v["stage"] = "approved"
        v["reached"]["approved"] = v["reached"]["review"]
        v["approval"] = {"status": "approved", "note": "Lovely. Upload Sunday.", "at": v["reached"]["review"], "id": "fixture-1"}
    if "pip_pilot_v1" in vids:
        v = vids["pip_pilot_v1"]
        v["approval"] = {"status": "needs_changes", "note": "Captions a little fast in the storm part.", "at": v["reached"]["review"], "id": "fixture-2"}
    f["counts"] = {st: sum(v["stage"] == st for v in f["videos"]) for st in f["stages"]}
    jobs = f["kaggle"]["jobs"]
    if jobs:
        run = copy.deepcopy(jobs[0])
        run.update(kernel="heartwick-gpu-fixture-running", state="running", wait_min=34.0, run_min=6.5, wall_min=None, worker_stages=None, video=None, story="blue_mitten")
        bad = copy.deepcopy(jobs[-1])
        bad.update(kernel="heartwick-gpu-fixture-error", state="error", worker_errors=["images: CUDA out of memory (fixture)"])
        f["kaggle"]["jobs"] = [run] + jobs + [bad]
    f["queue"]["jobs"] = [{"id": "fixture-q1", "label": "Queue short video of blue_mitten (blue_mitten_short)", "args": ["make"], "status": "pending",
                           "added_at": f["generated_at"], "source": "dashboard"}]
    f["research"] = {"present": True, "report_md": REPORT, "report_truncated": False, "report_mtime": f["generated_at"],
                     "samples": [x for v in f["videos"] for x in v.get("samples", [])][:2], "other_docs": []}
    f["actions"] = {"applied": [{"id": "fixture-1", "kind": "approve", "video": "pip_pilot_v0", "at": f["generated_at"], "applied_at": f["generated_at"],
                                 "via": "vercel inbox", "result": "ok", "msg": "pip_pilot_v0 -> approved"}]}
    json.dump(f, open(os.path.join(OUT, "sample-snapshot.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    empty = {"schema": 1, "generated_at": s["generated_at"], "stages": s["stages"], "counts": {}, "stories": [], "videos": [],
             "kaggle": {"jobs": []}, "queue": {}, "research": {"present": False}, "channel": {"exists": False}, "events": [], "truth": None}
    json.dump(empty, open(os.path.join(OUT, "empty-snapshot.json"), "w", encoding="utf-8"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
