"""Refine loop: draft -> critique -> fix -> re-render, then ONE final render of the best pass.

    python tools/refine.py stories/walk_test.md --art v1 --name walk_test_A --passes 5 [--final --parts 1]
    python tools/refine.py --baseline out/pip_short_B/pip_short_B_sheet.jpg out/pip_short_B/pip_short_B_report.json --name pip_short_B_baseline

Every pass renders a 540x960 / 30 fps preview on a Kaggle CPU kernel (no GPU queue), then:
  automatic checks  foot slide, longest still stretch per character, motion density (reference method), cuts in pauses,
                    loudness, near-black / blown frames
  vision critique   the 0.25 s contact sheet scored by a vision model (local 9Router, cc/claude-sonnet-5-5) against a fixed
                    rubric, returning bounded parameter fixes (engine2.direct.TUNE keys)
History: out/<name>/passes/pass_N/{sheet.jpg, report.json, critique.json, params.json, score.json}; scores.csv per video.
"""
import argparse, base64, csv, io, json, os, shutil, sys, time, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine2 import pipeline  # noqa: E402
from engine2.direct import TUNE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUBRIC = ["motion_naturalness", "walk_quality", "fluidity", "pose_expressiveness", "follow_through", "camera_dynamism",
          "composition", "readability", "art_consistency", "artifacts"]
BOUNDS = {"every_min": (0.45, 1.0), "every_max": (0.7, 1.4), "trans_min": (0.1, 0.3), "trans_max": (0.18, 0.45), "overshoot": (0.0, 2.2),
          "overlay_p": (0.2, 1.0), "char_scale": (0.75, 1.25), "handheld": (0.0, 12.0), "pan": (0.6, 1.8), "cadence": (0.85, 1.2),
          "key": (0.4, 1.8), "rim": (0.3, 1.8), "bloom": (0.3, 1.8), "grain": (0.0, 0.14)}
PROMPT = """You are the animation director reviewing a contact sheet from a vertical (9:16) animated storybook short.
Frames are {every} s apart, left to right, top to bottom; timestamps are printed on each frame. The quality bar is high-quality
anime acting at 60 fps: strong expressive key poses, fluid in-betweens, anticipation and overshoot, hair/cloth follow-through,
dynamic camera (push-ins, pans, slight handheld), real walking with planted feet and natural travel speed, characters never frozen.
Measured by the engine: {metrics}

Score each item 0-10 (10 = indistinguishable from a top anime studio short): {rubric}.
Then propose concrete fixes ONLY as changes to these engine parameters (current values given), within their ranges:
{params}
Meanings: every_min/every_max = seconds between pose changes; trans_min/trans_max = pose transition seconds; overshoot = anticipation/
overshoot strength; overlay_p = probability of a nod/glance inside a hold; char_scale = character size; handheld = camera drift px;
pan = camera move speed multiplier; cadence = walk cadence multiplier; key/rim/bloom = lighting strengths; grain = film grain.
Reply with ONLY a JSON object: {{"scores": {{...}}, "problems": ["..."], "fixes": {{"param": value, ...}}}}"""


def router(messages, model="cc/claude-sonnet-5-5", max_tokens=900):
    key = os.environ.get("NINEROUTER_API_KEY", "")
    body = {"model": model, "stream": False, "max_tokens": max_tokens, "messages": messages}
    req = urllib.request.Request("http://127.0.0.1:20128/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    return r["choices"][0]["message"]["content"]


def _img_b64(path, max_side=1568):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=82)
    return base64.b64encode(b.getvalue()).decode()


def _json(text):
    s, e = text.find("{"), text.rfind("}")
    return json.loads(text[s:e + 1])


def critique(sheet, metrics, params, every):
    msg = PROMPT.format(every=every, metrics=json.dumps(metrics), rubric=", ".join(RUBRIC),
                        params=json.dumps({k: [round(params[k], 3), list(BOUNDS[k])] for k in BOUNDS}))
    for attempt in range(3):
        try:
            txt = router([{"role": "user", "content": [{"type": "text", "text": msg},
                                                       {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _img_b64(sheet)}}]}])
            out = _json(txt)
            out["scores"] = {k: float(out.get("scores", {}).get(k, 0)) for k in RUBRIC}
            return out
        except Exception as e:  # noqa: BLE001
            err = repr(e)[:300]
            time.sleep(5)
    return {"scores": {k: 0.0 for k in RUBRIC}, "problems": [f"critique failed: {err}"], "fixes": {}}


def frame_checks(sheet):
    """Near-black / blown tiles in the contact sheet (each tile ~216 px wide)."""
    import cv2
    im = cv2.imread(sheet)
    if im is None:
        return {}
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    th, tw = 384, 220
    tiles = [g[r:r + th - 4, c:c + tw - 4] for r in range(0, g.shape[0] - th + 1, th) for c in range(0, g.shape[1] - tw + 1, tw)]
    tiles = [t for t in tiles if t.size and t.std() > 1]
    return {"near_black_tiles": int(sum(t.mean() < 18 for t in tiles)), "blown_tiles": int(sum((t > 250).mean() > 0.25 for t in tiles)), "tiles": len(tiles)}


def metrics_of(rep, checks):
    qa = rep.get("motion_qa") or []
    slides = [m["foot_slide_px"] for m in qa if m.get("foot_slide_px") is not None]
    speeds = [m["walk_speed_h_s"] for m in qa if m.get("walk_speed_h_s") is not None]
    stills = [m["max_still_s"] for m in qa if m.get("max_still_s") is not None]
    return {"motion_seconds_pct": rep.get("motion", {}).get("motion_seconds_pct"), "character_motion_seconds_pct": rep.get("character_motion_seconds_pct"),
            "foot_slide_px_max": max(slides) if slides else None, "walk_speed_h_s": speeds, "max_still_s": max(stills) if stills else None,
            "cuts_in_pauses": rep.get("cuts_in_pauses"), "shots_per_min": rep.get("shots_per_min"),
            "lufs": (rep.get("loudness_ffmpeg") or {}).get("lufs", (rep.get("audio") or {}).get("lufs")), **checks}


def score_of(crit, m):
    s = float(np.mean([crit["scores"][k] for k in RUBRIC]))
    pen = 0.0
    if m.get("foot_slide_px_max") is not None and m["foot_slide_px_max"] > 0.5:
        pen += 1.0
    if m.get("max_still_s") is not None and m["max_still_s"] > 0.8:
        pen += 0.3 * min(3.0, (m["max_still_s"] - 0.8) / 0.4)
    if (m.get("motion_seconds_pct") or 100) < 65:
        pen += 1.0
    if m.get("lufs") is not None and abs(m["lufs"] + 14) > 0.6:
        pen += 0.5
    pen += 0.3 * (m.get("near_black_tiles", 0) > 2) + 0.3 * (m.get("blown_tiles", 0) > 2)
    return round(s - pen, 3)


def apply_fixes(params, fixes):
    p = dict(params)
    for k, v in (fixes or {}).items():
        if k in BOUNDS:
            try:
                lo, hi = BOUNDS[k]
                p[k] = float(min(hi, max(lo, float(v))))
            except (TypeError, ValueError):
                pass
    p["every_max"] = max(p["every_max"], p["every_min"] + 0.15)
    p["trans_max"] = max(p["trans_max"], p["trans_min"] + 0.05)
    return p


def write_scores(base, rows):
    with open(os.path.join(base, "scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def refine(story, art, style, name, passes, final, parts, every=0.25, start=None):
    ctx = pipeline.prepare(story, style=style, art=art)
    base = os.path.join(ROOT, "out", name, "passes")
    os.makedirs(base, exist_ok=True)
    params = dict(TUNE)
    params.update(start or {})
    rows, best = [], None
    for n in range(1, passes + 1):
        pd = os.path.join(base, f"pass_{n}")
        os.makedirs(pd, exist_ok=True)
        pname = f"{name}_pass{n}"
        t0 = time.time()
        res = pipeline.render_job(ctx, pname, overrides={"tune": params}, preview=True, sheet_every=every)
        wall = time.time() - t0
        rep_p = os.path.join(res["dest"], f"{pname}_report.json")
        sheet = os.path.join(res["dest"], f"{pname}_sheet.jpg")
        if not os.path.exists(rep_p):
            print("pass", n, "failed to render", res)
            continue
        rep = json.load(open(rep_p))
        m = metrics_of(rep, frame_checks(sheet))
        crit = critique(sheet, m, params, every)
        sc = score_of(crit, m)
        for f, dst in ((sheet, "sheet.jpg"), (rep_p, "report.json")):
            shutil.copy(f, os.path.join(pd, dst))
        json.dump(crit, open(os.path.join(pd, "critique.json"), "w"), indent=1)
        json.dump(params, open(os.path.join(pd, "params.json"), "w"), indent=1)
        json.dump({"score": sc, "metrics": m, "kaggle_wall_s": round(wall, 1), "render_s": rep.get("render_s")}, open(os.path.join(pd, "score.json"), "w"), indent=1)
        row = {"pass": n, "score": sc, **{k: crit["scores"][k] for k in RUBRIC}, "foot_slide_px": m.get("foot_slide_px_max"),
               "max_still_s": m.get("max_still_s"), "motion_pct": m.get("motion_seconds_pct"), "kaggle_wall_s": round(wall, 1),
               "render_cpu_s": rep.get("render_s")}
        rows.append(row)
        write_scores(base, rows)
        print(f"pass {n}: score {sc}  {crit.get('problems', [])[:3]}  fixes {crit.get('fixes')}", flush=True)
        if best is None or sc > best[0]:
            best = (sc, n, dict(params))
        params = apply_fixes(params, crit.get("fixes"))
    if best is None:
        sys.exit("no pass rendered")
    json.dump({"best_pass": best[1], "score": best[0], "params": best[2]}, open(os.path.join(base, "best.json"), "w"), indent=1)
    print("best pass", best[1], "score", best[0])
    if final:  # ONE final render of the best pass: 1080x1920, 60 fps, supersampled, high bitrate
        res = pipeline.render_job(ctx, name, overrides={"tune": best[2]}, preview=False, parts=parts, sheet_every=0.25)
        print("final:", res)
    return best


def baseline(sheet, report, name, every):
    """Score an existing render (v1 pilot, earlier v2) with the same critique, for before/after numbers."""
    rep = json.load(open(report)) if report and os.path.exists(report) else {}
    m = metrics_of(rep, frame_checks(sheet))
    crit = critique(sheet, m, dict(TUNE), every)
    sc = score_of(crit, m)
    d = os.path.join(ROOT, "out", name)
    os.makedirs(d, exist_ok=True)
    json.dump({"score": sc, "metrics": m, "critique": crit, "sheet": sheet}, open(os.path.join(d, "baseline.json"), "w"), indent=1)
    print(name, "baseline score", sc, crit["scores"])
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("story", nargs="?")
    ap.add_argument("--art", default="c")
    ap.add_argument("--style", default="C")
    ap.add_argument("--name")
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--every", type=float, default=0.25)
    ap.add_argument("--baseline", nargs=2, metavar=("SHEET", "REPORT"))
    a = ap.parse_args()
    if a.baseline:
        baseline(a.baseline[0], a.baseline[1], a.name, a.every)
    else:
        refine(a.story, a.art, a.style, a.name, a.passes, a.final, a.parts, a.every)


if __name__ == "__main__":
    main()
