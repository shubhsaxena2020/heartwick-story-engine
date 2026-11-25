"""Round 3: merge GPU timings (rec_*.json), QA (qa.json), critic (critic.json) into results.csv + WINNERS.json.
usage: python aggregate.py   (reads research/round3/results/{s1,s2}, writes research/round3/results.csv, WINNERS.json)"""
import csv, glob, json, os, time

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(R, "results")

# id -> (axis list, human setting). Baseline = Q5_K_M, 480x832, 4 steps, plain I2V, round-1 Z-Image key, watercolour, seed 7.
VAR = {
    "q5_480_4s": (["baseline", "precision", "resolution", "steps", "mode"], "BASE: Q5_K_M, 480x832, 4 steps, plain I2V, round-1 prompt"),
    "r2_q4_480": (["precision"], "Q4_K_M (round 2 run, same key/prompt/seed)"),
    "r2_q5_480": (["repro"], "Q5_K_M round-2 run of the baseline (reproducibility)"),
    "r2_q5_480_s11": (["seed"], "baseline settings, seed 11 (round 2)"),
    "q8_480_4s": (["precision"], "Q8_0 sharded over both T4s"),
    "q5_544_4s": (["resolution"], "544x960"),
    "r2_q4_544": (["resolution"], "544x960 Q4 (round 2)"),
    "q5_720_4s": (["resolution"], "720x1280 sharded over both T4s"),
    "q5_480_6s": (["steps"], "6 steps"),
    "q5_480_8s": (["steps"], "8 steps"),
    "i2v_v3": (["mode", "prompt"], "plain I2V, improved prompt (one knee, Pip stays small)"),
    "flf_qp": (["mode", "keys"], "FLF2V, end key = Qwen-Edit chain + OpenPose"),
    "flf_qp_s11": (["seed"], "FLF2V Qwen-Edit+pose, seed 11"),
    "flf_qe": (["keys"], "FLF2V, end key = Qwen-Edit chain (no pose)"),
    "flf_qd": (["keys"], "FLF2V, end key = one direct Qwen-Edit + pose"),
    "flf_zi": (["keys"], "FLF2V, end key = Z-Image-Turbo alone (text only)"),
    "flf_jt": (["style"], "FLF2V, jewel-tone keys (restyled QP pair)"),
    "r1_wan22_i2v": (["reference"], "ROUND-1 reference clip (ZeroGPU, Lightning 6-step fp8)"),
}


def load(pat):
    out = {}
    for f in glob.glob(os.path.join(RES, "*", pat)):
        try:
            out.update(json.load(open(f)))
        except Exception:
            pass
    return out


def recs():
    out = {}
    for f in glob.glob(os.path.join(RES, "*", "rec_*.json")):
        d = json.load(open(f))
        for cid, c in d.get("clips", {}).items():
            out[cid] = {"gpu_s": c.get("gpu_s_total"), "gpus": c.get("gpus_used", 1), "steps": len(c.get("sigmas", [0] * 5)) - 1,
                        "peak": {k: v for k, v in d.get("phases", {}).items() if k.startswith("peak_vram")}}
    return out


def row(cid, qa, cr, rc):
    q = qa.get(f"{cid}.mp4", {}).get("summary", {})
    c = cr.get(f"{cid}_sheet.jpg", {})
    g = rc.get(cid, {})
    ep = q.get("end_pose", [])
    labels = [p.get("label") for p in ep]
    runs = [r for r in c.get("runs", []) if "scores" in r]
    r = {"id": cid, "axes": "+".join(VAR.get(cid, ([], ""))[0]), "setting": VAR.get(cid, ([], cid))[1],
         "critic_total": c.get("total"), **{f"c_{k}": v for k, v in c.get("mean", {}).items()},
         "critic_kneel": "/".join(str(x.get("kneel_one_knee")) for x in runs), "critic_pip": "/".join(str(x.get("pip_same_size")) for x in runs),
         "qa_body_sim_med": q.get("body_sim", {}).get("median"), "qa_head_sim_med": q.get("head_sim", {}).get("median"),
         "qa_head_sim_min": q.get("head_sim", {}).get("min"), "qa_pip_max_over_min": q.get("pip_scale", {}).get("max_over_min"),
         "qa_pip_growth": q.get("pip_growth_last_over_first"), "qa_headroom_min": q.get("headroom_min"), "qa_frames_cut": q.get("frames_cut"),
         "qa_end_pose": "/".join(str(x) for x in labels), "qa_dist_kneel": min([p.get("dist_kneel_hold", 9) for p in ep] or [None]) if ep else None,
         "gpu_s_clip": g.get("gpu_s"), "gpus": g.get("gpus"),
         "t4_minutes_per_clip": round(g["gpu_s"] * g.get("gpus", 1) / 60, 1) if g.get("gpu_s") else None}
    # composite: critic mean (0-10) with QA penalties for the three measurable flaws
    s = r["critic_total"]
    if s is not None:
        pg = r["qa_pip_growth"] or r["qa_pip_max_over_min"]
        if pg and pg > 1.2: s -= min(1.5, (pg - 1.2) * 2)
        if r["qa_frames_cut"]: s -= 0.15 * r["qa_frames_cut"]
        if "yes" in r["critic_kneel"]: s += 0.3 * r["critic_kneel"].count("yes") / max(1, len(runs))
        r["composite"] = round(s, 2)
    return r


def main():
    qa, cr, rc = load("qa.json"), load("critic.json"), recs()
    ids = [k for k in VAR if f"{k}.mp4" in qa or f"{k}_sheet.jpg" in cr]
    rows = [row(i, qa, cr, rc) for i in ids]
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(R, "results.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    json.dump(rows, open(os.path.join(R, "results.json"), "w"), indent=1)
    print(json.dumps([{k: r.get(k) for k in ("id", "composite", "critic_total", "qa_pip_growth", "qa_frames_cut", "qa_end_pose", "critic_kneel", "gpu_s_clip")} for r in rows], indent=0))


if __name__ == "__main__":
    main()
