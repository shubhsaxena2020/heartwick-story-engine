"""Engine v3 post stage (laptop, light): SEO pack (LLM via 9Router, with a deterministic fallback), review sheets copied
next to the video, GPU minutes per stage folded into qa_report.json, and worker_log.json for the dashboard."""
import glob, json, os, shutil

from . import shots as SH


def seo_pack(meta, shots):
    narration = " ".join(s["narration"] for s in shots["shots"])
    base = meta.get("seo") or {}
    try:
        txt = SH._chat([{"role": "system", "content": "You write YouTube Shorts SEO for Heartwick, a channel of warm animated "
                         "story Shorts for adult US viewers (not made for kids). Reply with JSON only."},
                        {"role": "user", "content": "Story narration:\n" + narration + "\n\nExisting notes: " + json.dumps(base) +
                         '\n\nReturn {"titles": [5 titles under 70 chars, the best first], "description": "3 short lines + a '
                         'question to the viewer", "tags": [15 tags], "hashtags": [3-5 hashtags incl. #shorts], '
                         '"pinned_comment": str, "filename": "kebab-case.mp4"}'}], max_tokens=16000)
        d = SH._json(txt)
    except Exception as e:
        d = {"titles": [base.get("shorts_title") or meta.get("title")], "description": base.get("summary", ""),
             "tags": [], "hashtags": base.get("hashtags", []), "error": repr(e)[:200]}
    d["made_for_kids"] = False
    return d


def finish(out, name, meta, shots, tl, st):
    seo = seo_pack(meta, shots)
    json.dump(seo, open(os.path.join(out, "SEO_PACK.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    md = [f"# {seo['titles'][0]}", "", "## Title options"] + [f"- {t}" for t in seo.get("titles", [])] + \
         ["", "## Description", seo.get("description", ""), "", "## Tags", ", ".join(seo.get("tags", [])), "",
          "## Hashtags", " ".join(seo.get("hashtags", [])), "", "## Pinned comment", seo.get("pinned_comment", ""), "",
          "Made for kids: **No** (adult audience)."]
    open(os.path.join(out, "SEO_PACK.md"), "w", encoding="utf-8").write("\n".join(md))
    kg = os.path.join(out, "kaggle")
    for pat, dst in (("keys/out/keys/keys_chosen.jpg", "review_keys_chosen.jpg"), ("keys/out/keys/keys_candidates.jpg", "review_keys_candidates.jpg"),
                     ("clips-*/out/clips_sheet_*.jpg", None), ("clips-*/out/clips_qa_sheet.jpg", None)):
        for p in sorted(glob.glob(os.path.join(kg, pat))):
            d = dst or ("review_" + os.path.basename(os.path.dirname(os.path.dirname(p))) + "_" + os.path.basename(p))
            shutil.copy(p, os.path.join(out, d))
    qp = os.path.join(out, "qa_report.json")
    qa = json.load(open(qp)) if os.path.exists(qp) else {}
    stages = {k: {"gpu": v.get("gpu"), "run_min": v.get("run_min"), "wall_min": v.get("wall_min")}
              for k, v in st.d["stages"].items() if v.get("kernel")}
    qa["engine"] = "v3"; qa["kaggle_stages"] = stages
    qa["gpu_min_total"] = round(sum((v["run_min"] or 0) for v in stages.values() if v["gpu"]), 1)
    qa["duration_s"] = qa.get("duration_s")
    if "audio" in qa:
        qa["audio"]["integrated_lufs"] = qa["audio"].get("delivered_lufs", qa["audio"].get("lufs"))
        qa["audio"]["true_peak_dbfs"] = qa["audio"].get("true_peak_db")
    json.dump(qa, open(qp, "w"), indent=1)
    json.dump({"stages": {k: round((v["run_min"] or 0) * 60, 1) for k, v in stages.items()}, "errors": [],
               "total_s": round(sum((v["wall_min"] or 0) for v in stages.values()) * 60, 1)},
              open(os.path.join(out, "worker_log.json"), "w"), indent=1)
