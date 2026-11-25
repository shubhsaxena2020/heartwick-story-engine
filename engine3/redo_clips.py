"""Animate chosen shots in their own GPU kernel beside a running clips kernel, without touching state.json (the main
pipeline process owns it while it runs). Result -> clips_redo.json; merge it afterwards with --merge.
  python -m engine3.redo_clips out/pip_short_v3 S08,S09,S10,S11 --keys K8=K8_s7.png,K9=K9_s3.png,...
  python -m engine3.redo_clips out/pip_short_v3 --merge        # once the main pipeline has stopped"""
import json, os, sys, time
from . import kaggle3 as K, stages as ST
from .pipeline import State, _od


def main():
    out = sys.argv[1]
    st = State(out)
    side = os.path.join(out, "clips_redo.json")
    ks = os.path.basename(out).rsplit("_v3", 1)[0].replace("_", "-")
    if "--merge" in sys.argv:
        r = json.load(open(side))
        chosen = dict(st.get("keys", "chosen") or {}); chosen.update(r["keys"])
        srcs = list(dict.fromkeys(list(st.get("keys", "sources") or []) + r["key_sources"]))
        st.set("keys", chosen=chosen, sources=srcs)
        cs = dict(st.get("clips", "sources") or {}); cs.update({sid: r["slug"] for sid in r["made"]})
        missing = [s["id"] for s in json.load(open(os.path.join(out, "shots.json"), encoding="utf-8"))["shots"] if s["id"] not in cs]
        st.set("clips", sources=cs, missing=missing, status="done" if not missing else "failed")
        st.set("assemble", status="pending")     # the next `make` re-assembles with the new shots
        print("merged:", r["made"], "missing", missing); return
    only = set(sys.argv[2].split(","))
    keys = dict(x.split("=") for x in sys.argv[sys.argv.index("--keys") + 1].split(","))
    shots = json.load(open(os.path.join(out, "shots.json"), encoding="utf-8"))
    tl = json.load(open(os.path.join(out, "timeline.json")))
    chosen = dict(st.get("keys", "chosen") or {}); chosen.update(keys)
    hist = json.load(open(os.path.join(out, "keys_redo.json")))
    ksrc = list(dict.fromkeys(list(st.get("keys", "sources") or []) + [h["slug"] for h in hist]))
    plan, script = ST.build_clips(shots, tl, chosen, only=only)
    slug = f"{K.PREFIX}clips-{ks}-fix{int(time.time()) % 100000}"
    while K.gpu_busy()[1] >= 2:
        print(K.now(), "account at 2 GPU kernels; waiting", flush=True); time.sleep(120)
    print("pushing", slug, sorted(only), [(c["shot"], c["start_file"], c.get("end_file")) for c in plan["clips"]], flush=True)
    rec = K.push(slug, script, True, kernel_sources=[f"{K.USER}/{s}" for s in ksrc], note=f"redo shots {sorted(only)}",
                 wait_slot=False)
    fin = K.wait(rec)
    dl = _od(K.download(slug, os.path.join(out, "kaggle", slug.replace(K.PREFIX, ""))))
    stt = json.load(open(os.path.join(dl, "status.json"))) if os.path.exists(os.path.join(dl, "status.json")) else {}
    made = [f[:-4] for f in (stt.get("finish", {}).get("done") or [])]
    json.dump({"slug": slug, "made": made, "keys": keys, "key_sources": ksrc, "kaggle_state": fin.get("state")},
              open(side, "w"), indent=1)
    print("redo clips done:", made, fin.get("state"))


if __name__ == "__main__":
    main()
