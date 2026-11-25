"""Re-make chosen keys in their own GPU kernel (fresh seeds = unique file names, so a clips kernel that mounts every
keys kernel never picks the old file). Approved keys the redo references ride along as inputs.
  python -m engine3.redo_keys out/pip_short_v3 K8,K9,K10,K11 [--seeds 5,33]
Then: python heartwick.py make stories/pip_short.md --engine v3 --redo <shots using them>"""
import json, os, sys
from . import kaggle3 as K, stages as ST
from .pipeline import State, _od


def main():
    out, ids = sys.argv[1], set(sys.argv[2].split(","))
    seeds = [int(x) for x in (sys.argv[sys.argv.index("--seeds") + 1] if "--seeds" in sys.argv else "5,33").split(",")]
    st = State(out); shots = json.load(open(os.path.join(out, "shots.json"), encoding="utf-8"))
    # --from K8=K6: paint K8 from approved key K6 instead of the character sheets (24 Nov: keys painted straight
    # from the 3-view sheets came out with a 2nd Nell / extra squirrels; keys chained from a clean key did not)
    opt = lambda name: dict(x.split("=") for x in sys.argv[sys.argv.index(name) + 1].split(",")) if name in sys.argv else {}
    for k in shots["keys"]:
        if k["id"] in opt("--from"):
            k["from"] = opt("--from")[k["id"]]
            k.setdefault("change", k["image"])
    for k in shots["keys"]:     # --nopose K11: no pose skeleton (24 Nov: in the window scenes it was painted as a person)
        if "--nopose" in sys.argv and k["id"] in sys.argv[sys.argv.index("--nopose") + 1].split(","):
            k["pose"] = "none"
    use = opt("--use")          # --use K9=K9_s3.png: take that file (from any keys kernel) as the approved key
    ks = os.path.basename(out).rsplit("_v3", 1)[0].replace("_", "-")
    srcs = list(st.get("keys", "sources") or [])
    kdir = lambda s: os.path.join(out, "kaggle", "keys" if s == f"{K.PREFIX}keys-{ks}" else s.replace(K.PREFIX, ""))
    chosen = dict(st.get("keys", "chosen") or {})
    redo_prev = os.path.join(out, "keys_redo.json")
    hist = json.load(open(redo_prev)) if os.path.exists(redo_prev) else []
    hist = hist if isinstance(hist, list) else [hist]
    for rp in hist:                        # earlier redo kernels are sources too
        if rp["slug"] not in srcs:
            srcs.append(rp["slug"])
    chosen.update(use)
    _kp = ST.key_prompt
    ST.key_prompt = lambda k, keys: dict(_kp(k, keys), seeds=seeds if "nell" in k.get("chars", []) else seeds[:1])
    jobs = ST.keys_jobs(shots, only=ids)
    used = {r[1:] for j in jobs["keys"] for r in j["refs"] if r.startswith("@") and not r.startswith("@IN@")} - ids
    files = {}
    for kid in used:
        for s in srcs:
            f = os.path.join(_od(kdir(s)), "keys", chosen[kid])
            if os.path.exists(f):
                files[kid] = f
    slug = f"{K.PREFIX}keys-{ks}-r{len(srcs)}"
    script = ST.build_keys(shots, music_s=None, given=files, only=ids)
    print("pushing", slug, "for", sorted(ids), "seeds", seeds, "given", sorted(files), len(script), "bytes")
    # Kaggle allows 2 GPU sessions: a redo may run beside the clips kernel (engine v3's push waits for none of its
    # own); still never a 3rd
    import time
    while K.gpu_busy()[1] >= 2:
        print(K.now(), "account at 2 GPU kernels; waiting", flush=True); time.sleep(120)
    rec = K.push(slug, script, True, note=f"redo keys {sorted(ids)}", wait_slot=False)
    fin = K.wait(rec)
    dl = _od(K.download(slug, kdir(slug)))
    r = json.load(open(os.path.join(dl, "keys", "keys_rec.json")))
    got = {k: v for k, v in r.get("chosen", {}).items() if k in ids}
    # the main pipeline process keeps state.json in memory and would overwrite an edit here: the result goes to
    # keys_redo.json and is merged into the keys stage (chosen + sources) before the --redo clips run
    hist.append({"slug": slug, "chosen": got, "use": use, "kaggle_state": fin.get("state")})
    json.dump(hist, open(redo_prev, "w"), indent=1)
    print("redo done:", got, "state", fin.get("state"))


if __name__ == "__main__":
    main()
