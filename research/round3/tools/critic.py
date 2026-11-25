"""Round 3 vision critic: scores a clip's contact sheet (8 evenly spaced frames, left to right) against the master key
with a FIXED rubric, through the local 9Router (stream false, cc/claude-sonnet-5-5). Two independent calls per sheet,
averaged; the spread is kept so we can see when the critic itself is noise.
usage: python critic.py --ref K0.png --sheets a_sheet.jpg b_sheet.jpg ... --out critic.json [--runs 2] [--keys]"""
import argparse, base64, io, json, os, sys, time, urllib.request
from PIL import Image

RUBRIC = ["motion_naturalness", "identity", "anatomy", "framing", "art_quality", "artifacts"]
CLIP_PROMPT = """You are a strict animation director judging ONE generated 5-second shot for a vertical 9:16 animated story Short.
Image 1 is the approved master keyframe (the character bible): Nell, an older woman with silver hair in a low bun, round glasses,
dark green cardigan, cream blouse, long rust-brown skirt; and Pip, a SMALL grey squirrel.
Image 2 is a contact sheet of the shot: 8 frames, evenly spaced in time, left to right.
The scripted action: Nell walks to the maple tree, KNEELS ON ONE KNEE (one knee on the ground, the other foot planted), lifts the
small grey squirrel in both hands and smiles. Pip must stay the same small size the whole time.

Score each item 0-10 (10 = flawless, broadcast quality; 5 = usable with visible flaws; 0 = broken):
- motion_naturalness: plausible, smooth body motion and weight shift across the frames
- identity: Nell and Pip look like the SAME characters as image 1 in every frame (face, hair, glasses, clothes, colours, squirrel design)
- anatomy: hands, limbs, face, squirrel body correct; no extra/missing fingers or melted limbs
- framing: good 9:16 composition, Nell's whole body/head inside the frame with headroom, nothing important cut off by an edge
- art_quality: line art, colour, background richness, appeal
- artifacts: 10 = no morphing, flicker, smearing, duplicated objects or warping; lower for each visible artifact
Also answer: "kneel_one_knee": "yes" | "no (crouch/squat)" | "no (stands)" | "unclear"; "pip_same_size": "yes" | "grows" | "shrinks" | "unclear";
"smiles": "yes" | "no" | "unclear".
Be consistent and harsh; small differences matter. Reply with ONLY a JSON object:
{"scores": {"motion_naturalness": n, "identity": n, "anatomy": n, "framing": n, "art_quality": n, "artifacts": n},
 "kneel_one_knee": "...", "pip_same_size": "...", "smiles": "...", "worst_problem": "one short sentence"}"""
KEY_PROMPT = """You are a strict animation art director judging ONE generated keyframe for a vertical 9:16 animated story Short.
Image 1 is the approved master keyframe (the character bible): Nell, an older woman with silver hair in a low bun, round glasses,
dark green cardigan, cream blouse, long rust-brown skirt; and Pip, a SMALL grey squirrel, in an autumn yard by a maple tree.
Image 2 is the candidate END keyframe of the beat. It must show: Nell KNEELING ON ONE KNEE (one knee on the ground, the other foot
planted flat), holding the small grey squirrel in both hands at her chest, smiling; the same characters, place and art style as image 1.
Score each item 0-10: "identity" (same Nell and Pip as image 1), "pose" (a true one-knee kneel, hands holding Pip), "pip_scale" (Pip still
small, hand-sized), "framing" (whole body inside 9:16, headroom, nothing cut), "art_quality", "artifacts" (10 = none), "continuity"
(same yard, tree, house, light and art style as image 1, so a video can move from image 1 to image 2 without a cut).
Reply with ONLY a JSON object: {"scores": {"identity": n, "pose": n, "pip_scale": n, "framing": n, "art_quality": n, "artifacts": n,
"continuity": n}, "kneel_one_knee": "yes|no|unclear", "worst_problem": "one short sentence"}"""


def b64(path, max_side=1568):
    im = Image.open(path).convert("RGB"); im.thumbnail((max_side, max_side))
    b = io.BytesIO(); im.save(b, "JPEG", quality=85); return base64.b64encode(b.getvalue()).decode()


def router(messages, model="cc/claude-sonnet-5-5", max_tokens=700):
    key = os.environ.get("NINEROUTER_API_KEY", "")
    body = {"model": model, "stream": False, "max_tokens": max_tokens, "temperature": 0, "messages": messages}
    req = urllib.request.Request("http://127.0.0.1:20128/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    return r["choices"][0]["message"]["content"]


def judge(ref, img, keys=False):
    content = [{"type": "text", "text": KEY_PROMPT if keys else CLIP_PROMPT},
               {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(ref, 768)}},
               {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(img)}}]
    for attempt in range(3):
        try:
            t = router([{"role": "user", "content": content}])
            return json.loads(t[t.find("{"):t.rfind("}") + 1])
        except Exception as e:
            err = repr(e)[:200]; time.sleep(5)
    return {"error": err}


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--ref", required=True); a.add_argument("--sheets", nargs="+", required=True)
    a.add_argument("--out", required=True); a.add_argument("--runs", type=int, default=2); a.add_argument("--keys", action="store_true")
    A = a.parse_args()
    res = json.load(open(A.out)) if os.path.exists(A.out) else {}
    for s in A.sheets:
        k = os.path.basename(s)
        if k in res and "mean" in res[k]:
            continue
        runs = [judge(A.ref, s, A.keys) for _ in range(A.runs)]
        good = [r for r in runs if "scores" in r]
        out = {"runs": runs}
        if good:
            names = list(good[0]["scores"])
            out["mean"] = {n: round(sum(float(r["scores"].get(n, 0)) for r in good) / len(good), 2) for n in names}
            out["spread"] = {n: round(max(float(r["scores"].get(n, 0)) for r in good) - min(float(r["scores"].get(n, 0)) for r in good), 1) for n in names}
            out["total"] = round(sum(out["mean"].values()) / len(names), 2)
        res[k] = out
        print(k, out.get("total"), out.get("mean"), [r.get("kneel_one_knee") for r in good], [r.get("pip_same_size") for r in good], flush=True)
        json.dump(res, open(A.out, "w"), indent=1)
