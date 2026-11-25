"""Parse a Heartwick story markdown file (YAML frontmatter + `## Section` + `::: shot` blocks).

Format (see stories/pip.md and README.md):
  ## Chapter title                -> chapter (SEO chapters, caption-free)
  ::: shot key=value ...          -> starts a shot; narration lines below belong to it
      bg=<location>  cam=push_in|pull_back|pan_left|pan_right|tilt_down|tilt_up|drift
      fx=motes,leaves,rain,snow,mist,fireflies,lightning   focus=x,y   id=<name>
      chars=<char>:<pose>@x,y,h[:anim][;...]   (x = centre, y = feet, h = height, all as fraction of frame)
  narration text with [pause 1s] markers and *emphasis* (emphasis -> caption highlight)
A story with no `::: shot` lines still works: each paragraph becomes a shot on an auto-chosen location.
"""
import re
import yaml

PAUSE_RE = re.compile(r"\[pause\s*([\d.]+)\s*s?\]", re.I)


def _parse_shot_args(s):
    args = {}
    for tok in s.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            args[k.strip()] = v.strip()
    out = {"bg": args.get("bg"), "cam": args.get("cam", "push_in"),
           "fx": [f for f in args.get("fx", "").split(",") if f], "id": args.get("id")}
    if "focus" in args:
        out["focus"] = [float(x) for x in args["focus"].split(",")]
    chars = []
    for c in filter(None, args.get("chars", "").split(";")):
        m = re.match(r"(\w+):(\w+)@(-?[\d.]+),([\d.]+),([\d.]+)(?::(\w+))?", c)  # x may be off-screen (walk-ins)
        if not m:
            raise ValueError(f"bad chars spec: {c}")
        chars.append({"char": m[1], "pose": m[2], "x": float(m[3]), "y": float(m[4]), "h": float(m[5]),
                      "anim": m[6] or "idle"})
    out["chars"] = chars
    return out


def split_utterances(text):
    """Split narration into (sentence_text, pause_after_s, emphasis_words). Pause markers end an utterance."""
    text = re.sub(r"\s+", " ", text).strip()
    parts = PAUSE_RE.split(text)  # [txt, p, txt, p, ...]
    utts = []
    for i in range(0, len(parts), 2):
        seg = parts[i].strip()
        pause = float(parts[i + 1]) if i + 1 < len(parts) else 0.3
        if not seg:
            if utts and i + 1 < len(parts):
                utts[-1]["pause"] += pause
            continue
        emph = re.findall(r"\*(\w+)\*", seg)
        clean = re.sub(r"\*(\w+)\*", r"\1", seg)
        utts.append({"text": clean, "pause": pause, "emph": emph})
    return utts


def parse_story(path):
    raw = open(path, encoding="utf-8").read()
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw, re.S)
    if not m:
        raise ValueError("story needs YAML frontmatter between --- lines")
    meta = yaml.safe_load(m[1])
    body = m[2]
    sections, cur_sec, cur_shot = [], None, None
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("## "):
            cur_sec = {"title": s[3:].strip(), "shots": []}
            sections.append(cur_sec)
            cur_shot = None
        elif s.startswith("::: shot"):
            if cur_sec is None:
                cur_sec = {"title": meta.get("title", "Story"), "shots": []}
                sections.append(cur_sec)
            cur_shot = _parse_shot_args(s[len("::: shot"):])
            cur_shot["text"] = ""
            cur_sec["shots"].append(cur_shot)
        elif s:
            if cur_sec is None:
                cur_sec = {"title": meta.get("title", "Story"), "shots": []}
                sections.append(cur_sec)
            if cur_shot is None:  # auto shot (no storyboard)
                cur_shot = {"bg": None, "cam": None, "fx": [], "chars": [], "id": None, "text": ""}
                cur_sec["shots"].append(cur_shot)
            cur_shot["text"] += " " + s
        else:
            if cur_shot is not None and cur_shot.get("bg") is None:
                cur_shot = None  # blank line ends an auto shot
    for sec in sections:
        for sh in sec["shots"]:
            sh["utts"] = split_utterances(sh.pop("text"))
    _autofill(meta, sections)
    return meta, sections


def _autofill(meta, sections):
    """Give storyboard-less shots a location + camera so plain text stories still render."""
    locs = list((meta.get("locations") or {}).keys())
    cams = ["push_in", "pan_right", "pull_back", "pan_left", "drift"]
    k = 0
    for sec in sections:
        for sh in sec["shots"]:
            if not sh.get("bg"):
                sh["bg"] = locs[k % len(locs)] if locs else "auto_%d" % k
                sh["fx"] = sh.get("fx") or ["motes"]
            if not sh.get("cam"):
                sh["cam"] = cams[k % len(cams)]
            k += 1


def select_sections(sections, spec):
    """spec like '1-2' or '1,3,5' (1-based)."""
    if not spec:
        return sections
    idx = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            idx.update(range(int(a), int(b) + 1))
        else:
            idx.add(int(part))
    return [s for i, s in enumerate(sections, 1) if i in idx]


if __name__ == "__main__":
    import sys, json
    meta, secs = parse_story(sys.argv[1])
    for s in select_sections(secs, sys.argv[2] if len(sys.argv) > 2 else None):
        print("##", s["title"])
        for sh in s["shots"]:
            print("  shot", sh["bg"], sh["cam"], sh["fx"], [c["char"] + ":" + c["pose"] for c in sh["chars"]])
            for u in sh["utts"]:
                print("     ", round(u["pause"], 2), u["emph"], u["text"][:70])
