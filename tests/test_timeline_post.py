import json, re
from engine import story as S, plan as P, timeline as T, post
from conftest import fake_tts


def _build(mini_story, tmp_path, **cfg):
    meta, secs = S.parse_story(mini_story)
    plan = P.build_plan(meta, secs)
    d = tmp_path / "assets"
    d.mkdir(exist_ok=True)
    for s in plan["sections"]:
        for sh in s["shots"]:
            for u in sh["utts"]:
                fake_tts(str(d), u["id"], u["text"])
    c = {"fps": 30, "pause_scale": 0.5, "max_shot_s": 4.0, "end_card_s": 3.0}
    c.update(cfg)
    return meta, plan, T.build(plan, str(d), c), d


def test_caption_chunks(mini_story, tmp_path):
    _, _, tl, _ = _build(mini_story, tmp_path)
    caps = tl["captions"]
    assert caps
    for c in caps:
        assert 1 <= len(c["words"]) <= 5
        assert len(" ".join(c["words"])) <= 26
        assert -1 <= c["hl"] < len(c["words"])
        assert c["e"] > c["s"]
    assert sum(len(c["words"]) == 1 for c in caps) <= 1  # orphan words are merged
    for a, b in zip(caps, caps[1:]):  # in order, never overlapping
        assert a["e"] <= b["s"] + 1e-6
    words = " ".join(w for c in caps for w in c["words"])
    assert "Nell made her coffee." in words and '"Oh, little one,"' in words
    one = next(c for c in caps if "one" in c["words"])
    assert one["words"][one["hl"]] == "one"  # *emphasis* wins the highlight


def test_caption_times_follow_audio_offsets(mini_story, tmp_path):
    _, _, tl, d = _build(mini_story, tmp_path)
    first = tl["audio"][0]
    w0 = json.load(open(d / (first["id"] + ".json")))["words"][0]
    assert abs(tl["captions"][0]["s"] - (first["t"] + w0["s"])) < 1e-6
    for a, b in zip(tl["audio"], tl["audio"][1:]):
        assert b["t"] >= a["t"] + a["dur"]  # no overlapping narration


def test_shots_contiguous_and_split(mini_story, tmp_path):
    _, _, tl, _ = _build(mini_story, tmp_path)
    sh = tl["shots"]
    assert sh[0]["s"] == 0
    for a, b in zip(sh, sh[1:]):
        assert abs(a["e"] - b["s"]) < 1e-3
    assert abs(sh[-1]["e"] - (tl["total"] - tl["end_card_s"])) < 1e-3
    assert max(s["e"] - s["s"] for s in sh) <= 4.0 * 1.3
    long = [s for s in sh if s["nsub"] > 1]
    assert long and len({s["framing"] for s in long}) > 1
    assert all(s["trans_in"] in (0.0, 0.3, 0.6) for s in sh)
    assert [c["title"] for c in tl["chapters"]] == ["Opening chapter", "Second chapter"]


def test_srt_and_seo(mini_story, tmp_path):
    meta, _, tl, _ = _build(mini_story, tmp_path)
    srt = tmp_path / "x.srt"
    n = post.write_srt(tl, str(srt))
    blocks = [b for b in srt.read_text(encoding="utf-8").strip().split("\n\n") if b]
    assert len(blocks) == n
    last = 0
    for i, b in enumerate(blocks, 1):
        lines = b.split("\n")
        assert lines[0] == str(i)
        m = re.match(r"(\d\d):(\d\d):(\d\d),(\d{3}) --> (\d\d):(\d\d):(\d\d),(\d{3})", lines[1])
        assert m
        s = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000
        e = int(m[5]) * 3600 + int(m[6]) * 60 + int(m[7]) + int(m[8]) / 1000
        assert e > s >= last - 1e-3 and len(lines[2]) <= 48
        last = e
    md = tmp_path / "SEO_PACK.md"
    post.seo_pack(meta, tl, "mini", True, str(md))
    js = json.load(open(str(md).replace(".md", ".json"), encoding="utf-8"))
    assert js["hashtags"] == ["#a", "#b", "#c"] and len(", ".join(js["tags"])) <= 500
    assert "AI-generated" in js["description"] and js["filename"].endswith("-short.mp4")
    assert "not made for kids" in md.read_text(encoding="utf-8")


def test_chapters_rules():
    tl = {"total": 100.0, "chapters": [{"title": "a", "t": 0.4}, {"title": "b", "t": 5}, {"title": "c", "t": 30},
                                       {"title": "d", "t": 60}, {"title": "e", "t": 95}]}
    ch = post.chapters(tl)
    assert ch[0]["t"] == 0 and [c["title"] for c in ch] == ["a", "c", "d"]
    assert post.chapters({"total": 30, "chapters": [{"title": "a", "t": 0}]}) == []


def test_asr_check_wer_and_offsets():
    tl = {"captions": [{"words": ["She", "set", "out"], "s": 1.0}, {"words": ["one", "mug."], "s": 2.0}, {"words": ["Very", "quiet."], "s": 4.0}]}
    asr = {"chunks": [{"w": " She", "t": [1.05, 1.2]}, {"w": " set", "t": [1.2, 1.4]}, {"w": " out", "t": [1.4, 1.6]},
                      {"w": " one", "t": [2.02, 2.3]}, {"w": " mug.", "t": [2.3, 2.6]}, {"w": " very", "t": [4.1, 4.3]}, {"w": " quite.", "t": [4.3, 4.7]}]}
    r = post.asr_check(tl, asr)
    assert r["ref_words"] == 7 and r["wer"] == round(1 / 7, 3)
    assert r["matched_chunk_starts"] == 3 and abs(r["caption_vs_speech_median_s"] - 0.05) < 1e-6


def test_opening_quote_after_space_starts_new_word():
    toks = [{"w": "whispered", "ws": "", "s": 0, "e": .5}, {"w": ".", "ws": " ", "s": .5, "e": .5}, {"w": '"', "ws": "", "s": .6, "e": .6},
            {"w": "Where's", "ws": " ", "s": .6, "e": .9}, {"w": "mama", "ws": "", "s": 1, "e": 1.3}, {"w": "?", "ws": "", "s": 1.3, "e": 1.3},
            {"w": '"', "ws": " ", "s": 1.3, "e": 1.3}]
    assert [w["w"] for w in T.merge_tokens(toks)] == ["whispered.", '"Where\'s', 'mama?"']
