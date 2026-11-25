import os
from engine import story as S, plan as P
from conftest import ROOT


def test_parse_mini(mini_story):
    meta, secs = S.parse_story(mini_story)
    assert [s["title"] for s in secs] == ["Opening chapter", "Second chapter"]
    sh = secs[0]["shots"][0]
    assert sh["bg"] == "kitchen" and sh["cam"] == "push_in" and sh["fx"] == ["motes"]
    assert sh["chars"][0] == {"char": "nell", "pose": "stand", "x": 0.5, "y": 0.95, "h": 0.6, "anim": "idle"}
    u = sh["utts"]
    assert [x["text"] for x in u] == ["Nell made her coffee.", "She set out one mug."]
    assert u[0]["pause"] == 0.5 and u[1]["pause"] == 1.0 and u[1]["emph"] == ["one"]
    assert secs[0]["shots"][1]["chars"][0]["anim"] == "tremble"


def test_no_phantom_pause(mini_story):
    _, secs = S.parse_story(mini_story)
    assert secs[1]["shots"][0]["utts"][-1]["pause"] == 1.0


def test_plain_text_story_autofills(tmp_path):
    p = tmp_path / "plain.md"
    p.write_text("---\nslug: plain\nlocations: {a: {prompt: x}, b: {prompt: y}}\n---\nFirst paragraph here.\n\nSecond one.\n", encoding="utf-8")
    _, secs = S.parse_story(str(p))
    shots = secs[0]["shots"]
    assert len(shots) == 2 and shots[0]["bg"] == "a" and shots[1]["bg"] == "b" and shots[0]["cam"]


def test_select_sections():
    meta, secs = S.parse_story(os.path.join(ROOT, "stories", "sample.md"))
    assert len(secs) == 3
    assert [s["title"] for s in S.select_sections(secs, "1-2")] == [secs[0]["title"], secs[1]["title"]]
    assert len(S.select_sections(secs, "1,3")) == 2


def test_real_stories_reference_known_assets():
    for f in ("sample.md",):
        meta, secs = S.parse_story(os.path.join(ROOT, "stories", f))
        for s in secs:
            for sh in s["shots"]:
                assert sh["bg"] in meta["locations"], (f, sh["bg"])
                for c in sh["chars"]:
                    assert c["pose"] in meta["characters"][c["char"]]["poses"], (f, c)


def test_content_addressed_ids_and_cache(mini_story):
    meta, secs = S.parse_story(mini_story)
    p1 = P.build_plan(meta, secs, speed=1.0)
    p2 = P.build_plan(meta, secs, speed=1.0)
    p3 = P.build_plan(meta, secs, speed=0.94)
    ids = lambda p: [u["id"] for s in p["sections"] for sh in s["shots"] for u in sh["utts"]]
    assert ids(p1) == ids(p2) and ids(p1) != ids(p3)
    items, need = P.asset_items(meta, p1, set(), set())
    assert need == ["nell", "pip"] and {i["id"] for i in items} == {"bg_kitchen", "bg_yard"}
    items, need = P.asset_items(meta, p1, {"bg_kitchen", "bg_kitchen_depth"}, {"nell", "pip"})
    assert {i["id"] for i in items} == {"bg_yard", "pose_nell_stand", "pose_pip_sit"}
    tall = next(i for i in items if i["id"] == "bg_yard")
    assert (tall["w"], tall["h"]) == (640, 1536)
    pose = next(i for i in items if i["id"] == "pose_pip_sit")
    assert pose["ip_ref"] == "master_pip" and pose["prompt"].startswith("storybook")
    items, _ = P.asset_items(meta, p1, set(), set(), inline_masters=True)
    assert any(i["id"].startswith("pose_") for i in items)
    kinds = {c["char"]: c["kind"] for s in p1["sections"] for sh in s["shots"] for c in sh["chars"]}
    assert kinds == {"nell": "human", "pip": "animal"}


def test_prompts_style_first_and_fit_clip():
    for f in ("sample.md",):
        meta, secs = S.parse_story(os.path.join(ROOT, "stories", f))
        plan = P.build_plan(meta, secs)
        items, _ = P.asset_items(meta, plan, set(), set(meta["characters"]))
        items += [P.master_item(n, meta["characters"][n]) for n in meta["characters"]]
        for i in items:
            assert i["prompt"].startswith("storybook illustration")
            assert "candle" in i["neg"]
            assert P.approx_tokens(i["prompt"]) <= 80, i["id"]
