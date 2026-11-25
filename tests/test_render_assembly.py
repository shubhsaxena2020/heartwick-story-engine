"""Tiny real render: synthetic plates/cut-outs/narration -> mp4 with audio, loudness-normalised. CPU-light (270x480, 10 fps)."""
import json, os, subprocess
import numpy as np
from PIL import Image, ImageDraw
from engine import story as S, plan as P, timeline as T, render, post
from conftest import fake_tts


def _assets(d, meta, plan):
    for loc, L in meta["locations"].items():
        w, h = (640, 1536) if L.get("tall") else (768, 1344)
        im = Image.new("RGB", (w, h), (90, 120, 160))
        dr = ImageDraw.Draw(im)
        dr.ellipse([-100, h * 0.6, w + 100, h * 1.3], fill=(70, 110, 70))
        dr.rectangle([w * .3, h * .5, w * .6, h * .7], fill=(230, 225, 210))
        im.save(os.path.join(d, f"bg_{loc}.png"))
        Image.fromarray((np.linspace(0, 255, h)[:, None] * np.ones((1, w))).astype(np.uint8)).save(os.path.join(d, f"bg_{loc}_depth.png"))
    for c, ch in meta["characters"].items():
        for pose in ch["poses"]:
            im = Image.new("RGBA", (300, 600), (0, 0, 0, 0))
            dr = ImageDraw.Draw(im)
            dr.ellipse([80, 0, 220, 140], fill=(230, 190, 160, 255))
            dr.rounded_rectangle([60, 130, 240, 600], 30, fill=(60, 130, 80, 255))
            im.save(os.path.join(d, f"pose_{c}_{pose}_cut.png"))
    for s in plan["sections"]:
        for sh in s["shots"]:
            for u in sh["utts"]:
                fake_tts(d, u["id"], u["text"])


def test_tiny_render_end_to_end(mini_story, tmp_path):
    meta, secs = S.parse_story(mini_story)
    plan = P.build_plan(meta, secs)
    d = str(tmp_path / "assets")
    os.makedirs(d)
    _assets(d, meta, plan)
    cfg = {"W": 270, "H": 480, "fps": 10, "pause_scale": 0.3, "max_shot_s": 4.0, "end_card_s": 1.5, "lufs": -15.0}
    tl = T.build(plan, d, cfg)
    out = str(tmp_path / "out" / "mini.mp4")
    render.render_video(tl, d, d, out, cfg, workers=1, low=True)
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", out],
                                      capture_output=True, text=True).stdout)
    kinds = {s["codec_type"]: s for s in probe["streams"]}
    assert kinds["video"]["width"] == 270 and kinds["video"]["height"] == 480
    assert kinds["audio"]["sample_rate"] == "48000"
    assert abs(float(probe["format"]["duration"]) - tl["total"]) < 0.4
    lo = post.loudness(out)
    assert abs(lo["integrated_lufs"] + 15) < 1.5 and lo["true_peak_dbfs"] <= -0.5
    sync = post.caption_sync(tl, out)
    assert sync["speech_present"] >= 0.9 * sync["chunks"]
    assert sync["onset_chunks"] >= 2 and sync["onset_p90_abs_s"] <= 0.1
    # frames are not static: camera + puppets + particles move during the story
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", out, "-vf", "scale=90:160", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, 160, 90).astype(np.float32)
    diffs = np.abs(np.diff(fr[: int(tl["speech_end"] * 10)], axis=0)).mean(axis=(1, 2))
    assert (diffs < 0.05).mean() < 0.2


def test_every_fx_renders_finite_full_frame():
    """Regression: a float32 sin() feather once produced NaN rows (black bars) under the mist layer."""
    tl = {"locations": {"x": {"tint": "storm"}}, "shots": []}
    W, H = 270, 480
    p = render.Particles(["motes", "fireflies", "rain", "snow", "mist", "leaves", "lightning"], 3, W, H, W / 1080)
    for t in (0.0, 0.7, 2.9, 5.1):
        img = p.draw(np.full((H, W, 3), 0.5, np.float32), t, 6.0, (12.0, -8.0))
        assert np.isfinite(img).all()
        assert img[-40:].mean() > 0.2  # bottom rows are not blacked out
