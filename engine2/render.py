"""Engine v2 renderer: story -> frames (skia, CPU) -> H.264 + mastered AAC.

    python -m engine2.render <story.md> <tts_dir> <out.mp4> --style B|C [--sections 1-2] [--W 1080 --fps 30 --workers 4]

Frames are rendered by `workers` processes, each owning a contiguous frame range and its own ffmpeg encoder;
segments are concatenated losslessly, then muxed with the mastered audio.
"""
import argparse, json, math, os, subprocess, sys, time
import numpy as np
import skia
from engine import story as S1, plan as P1
from engine import brand as B1
from . import edit as ED, story2, direct, style as ST, audio as AU
from .fx import native
from .mathx import smooth as smooth_
from .post import Finisher


def painter_for(style, seed=0):
    return ST.PaperTheatre(seed) if style == "C" else ST.VectorCel(seed)


def pil_to_skia(im):
    a = np.ascontiguousarray(np.asarray(im.convert("RGBA")))
    return native(skia.Image.fromarray(a, skia.kRGBA_8888_ColorType, skia.kUnpremul_AlphaType))


class Film:
    def __init__(self, story_path, tts_dir, cfg):
        self.cfg = cfg = dict(cfg)
        meta, sections = story2.parse(story_path)
        sections = S1.select_sections(sections, cfg.get("sections"))
        self.meta = meta
        voice, speed = cfg.get("voice", meta.get("voice", "af_heart")), cfg.get("speed", 1.0)
        plan = P1.build_plan(meta, sections, voice=voice, speed=speed)
        ids = iter([u["id"] for s in plan["sections"] for sh in s["shots"] for u in sh["utts"]])
        for sec in sections:  # keep v2 shot data, take content-addressed utterance ids from the v1 planner
            for sh in sec["shots"]:
                for u in sh["utts"]:
                    u["id"] = next(ids)
        if cfg.get("utt_map"):  # an alternative voice: {text: utterance id}
            for sec in sections:
                for sh in sec["shots"]:
                    for u in sh["utts"]:
                        u["id"] = cfg["utt_map"].get(u["text"], u["id"])
        self.tts_dir = tts_dir
        tts = {}
        for sec in sections:
            for sh in sec["shots"]:
                for u in sh["utts"]:
                    tts[u["id"]] = json.load(open(os.path.join(tts_dir, u["id"] + ".json")))
        cfg.setdefault("highlight_words", meta.get("highlight_words", []))
        self.scenes, self.words, self.audio, caps, self.speech_end = ED.place(sections, tts, cfg)
        self.shots, self.story_end = ED.cut_plan(self.scenes, self.words, cfg)
        self.caps = ED.captions(caps)
        self.end_card_s = cfg.get("end_card_s", 3.6)
        self.total = self.story_end + self.end_card_s
        self.ss = cfg.get("ss", 1.0)  # supersampling: frames are drawn ss x larger, then downscaled with mipmaps
        self.out_W, self.out_H = cfg.get("W", 1080), cfg.get("H", 1920)
        self.W, self.H, self.fps = int(round(self.out_W * self.ss)), int(round(self.out_H * self.ss)), cfg.get("fps", 30)
        from .direct import tune
        self.T = tune(cfg)
        self.fin = Finisher(self.W, self.H, seed=cfg.get("seed", 0), grain=self.T["grain"], gains={"key": self.T["key"], "rim": self.T["rim"], "bloom": self.T["bloom"]})
        self._hero = {}
        self.P = painter_for(cfg.get("style", "B"), cfg.get("seed", 0))
        self._setups = {}
        self._sprites = {}
        self.hook = cfg.get("hook", meta.get("hook"))
        self.series_tag = cfg.get("series_tag", meta.get("series_tag"))
        self._end = None

    # ---------------------------------------------------------------- scene setups (built lazily per worker)
    def setup(self, i):
        if i not in self._setups:
            shots = [s for s in self.shots if s["scene"] == i]
            self._setups[i] = direct.setup(self.scenes[i], shots, self.P, self.cfg, seed=self.cfg.get("seed", 0) * 101 + i)
            # free scenes far behind (memory: cached static layers)
            for k in [k for k in self._setups if k < i - 1]:
                del self._setups[k]
        return self._setups[i]

    def shot_index(self, t):
        for j, s in enumerate(self.shots):
            if s["s"] <= t < s["e"]:
                return j
        return len(self.shots) - 1

    def draw_shot(self, c, j, t):
        sh = self.shots[j]
        st = self.setup(sh["scene"])
        scene = st["scene"]
        c.saveLayer(None, self.fin.grade_paint(scene.mood))  # per-scene mood grade
        clip = self.hero_clip(sh)
        if clip is not None:
            self.draw_hero(c, clip, sh, t)
            self.fin.key_light(c, scene.mood, (self.W * 0.3, self.H * 0.15))
        else:
            cam = st["cams"][j].at(t)
            q = min(2.0, max(0.5, self.W / 1080.0 * cam[2] * 1.05))
            q = round(q * 4) / 4
            scene.render(c, t, cam, st["actors"], self.W, self.H, q=q, fin=self.fin)
        c.restore()

    # ---------------------------------------------------------------- hero shots (image-to-video clips)
    def hero_clip(self, sh):
        """The first sub-shot of a story shot with `hero=<beat>` plays assets/hero/<slug>/<beat>_<model>.mp4."""
        beat = self.scenes[sh["scene"]]["shot"].get("v2", {}).get("hero")
        if not beat or sh["sub"] != 0:
            return None
        if beat not in self._hero:
            d = self.cfg.get("hero_dir") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "hero",
                                                         self.meta.get("slug", ""))
            pref = self.cfg.get("hero_model", "wan")
            path = next((os.path.join(d, f"{beat}_{m}.mp4") for m in (pref, "wan", "ltx") if os.path.exists(os.path.join(d, f"{beat}_{m}.mp4"))), None)
            self._hero[beat] = self._load_clip(path) if path else None
        return self._hero[beat]

    def _load_clip(self, path):
        import imageio
        frames = []
        for f in imageio.get_reader(path):
            a = np.ascontiguousarray(np.dstack([np.asarray(f)[..., :3], np.full(f.shape[:2], 255, np.uint8)]))
            frames.append(native(skia.Image.fromarray(a, skia.kRGBA_8888_ColorType)))
        return frames

    def draw_hero(self, c, frames, sh, t):
        d = sh["e"] - sh["s"]
        n = len(frames)
        clip_s = n / 30.0
        rate = min(1.0, clip_s / max(d, 1e-3)) if clip_s < d else 1.0
        rate = max(rate, 1 / 1.6)  # slow the clip at most 1.6x to fit the line, then hold its last frame
        u = (t - sh["s"]) * 30 * rate
        img = frames[int(min(n - 1, max(0, u)))]
        k = max(self.W / img.width(), self.H / img.height()) * (1.04 + 0.03 * (t - sh["s"]) / max(d, 1e-3))  # cover + slow push
        p = skia.Paint(AntiAlias=True)
        p.setFilterQuality(skia.FilterQuality.kHigh_FilterQuality)
        c.save()
        c.translate(self.W / 2, self.H / 2)
        c.scale(k, k)
        c.drawImage(img, -img.width() / 2, -img.height() / 2, p)
        c.restore()
        if self.P.name == "C":  # same stage as the puppets: the paper proscenium frames the anime moment too
            ov = self.setup(sh["scene"])["scene"].overlay
            if ov:
                ov(c, t, self.W, self.H)

    # ---------------------------------------------------------------- overlays
    def sprite(self, key, make):
        if key not in self._sprites:
            self._sprites[key] = pil_to_skia(make())
        return self._sprites[key]

    def overlays(self, c, t):
        k = self.W / 1080.0
        W, H = self.W, self.H
        p = skia.Paint(AntiAlias=True)
        p.setFilterQuality(skia.FilterQuality.kMedium_FilterQuality)
        hook_end = 2.0 if self.hook else 0.0
        if self.hook and t < 2.3:
            img = self.sprite("hook", lambda: B1.hook_sprite(self.hook["kicker"], self.hook["lines"], self.hook.get("highlight"), k, W))
            a = min(1.0, t / 0.25) * min(1.0, max(0.0, (2.3 - t) / 0.4))
            p.setAlphaf(a)
            c.drawImage(img, 0, int(H * 0.17), p)
        if self.series_tag and t >= hook_end and t < self.story_end:
            img = self.sprite("tag", lambda: B1.series_tag(self.series_tag, k))
            p.setAlphaf(min(1.0, (t - hook_end) / 0.35) if self.hook else 1.0)
            c.drawImage(img, 64 * k, 210 * k, p)
        for i, cp in enumerate(self.caps):
            if cp["s"] <= t < cp["e"] and t < self.story_end:
                img = self.sprite(("cap", i), lambda: B1.caption_sprite(cp["words"], cp["hl"], k))
                p.setAlphaf(min(1.0, (t - cp["s"]) / cp["fade"]))
                c.drawImage(img, (W - img.width()) / 2, H * 0.71 - img.height() / 2, p)
                break

    def end_card(self, c, t):
        if self._end is None:
            self._end = B1.EndCard(self.W, self.H, self.cfg.get("cta", "Follow for more Heartwick stories"))
        fr = (np.clip(self._end.frame(t), 0, 1) * 255).astype(np.uint8)
        rgba = np.dstack([fr, np.full(fr.shape[:2], 255, np.uint8)])
        c.drawImage(skia.Image.fromarray(np.ascontiguousarray(rgba), skia.kRGBA_8888_ColorType), 0, 0)

    # ---------------------------------------------------------------- one frame
    def compose(self, c, t):
        """The picture without overlays: shot(s) + transition. Dissolves only within one location; scene changes
        dip through a warm dark (a two-scene double exposure reads as mud)."""
        if t >= self.story_end:
            self.end_card(c, t - self.story_end)
            fade = 1 - min(1.0, (t - self.story_end) / 0.6)
            if fade > 0:
                c.saveLayerAlpha(None, int(255 * fade))
                self.draw_shot(c, len(self.shots) - 1, t)
                c.restore()
            return
        j = self.shot_index(t)
        sh = self.shots[j]
        nx = self.shots[j + 1] if j + 1 < len(self.shots) else None
        self.draw_shot(c, j, t)
        if nx and nx["trans"] == "dissolve" and t >= nx["s"] - nx["trans_d"] / 2:
            a = (t - (nx["s"] - nx["trans_d"] / 2)) / nx["trans_d"]
            c.saveLayerAlpha(None, int(255 * smooth_(min(1, a))))
            self.draw_shot(c, j + 1, t)
            c.restore()
        if sh["trans"] == "dissolve" and j > 0 and t < sh["s"] + sh["trans_d"] / 2:
            a = 0.5 + (t - sh["s"]) / sh["trans_d"]
            c.saveLayerAlpha(None, int(255 * (1 - smooth_(min(1, a)))))
            self.draw_shot(c, j - 1, t)
            c.restore()
        dark = 0.0
        if nx and nx["trans"] == "dip" and t >= nx["s"] - nx["trans_d"] / 2:
            dark = (t - (nx["s"] - nx["trans_d"] / 2)) / (nx["trans_d"] / 2)
        if sh["trans"] == "dip" and j > 0 and t < sh["s"] + sh["trans_d"] / 2:
            dark = 1 - (t - sh["s"]) / (sh["trans_d"] / 2)
        if dark > 0:
            c.drawRect(skia.Rect(0, 0, self.W, self.H), skia.Paint(Color=skia.Color(18, 10, 8, int(255 * 0.96 * smooth_(min(1, dark))))))

    def motion_speed(self, t):
        """px per frame (at 1080) of the fastest thing on screen: camera travel or a character's limbs."""
        if t >= self.story_end:
            return 0.0
        j = self.shot_index(t)
        st = self.setup(self.shots[j]["scene"])
        a, b = st["cams"][j].at(t), st["cams"][j].at(t + 1 / self.fps)
        v = math.hypot(a[0] - b[0], a[1] - b[1]) * a[2]
        for act in st["actors"]:
            _, A0 = act.angles_at(t)
            _, A1 = act.angles_at(t + 1 / self.fps)
            dmax = max([abs(A1.get(k, 0) - A0.get(k, 0)) for k in A1 if not k.startswith(self.SECONDARY)] or [0])
            v = max(v, math.radians(dmax) * act.height * 0.25)  # a limb ~ a quarter of the figure
        return v

    def frame(self, c, t, surf=None):
        c.clear(skia.ColorBLACK)
        fi = int(round(t * self.fps))
        mb = self.cfg.get("motion_blur", True) and self.motion_speed(t) > 6.0
        self.compose(c, t)
        if mb:  # sub-frame sampling across the shutter (180 deg): running average of 3 samples
            for k, dt in enumerate((-1 / (3 * self.fps), 1 / (3 * self.fps)), start=2):
                c.saveLayerAlpha(None, int(255 / k))
                self.compose(c, t + dt)
                c.restore()
        if surf is not None and t < self.story_end and self.cfg.get("bloom", True):
            mood = self.setup(self.shots[self.shot_index(t)]["scene"])["scene"].mood
            self.fin.bloom(c, surf.makeImageSnapshot(), mood)
        if self.cfg.get("grain", True):
            self.fin.film_grain(c, fi)
        self.overlays(c, t)

    def n_frames(self):
        return int(math.ceil(self.total * self.fps))

    def music_and_sfx(self):
        """-> (music stem, sfx stem or None) for the whole film."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lib = AU.library(self.cfg.get("music_dir") or os.path.join(root, "assets", "music"))
        segs, sfx, swell = [], [], []
        seen_creak = False
        for i, sc in enumerate(self.scenes):
            shots = [s for s in self.shots if s["scene"] == i]
            t0, t1 = shots[0]["s"], shots[-1]["e"]
            if i == len(self.scenes) - 1:
                t1 = self.total
            scene = self.setup(i)["scene"]
            v2 = sc["shot"].get("v2", {})
            segs.append((t0, t1, v2.get("music") or AU.MUSIC_OF_MOOD.get(scene.mood, "tender")))
            kinds = list(v2.get("sfx") or [])
            if not kinds and scene.mood == "storm":
                kinds = ["wind", "rain"]
            if not kinds and sc["shot"].get("bg") == "window_ramp" and not seen_creak:
                kinds, seen_creak = ["creak"], True
            for k in kinds:  # never on the cut itself: beds fade in, one-shots land 0.5 s into the shot
                sfx.append((t0 + (0.5 if k == "creak" else 0.0), t1, k))
            for j, s in enumerate(shots):
                if self.setup(i)["cams"][s["idx"]].kind.startswith("tilt") or self.hero_clip(s) is not None:
                    swell.append((s["s"], min(4.0, s["e"] - s["s"])))
        if lib and self.cfg.get("music", True) is not False:
            music = AU.swells(AU.music_track(segs, self.total, lib), swell)
        else:
            music = AU.pad(self.total, swells=swell)
        return music, (AU.sfx_track(sfx, self.total) if sfx else None)

    SECONDARY = ("scarf", "hair", "tail2", "tail3")

    def motion_qa(self):
        """Per actor: foot slide (px/frame of the planted foot), walk speed (heights/s), longest still stretch (s)."""
        out = []
        for i in range(len(self.scenes)):
            for a in self.setup(i)["actors"]:
                fs, v = a.foot_slide(), a.walk_speed()
                out.append({"scene": i, "rig": a.rig.name, "foot_slide_px": None if fs is None else round(fs, 4),
                            "walk_speed_h_s": None if v is None else round(v, 3), "max_still_s": round(a.max_still() or 0, 2)})
        return out

    def pose_motion_pct(self, thr_deg=6.0):
        """% of story seconds in which an on-screen character's main joints move > thr_deg (summed per-frame max
        joint change over the second). Stricter than the pixel metric: particles, springs and camera don't count."""
        secs = int(self.story_end)
        hit = 0
        for s in range(secs):
            tot = 0.0
            for f in range(self.fps):
                t0, t1 = s + f / self.fps, s + (f + 1) / self.fps
                sh = self.shots[self.shot_index(t0)]
                best = 0.0
                for a in self.setup(sh["scene"])["actors"]:
                    _, A0 = a.angles_at(t0)
                    _, A1 = a.angles_at(t1)
                    d = [abs(A1.get(k, 0) - A0.get(k, 0)) for k in A1 if not k.startswith(self.SECONDARY)]
                    best = max([best] + d)
                tot += best
            hit += tot > thr_deg
        return round(100 * hit / max(1, secs), 1)

    # ---------------------------------------------------------------- QA facts known without looking at pixels
    def edit_report(self):
        dur = self.story_end
        cams = []
        for j, s in enumerate(self.shots):
            st = self.setup(s["scene"])
            px, zp = st["cams"][j].speeds(self.fps)
            cams.append({"shot": j, "kind": st["cams"][j].kind, "px_per_frame": round(px, 3), "zoom_pct_s": round(zp, 3)})
        changes = []
        for i in range(len(self.scenes)):
            for a in self.setup(i)["actors"]:
                ks = [k[0] for k in a.track.keys]
                changes += [round(b - a_, 2) for a_, b in zip(ks, ks[1:])]
        return {"duration_s": round(self.total, 2), "story_s": round(dur, 2), "shots": len(self.shots),
                "character_motion_seconds_pct": self.pose_motion_pct(),
                "motion_qa": self.motion_qa(),
                "shots_per_min": round(len(self.shots) / (dur / 60), 1), "cuts_in_pauses": round(ED.cuts_in_pauses(self.shots, self.words), 3),
                "dissolves": [s["trans_d"] for s in self.shots if s["trans"] == "dissolve"],
                "pose_change_interval_s": {"median": float(np.median(changes)) if changes else None,
                                           "p10": float(np.percentile(changes, 10)) if changes else None,
                                           "p90": float(np.percentile(changes, 90)) if changes else None},
                "camera": cams, "captions": len(self.caps)}


# ---------------------------------------------------------------- video + audio
_FILM = None


def _render_range(args):
    story, tts_dir, cfg, a, b, seg, low = args
    global _FILM
    if low:
        _lower_priority()
    if _FILM is None:
        _FILM = Film(story, tts_dir, cfg)
    F = _FILM
    W, H = F.out_W, F.out_H
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}", "-r", str(F.fps), "-i", "-",
                          "-c:v", "libx264", "-preset", cfg.get("preset", "medium"), "-crf", str(cfg.get("crf", 16)), "-tune", "animation",
                          "-profile:v", "high", "-pix_fmt", "yuv420p", "-threads", "1", seg], stdin=subprocess.PIPE)
    s = skia.Surface.MakeRasterN32Premul(F.W, F.H)
    c = s.getCanvas()
    out = s if F.ss == 1.0 else skia.Surface.MakeRasterN32Premul(W, H)
    dp = skia.Paint(AntiAlias=True)
    dp.setFilterQuality(skia.FilterQuality.kHigh_FilterQuality)
    info = skia.ImageInfo.Make(W, H, skia.kBGRA_8888_ColorType, skia.kPremul_AlphaType)
    buf = np.empty((H, W, 4), np.uint8)
    t0 = time.time()
    for f in range(a, b):
        F.frame(c, f / F.fps, s)
        if out is not s:  # supersampled anti-aliasing: high-quality (mipmapped) downscale
            oc = out.getCanvas()
            oc.save()
            oc.scale(W / F.W, H / F.H)
            oc.drawImage(s.makeImageSnapshot(), 0, 0, dp)
            oc.restore()
        out.readPixels(info, buf)
        p.stdin.write(buf.tobytes())
    p.stdin.close()
    p.wait()
    return seg, round(time.time() - t0, 1), b - a


def _lower_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass


def render(story, tts_dir, out, cfg, workers=4, low=False, frame_range=None, with_audio=True):
    """frame_range=(a, b): render only those frames (one part of a render split across parallel kernels); the part
    without audio is a video-only mp4 that tools concatenate losslessly; with_audio adds the mastered mix."""
    t_start = time.time()
    F = Film(story, tts_dir, cfg)
    n_all = F.n_frames()
    fa, fb = frame_range or (0, n_all)
    fb = min(fb, n_all)
    n = fb - fa
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    base = os.path.splitext(out)[0]
    chunks = [(story, tts_dir, cfg, fa + n * i // workers, fa + n * (i + 1) // workers, f"{base}_seg{i}.mp4", low) for i in range(workers)]
    if workers == 1:
        res = [_render_range(chunks[0])]
    else:
        import multiprocessing as mp
        # fork on Linux (Kaggle): spawn would re-import the kernel script, which has no __main__ guard
        with mp.get_context("spawn" if os.name == "nt" else "fork").Pool(workers) as pool:
            res = pool.map(_render_range, chunks)
    lst = base + "_segs.txt"
    open(lst, "w").write("".join(f"file '{os.path.abspath(r[0])}'\n" for r in res))
    video = base + "_video.mp4" if with_audio else out
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", video], check=True)
    if not with_audio:
        for r_ in res:
            os.remove(r_[0])
        os.remove(lst)
        rep = {"frames": n, "frame_range": [fa, fb], "render_s": round(time.time() - t_start, 1),
               "s_per_frame": round(sum(r_[1] for r_ in res) / max(1, n), 3)}
        return rep
    # audio: narration + music (ACE-Step library by scene mood, else the synthesised pad) + sparse SFX, mastered
    voice = AU.voice_track(F.audio, tts_dir, F.total)
    mix, arep = AU.master(voice, *F.music_and_sfx(), target=cfg.get("lufs", -14.0), pad_gap_db=cfg.get("pad_gap_db", 20.0))
    mix = mix[: int(F.total * AU.SR)]
    wav = base + "_mix.wav"
    import soundfile as sf
    sf.write(wav, mix, AU.SR)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
                    "-shortest", "-movflags", "+faststart", out], check=True)
    for r in res:
        os.remove(r[0])
    os.remove(lst)
    os.remove(video)
    rep = {"frames": n, "render_s": round(time.time() - t_start, 1),
           "s_per_frame": round(sum(r[1] for r in res) / max(1, n), 3), "audio": arep}
    rep.update(F.edit_report())
    json.dump({"shots": F.shots, "captions": F.caps, "audio": F.audio, "words": F.words}, open(base + "_timeline.json", "w"), indent=0)
    return rep


def mix_audio(story, tts_dir, out_base, cfg):
    """Part 0 of a split render: the mastered mix for the whole film + the edit report and timeline."""
    import soundfile as sf
    F = Film(story, tts_dir, cfg)
    voice = AU.voice_track(F.audio, tts_dir, F.total)
    mix, arep = AU.master(voice, *F.music_and_sfx(), target=cfg.get("lufs", -14.0), pad_gap_db=cfg.get("pad_gap_db", 20.0))
    sf.write(out_base + "_mix.wav", mix[: int(F.total * AU.SR)], AU.SR)
    rep = {"audio": arep, "total_frames": F.n_frames()}
    rep.update(F.edit_report())
    json.dump({"shots": F.shots, "captions": F.caps, "audio": F.audio, "words": F.words}, open(out_base + "_timeline.json", "w"), indent=0)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("story")
    ap.add_argument("tts_dir")
    ap.add_argument("out")
    ap.add_argument("--style", default="B")
    ap.add_argument("--sections")
    ap.add_argument("--W", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cfg", help="JSON file with extra config")
    ap.add_argument("--low", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(a.cfg)) if a.cfg else {}
    cfg.update(style=a.style, W=a.W, H=a.W * 16 // 9, fps=a.fps)
    if a.sections:
        cfg["sections"] = a.sections
    rep = render(a.story, a.tts_dir, a.out, cfg, workers=a.workers, low=a.low)
    json.dump(rep, open(os.path.splitext(a.out)[0] + "_report.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "camera"}, indent=1))


if __name__ == "__main__":
    main()
