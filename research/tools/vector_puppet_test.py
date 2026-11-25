"""Feasibility prototype: the reference's technique (layered vector puppet + parallax + particles) in Heartwick's own art.

Everything is SVG built per frame in Python, rasterised with resvg (CPU). Bones are nested <g transform="rotate(..)">
groups, so forward kinematics comes for free: hips > torso > neck > head, shoulder > upper arm > forearm > hand.
Pose-to-pose keys with holds every ~1-1.5 s, breathing, sway, blinks, mouth, swinging shawl tassels (secondary motion),
4 parallax layers + foreground, falling leaves, twinkling stars, camera push-in and drift.
Output: research/animation/vector_puppet/vector_puppet.mp4 (+ strip). This is a prototype, not engine code.
Usage: PYTHONPATH=<dir with resvg_py> python vector_puppet_test.py <out_dir> <fonts_dir>
"""
import io, math, os, subprocess, sys
import numpy as np
from PIL import Image
import resvg_py

W, H, FPS, DUR = 540, 960, 24, 7.0
rng = np.random.default_rng(3)


def ease(x):
    x = min(1.0, max(0.0, x)); return x * x * (3 - 2 * x)


def keyed(t, keys):
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            return v0 + (v1 - v0) * ease((t - t0) / max(1e-6, t1 - t0))
    return keys[-1][1]


DEFS = """
<defs>
 <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0A1636"/><stop offset=".55" stop-color="#22336A"/><stop offset="1" stop-color="#7A4B6E"/></linearGradient>
 <radialGradient id="moon" cx=".5" cy=".5" r=".5"><stop offset="0" stop-color="#FFF3C9"/><stop offset=".35" stop-color="#FFE29A" stop-opacity=".55"/><stop offset="1" stop-color="#FFE29A" stop-opacity="0"/></radialGradient>
 <radialGradient id="win" cx=".5" cy=".5" r=".5"><stop offset="0" stop-color="#FFC56B" stop-opacity=".9"/><stop offset="1" stop-color="#FFC56B" stop-opacity="0"/></radialGradient>
 <linearGradient id="hill1" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#2A3F78"/><stop offset="1" stop-color="#16244A"/></linearGradient>
 <linearGradient id="hill2" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#1E5A4C"/><stop offset="1" stop-color="#0E3229"/></linearGradient>
 <linearGradient id="ground" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#2E6B3F"/><stop offset="1" stop-color="#123522"/></linearGradient>
 <linearGradient id="skirt" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#5E0F22"/><stop offset=".45" stop-color="#A3203A"/><stop offset="1" stop-color="#5E0F22"/></linearGradient>
 <linearGradient id="bodice" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#0C4536"/><stop offset=".5" stop-color="#1F7A5C"/><stop offset="1" stop-color="#0C4536"/></linearGradient>
 <linearGradient id="sleeve" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#0E4A3A"/><stop offset=".5" stop-color="#25876A"/><stop offset="1" stop-color="#0E4A3A"/></linearGradient>
 <linearGradient id="shawl" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#2B4FA8"/><stop offset="1" stop-color="#172E6E"/></linearGradient>
 <radialGradient id="skin" cx=".4" cy=".35" r=".7"><stop offset="0" stop-color="#F6D3B5"/><stop offset="1" stop-color="#D49A77"/></radialGradient>
 <radialGradient id="hair" cx=".35" cy=".3" r=".8"><stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#B9BED0"/></radialGradient>
 <radialGradient id="gold" cx=".35" cy=".3" r=".8"><stop offset="0" stop-color="#FFF1B0"/><stop offset=".5" stop-color="#E7B53C"/><stop offset="1" stop-color="#9A6A12"/></radialGradient>
 <pattern id="band" width="18" height="18" patternUnits="userSpaceOnUse"><rect width="18" height="18" fill="#7A1328"/><path d="M9 2 L16 9 L9 16 L2 9Z" fill="#E7B53C"/><circle cx="9" cy="9" r="2.2" fill="#2B4FA8"/></pattern>
 <pattern id="shawlpat" width="22" height="22" patternUnits="userSpaceOnUse"><rect width="22" height="22" fill="url(#shawl)"/><circle cx="11" cy="11" r="3" fill="#E7B53C" opacity=".85"/><circle cx="0" cy="0" r="1.6" fill="#F3D27A"/><circle cx="22" cy="22" r="1.6" fill="#F3D27A"/></pattern>
 <filter id="soft"><feGaussianBlur stdDeviation="6"/></filter>
</defs>"""


def leaf(x, y, s, rot, col):
    return (f'<g transform="translate({x:.1f},{y:.1f}) rotate({rot:.1f}) scale({s:.2f})">'
            f'<path d="M0 -14 L4 -5 L13 -8 L8 1 L12 9 L3 6 L0 14 L-3 6 L-12 9 L-8 1 L-13 -8 L-4 -5Z" fill="{col}"/></g>')


STARS = [(rng.uniform(0, W), rng.uniform(0, 420), rng.uniform(0.8, 2.2), rng.uniform(0, 6.28)) for _ in range(60)]
FALL = [(rng.uniform(-40, W + 40), rng.uniform(-200, H), rng.uniform(0.6, 1.3), rng.uniform(0, 6.28), rng.choice(["#C8332B", "#E7892E", "#F2C14E", "#9E1F2F"])) for _ in range(22)]
TREE = [(rng.normal(0, 70), rng.normal(0, 45), rng.uniform(16, 30), rng.choice(["#B3243A", "#D9572A", "#E9A53A", "#8C1A2C", "#F2C14E"])) for _ in range(70)]


def scene_back(t, cam):
    s = [f'<rect width="{W}" height="{H}" fill="url(#sky)"/>']
    g = lambda f: f'transform="translate({cam[0]*f:.2f},{cam[1]*f:.2f})"'
    s.append(f'<g {g(0.1)}><circle cx="400" cy="190" r="110" fill="url(#moon)"/><circle cx="400" cy="190" r="34" fill="#FFF6D8"/>')
    for x, y, r, ph in STARS:
        a = 0.45 + 0.55 * (0.5 + 0.5 * math.sin(2 * math.pi * 0.6 * t + ph))
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.2f}" fill="#F7D98A" opacity="{a:.2f}"/>')
    s.append('</g>')
    s.append(f'<g {g(0.25)}><path d="M-60 560 Q 90 430 220 520 T 520 470 T 700 540 V 1100 H -60Z" fill="url(#hill1)"/></g>')
    # mid: farmhouse with glowing windows + maple tree
    s.append(f'<g {g(0.5)}><path d="M-60 640 Q 150 560 330 620 T 700 600 V 1100 H -60Z" fill="url(#hill2)"/>'
             '<g transform="translate(70,520)"><rect x="0" y="40" width="120" height="80" fill="#E9E1CF"/><path d="M-10 44 L60 -4 L130 44Z" fill="#7A1328"/>'
             '<rect x="20" y="62" width="22" height="26" fill="#FFC56B"/><rect x="78" y="62" width="22" height="26" fill="#FFC56B"/>'
             f'<circle cx="31" cy="75" r="40" fill="url(#win)" opacity="{0.75 + 0.2*math.sin(t*3.1):.2f}"/><circle cx="89" cy="75" r="40" fill="url(#win)" opacity="{0.75 + 0.2*math.sin(t*2.3+1):.2f}"/></g>'
             '<path d="M418 700 C 412 600 430 540 400 470 L 425 470 C 440 520 455 560 470 470 L 488 474 C 470 560 452 620 450 700Z" fill="#3B2418"/>')
    sw = 3 * math.sin(2 * math.pi * 0.2 * t)
    s.append(f'<g transform="rotate({sw:.2f} 440 560)">')
    for x, y, r, c in TREE:
        s.append(f'<circle cx="{440 + x:.1f}" cy="{420 + y:.1f}" r="{r:.1f}" fill="{c}"/>')
    s.append('</g></g>')
    s.append(f'<g {g(0.8)}><path d="M-60 760 Q 200 700 400 740 T 700 730 V 1100 H -60Z" fill="url(#ground)"/></g>')
    return "".join(s)


def character(t):
    """Jointed puppet. Pivots in character space (feet at y=880, x~270)."""
    breath = 1 + 0.012 * math.sin(2 * math.pi * 0.28 * t)
    sway = 1.0 * math.sin(2 * math.pi * 0.16 * t)
    # pose keys (degrees): reach out to the fox (t 0.8-1.4), hold, hand to heart (3.2-3.8), hold
    near_up = keyed(t, [(0, 8), (0.8, 8), (1.4, -62), (2.9, -58), (3.5, -20), (7, -18)]) + 1.5 * math.sin(2 * math.pi * 0.3 * t)
    near_fo = keyed(t, [(0, 10), (0.8, 10), (1.4, -18), (2.9, -10), (3.5, -115), (7, -110)])
    far_up = keyed(t, [(0, 14), (4.6, 14), (5.1, -30), (7, -28)]) + 1.2 * math.sin(2 * math.pi * 0.3 * t + 2)
    far_fo = keyed(t, [(0, 6), (4.6, 6), (5.1, -40), (7, -36)])
    head = keyed(t, [(0, 4), (1.0, 4), (1.4, 12), (2.6, 12), (3.0, -2), (3.4, 6), (5.2, 6), (5.6, -6), (7, -6)]) + 0.8 * math.sin(2 * math.pi * 0.21 * t)
    torso = keyed(t, [(0, 0), (0.9, 0), (1.4, 6), (2.9, 5), (3.4, -2), (7, -2)]) + sway
    blink = 1.0
    for tb in (0.5, 2.3, 4.1, 5.9):
        if tb <= t <= tb + 0.16:
            blink = abs(math.cos(math.pi * (t - tb) / 0.16))
    mouth = 2 + 3 * max(0.0, math.sin(2 * math.pi * 3.2 * t)) * (1 if 3.6 < t < 5.0 else 0)  # small 'whisper'
    tassel = [12 * math.sin(2 * math.pi * 0.9 * t + i * 0.7) + 6 * math.sin(2 * math.pi * 0.37 * t) for i in range(6)]

    def arm(sx, sy, up, fo, near):
        sleeve = "url(#sleeve)"
        return (f'<g transform="rotate({up:.2f} {sx} {sy})">'
                f'<rect x="{sx-15}" y="{sy-6}" width="30" height="96" rx="15" fill="{sleeve}"/>'
                f'<g transform="rotate({fo:.2f} {sx} {sy+86})">'
                f'<rect x="{sx-13}" y="{sy+80}" width="26" height="82" rx="13" fill="{sleeve}"/>'
                f'<rect x="{sx-14}" y="{sy+148}" width="28" height="10" rx="4" fill="url(#gold)"/>'
                f'<ellipse cx="{sx}" cy="{sy+170}" rx="13" ry="16" fill="url(#skin)"/>'
                f'</g><circle cx="{sx}" cy="{sy+86}" r="14" fill="#1A6B52"/><circle cx="{sx}" cy="{sy+86}" r="5" fill="url(#gold)"/></g>')

    s = [f'<g transform="translate(0,{880*(1-breath):.2f}) scale(1,{breath:.4f})">']
    s.append(f'<g transform="rotate({torso:.2f} 270 600)">')
    s.append(arm(238, 452, far_up, far_fo, False))
    # skirt + band + shoes
    s.append('<ellipse cx="240" cy="878" rx="22" ry="9" fill="#2A1A14"/><ellipse cx="300" cy="880" rx="22" ry="9" fill="#2A1A14"/>')
    s.append('<path d="M228 570 L312 570 Q 360 760 352 870 L 188 870 Q 180 760 228 570Z" fill="url(#skirt)"/>')
    s.append('<path d="M196 820 L346 820 L352 864 L188 864Z" fill="url(#band)"/>')
    s.append('<rect x="226" y="440" width="88" height="140" rx="30" fill="url(#bodice)"/>')
    for i in range(4):
        s.append(f'<circle cx="284" cy="{474 + i*26}" r="4.5" fill="url(#gold)"/>')
    # shawl with tassels (secondary motion)
    s.append('<path d="M214 446 Q 270 410 330 446 L 318 528 Q 270 548 222 528Z" fill="url(#shawlpat)"/>')
    for i, a in enumerate(tassel):
        x = 226 + i * 18
        s.append(f'<g transform="rotate({a:.1f} {x} 532)"><line x1="{x}" y1="532" x2="{x}" y2="560" stroke="#E7B53C" stroke-width="3"/><circle cx="{x}" cy="562" r="4" fill="url(#gold)"/></g>')
    # neck + head
    s.append(f'<g transform="rotate({head:.2f} 276 438)">')
    s.append('<rect x="262" y="400" width="26" height="44" rx="10" fill="#D9A07F"/>')
    s.append('<circle cx="252" cy="326" r="26" fill="url(#hair)"/><circle cx="252" cy="326" r="8" fill="#E7B53C" opacity=".8"/>')
    s.append('<path d="M240 380 Q 236 330 286 332 Q 330 334 330 378 L 334 392 L 326 396 Q 328 420 300 428 Q 262 432 246 412Z" fill="url(#skin)"/>')
    s.append('<path d="M238 382 Q 232 330 286 330 Q 318 330 326 352 Q 296 344 276 360 Q 262 372 258 398 Z" fill="url(#hair)"/>')
    s.append('<circle cx="262" cy="392" r="7" fill="#E9A88A"/><circle cx="262" cy="402" r="3" fill="url(#gold)"/>')
    s.append(f'<ellipse cx="309" cy="378" rx="4.2" ry="{4.6*blink+0.4:.2f}" fill="#2A1A14"/>')
    s.append('<path d="M300 366 Q 309 362 318 366" stroke="#9AA0B5" stroke-width="2.5" fill="none"/>')
    s.append('<circle cx="309" cy="378" r="11" fill="none" stroke="#C9962A" stroke-width="2.2"/>')
    s.append('<circle cx="314" cy="400" r="6" fill="#E98B7F" opacity=".45"/>')
    s.append(f'<ellipse cx="318" cy="412" rx="5" ry="{mouth/2:.2f}" fill="#8A3A3A"/>')
    s.append('</g>')
    s.append(arm(300, 452, near_up, near_fo, True))
    s.append('</g></g>')
    return "".join(s)


def fox(t):
    hop = 4 * abs(math.sin(2 * math.pi * 0.9 * t)) if 1.0 < t < 2.4 else 0
    tail = 10 * math.sin(2 * math.pi * 1.1 * t)
    return (f'<g transform="translate(410,{830-hop:.1f})">'
            f'<g transform="rotate({tail:.1f} -20 10)"><path d="M-20 10 Q -70 -10 -64 30 Q -44 40 -20 22Z" fill="#E0702C"/><path d="M-60 26 Q -64 34 -52 36 Q -50 30 -60 26Z" fill="#FFF3E2"/></g>'
            '<ellipse cx="0" cy="14" rx="28" ry="20" fill="#E0702C"/><circle cx="22" cy="-8" r="16" fill="#E8833A"/>'
            '<path d="M14 -20 L 18 -40 L 26 -22Z" fill="#E0702C"/><path d="M26 -20 L 34 -38 L 36 -18Z" fill="#E0702C"/>'
            '<path d="M30 -2 L 44 0 L 30 6Z" fill="#FFF3E2"/><circle cx="27" cy="-10" r="2.6" fill="#1A1210"/></g>')


def frame_svg(t, fonts_ok=True):
    push = 1.0 + 0.06 * ease(t / DUR)
    cam = (-14 * math.sin(2 * math.pi * t / DUR), -10 * ease(t / DUR))
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', DEFS]
    s.append(f'<g transform="translate({W/2},{H*0.62}) scale({push:.4f}) translate({-W/2},{-H*0.62})">')
    s.append(scene_back(t, cam))
    s.append(f'<g transform="translate({cam[0]:.2f},{cam[1]:.2f})">{character(t)}{fox(t)}</g>')
    # foreground leaves at 1.5x parallax
    s.append(f'<g transform="translate({cam[0]*1.5:.2f},{cam[1]*1.5:.2f})">')
    s.append(leaf(30, 930, 4.2, 20, "#7A1328") + leaf(520, 945, 3.6, -30, "#B3243A") + leaf(470, 905, 2.2, 60, "#E9A53A"))
    s.append('</g></g>')
    for x, y, sc, ph, c in FALL:
        yy = (y + 55 * t * sc) % (H + 200) - 100
        xx = x + 25 * math.sin(2 * math.pi * 0.35 * t + ph)
        s.append(leaf(xx, yy, sc, (t * 90 * sc + ph * 57) % 360, c))
    s.append(f'<rect width="{W}" height="{H}" fill="none" stroke="#000" stroke-opacity=".35" stroke-width="80" filter="url(#soft)"/>')
    s.append('<text x="32" y="118" font-family="Nunito" font-weight="800" font-size="13" letter-spacing="2.5" fill="#FFF4E2" fill-opacity=".85">HEARTWICK STORIES</text>'
             '<text x="32" y="142" font-family="Fraunces" font-style="italic" font-size="17" fill="#F3C77A">The Fox Under the Maple</text>')
    chunks = [(0, 1.3, "She found him", "found"), (1.3, 2.6, "under the old maple", "maple"), (2.6, 3.9, "shivering in", None),
              (3.9, 5.2, "the red leaves", "leaves"), (5.2, 7.0, "and smiled.", None)]
    for a, b, txt, hi in chunks:
        if a <= t < b:
            op = min(1.0, (t - a) / 0.22)
            words = txt.split(" ")
            tsp = "".join(f'<tspan fill="{"#FFA552" if w.strip(".,") == hi else "#FFF4E2"}">{w} </tspan>' for w in words)
            s.append(f'<ellipse cx="{W/2}" cy="{H*0.70-10}" rx="{len(txt)*9+30}" ry="34" fill="#10233A" opacity="{0.55*op:.2f}" filter="url(#soft)"/>')
            s.append(f'<text x="{W/2}" y="{H*0.70}" text-anchor="middle" font-family="Nunito" font-weight="900" font-size="33" opacity="{op:.2f}">{tsp}</text>')
    s.append('</svg>')
    return "".join(s)


def main(out_dir, fonts_dir):
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "vector_puppet.mp4")
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-threads", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out], stdin=subprocess.PIPE)
    keep = []
    import time
    t0 = time.time()
    n = int(DUR * FPS)
    for f in range(n):
        t = f / FPS
        png = resvg_py.svg_to_bytes(svg_string=frame_svg(t), font_dirs=[fonts_dir])
        im = Image.open(io.BytesIO(bytes(png))).convert("RGBA")
        p.stdin.write(im.tobytes())
        if f % 14 == 0:
            keep.append(im.convert("RGB"))
    p.stdin.close(); p.wait()
    dt = time.time() - t0
    strip = Image.new("RGB", (W // 2 * len(keep[:12]), H // 2))
    for i, im in enumerate(keep[:12]):
        strip.paste(im.resize((W // 2, H // 2)), (i * W // 2, 0))
    strip.save(os.path.join(out_dir, "vector_puppet_strip.jpg"), quality=88)
    keep[len(keep) // 2].save(os.path.join(out_dir, "vector_puppet_frame.png"))
    print(f"rendered {n} frames in {dt:.1f}s ({dt/n*1000:.0f} ms/frame at {W}x{H})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
