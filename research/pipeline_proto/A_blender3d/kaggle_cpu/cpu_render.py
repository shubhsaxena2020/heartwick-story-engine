# Prototype A on a Kaggle CPU kernel (no GPU, no queue): portable Blender 4.5 LTS, CC0 assets from OpenGameArt,
# render the first 30 frames of the Nell beat with EEVEE (software GL) and with Cycles CPU, report s/frame.
import base64, json, os, subprocess, time
from pathlib import Path
W = Path("/kaggle/working"); W.mkdir(parents=True, exist_ok=True)
def sh(c, t=None):
    t0 = time.time(); r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
    open(W / "log.txt", "a").write(f"\n$ {c}\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}\n"); return r.returncode, round(time.time() - t0, 1), r.stdout
(W / "nell_beat.py").write_text(base64.b64decode("__SCRIPT__").decode())
(W / "bg_plate.jpg").write_bytes(base64.b64decode("__PLATE__"))
st = {}
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/130 Safari/537.36"
st["dl"] = sh(f"cd /tmp && wget -q https://download.blender.org/release/Blender4.5/blender-4.5.9-linux-x64.tar.xz && tar xf blender-4.5.9-linux-x64.tar.xz && "
              f"mkdir -p ual && cd ual && wget -q -U '{UA}' https://opengameart.org/sites/default/files/universal_animation_librarystandard.zip && "
              f"wget -q -U '{UA}' https://opengameart.org/sites/default/files/universal_animation_library_2standard.zip && unzip -q -o '*.zip'", 1200)[:2]
sh("apt-get install -y -qq libegl1 libgl1 libxi6 libxkbcommon0 libsm6 > /dev/null 2>&1; nproc; free -g")
B = "/tmp/blender-4.5.9-linux-x64/blender"
# limit to 30 frames: patch frame_end
src = (W / "nell_beat.py").read_text().replace("bpy.ops.render.render(animation=True)", "sc.frame_end = 30\nbpy.ops.render.render(animation=True)")
(W / "nell_30.py").write_text(src)
for eng in ["EEVEE", "CYCLES"]:
    out = f"/kaggle/working/{eng.lower()}"; os.makedirs(out, exist_ok=True)
    rc, s, o = sh(f"{B} -b --factory-startup -P /kaggle/working/nell_30.py -- /tmp/ual /kaggle/working/bg_plate.jpg {out} {eng} 0.75 2>&1 | grep -E 'RENDER_S|Error|rror|Trace' | tail -5", 3000)
    st[eng] = {"rc": rc, "wall_s": s, "out": o[-600:]}
    (W / "status.json").write_text(json.dumps(st, indent=1))
sh("rm -f /kaggle/working/*/nell_beat.blend")
print(json.dumps(st, indent=1))
