# Heartwick assets: rich characters and layered scenes for engine v2

Art direction B (vector cel), raised toward the reference's richness in our own Western folk-tale world:

- jewel tones (ruby, emerald, sapphire, saffron) with gold trim, gem cabochons and gold buttons;
- patterned textiles: cable knit, quilting, tartan, calico sprigs, a folk damask of tulips and rosettes;
- airbrushed cylinder shading with rim light baked into every part;
- 5-6 depth layers per scene, with aerial-perspective haze.

Everything is drawn in code (`assets/src/`), so it is crisp, consistent between shots and cheap to change. Nothing
comes from the reference: no deities, logo, lettering or patterns, and no candles or flames.

Comparison sheets:

- `previews/before_after.jpg`: the engine-v2 style-B test frames against the new assets, rendered by engine v2's own
  renderer.
- `research/parabrahma/sheets/art_vs_reference.jpg`: the same frames next to the reference. This file is local only
  and git-ignored, because it contains third-party frames.

## Characters (`characters/<name>/rig/`)

| Name | Kind | Design | Cloth / spring chains |
|---|---|---|---|
| `nell` | humanoid | Older Vermont widow. Silver braided bun with a tortoiseshell comb, gold spectacles, emerald cable cardigan with gold buttons and cuffs, sprig-print blouse, ruby skirt with a sapphire folk-embroidered hem | `scarf1-3`: quilted sapphire shawl ends with a gold-beaded fringe |
| `dee` | humanoid | Wildlife rehabber in her fifties. Ruby tartan shirt, quilted slate-blue vest with a folk rosette patch, jeans, glossy green boots | `hair1-2`: braid |
| `eli` | humanoid | Boy of nine. Fair-Isle yoke under a ruby zip jacket, freckles, sandy tufts, sneakers (child proportions) | none |
| `pip` | squirrel | A real Eastern grey squirrel: agouti grey with a rusty wash, white belly and eye ring, ear tufts, a long bushy tail with a frosted silver fringe | `tail2-3` |
| `pip_kit` | squirrel | Pip as an orphaned kit: fuzzy grey, eyes barely open, short fluffy tail. No longer pink (see "Decisions" below) | `tail2-3` |
| `king` | humanoid | Folk-tale king. Jewelled crown, sapphire tunic with a gold folk damask, gem belt, chain of office, ruby cape with an ermine hem | `cape1-3` |
| `elder` | humanoid | Wise elder. Long white hair and beard, deep-green hooded robe with an embroidered plum hem, rope belt, satchel, carved staff | none (staff in `hand_n`) |
| `hare` | squirrel rig | Folk-tale animal: a brown hare with black-tipped ears and a sapphire ribbon with a gold bell | none |

`preview.png` in each character folder shows the sheet loaded with `engine2.rig.load_partsheet`, posed by
`engine2/poses.py` (stand, point, heart, arms_open, kneel_reach, sit_hold, walk; squirrels: sit, sit_look_up,
nibble, alert, crouch, look_back, run).

### How the rig uses the parts

The format is engine v2's raster part sheet (`engine2/rig.py: load_partsheet`): `rig.json` plus one PNG per part.
There are no new formats. Today the engine loads these with `cfg["raster_dir"] = "assets/characters"`, because
`direct.rig_for` looks for `<raster_dir>/<name>/rig/rig.json`.

**Bones and pivots**

- Every bone has `rest_deg: 0`. The parts are drawn in the bone's canonical frame: limbs along +y, the spine, neck
  and head along -y.
- So `pivot_px` is simply the rest-pose joint position.
- Each part's `anchor_px` is where its joint pivot sits inside its PNG. The part rotates about that point.
- The bone names and lengths match `engine2/characters.py`: thigh 232, shin 222, spine 172, chest 150, neck 38,
  upper arm 168, forearm 148, and Eli's child sizes. So every pose, the walk cycle and the sitting heights
  (`root_y`) fit unchanged.

**Parts per humanoid**

- Body: `pelvis`, `spine` (waist), `chest`, `neck`, `head`.
- Each side, near `_n` and far `_f`:
  - upper arm, forearm and hand;
  - thigh, shin and foot.
- Far-side parts are painted about 20% darker.
- Joints overlap with round caps, so rotation never opens a gap.

**Long skirts and robes (Nell, Elder)** are rigid cloth panels:

| Panel | Attached to | z |
|---|---|---|
| `skirt_hip` (hip bell) | `pelvis` | 20 |
| `skirt_drape` (inner drape; fills the space under the lap when she kneels or sits) | `pelvis` | 12.5 |
| `skirt_n` / `skirt_f` (one per thigh) | `thigh_n` / `thigh_f` | 21 / 9.5 |
| `hem_n` / `hem_f` (embroidered hem, tucked behind its thigh panel) | `shin_n` / `shin_f` | 20.5 / 9 |

**Draw order (`z`, low = behind)**

| z | Parts |
|---|---|
| 3-5 | cape / tail |
| 9-12 | far leg and far skirt panels |
| 12-15 | far arm |
| 16-18 | near leg |
| 20-21 | skirt |
| 29-31 | hips, waist, chest |
| 32 | neck |
| 33 | hair, braid, collar |
| 34 | head |
| 35-36 | scarf |
| 40-43 | near arm and hand |

The order is the same as the vector cast.

**Extra keys in `rig.json`.** The current loader ignores them; each needs a small engine change to use:

- `"springs"`: the spring chains, in the same dicts as `engine2/characters.py`.
  - `direct.rig_for` currently returns `[]` springs for raster rigs.
  - Returning `json.load(...)["springs"]` there makes the scarf, cape, braid and tails sway.
- `"variants"`: alternative PNGs for a part, each with its own anchor.
  - Hands: `hand_n__point / fist / cup` and `hand_f__…`, matching the poses' `hand_n` / `hand_f` field.
  - Blinks: `head__blink`.
  - Swapping a `RasterPart.image` per frame, from `face["hand_n"]` or `face["blink"] < 0.5`, gives hand shapes and
    blinks.
- `"props"` (elder): the staff is baked into `hand_n`. Its `open / point / cup` variants are empty hands.

**Squirrel rigs** (`pip`, `pip_kit`, `hare`):

- They use the vector squirrel's skeleton exactly: body along +x, head at (128, -14), and tail bones running along -x.
- `units` is 260 for Pip, 200 for the kit and 320 for the hare.
- The hare's sitting posture fits the squirrel pose library (sit, alert, nibble, crouch, run).

## Scenes (`scenes/<name>/`)

| Scene | World | Layers (depth) | Engine fx in `scene.json` |
|---|---|---|---|
| `village_autumn` | 1400x3000, tall (for tilt reveals) | sky .08, far .3 (three Green-Mountain ridges with autumn forest), mid .6 (white steeple church, covered bridge, barn with a barn star, houses, maples), ground .92 (farmhouse, giant maple, stone walls, road, pumpkins), actors, fore 1.25 (grass, a framing maple branch) | leaves, motes, sun rays, mist, birds |
| `maple_yard` | 1400x2100 | sky, far (ridges, church), mid (barn, picket fence, maples), ground (Nell's farmhouse with porch, the big maple, leaf pile, storm branch, rake), actors, fore | leaves, motes, rays, birds |
| `kitchen` | 1400x2100 | outside .3 (the maple through the glass), room .55 (folk-tulip wallpaper, sage beadboard with a gold rail, ruby curtains with gold tiebacks, plate shelf, jars, copper pots, clock, braided rug), furniture .85 (closed cast-iron cookstove, table with a quilted runner, mugs, pie, geraniums), actors, fore 1.2 (dried herbs, apple basket) | motes, window rays, lamp glow |
| `night_sky` | 1400x3000, tall | sky (Milky-Way band in teal and rose, moon, stars), far (ridges, church), mid (houses with lit windows, maples), ground (house, fence), actors, fore | twinkling stars, fireflies, mist |
| `storm` | 1400x2100 | sky (storm banks, lightning bolt), far, mid (house with a lit window, fence), ground (wind-bent maple, puddles reflecting the window), actors, fore | rain, leaves, lightning flash |
| `castle_court` | 1400x2100 | sky, far (hilltop castle with jewel-roofed towers and banners), mid (half-timbered houses, bunting), ground (cobbles, gilded fountain), actors, fore | petals, motes, rays, birds |

**Loading.** `scenes/load_scene.py` turns a folder into an `engine2.scene.Scene`:

```python
from assets.scenes.load_scene import load_scene
scene = load_scene("assets/scenes/kitchen", painter)
```

- It is a drop-in for `scenery.build_location`: same world size, ground line, parallax depths, margin and `spots`.
- Static plates are cached once per scene by `engine2.scene`.
- Movement comes from engine v2's own `fx`: particles, god rays, mist, twinkles, lightning and birds.
- The loader only imports from `engine2`; it does not change the engine.

**Storage.**

- Layers are PNG at 1.25 px per world px, so close framings stay sharp. Opaque skies are JPEG.
- The whole set is about 15 MB.

## Regenerate

```
python assets/src/chars.py [nell dee eli king elder]
python assets/src/animals.py [pip pip_kit hare]
python assets/src/scenes.py [village_autumn maple_yard night_sky storm kitchen castle_court]
python assets/src/preview.py
python assets/src/compose.py
python assets/src/compare.py
```

Each step is single-threaded CPU work: about 1 minute for everything.

## Known limits

- **Skirts kneel as rigid panels.** That is puppet-like and reads fine at Shorts size, but the vector rig's
  hull-deformed skirt is smoother. A hybrid would be better: keep the vector skirt for Nell and use raster parts
  for the rest of her.
- **No mouth shapes.** The reference has no lip sync either.
- **Hand shapes and blinks** need the variant swap described above.
- **AI raster parts** (Z-Image-Turbo or SDXL, then cut out and inpainted) were not made in this pass:
  - the research report found that limbs tear when cut out of a single painting;
  - the previous Z-Image run on a T4 produced NaN (black) images and was then killed (rc -9).

  The vector library already beats the earlier test clearly, rigs exactly, and needs no GPU.
