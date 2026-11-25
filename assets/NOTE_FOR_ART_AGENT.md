# Note for the art/voice agent (from the engine agent, 21 Nov 2026, 21:15 IST)

Shubh's direction changed: **the target art is now V1's painted storybook style** (out/pip_pilot_v2, the SDXL +
IP-Adapter gouache look of `characters/<name>/master.png` and the `cache/pip/assets/bg_*.png` plates). Style C and
the vector parts are now secondary, though they're still rendered as the "B" comparison. Please make new character parts match V1's painted look.

How the engine (engine2/) loads your parts, so they drive the new motion system (60 fps, foot-locked walks, multi-step
kneel/sit/stand, follow-through springs):

- **Format:** `assets/characters/<name>/rig/rig.json` plus part PNGs (the format you already use: absolute `pivot_px`,
  `rest_deg`, `anchor_px`, `contacts` in image px, `springs`, `variants` for `head__blink` and `hand_*__point|fist|cup`).
- **Feet for walking:** keep `thigh_*`, `shin_*` and `foot_*` as separate parts, with `foot_*` contacts at heel and toe.
  The walk locks the stance foot's lowest contact to the ground.
- **Skirts:** a long skirt cut into thigh and shin pieces reads as trousers when kneeling or walking. If possible,
  give the skirt its own part on `pelvis` that covers both legs (the legs underneath become stockings).
- **The newborn kit:** `pip_kit` currently looks like a small adult grey squirrel. The story's kit is newborn: pink
  skin, thin grey fuzz, eyes closed.
- **Squirrel rigs:** use the engine's squirrel bone names (body, head, paw_n/f, leg_n/f, tail1-3).
- **Art source A:** I generate V1-style rig-ready painted figures on Kaggle myself (tools/hero_music.py --v1-chars) into
  `characters/<name>/v1rig/`. If you produce better painted parts, put them in `assets/characters/<name>/rig/`
  and tell the coordinator, and I'll point art source A at them.

Voices are unchanged.
