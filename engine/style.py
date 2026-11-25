"""Heartwick look: palette, prompts, type. Single source of truth (see STYLE_BIBLE.md)."""

PALETTE = {
    "night_pine": (16, 35, 58),
    "dusk_teal": (31, 78, 92),
    "maple": (184, 67, 47),
    "ember": (255, 165, 82),
    "honey": (243, 199, 122),
    "cream": (255, 244, 226),
    "sage": (141, 166, 126),
}

# CLIP truncates at 77 tokens: the style goes FIRST and prompts stay short. A style reference image
# (brand/style_ref.png, IP-Adapter InstantStyle) keeps every plate and character in the same hand.
STYLE_PREFIX = "storybook illustration, flat 2D cartoon, gouache, soft shading"
STYLE_NEGATIVE = ("photo, photograph, photorealistic, realistic, 3d render, cgi, anime, text, letters, watermark, signature, "
                  "logo, blurry, lowres, deformed, bad anatomy, extra limbs, candle, flame, fire")
BG_SUFFIX = "empty scene, no people, vertical composition"
BG_NEG_EXTRA = "people, person, human, animal, squirrel, character"
CHAR_SUFFIX = "full body, plain light gray background"
CHAR_NEG_EXTRA = "scenery, landscape, room, multiple characters, cropped, close-up"
STYLE_REF = "brand/style_ref.png"

TINTS = {  # multiplicative RGB grade + lift per location tint
    "warm": ((1.04, 1.0, 0.94), (6, 3, 0)),
    "cool": ((0.96, 1.0, 1.05), (0, 2, 6)),
    "storm": ((0.85, 0.92, 1.05), (0, 4, 12)),
    "night": ((0.9, 0.92, 1.02), (2, 4, 10)),
    "dusk": ((1.05, 0.96, 0.98), (8, 2, 6)),
}

FONT_CAPTION = "brand/fonts/Nunito-VF.ttf"
FONT_TITLE = "brand/fonts/Fraunces-VF.ttf"
