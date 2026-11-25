"""Series bible for engine v3: how each recurring character looks (the words every key prompt repeats), the fixed
framing sentences per framing class, and the pose-skeleton library ids the shot writer may use."""

LOOK = {
    "nell": ("Nell, a kind older woman about 70 with silver hair in a low bun, round glasses, a dark green knitted cardigan "
             "over a cream blouse, a long rust-brown skirt and brown shoes"),
    "pip": "Pip, a small grey squirrel with a pale belly, big dark eyes and a bushy tail curled like a question mark",
    "pip_baby": "Pip as a tiny baby grey squirrel, damp fur, eyes barely open, small enough to fit in one palm",
}

STYLE = ("Hand-drawn 2D anime film still, Studio-Ghibli-like warm watercolor backgrounds, clean line art, soft light, "
         "vertical 9:16 frame. No text, no watermark.")

KEEP = ("Keep every character's face, hair, glasses, clothes and colours identical to the reference images, keep the exact "
        "same hand-drawn anime art style, line art and colour palette. No text, no skeleton lines in the result.")

FRAME = {
    "wide": ("Vertical 9:16 wide shot: everyone is fully inside the frame with generous space around them, the setting "
             "clearly visible, nothing cut off by the frame edges."),
    "medium-full": ("Vertical 9:16 composition: her whole body from the top of her hair to her shoes is inside the frame, "
                    "her head in the upper third with clear space above her hair, nothing cut off by the frame edges."),
    "medium": ("Vertical 9:16 medium shot from the waist up: her head in the upper third with clear space above her hair, "
               "hands and anything she holds inside the frame, nothing important cut off."),
    "close": ("Vertical 9:16 close shot: her face and hands fill the middle of the frame with space above her hair, "
              "nothing important cut off."),
    "insert": "Vertical 9:16 close shot centred on the subject, with space around it, nothing important cut off.",
}

POSES = ["none", "stand_look", "stand_walk", "kneel_reach", "kneel_hold", "sit_table", "stand_window"]

# Pip's size relative to Nell's head box (sqrt-area ratio from the QA gate), per size class; QA allows +-15 %.
PIP_SCALE = {"baby": 0.55, "adult": 1.0}
