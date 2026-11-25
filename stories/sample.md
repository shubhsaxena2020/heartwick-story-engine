---
slug: sample
title: "Sample - The Borrowed Scarf"
series_tag: "Sample"
hook:
  kicker: "A HEARTWICK STORY"
  lines: ["Her scarf kept vanishing.", "Then she found out why."]
  highlight: "why."
voice: af_heart
highlight_words: [Ada, scarf, squirrel, nest, wool, winter, gift, porch]
seo:
  primary_keyword: "sample story"
  shorts_title: "The Borrowed Scarf (sample story) #shorts"
  summary: "A placeholder story used to demo the engine: a gardener's scarf keeps disappearing, and a squirrel is to blame."
  hashtags: ["#samplestory", "#shorts"]
characters:
  nell:
    desc: "older gardener, gray hair, round glasses, blue knitted scarf, brown coat"
    seed: 1101
    poses:
      stand_mug: "standing, holding a mug with both hands, looking out, three-quarter view"
      kneel: "kneeling on the ground, reaching gently down, curious expression, side view"
      smile: "standing, hands clasped in front, warm surprised smile"
      sit_smile: "sitting on a chair holding a teacup, soft content smile"
  pip:
    desc: "young gray squirrel, big dark eyes, fluffy gray fur, bushy curled tail"
    seed: 3303
    poses:
      sit: "sitting upright on its hind legs, front paws together, looking at the viewer"
      run: "running, body stretched out, tail streaming behind, side view facing right"
      look_back: "standing on all four paws, looking back over its shoulder"
locations:
  yard_morning: {tint: cool, prompt: "a backyard lawn in late autumn, fallen leaves, a big tree, a wooden fence, soft morning light"}
  kitchen_warm: {prompt: "a cozy kitchen in warm lamplight, wooden table, a window with gray daylight, a basket of yarn"}
  porch_dusk: {tint: dusk, prompt: "a porch railing at dusk outside a house, a glowing window behind it, fading violet sky"}
---

## The missing scarf

::: shot bg=kitchen_warm cam=push_in fx=motes chars=nell:stand_mug@0.5,0.96,0.62
Every morning that November, Ada hung her blue scarf on the porch hook. And every evening, it was gone. [pause 1s]

::: shot bg=yard_morning cam=drift fx=leaves chars=pip:run@0.6,0.95,0.09
She blamed the wind. She blamed the neighbors' dog. She even blamed herself, and bought a new scarf at the market. The next week, that one vanished too. [pause 0.5s]

## The culprit

::: shot bg=yard_morning cam=push_in fx=leaves chars=nell:kneel@0.38,0.93,0.5;pip:look_back@0.72,0.92,0.09
Then, one frosty afternoon, she saw a gray tail dragging a long blue thread up the oak tree. [pause 0.5s]

::: shot bg=yard_morning cam=pull_back fx=leaves chars=nell:smile@0.4,0.95,0.55;pip:sit@0.7,0.92,0.09
High in the branches sat a nest, lined in soft blue wool. Inside, three sleepy young squirrels were curled up, warm as toast, while the wind rattled the bare branches. [pause 1s]

## A fair trade

::: shot bg=kitchen_warm cam=drift fx=motes chars=nell:sit_smile@0.45,0.97,0.58
Ada laughed, found her knitting needles, and made a second scarf, small, warm and a little crooked. [pause 0.5s]

::: shot bg=porch_dusk cam=push_in fx=leaves chars=pip:sit@0.55,0.92,0.1
She left it on the porch rail. By morning, it had disappeared, and her own scarf hung on its hook, neatly folded. Ada never did find out how. [pause 2s]
