import json, time, sys
from engine import timeline, render
d='cache/_test/assets'
plan={"slug":"t","hook":{"kicker":"A HEARTWICK STORY","lines":["Something stirred","in the leaves"],"highlight":"leaves"},"series_tag":"Sample · Part 1",
 "highlight_words":["vermont","mug"],"locations":{"test":{"prompt":"x","tint":"cool"}},
 "sections":[{"title":"A","shots":[{"bg":"test","cam":"push_in","fx":["motes","leaves","mist"],"chars":[{"char":"nell","pose":"stand_mug","x":0.5,"y":0.92,"h":0.5,"anim":"idle"}],"utts":[{"id":"u_a","pause":0.5,"emph":[]}]},
  {"bg":"test","cam":"pan_right","fx":["rain","snow","lightning","fireflies"],"chars":[],"utts":[{"id":"u_b","pause":0.5,"emph":["one"]}]}]}]}
cfg={"W":540,"H":960,"fps":15,"pause_scale":0.5,"max_shot_s":4.0,"end_card_s":2.0,"cta":"Follow for Part 2"}
tl=timeline.build(plan,d,cfg); print(len(tl["shots"]),"shots", tl["total"], [(c["words"],c["hl"]) for c in tl["captions"]])
t0=time.time(); render._lower_priority()
render.render_video(tl,d,d,'cache/_test/test.mp4',cfg,workers=1,low=True); print('secs',round(time.time()-t0,1), 'frames', int(tl["total"]*15))
