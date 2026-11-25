import subprocess, sys, numpy as np
from PIL import Image, ImageDraw
src,out=sys.argv[1],sys.argv[2]; spans=[tuple(map(float,s.split(':'))) for s in sys.argv[3:]]
W,H=180,320; rows=[]
for a,b in spans:
    raw=subprocess.run(["ffmpeg","-v","error","-threads","2","-ss",str(a),"-t",str(b-a),"-i",src,"-vf",f"fps=2,scale={W}:{H}","-f","rawvideo","-pix_fmt","rgb24","-"],capture_output=True).stdout
    fr=np.frombuffer(raw,np.uint8).reshape(-1,H,W,3); rows.append((a,fr))
cols=max(len(f) for _,f in rows); img=Image.new("RGB",(cols*W,len(rows)*H)); d=ImageDraw.Draw(img)
for r,(a,fr) in enumerate(rows):
    for c,f in enumerate(fr):
        img.paste(Image.fromarray(f),(c*W,r*H)); d.text((c*W+3,r*H+3),f"{a+c*0.5:.1f}",fill=(255,255,0))
img.save(out); print(out,img.size)
