import subprocess, numpy as np, cv2, json, sys
src=sys.argv[1]; W,H,FPS=135,240,10
raw=subprocess.run(["ffmpeg","-v","error","-threads","2","-i",src,"-vf",f"fps={FPS},scale={W}:{H}","-f","rawvideo","-pix_fmt","rgb24","-"],capture_output=True).stdout
fr=np.frombuffer(raw,np.uint8).reshape(-1,H,W,3)
mask=np.ones((H,W),np.uint8); mask[int(H*.62):int(H*.80)]=0; mask[:int(H*.12)]=0
def h(f):
    hh=cv2.calcHist([cv2.cvtColor(f,cv2.COLOR_RGB2HSV)],[0,1],mask,[18,8],[0,180,0,256]).ravel(); return hh/hh.sum()
hs=np.array([h(f) for f in fr]); L=FPS  # 1 s lag
d=np.array([0.5*np.abs(hs[i]-hs[i-L]).sum() if i>=L else 0 for i in range(len(hs))])
# peaks above 0.3, min separation 1.0 s
peaks=[]
for i in range(L,len(d)-1):
    if d[i]>0.28 and d[i]==d[max(0,i-5):i+6].max():
        if not peaks or i-peaks[-1]>=FPS: peaks.append(i)
ts=[round((p-L/2)/FPS,1) for p in peaks]
print(len(ts),"transitions:",ts)
dur=len(fr)/FPS; print("dur",dur,"shots/min",round((len(ts)+1)/dur*60,1))
gaps=np.diff([0]+ts+[dur]); print("mean shot s",round(gaps.mean(),2),"median",round(float(np.median(gaps)),2),"min",round(gaps.min(),2),"max",round(gaps.max(),2))
json.dump({"transitions":ts,"shot_lengths":[round(g,2) for g in gaps]},open("analysis/transitions.json","w"))
