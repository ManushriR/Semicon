import os,math,random,argparse,shutil,json
import numpy as np,pandas as pd
from PIL import Image,ImageDraw,ImageFilter

p=argparse.ArgumentParser();p.add_argument('--samples',type=int,default=300);p.add_argument('--out',default='dataset');p.add_argument('--seed',type=int,default=20260817);a=p.parse_args()
rng=np.random.default_rng(a.seed);random.seed(a.seed); S=1000
if os.path.exists(a.out): shutil.rmtree(a.out)
for d in ['reference','search','visualization']: os.makedirs(os.path.join(a.out,d))

def edge(x,k):
 gx=np.zeros_like(x);gy=np.zeros_like(x);gx[:,1:-1]=(x[:,2:]-x[:,:-2])*.5;gy[1:-1]=(x[2:,:]-x[:-2,:])*.5
 g=np.sqrt(gx*gx+gy*gy);g=(g-g.min())/(g.max()-g.min()+1e-8);return np.clip(x+k*g,0,1)
def noise(x,p,s): return np.clip(rng.poisson(np.clip(x,0,1)*p)/p+rng.normal(0,s,x.shape),0,1)
def dram(px,py,lw,vr,ox,oy,jit):
 a=np.zeros((S,S),np.float32);xs=[];ys=[];v=ox
 while v<S: xs.append(int(v));v+=px
 v=oy
 while v<S: ys.append(int(v));v+=py
 for x in xs:
  w=max(1,int(lw*rng.uniform(.85,1.15)));a[:,max(0,x-w//2):min(S,x+w//2+1)]=rng.uniform(.82,1)
 for y in ys:
  w=max(1,int(lw*rng.uniform(.85,1.15)));a[max(0,y-w//2):min(S,y+w//2+1),:]=rng.uniform(.80,1)
 for x in xs:
  for y in ys:
   xx=x+int(rng.normal(0,jit));yy=y+int(rng.normal(0,jit));r=max(1,int(vr*rng.uniform(.8,1.2)))
   y0,y1=max(0,yy-r),min(S,yy+r+1);x0,x1=max(0,xx-r),min(S,xx+r+1);Y,X=np.ogrid[y0:y1,x0:x1];m=(X-xx)**2+(Y-yy)**2<=r*r;a[y0:y1,x0:x1][m]=1
 return a
def save(x,path): Image.fromarray(np.uint8(np.clip(x,0,1)*255)).save(path)
rows=[]
for i in range(1,a.samples+1):
 px=float(rng.uniform(18,42));py=float(rng.uniform(18,42));lw=float(rng.uniform(2,6));vr=float(rng.uniform(1,3));jit=float(rng.uniform(0,.8));ox=float(rng.uniform(0,px));oy=float(rng.uniform(0,py));
 diff=rng.choice(['easy','medium','hard','extreme'],p=[.25,.4,.25,.1]);cfg={'easy':(-1,1,.99,1.01,.1,.4,.004,.015,250,450),'medium':(-3,3,.96,1.04,.3,.8,.01,.03,150,300),'hard':(-5,5,.93,1.07,.5,1.2,.02,.055,80,180),'extreme':(-8,8,.90,1.10,.8,1.8,.04,.09,40,100)}[diff]
 amin,amax,smin,smax,bmin,bmax,nmin,nmax,pmin,pmax=cfg;ang=float(rng.uniform(amin,amax));scale=float(rng.uniform(smin,smax));blur=float(rng.uniform(bmin,bmax));ns=float(rng.uniform(nmin,nmax));ps=int(rng.integers(pmin,pmax));es=float(rng.uniform(.08,.28));base=dram(px,py,lw,vr,ox,oy,jit);tx=int(rng.integers(150,850));ty=int(rng.integers(150,850))
 ref=base[ty-50:ty+50,tx-50:tx+50].copy();ref=np.asarray(Image.fromarray(np.uint8(ref*255)).filter(ImageFilter.GaussianBlur(rng.uniform(.05,.35))),dtype=np.float32)/255;ref=edge(ref,rng.uniform(.06,.16));ref=noise(ref,int(rng.integers(220,450)),rng.uniform(.003,.015))
 im=Image.fromarray(np.uint8(base*255)).rotate(ang,Image.Resampling.BICUBIC,expand=False,fillcolor=0);sea=np.asarray(im,dtype=np.float32)/255;c=499.5;t=math.radians(ang);dx=tx-c;dy=ty-c;rx=math.cos(t)*dx-math.sin(t)*dy;ry=math.sin(t)*dx+math.cos(t)*dy;gx=float(np.clip(c+rx*scale,150,850));gy=float(np.clip(c+ry*scale,150,850));sea=np.asarray(Image.fromarray(np.uint8(sea*255)).filter(ImageFilter.GaussianBlur(blur)),dtype=np.float32)/255;sea=edge(sea,es);sea=noise(sea,ps,ns);sea=np.clip((sea-.5)*rng.uniform(.85,1.15)+.5+rng.uniform(-.05,.05),0,1)
 rn=f'ref_{i:05d}.png';sn=f'search_{i:05d}.png';save(ref,os.path.join(a.out,'reference',rn));save(sea,os.path.join(a.out,'search',sn));v=Image.fromarray(np.uint8(sea*255)).convert('RGB');ImageDraw.Draw(v).rectangle([gx-50,gy-50,gx+50,gy+50],outline='red',width=3);v.save(os.path.join(a.out,'visualization',f'pair_{i:05d}.png'))
 rows.append(dict(sample_id=i,reference_image=f'reference/{rn}',search_image=f'search/{sn}',center_x=gx,center_y=gy,pitch_x=px,pitch_y=py,line_width=lw,via_radius=vr,rotation_deg=ang,scale=scale,blur_sigma=blur,noise_sigma=ns,poisson_strength=ps,edge_strength=es,difficulty=diff))
pd.DataFrame(rows).to_csv(os.path.join(a.out,'metadata.csv'),index=False)
json.dump({str(r['sample_id']):{'x':r['center_x'],'y':r['center_y']} for r in rows},open(os.path.join(a.out,'ground_truth.json'),'w'),indent=2)
print('Generated',a.samples)
