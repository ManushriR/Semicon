
import argparse, json, math
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

S=1000

def edge_bright(a, strength):
    gx=cv2.Sobel(a,cv2.CV_32F,1,0,3); gy=cv2.Sobel(a,cv2.CV_32F,0,1,3)
    mag=cv2.magnitude(gx,gy); mag=cv2.normalize(mag,None,0,1,cv2.NORM_MINMAX)
    return np.clip(a + strength*mag,0,1)

def sensor_noise(a, rng, shot=(70,150), read=(.012,.03)):
    p=float(rng.integers(*shot))
    out=rng.poisson(np.clip(a,0,1)*p).astype(np.float32)/p
    out += rng.normal(0,float(rng.uniform(*read)),a.shape).astype(np.float32)
    return np.clip(out,0,1)

def dram_tile(rng):
    # A 1000x1000 high-mag DRAM-style field.
    img=np.zeros((S,S),np.float32)
    px=float(rng.uniform(18,32)); py=float(rng.uniform(18,32))
    lw=float(rng.uniform(2.0,5.0)); vr=float(rng.uniform(1.0,2.8))
    phx=float(rng.uniform(0,px)); phy=float(rng.uniform(0,py))
    for x in np.arange(phx,S,px):
        c=int(round(x)); w=max(1,int(round(lw*rng.uniform(.9,1.1))))
        img[:,max(0,c-w//2):min(S,c+w//2+1)] = rng.uniform(.72,.94)
    for y in np.arange(phy,S,py):
        c=int(round(y)); w=max(1,int(round(lw*rng.uniform(.9,1.1))))
        img[max(0,c-w//2):min(S,c+w//2+1),:] = rng.uniform(.70,.92)
    for x in np.arange(phx,S,px):
        for y in np.arange(phy,S,py):
            cx,cy=int(round(x)),int(round(y)); r=max(1,int(round(vr*rng.uniform(.85,1.15))))
            yy0=max(0,cy-r); yy1=min(S,cy+r+1); xx0=max(0,cx-r); xx1=min(S,cx+r+1)
            yy,xx=np.ogrid[yy0:yy1,xx0:xx1]
            m=(xx-cx)**2+(yy-cy)**2<=r*r
            img[yy0:yy1,xx0:xx1][m]=1.0
    return img, (px,py,lw,vr)

def transform(a, angle, scale):
    M=cv2.getRotationMatrix2D((499.5,499.5),angle,scale)
    return cv2.warpAffine(a,M,(S,S),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)

def center_nearest_periodic(gt_candidates):
    return min(gt_candidates,key=lambda p: math.hypot(p[0]-499.5,p[1]-499.5))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--architecture",choices=["DRAM","FinFET"],default="DRAM")
    ap.add_argument("--samples",type=int,default=50)
    ap.add_argument("--out",default="dataset")
    ap.add_argument("--seed",type=int,default=20260817)
    args=ap.parse_args()
    if args.architecture!="DRAM":
        raise NotImplementedError("Fast submission build currently uses DRAM.")
    rng=np.random.default_rng(args.seed)
    out=Path(args.out); (out/"reference").mkdir(parents=True,exist_ok=True); (out/"search").mkdir(parents=True,exist_ok=True)
    rows=[]
    for i in range(1,args.samples+1):
        # Generate a high-mag unit cell/field. Reference is a crop of it.
        field, params=dram_tile(rng)
        px,py,lw,vr=params
        # Reference is deliberately a full 1000px high-mag field.
        ref=field.copy()
        ref=transform(ref,float(rng.uniform(-1.5,1.5)),float(rng.uniform(.985,1.015)))
        ref=np.asarray(Image.fromarray(np.uint8(ref*255)).filter(ImageFilter.GaussianBlur(float(rng.uniform(.08,.25)))),dtype=np.float32)/255
        ref=edge_bright(ref,float(rng.uniform(.08,.16)))
        ref=sensor_noise(ref,rng,shot=(180,360),read=(.006,.016))
        # Build 10x FOV by tiling the same physical architecture, then downsample.
        # This deliberately creates periodic ambiguity; official tie rule resolves it.
        physical=np.tile(field,(10,10))
        search=np.asarray(Image.fromarray(np.uint8(physical*255)).resize((S,S),Image.Resampling.BOX),dtype=np.float32)/255
        angle=float(rng.uniform(-8,8)); scale=float(rng.uniform(.94,1.06))
        search=transform(search,angle,scale)
        # Every 100x100 tile is a legitimate occurrence before the global transform.
        candidates=[]
        for cy in np.arange(50,1000,100):
            for cx in np.arange(50,1000,100):
                x,y=transform_point(cx,cy,angle,scale)
                if 45<x<955 and 45<y<955: candidates.append((x,y))
        gt=center_nearest_periodic(candidates)
        search=np.asarray(Image.fromarray(np.uint8(search*255)).filter(ImageFilter.GaussianBlur(float(rng.uniform(.20,.60)))),dtype=np.float32)/255
        search=edge_bright(search,float(rng.uniform(.12,.30)))
        search=sensor_noise(search,rng,shot=(45,95),read=(.025,.055))
        search=np.clip((search-.5)*rng.uniform(.85,1.15)+.5+rng.uniform(-.04,.04),0,1)
        rp=f"ref_{i:04d}.png"; sp=f"search_{i:04d}.png"
        Image.fromarray(np.uint8(ref*255),"L").save(out/"reference"/rp)
        Image.fromarray(np.uint8(search*255),"L").save(out/"search"/sp)
        rows.append({"sample_id":i,"reference_image":f"reference/{rp}","search_image":f"search/{sp}",
                     "center_x":gt[0],"center_y":gt[1],"reference_width":1000,"reference_height":1000,
                     "search_width":1000,"search_height":1000,"physical_scale_ratio":10,
                     "pitch_x_high_px":px,"pitch_y_high_px":py,"line_width_high_px":lw,"via_radius_high_px":vr,
                     "rotation_deg":angle,"scale":scale,"independent_noise":True,"architecture":"DRAM"})
        if i%10==0: print("generated",i)
    pd.DataFrame(rows).to_csv(out/"metadata.csv",index=False)
    with open(out/"ground_truth.json","w") as f:
        json.dump({str(r["sample_id"]):{"center_x":r["center_x"],"center_y":r["center_y"]} for r in rows},f,indent=2)
    print(f"Generated {args.samples} DRAM pairs; GT uses the closest matching periodic occurrence to search center.")

def transform_point(x,y,angle,scale):
    c=499.5; t=np.deg2rad(angle); dx=x-c; dy=y-c
    return c+scale*(np.cos(t)*dx-np.sin(t)*dy), c+scale*(np.sin(t)*dx+np.cos(t)*dy)

if __name__=="__main__": main()
