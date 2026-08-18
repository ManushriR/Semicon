
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

def add_fiducial(img, cx, cy, size, thickness):
    """
    Stamp a cross-shaped alignment fiducial centered at (cx, cy).

    Real photolithography reticles/wafers carry unique alignment marks
    distinct from the repeating memory array specifically so a navigation
    system can anchor itself despite the array's periodicity. Without a
    landmark like this, every occurrence of a purely periodic tile is
    pixel-identical and the "true" match is undecidable from image content
    alone.
    """
    x0=max(0,cx-size); x1=min(S,cx+size+1)
    y0=max(0,cy-size); y1=min(S,cy+size+1)
    tx0=max(0,cx-thickness//2); tx1=min(S,cx+thickness//2+1)
    ty0=max(0,cy-thickness//2); ty1=min(S,cy+thickness//2+1)
    img[ty0:ty1, x0:x1] = 1.0
    img[y0:y1, tx0:tx1] = 1.0
    return img

def transform(a, angle, scale):
    M=cv2.getRotationMatrix2D((499.5,499.5),angle,scale)
    return cv2.warpAffine(a,M,(S,S),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)

def transform_point(x,y,angle,scale):
    """
    Maps a point through the same geometric transform cv2.warpAffine applies
    with getRotationMatrix2D(center, angle, scale). Note: to match the actual
    pixel motion produced by warpAffine, call this with the *negated* angle
    (see transform() call sites below) -- OpenCV's affine convention rotates
    opposite to the naive forward formula.
    """
    c=499.5; t=np.deg2rad(angle); dx=x-c; dy=y-c
    return c+scale*(np.cos(t)*dx-np.sin(t)*dy), c+scale*(np.sin(t)*dx+np.cos(t)*dy)

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
        # Generate a high-mag unit cell/field. This is the purely periodic
        # background shared by every tile occurrence -- on its own it makes
        # every occurrence indistinguishable, same as a real DRAM array.
        field, params=dram_tile(rng)
        px,py,lw,vr=params

        # A unique alignment fiducial anchors ground truth to exactly one
        # physical tile. Sized in high-mag (field/physical) pixels so that
        # after the 10x downsample to the search image it is still several
        # pixels wide, not sub-resolution.
        fid_size=int(round(rng.uniform(45,75)))
        fid_thick=int(round(rng.uniform(14,20)))
        fx=float(rng.uniform(150,850)); fy=float(rng.uniform(150,850))

        # Reference is a full 1000px high-mag field that includes the fiducial,
        # since the reference is meant to uniquely identify one real location.
        ref=add_fiducial(field.copy(), int(round(fx)), int(round(fy)), fid_size, fid_thick)
        ref=transform(ref,float(rng.uniform(-1.5,1.5)),float(rng.uniform(.985,1.015)))
        ref=np.asarray(Image.fromarray(np.uint8(ref*255)).filter(ImageFilter.GaussianBlur(float(rng.uniform(.08,.25)))),dtype=np.float32)/255
        ref=edge_bright(ref,float(rng.uniform(.08,.16)))
        ref=sensor_noise(ref,rng,shot=(180,360),read=(.006,.016))

        # Build 10x FOV by tiling the same periodic architecture everywhere,
        # then stamp the identical fiducial into exactly one tile so only
        # that tile truly matches the reference's content.
        physical=np.tile(field,(10,10))
        attempt=0
        while True:
            attempt+=1
            tile_row=int(rng.integers(1,9)); tile_col=int(rng.integers(1,9))
            angle=float(rng.uniform(-8,8)); scale=float(rng.uniform(.94,1.06))
            pre_x=tile_col*100.0+fx/10.0; pre_y=tile_row*100.0+fy/10.0
            gt_x,gt_y=transform_point(pre_x,pre_y,-angle,scale)
            if 60<gt_x<940 and 60<gt_y<940:
                break
            if attempt>50:
                raise RuntimeError("Could not place fiducial in-bounds after 50 attempts.")

        py0,px0=tile_row*S,tile_col*S
        physical[py0:py0+S,px0:px0+S]=add_fiducial(
            physical[py0:py0+S,px0:px0+S].copy(), int(round(fx)), int(round(fy)), fid_size, fid_thick
        )

        search=np.asarray(Image.fromarray(np.uint8(physical*255)).resize((S,S),Image.Resampling.BOX),dtype=np.float32)/255
        search=transform(search,angle,scale)
        gt=(gt_x,gt_y)

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
                     "fiducial_size_high_px":fid_size,"fiducial_thickness_high_px":fid_thick,
                     "rotation_deg":angle,"scale":scale,"independent_noise":True,"architecture":"DRAM"})
        if i%10==0: print("generated",i)
    pd.DataFrame(rows).to_csv(out/"metadata.csv",index=False)
    with open(out/"ground_truth.json","w") as f:
        json.dump({str(r["sample_id"]):{"center_x":r["center_x"],"center_y":r["center_y"]} for r in rows},f,indent=2)
    print(f"Generated {args.samples} DRAM pairs; GT is anchored by a unique fiducial embedded in exactly one periodic tile.")

if __name__=="__main__": main()
