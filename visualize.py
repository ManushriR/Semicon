
import argparse
from pathlib import Path
import cv2, pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",default="dataset"); ap.add_argument("--results",default="results")
    ap.add_argument("--sample",type=int,default=1); ap.add_argument("--out",default="results/example.png")
    a=ap.parse_args(); ds=Path(a.dataset); meta=pd.read_csv(ds/"metadata.csv"); row=meta.loc[meta.sample_id==a.sample].iloc[0]
    ref=cv2.imread(str(ds/row.reference_image),0); sea=cv2.imread(str(ds/row.search_image),0)
    # Save a side-by-side contact sheet with GT marker.
    ref=cv2.resize(ref,(500,500)); sea=cv2.resize(sea,(500,500)); canvas=cv2.hconcat([ref,sea])
    x=int(row.center_x/2)+500; y=int(row.center_y/2)
    cv2.drawMarker(canvas,(x,y),(255,),cv2.MARKER_CROSS,30,2)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(a.out,canvas); print(a.out)
if __name__=="__main__": main()
