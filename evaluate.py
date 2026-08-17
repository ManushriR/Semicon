
import argparse, math, time, json
from pathlib import Path
import numpy as np, pandas as pd
from inference import match

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",default="dataset"); ap.add_argument("--out",default="results")
    a=ap.parse_args(); ds=Path(a.dataset); out=Path(a.out); out.mkdir(exist_ok=True)
    meta=pd.read_csv(ds/"metadata.csv"); rows=[]
    for i,r in meta.iterrows():
        t=time.perf_counter()
        pred=match(ds/r["reference_image"],ds/r["search_image"])
        ms=(time.perf_counter()-t)*1000
        e=math.hypot(pred["x"]-r["center_x"],pred["y"]-r["center_y"])
        rows.append({"sample_id":int(r["sample_id"]),"gt_x":r["center_x"],"gt_y":r["center_y"],
                     "pred_x":pred["x"],"pred_y":pred["y"],"error_px":e,"runtime_ms":ms,
                     "score":pred["score"],"scale":pred["scale"],"angle":pred["angle"]})
        if (i+1)%10==0: print("evaluated",i+1)
    d=pd.DataFrame(rows); d.to_csv(out/"predictions.csv",index=False); e=d.error_px.to_numpy()
    s={"samples":len(d),"mean_error_px":float(e.mean()),"median_error_px":float(np.median(e)),
       "rmse_px":float(np.sqrt(np.mean(e*e))),"p90_px":float(np.percentile(e,90)),
       "p95_px":float(np.percentile(e,95)),"within_1px":float(np.mean(e<=1)),
       "within_2px":float(np.mean(e<=2)),"within_5px":float(np.mean(e<=5)),
       "within_10px":float(np.mean(e<=10)),"mean_runtime_ms":float(d.runtime_ms.mean()),
       "p95_runtime_ms":float(d.runtime_ms.quantile(.95))}
    (out/"summary.json").write_text(json.dumps(s,indent=2)); print(s)
if __name__=="__main__": main()
