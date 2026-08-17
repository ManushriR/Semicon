import os,argparse,json,numpy as np,pandas as pd,matplotlib.pyplot as plt
from hybrid import localize
p=argparse.ArgumentParser();p.add_argument('--dataset',default='dataset');p.add_argument('--model',default='models/model.pth');p.add_argument('--output',default='results');a=p.parse_args();os.makedirs(a.output,exist_ok=True);df=pd.read_csv(os.path.join(a.dataset,'metadata.csv'));rows=[]
for n,(_,r) in enumerate(df.iterrows(),1):
 try:
  q=localize(os.path.join(a.dataset,r.reference_image),os.path.join(a.dataset,r.search_image),a.model);e=float(np.hypot(q['x_refined']-r.center_x,q['y_refined']-r.center_y));rows.append({'sample_id':r.sample_id,'gt_x':r.center_x,'gt_y':r.center_y,'pred_x':q['x_refined'],'pred_y':q['y_refined'],'error_px':e,'dl_score':q['dl_score'],'ncc':q['ncc'],'confidence':q['confidence'],'difficulty':r.difficulty})
 except Exception as ex: print('failed',r.sample_id,ex)
 if n%25==0: print('evaluated',n)
r=pd.DataFrame(rows);r.to_csv(os.path.join(a.output,'results.csv'),index=False);e=r.error_px.to_numpy();s={'samples':len(r),'mean_error_px':float(e.mean()),'median_error_px':float(np.median(e)),'rmse_px':float(np.sqrt(np.mean(e**2))),'p95_px':float(np.percentile(e,95)),'within_1px':float(np.mean(e<=1)),'within_2px':float(np.mean(e<=2)),'within_5px':float(np.mean(e<=5))};json.dump(s,open(os.path.join(a.output,'summary.json'),'w'),indent=2);plt.figure(figsize=(8,5));plt.hist(e,bins=30);plt.xlabel('Localization error (pixels)');plt.ylabel('Samples');plt.tight_layout();plt.savefig(os.path.join(a.output,'error_histogram.png'),dpi=180);plt.close();r.groupby('difficulty').error_px.agg(['mean','median','count']).to_csv(os.path.join(a.output,'difficulty_results.csv'));print(s)
