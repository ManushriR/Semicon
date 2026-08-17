import os,cv2,argparse,pandas as pd
from candidate_utils import candidates,crop
p=argparse.ArgumentParser();p.add_argument('--dataset',default='dataset');p.add_argument('--topk',type=int,default=30);a=p.parse_args();df=pd.read_csv(os.path.join(a.dataset,'metadata.csv'));os.makedirs('candidate_data/patches',exist_ok=True);rows=[]
for n,(_,r) in enumerate(df.iterrows(),1):
 rp=os.path.join(a.dataset,r.reference_image);sp=os.path.join(a.dataset,r.search_image)
 for j,c in enumerate(candidates(rp,sp,a.topk)):
  d=((c['x']-r.center_x)**2+(c['y']-r.center_y)**2)**.5;lab=int(d<=8)
  if lab or j<15:
   fn=f"{int(r.sample_id):05d}_{j:02d}.png";fp=os.path.join('candidate_data/patches',fn);cv2.imwrite(fp,crop(sp,c['x'],c['y']));rows.append(dict(sample_id=int(r.sample_id),reference=rp,candidate=fp,candidate_x=c['x'],candidate_y=c['y'],ncc=c['ncc'],edge_ncc=c['edge_ncc'],distance=d,label=lab))
 if n%25==0: print('processed',n)
out=pd.DataFrame(rows);pos=out[out.label==1];neg=out[out.label==0];
if len(neg)>max(1,len(pos))*5: neg=neg.sample(len(pos)*5,random_state=2026)
pd.concat([pos,neg]).sample(frac=1,random_state=2026).to_csv('candidate_data/candidates.csv',index=False);print('positive/negative',len(pos),len(neg))
