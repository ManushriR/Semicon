import cv2,numpy as np,torch
from models import make
from candidate_utils import read,candidates,crop,periodicity
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
def load(path):
 c=torch.load(path,map_location=DEVICE);m=make(c['architecture']).to(DEVICE);m.load_state_dict(c['model']);m.eval();return m
def localize(rp,sp,model_path,topk=30):
 r=read(rp);m=load(model_path);out=[]
 for c in candidates(rp,sp,topk):
  p=crop(sp,c['x'],c['y']).astype('float32')/255
  a=torch.tensor(r).unsqueeze(0).unsqueeze(0).to(DEVICE);b=torch.tensor(p).unsqueeze(0).unsqueeze(0).to(DEVICE)
  with torch.no_grad(): z,d=m(a,b);prob=torch.sigmoid(z).item();d=d.cpu().numpy()[0]
  per=periodicity(r,p);dist=np.hypot(c['x']-499.5,c['y']-499.5);center=1/(1+dist/500);score=.50*prob+.20*max(0,c['ncc'])+.12*max(0,c['edge_ncc'])+.13*np.tanh(per)+.05*center
  out.append({**c,'dl_score':prob,'periodicity':per,'dx':float(np.clip(d[0],-8,8)),'dy':float(np.clip(d[1],-8,8)),'final_score':score})
 out.sort(key=lambda q:q['final_score'],reverse=True);b=out[0];b['x_refined']=b['x']+b['dx'];b['y_refined']=b['y']+b['dy'];b['confidence']=b['final_score']-(out[1]['final_score'] if len(out)>1 else 0);return b
