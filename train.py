import argparse,os,cv2,numpy as np,pandas as pd,torch
from torch.utils.data import Dataset,DataLoader
from models import make
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
class DS(Dataset):
 def __init__(self,df): self.d=df.reset_index(drop=True)
 def __len__(self): return len(self.d)
 def __getitem__(self,i):
  r=self.d.iloc[i];a=cv2.resize(cv2.imread(r.reference,0),(100,100)).astype('float32')/255;b=cv2.resize(cv2.imread(r.candidate,0),(100,100)).astype('float32')/255
  if np.random.rand()<.5:a=np.clip(a*np.random.uniform(.9,1.1)+np.random.uniform(-.03,.03),0,1)
  if np.random.rand()<.5:b=np.clip(b*np.random.uniform(.85,1.15)+np.random.uniform(-.05,.05),0,1)
  return torch.tensor(a).unsqueeze(0),torch.tensor(b).unsqueeze(0),torch.tensor(float(r.label)),torch.zeros(2)
p=argparse.ArgumentParser();p.add_argument('--model',choices=['cnn','mobilenet','attention'],default='mobilenet');p.add_argument('--epochs',type=int,default=10);p.add_argument('--batch',type=int,default=64);a=p.parse_args();df=pd.read_csv('candidate_data/candidates.csv');ids=np.array(df.sample_id.unique());np.random.default_rng(2026).shuffle(ids);cut=int(.8*len(ids));tr=df[df.sample_id.isin(ids[:cut])];loader=DataLoader(DS(tr),batch_size=a.batch,shuffle=True);m=make(a.model).to(DEVICE);pos=max(1,(tr.label==1).sum());neg=max(1,(tr.label==0).sum());lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos],device=DEVICE));opt=torch.optim.AdamW(m.parameters(),lr=1e-3 if a.model=='cnn' else 2e-4,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs)
print('device',DEVICE,'model',a.model,'train',len(tr))
for e in range(a.epochs):
 m.train();tot=0
 for x,y,l,d in loader:
  x=x.float().to(DEVICE);y=y.float().to(DEVICE);l=l.float().to(DEVICE);opt.zero_grad();z,delta=m(x,y);loss=lossfn(z,l);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2);opt.step();tot+=loss.item()
 sch.step();print(f'epoch {e+1}/{a.epochs} loss={tot/len(loader):.5f}')
os.makedirs('models',exist_ok=True);torch.save({'model':m.state_dict(),'architecture':a.model},'models/model.pth');print('saved models/model.pth')
