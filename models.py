import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class Encoder(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.GELU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.GELU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.GELU(),nn.MaxPool2d(2),nn.Conv2d(64,96,3,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d(1))
 def forward(self,x): return self.net(x).flatten(1)
class SiameseCNN(nn.Module):
 def __init__(self):
  super().__init__();self.enc=Encoder();self.head=nn.Sequential(nn.Linear(96,48),nn.GELU(),nn.Dropout(.15),nn.Linear(48,16),nn.GELU(),nn.Linear(16,1));self.ref=nn.Sequential(nn.Linear(96,32),nn.GELU(),nn.Linear(32,2))
 def forward(self,a,b):
  d=torch.abs(self.enc(a)-self.enc(b));return self.head(d).squeeze(1),self.ref(d)
class MobileNetSiamese(nn.Module):
 def __init__(self):
  super().__init__();m=mobilenet_v3_small(weights=None);old=m.features[0][0];m.features[0][0]=nn.Conv2d(1,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False);self.enc=m.features;self.pool=nn.AdaptiveAvgPool2d(1);self.head=nn.Sequential(nn.Linear(576,128),nn.Hardswish(),nn.Dropout(.2),nn.Linear(128,32),nn.Hardswish(),nn.Linear(32,1));self.ref=nn.Sequential(nn.Linear(576,64),nn.Hardswish(),nn.Linear(64,2))
 def feat(self,x): return self.pool(self.enc(x)).flatten(1)
 def forward(self,a,b):
  d=torch.abs(self.feat(a)-self.feat(b));return self.head(d).squeeze(1),self.ref(d)
class AttentionModel(nn.Module):
 def __init__(self):
  super().__init__();self.stem=nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.GELU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.GELU(),nn.MaxPool2d(2));self.att=nn.MultiheadAttention(64,4,batch_first=True);self.norm=nn.LayerNorm(64);self.head=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,1));self.ref=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,2))
 def tok(self,x): return self.stem(x).flatten(2).transpose(1,2)
 def forward(self,a,b):
  q=self.tok(a);k=self.tok(b);z,_=self.att(q,k,k,need_weights=False);z=self.norm(z).mean(1);return self.head(z).squeeze(1),self.ref(z)
def make(name):
 if name=='cnn': return SiameseCNN()
 if name=='mobilenet': return MobileNetSiamese()
 if name=='attention': return AttentionModel()
 raise ValueError(name)
