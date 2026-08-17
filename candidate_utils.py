import cv2,numpy as np
from scipy.ndimage import maximum_filter

def read(p):
 x=cv2.imread(p,0)
 if x is None: raise FileNotFoundError(p)
 return x.astype(np.float32)/255
def edge(x):
 gx=cv2.Sobel(x,cv2.CV_32F,1,0,3);gy=cv2.Sobel(x,cv2.CV_32F,0,1,3);return cv2.normalize(cv2.magnitude(gx,gy),None,0,1,cv2.NORM_MINMAX)
def candidates(rp,sp,topk=30):
 r=read(rp);s=read(sp);re=edge(r);se=edge(s);out=[]
 for sc in np.linspace(.88,1.12,13):
  w=h=max(20,int(100*sc));rr=cv2.resize(r,(w,h));ee=cv2.resize(re,(w,h));a=cv2.matchTemplate(s,rr,cv2.TM_CCOEFF_NORMED);b=cv2.matchTemplate(se,ee,cv2.TM_CCOEFF_NORMED);f=.65*a+.35*b;mx=maximum_filter(f,17);th=np.percentile(f,99.3);ys,xs=np.where((f==mx)&(f>=th))
  for x,y in zip(xs,ys): out.append(dict(x=x+w/2,y=y+h/2,w=w,h=h,ncc=float(a[y,x]),edge_ncc=float(b[y,x]),base=float(f[y,x])))
 out.sort(key=lambda z:z['base'],reverse=True);sel=[]
 for c in out:
  if all(np.hypot(c['x']-q['x'],c['y']-q['y'])>=20 for q in sel): sel.append(c)
  if len(sel)>=topk: break
 return sel
def crop(path,x,y,size=100):
 im=cv2.imread(path,0);H,W=im.shape;x0=max(0,min(int(x-size/2),W-size));y0=max(0,min(int(y-size/2),H-size));return cv2.resize(im[y0:y0+size,x0:x0+size],(100,100))
def periodicity(a,b):
 def f(x):
  x=(x-x.mean())/(x.std()+1e-6);z=np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(x))));return z/(np.linalg.norm(z)+1e-6)
 return float(np.sum(f(a)*f(b)))
