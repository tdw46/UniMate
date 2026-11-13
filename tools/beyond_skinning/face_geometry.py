"""Independent face classifier: generic segments, eye spheres and radius/cap fallback."""
import math
import numpy as np
from .teeth_geometry import segment_distance,F

def norm(v):return F(np.sqrt(F(F(v[0]*v[0]+v[1]*v[1])+v[2]*v[2])))
def point_distance(a,b):return segment_distance(a,b,b)
def extent(points):
 # Native f32 bounds use minimumNumber/maximumNumber reductions and return
 # zero when the first lower coordinate is nonfinite (0x118b98).
 a=np.asarray(points,'f4')
 if not len(a):return F(0)
 lo=np.fmin.reduce(a,axis=0,initial=F(np.inf));hi=np.fmax.reduce(a,axis=0,initial=F(-np.inf))
 if not np.isfinite(lo[0]):return F(0)
 return F(np.fmax.reduce(hi-lo))

def eye_metrics(v,h,t,radii=None):
 if not len(v) or not len(h):return 'eyes',F(np.inf),False
 dist=np.array([[point_distance(q,a) for a in h] for q in v],'f4');which=np.argmin(dist,axis=1);near=dist[np.arange(len(v)),which]
 score=F(sum(map(float,near))/len(v));cv=F(0);offset=F(0)
 for i in range(len(h)):
  q=v[which==i]
  if len(q)<=3:continue
  center=np.array([sum(float(a[j]) for a in q)/len(q) for j in range(3)],'f4');size=max(F(1e-6),F(np.ptp(q,axis=0).max()))
  rr=[float(point_distance(a,h[i])) for a in q];mean=sum(rr)/len(rr);variance=sum((x-mean)*(x-mean) for x in rr)/len(rr)
  cv=max(cv,F(math.sqrt(variance)/max(mean,1e-9)));offset=max(offset,F(point_distance(center,h[i])/size))
 size=float(extent(v));anchor=float(extent(np.concatenate([h,t])))
 accepted=cv<F(.25) and float(offset)<.35 and score<max(size,anchor) and size<=anchor*3.
 if accepted:return 'eyes',score,True
 if radii is None:return 'eyes',score,False
 radii=np.asarray(radii,'f4')
 if radii.shape!=(len(h),) or not np.isfinite(radii).all():raise ValueError('invalid eye radii')
 ratio=max(F(d/max(radii[i],F(1e-9))) for d,i in zip(near,which));angle=F(0)
 for i in range(len(h)):
  q=v[which==i]
  if not len(q):continue
  dirs=np.array([(a-h[i])/max(norm(a-h[i]),F(1e-12)) for a in q],'f4');center=np.zeros(3,'f4')
  for d in dirs:center+=d
  center/=max(norm(center),F(1e-12));dot=min(F(1),min(F(F(d[0]*center[0]+d[1]*center[1])+d[2]*center[2]) for d in dirs));dot=max(F(-1),dot)
  angle=max(angle,F(F(math.acos(float(dot)))*F(57.29578)))
 q=v.astype('f8');mean=np.array([sum(q[:,j])/len(q) for j in range(3)]);delta=q-mean
 cov=np.array([[sum(delta[:,i]*delta[:,j])/len(q) for j in range(3)] for i in range(3)])
 flat=False
 if np.isfinite(cov).all():
  eig=np.linalg.eigvalsh(cov);flat=eig[-1]>1e-18 and F(math.sqrt(max(0.,eig[0])/eig[-1]))<F(.03)
 if ratio<=F(1) and angle<F(40):return ('eye_cards' if flat else 'eye_caps'),score,True
 return 'eyes',score,False

def classify(v,shapes,*,eye_radii=None,exclude=None,bounded=True):
 v=np.asarray(v,'f4')
 if v.ndim!=2 or v.shape[1]!=3 or (bounded and (len(v)>4096 or not np.isfinite(v).all())):raise ValueError('invalid classifier geometry')
 best=None
 for s in shapes:
  label=s['label']
  if label==exclude:continue
  if label not in ('teeth','teeth_top','teeth_bot','tongue','eyes','eye_caps','eye_cards'):raise ValueError('unknown part classifier')
  h=np.asarray(s['heads'],'f4').reshape(-1,3);t=np.asarray(s['tails'],'f4').reshape(-1,3)
  if len(h)!=len(t) or (bounded and (len(h)>64 or not np.isfinite(h).all() or not np.isfinite(t).all())):raise ValueError('invalid classifier shapes')
  if label=='eyes':label,score,ok=eye_metrics(v,h,t,eye_radii)
  elif not len(v) or not len(h):score=F(np.inf);ok=False
  else:
   score=F(sum(float(np.fmin.reduce([segment_distance(a,b,c) for b,c in zip(h,t)],initial=F(np.inf))) for a in v)/len(v));size=float(extent(v));anchor=float(extent(np.concatenate([h,t])));ok=float(score)<max(size,anchor)*.6 and size<=anchor*3.
  if best is None or score<best[1]:best=(label,score,ok)
 return (best[0],best[1]) if best is not None and best[2] else None
