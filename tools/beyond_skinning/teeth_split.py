"""Independent default weighted arch splitting; no original-module fallback."""
import numpy as np

def weighted_split(samples,*,bounded=True):
 a=np.asarray(samples,'f8')
 if a.ndim!=2 or a.shape[1]!=2 or (bounded and len(a)>4096) or (bounded and not np.isfinite(a).all()) or np.any(a[:,1]<0):raise ValueError('invalid weighted split')
 if len(a)<2:return None
 a=a[np.argsort(a[:,0],kind='stable')]
 total=-0.;moment=-0.
 for x,w in a:total+=float(w)
 for x,w in a:moment+=float(x)*float(w)
 left=0.;left_moment=0.;best=None;score=None
 for i,(x,w) in enumerate(a[:-1]):
  left+=float(w);left_moment+=float(w)*float(x);right=total-left
  if left<=0 or right<=0:continue
  lo=left_moment/left;hi=(moment-left_moment)/right;delta=hi-lo;s=delta*(left*right*delta)
  if score is None or s>score:score=s;best=((float(x)+float(a[i+1,0]))*.5,lo,hi)
 return best

def arch_mask(vertices,upper,lower,components,hint=None,*,bounded=True):
 v=np.asarray(vertices,'f4');u=np.asarray(upper,'f4');l=np.asarray(lower,'f4');c=np.asarray(components,'i4')
 if v.ndim!=2 or v.shape[1]!=3 or (bounded and len(v)>4096) or c.shape!=(len(v),) or any(a.ndim!=2 or a.shape[1]!=3 or (bounded and not np.isfinite(a).all()) for a in (v,u,l)) or not len(u) or not len(l):raise ValueError('invalid arch inputs')
 def center(a):
  result=np.zeros(3,'f8')
  for row in a:result+=row.astype('f8')/float(len(a))
  return result
 cu=center(u);cl=center(l);delta=cu-cl;length=float(np.sqrt((delta[0]*delta[0]+delta[1]*delta[1])+delta[2]*delta[2]))
 result=np.zeros(len(v),'u1')
 if length<1e-9:return result
 axis=delta/length;dot=lambda a:(a[0]*axis[0]+a[1]*axis[1])+a[2]*axis[2]
 values=np.array([dot(row.astype('f8')) for row in v]);up=dot(cu);low=dot(cl)
 groups=[]
 for key in sorted(set(c[c>=0].tolist())):
  ids=np.flatnonzero(c==key);vs=values[ids];mean=sum(float(x) for x in vs)/len(ids)
  groups.append((ids,mean,float(np.ptp(vs))>=2*length))
 if not groups:return result
 samples=[(mean,len(ids)) for ids,mean,wide in groups if not wide]
 from_centroids=len(samples)>=2
 if not from_centroids:samples=[(float(values[i]),1.) for ids,_,_ in groups for i in ids]
 split=weighted_split(samples,bounded=bounded)
 if hint is not None:
  hint=np.asarray(hint,'f4')
  if hint.shape!=(3,) or (bounded and not np.isfinite(hint).all()):raise ValueError('invalid arch hint')
 enabled=False
 if split:
  threshold,lo,hi=split
  aligned=abs(lo-low)<abs(lo-up) and abs(hi-up)<abs(hi-low)
  balanced=False
  if from_centroids and float(np.ptp(values))>=length and len(samples)>5:
   below=sum(x<=threshold for x,_ in samples);above=len(samples)-below
   balanced=max(below,above)>0 and min(below,above)>=max(below,above)*.85
  enabled=aligned or balanced
  if hint is not None:
   hp=dot(hint.astype('f8'));inside=lo<hp<hi and len(groups)<=5
   enabled=(enabled or inside) and hp-hi<length*.5 and lo-hp<length*.5
   if not (aligned or balanced):threshold=hp
 if enabled:
  for ids,mean,wide in groups:
   if wide:result[ids]=values[ids]>threshold
   elif from_centroids:result[ids]=mean>threshold
   else:result[ids]=abs(mean-up)<abs(mean-low)
 else:
  mean=-0.;count=0
  for ids,m,_ in groups:mean+=m*len(ids);count+=len(ids)
  mean/=count
  for ids,_,_ in groups:result[ids]=abs(mean-up)<abs(mean-low)
 return result
