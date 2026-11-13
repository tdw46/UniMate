"""Bounded independently recovered rigid eye and blended tongue writeback kernels."""
import numpy as np
from .teeth_geometry import segment_distance,F

def skin(vertices,names,heads,tails,*,rigid=False,bounded=True):
 v=np.asarray(vertices,'f4');h=np.asarray(heads,'f4');t=np.asarray(tails,'f4')
 if v.ndim!=2 or v.shape[1]!=3 or (bounded and len(v)>4096) or len(names)<1 or (bounded and len(names)>20) or h.shape!=(len(names),3) or t.shape!=h.shape or (bounded and any(not np.isfinite(x).all() for x in (v,h,t))):raise ValueError('invalid part skin inputs')
 k=1 if rigid else min(4,len(names));eps=max(F(1e-6),F(F(np.ptp(np.concatenate([h,t]),axis=0).max())*F(.01)))
 weights=np.zeros((len(v),k),'f4');indices=np.zeros((len(v),k),'i4')
 for row,a in enumerate(v):
  scores=[]
  for i,(b,c) in enumerate(zip(h,t)):
   distance=F(eps+segment_distance(a,b,c));scores.append(F(F(1)/F(distance*distance)))
  order=sorted(range(len(names)),key=lambda i:-float(scores[i]))[:k];total=F(-0.)
  for i in order:total=F(total+scores[i])
  for j,i in enumerate(order):weights[row,j]=F(scores[i]/total);indices[row,j]=i
 label='rigid per vertex on the nearest of ' if rigid else 'blended over '
 return (list(names),weights,indices,label+', '.join(names))
