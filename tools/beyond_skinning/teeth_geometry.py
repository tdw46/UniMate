"""Independent generic segment-distance face-part classifier branch (0x118b98).
Eye/cap/card and tongue-specific arbitration are deliberately not approximated.
"""
import numpy as np
F=np.float32

def segment_distance(point,head,tail):
 d=tail-head;delta=point-head
 dd=F(F(d[0]*d[0]+d[1]*d[1])+d[2]*d[2]);u=F(0)
 if dd>F(1e-12):u=min(F(1),max(F(0),F(F(F(delta[0]*d[0]+delta[1]*d[1])+delta[2]*d[2])/dd)))
 q=point-(head+d*u)
 return F(np.sqrt(F(q[2]*q[2]+F(q[0]*q[0]+q[1]*q[1]))))

def classify(vertices,shapes):
 v=np.asarray(vertices,'f4')
 if v.ndim!=2 or v.shape[1]!=3 or len(v)>4096 or not np.isfinite(v).all():raise ValueError('invalid classifier geometry')
 best=None
 for shape in shapes:
  label=shape['label']
  if label not in ('teeth','teeth_top','teeth_bot'):raise ValueError('non-teeth classification branch remains unqualified')
  h=np.asarray(shape['heads'],'f4').reshape(-1,3);t=np.asarray(shape['tails'],'f4').reshape(-1,3)
  if len(h)!=len(t) or len(h)>64 or not np.isfinite(h).all() or not np.isfinite(t).all():raise ValueError('invalid classifier shape')
  if not len(v) or not len(h):score=F(np.inf);ok=False
  else:
   total=0.
   for point in v:total+=float(min(segment_distance(point,a,b) for a,b in zip(h,t)))
   score=F(total/len(v));size=float(np.ptp(v,axis=0).max());anchor_size=float(np.ptp(np.concatenate([h,t]),axis=0).max())
   ok=float(score)<max(size,anchor_size)*.6 and size<=anchor_size*3.
  # Closest candidate wins even when its geometry gate rejects it.
  if best is None or score<best[1]:best=(label,score,ok)
 return (best[0],best[1]) if best is not None and best[2] else None
