"""Recovered bounded skull-axis bands and head-seed reassignment.

Explicit anatomical role IDs; complete automatic rig/part preparation is open.
"""
import numpy as np
F=np.float32

def dot(a,b):return F(F(a[0]*b[0]+a[1]*b[1])+a[2]*b[2])
def construct(heads,tails,head,skulls,jaw=None):
 h=np.asarray(heads,'f4');t=np.asarray(tails,'f4')
 if h.ndim!=2 or h.shape[1]!=3 or t.shape!=h.shape or not np.isfinite(h).all() or not np.isfinite(t).all() or not 0<=head<len(h) or any(not 0<=i<len(h) for i in skulls) or (jaw is not None and not 0<=jaw<len(h)):raise ValueError('invalid skull skeleton')
 axis=t[head]-h[head];length=dot(axis,axis)
 if length<=F(1e-12):return []
 limit=F(F(np.sqrt(length))*F(.05));bands=[]
 for i in skulls:
  bounds=[];near=True
  for point in (h[i],t[i]):
   u=F(dot(axis,point-h[head])/length);delta=point-(h[head]+axis*u)
   if F(np.sqrt(F(delta[2]*delta[2]+F(delta[0]*delta[0]+delta[1]*delta[1]))))>limit:near=False
   bounds.append(u)
  if near:bands.append([i,float(max(F(0),min(F(1),min(bounds)))),float(max(F(0),min(F(1),max(bounds))))])
 if not bands:return []
 intervals=sorted((F(a),F(b)) for _,a,b in bands);start,end=intervals[0];total=F(0)
 for a,b in intervals[1:]:
  if end<a:total=F(total+F(end-start));start=a;end=b
  else:end=max(end,b)
 covered=F(F(total+end)-start)
 if covered<F(.8):return []
 if jaw is not None and len(bands)>=2:
  bands.sort(key=lambda b:b[1]);u=F(dot(axis,h[jaw]-h[head])/max(length,F(1e-12)))
  if bands[0][1]<u<bands[0][2]:bands[0][2]=float(u);bands[1][1]=float(min(F(bands[1][1]),u))
 return bands

def reassign(points,labels,head,head_position,tail_position,bands):
 p=np.asarray(points,'f4').reshape(-1,3);out=np.asarray(labels,'u4').copy();a=np.asarray(head_position,'f4');axis=np.asarray(tail_position,'f4')-a;length=max(dot(axis,axis),F(1e-12))
 if len(out)!=len(p):raise ValueError('label count mismatch')
 if not bands:return out
 for i in np.flatnonzero(out==head):
  u=F(dot(axis,p[i]-a)/length)
  def distance(b):return F(b[1])-u if u<F(b[1]) else u-F(b[2]) if u>F(b[2]) else F(0)
  out[i]=min(bands,key=distance)[0]
 return out
