"""Bounded geometry-only proactive tongue selection from planner 0x11b8ac.

Five points per segment, 200 sphere rays per point, connected-component voting,
classification and surface-centre guard. No mesh names, oracle calls or skins.
"""
import math
import numpy as np
from .teeth_bind import components
from .face_geometry import classify
F=np.float32

def directions():
 angle=np.array([0x4122a99b],'u4').view('f4')[0];out=[]
 for j in range(200):
  a=F(F(j)+F(.5));z=F(F(F(a+a)/F(-200))+F(1));r=F(np.sqrt(max(F(0),F(F(1)-F(z*z)))));theta=F(F(j)*angle)
  out.append([F(F(math.cos(float(theta)))*r),F(F(math.sin(float(theta)))*r),z])
 return out

def surface_guard(runtime,v,f,indices):
 if not len(indices):return False
 center=np.zeros(3,'f8')
 for i in indices:center+=v[i].astype('f8')/len(indices)
 selected=set(map(int,indices));minimum=math.inf
 for tri in f:
  if all(int(i) in selected for i in tri):
   q=np.asarray(runtime.triangle_closest_point(center.tolist(),*v[tri].astype('f8').tolist()));d=q-center
   minimum=min(minimum,math.sqrt(float((d[0]*d[0]+d[1]*d[1])+d[2]*d[2])))
 span=float(np.ptp(v[indices],axis=0).max())
 return bool(math.isfinite(minimum) and span>0 and minimum/span>.15)

def select(runtime,meshes,plan,*,bounded=True):
 shape=next((s for s in plan['shapes'] if s['label']=='tongue'),None)
 if shape is None:return None
 h=np.array(shape['heads'],'f4').reshape(-1,3);t=np.array(shape['tails'],'f4').reshape(-1,3)
 if len(h)<1 or (bounded and len(h)>20):raise ValueError('tongue segments outside research bounds')
 vv=[];ff=[];owners=[];offset=0
 for idx,(_,v,f,e) in enumerate(meshes):
  vv.append(v);ff.append(f+offset);owners.extend([(idx,tri.copy()) for tri in f]);offset+=len(v)
 bvh=runtime.MeshBvh(np.concatenate(vv).astype('f4'),np.concatenate(ff).astype('i4'))
 votes={};dirs=directions()
 for a,b in zip(h,t):
  for step in range(5):
   origin=a+(b-a)*F(F(F(step)+F(.5))/F(5))
   for d in dirs:
    hit=bvh._raycast(origin.tolist(),list(map(float,d)))
    if hit is None:continue
    distance,triangle=hit;idx,tri=owners[triangle]
    votes.setdefault(idx,[]).extend((int(i),distance) for i in tri)
 best=None;minimum_span=float(F(np.ptp(np.concatenate([h,t]),axis=0)[0])*F(.5))
 for idx,records in votes.items():
  _,v,f,e=meshes[idx];c=components(len(v),e,bounded=bounded)
  for component in sorted(set(c.tolist())):
   indices=np.flatnonzero(c==component);hits=[d for i,d in records if c[i]==component]
   if len(indices)<12 or not hits or float(np.ptp(v[indices],axis=0).max())<minimum_span:continue
   hit=classify(v[indices],plan['shapes'],eye_radii=plan['eye_radii'],bounded=bounded)
   if hit is None or hit[0]!='tongue' or surface_guard(runtime,v,f,indices):continue
   mean=F(sum(float(d) for d in hits)/len(hits));key=(len(hits),-float(mean))
   if best is None or key>best['rank']:best=dict(mesh=idx,indices=indices,rank=key,hits=len(hits),mean=float(mean))
 return best
