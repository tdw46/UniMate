"""Bounded independent mouth surface projector, recovered from 0x102e1c.

Consumes explicit ring role records; complete automatic preparation remains separate.
"""
import numpy as np
from .mouth_ring_plane import ring_order
from .mouth_surface_path import shortest_path,surface_distances
F=np.float32

def dot(a,b):return F(F(a[0]*b[0]+a[1]*b[1])+a[2]*b[2])
def distance(a,b):
 d=a-b;return F(np.sqrt(dot(d,d)))
def distances(points,point):
 """Same rounded f32 subtract/products/left sum/sqrt, in independent lanes."""
 d=points-point
 q=d*d
 return np.sqrt((q[:,0]+q[:,1])+q[:,2])

def nearest_candidate(vertices,candidates,point):
 """Ordered scalar min semantics with at most 4096 independent f32 lanes.

 Strict improvements only: first ties win, later NaNs are ignored, and a NaN
 in the very first candidate remains the winner just like Python min(key=...).
 No vertex-by-candidate matrix or retained distance field is allocated.
 """
 if not candidates:raise ValueError('min() iterable argument is empty')
 best=candidates[0];best_d=distance(point,vertices[best])
 if np.isnan(best_d):return best,best_d
 for begin in range(0,len(candidates),4096):
  block=candidates[begin:begin+4096]
  ds=distances(vertices[block],point)
  # Subtract direction only changes signed zero before squaring; retain the
  # scalar recomputation for the returned winner below, including NaN payloads.
  at=int(np.argmin(np.where(np.isnan(ds),np.inf,ds)));d=ds[at]
  if d<best_d:best=block[at];best_d=d
 return best,distance(point,vertices[best])

def bridge_vertices(vertices,segments,bridge):
 """Bounded f32 segment tests; preserve scalar clamp semantics, including NaN."""
 for begin in range(0,len(vertices),4096):
  points=vertices[begin:begin+4096];hit=np.zeros(len(points),bool)
  for a,b in segments:
   d=b-a;dd=dot(d,d)
   if dd<=F(1e-12):u=np.zeros(len(points),'f4')
   else:
    delta=points-a
    numerator=(delta[:,0]*d[0]+delta[:,1]*d[1])+delta[:,2]*d[2]
    ratio=numerator/dd
    # Python max(0,NaN) returns its first argument, unlike np.maximum.
    u=np.where(ratio>F(0),ratio,F(0));u=np.where(u<F(1),u,F(1))
   q=points-(a+d*u[:,None]);square=q*q
   separation=np.sqrt(square[:,2]+(square[:,0]+square[:,1]))
   hit|=separation<=bridge
  for index in np.flatnonzero(hit):yield begin+int(index)

def project(vertices,bone_points,lip_ids,flags,params,graph,weights,bones,categories=None,*,allow_nonfinite_vertices=False,bounded=True):
 v=np.asarray(vertices,'f4').reshape(-1,3);t=np.asarray(bone_points,'f4').reshape(-1,3);out=np.asarray(weights,'f4');ids=np.asarray(bones,'i4');k=out.shape[1]
 if (bounded and (len(v)>4096 or len(lip_ids)>64)) or len(lip_ids)<2 or out.shape!=ids.shape or len(out)!=len(v):raise ValueError('mouth projector research bounds')
 if len(flags)!=len(lip_ids) or any(x not in (0,1,2) for x in flags):raise ValueError('invalid lip flags')
 if len(params)!=9 or not np.isfinite(params).all() or any(params[i]<=0 for i in (4,5,6)) or not 0<=params[8]<=8:raise ValueError('invalid bounded mouth parameters')
 if (not allow_nonfinite_vertices and not np.isfinite(v).all()) or not np.isfinite(t).all() or any(not 0<=i<len(t) for i in lip_ids):raise ValueError('invalid mouth geometry')
 pp=np.asarray(params,'f4');bridge,reach,path_cap,plateau,band,spread,corner_spread,corner_multiplier=pp[:8];grow=int(params[8])
 r,c,w=[graph[x] for x in ('rows','columns','weights')]
 reuse_distances=True
 if not bounded:
  from .face_memory import require
  try:require(len(v),len(c),slots=k,fields=2*len(lip_ids))
  except ValueError as exc:
   if not str(exc).startswith('facial working-set research budget exceeded:'):raise
   # Keep the previously admitted scalar route when the optional cache won't fit.
   require(len(v),len(c),slots=k,fields=len(lip_ids));reuse_distances=False
 out=out.copy();ids=ids.copy()
 # Retain only vertex-by-anchor distances (not vertex-by-vertex pairs).
 # Existing graph reserve covers bounded temporary lanes; extra retained field
 # storage is explicitly charged alongside the later geodesic fields.
 euclidean_fields=np.empty((len(lip_ids),len(v)),dtype='f4') if reuse_distances else None
 anchors=[];seeds=[]
 for anchor,(bone,flag) in enumerate(zip(lip_ids,flags)):
  if reuse_distances:
   ds=euclidean_fields[anchor]
   for begin in range(0,len(v),4096):
    ds[begin:begin+4096]=distances(v[begin:begin+4096],t[bone])
  else:ds=[distance(x,t[bone]) for x in v]
  index=int(np.argmin(np.where(np.isnan(ds),np.inf,ds)));anchors.append((bone,index,flag,ds[index]));seeds.append((index,ds[index]))
 order=ring_order(t[lip_ids]);cap=max(path_cap,F(band+band))
 for a,b in zip(order,order[1:]+order[:1]):
  path=shortest_path(r,c,w,anchors[a][1],anchors[b][1],cap,allow_nonfinite_edges=allow_nonfinite_vertices,bounded=bounded)
  if path:seeds.extend((i,F(0)) for i in path)
 if bridge>0:
  lower=[i for i,f in enumerate(flags) if f==2];segments=[]
  for i,flag in enumerate(flags):
   if flag==1 and lower:
    j=min(lower,key=lambda j:distance(t[lip_ids[i]],t[lip_ids[j]]));segments.append((t[lip_ids[i]],t[lip_ids[j]]))
  seeds.extend((i,F(0)) for i in bridge_vertices(v,segments,bridge))
 seeds=sorted(seeds,key=lambda x:(x[0],x[1]));unique=[]
 for i,d in seeds:
  if not unique or unique[-1][0]!=i:unique.append((i,d))
 all_dist=surface_distances(r,c,w,unique,F(band*max(F(1),corner_multiplier)),allow_nonfinite_edges=allow_nonfinite_vertices,bounded=bounded)
 bridged=np.full(len(v),-1,'i4')
 if reach>0 and unique:
  candidates=[x[0] for x in unique];lo=v[candidates].min(0)-reach;hi=v[candidates].max(0)+reach
  for i,x in enumerate(v):
   if np.any(x<lo) or np.any(x>hi):continue
   j,d=nearest_candidate(v,candidates,x)
   if d<=reach and d<all_dist[i]:all_dist[i]=d;bridged[i]=j
 fields=[surface_distances(r,c,w,[(index,d)],F(band+band),allow_nonfinite_edges=allow_nonfinite_vertices,bounded=bounded) for _,index,_,d in anchors]
 alpha=np.zeros(len(v),'f4');tags=np.zeros(len(v),'u1')
 cat=[] if categories is None else categories
 for i,x in enumerate(v):
  radius=band
  if corner_multiplier>1:
   closest=min(range(len(anchors)),key=lambda j:min(fields[j][i],euclidean_fields[j,i] if reuse_distances else distance(x,t[anchors[j][0]])))
   if anchors[closest][2]==0:radius=F(corner_multiplier*band)
  d=all_dist[i]
  if d>=radius:continue
  a=F(1)
  if d>plateau:
   q=F(F(1)-F(F(d-plateau)/F(radius-plateau)));a=F(F(q*q)*F(F(3)-F(q+q)))
  flag=cat[i] if len(cat) and cat[i] in (1,2) else 0
  if bridged[i]>=0 and not flag:
   eligible=[j for j,a in enumerate(anchors) if a[2]]
   if eligible:flag=anchors[min(eligible,key=lambda j:euclidean_fields[j,i] if reuse_distances else distance(x,t[anchors[j][0]]))][2]
  alpha[i]=a;tags[i]=flag
 grown=0
 for _ in range(grow):
  updates=[]
  for i in range(len(v)):
   if alpha[i]>=F(.5) or bridged[i]<0:continue
   near={int(tags[c[j]]) for j in range(r[i],r[i+1]) if alpha[c[j]]>=F(.5) and tags[c[j]]}
   if len(near)==1:updates.append((i,next(iter(near))))
  if not updates:break
  grown+=len(updates)
  for i,tag in updates:alpha[i]=1;tags[i]=tag
 changed=0
 for i,x in enumerate(v):
  a=alpha[i]
  if a<=0:continue
  pairs=[];total=F(0);best=None;best_d=F(np.inf)
  for j,(bone,index,flag,d0) in enumerate(anchors):
   euclidean=euclidean_fields[j,i] if reuse_distances else distance(x,t[bone]);d=euclidean if bridged[i]>=0 else min(fields[j][i],euclidean) if flag==0 or flag==tags[i] else fields[j][i]
   if not np.isfinite(d) or (tags[i] in (1,2) and flag not in (0,tags[i])):continue
   if d<best_d:best=bone;best_d=d
   q=F(d/(spread if flag else corner_spread))
   if q<1:
    inv=F(1)-q;value=F(F(F(q*F(4))+F(1))*F(inv*F(inv*F(inv*inv))))
    if value>0:pairs.append([bone,value]);total=F(total+value)
  if total<=0:
   if best is None:continue
   pairs.append([best,F(1)]);total=F(1)
  scale=F(a/total)
  for pair in pairs:pair[1]=F(pair[1]*scale)
  for bone,value in zip(ids[i],out[i]):
   value=F(F(F(1)-a)*value)
   if bone<0 or value<=0:continue
   found=next((p for p in pairs if p[0]==bone),None)
   if found is None:pairs.append([int(bone),value])
   else:found[1]=F(found[1]+value)
  pairs.sort(key=lambda x:-float(x[1]));pairs=pairs[:k];total=F(-0.)
  for _,value in pairs:total=F(total+value)
  if total<=0:continue
  out[i]=0;ids[i]=-1
  for j,(bone,value) in enumerate(pairs):ids[i,j]=bone;out[i,j]=F(value/total)
  changed+=1
 return dict(weights=out,bones=ids,stats=[changed,len(unique),int(np.count_nonzero(bridged>=0)),grown],alpha=alpha,tags=tags,seeds=unique)
