"""Public seam composition from recovered rig map, mouth record and split kernels."""
import numpy as np
from .mouth_project import F

def assemble(vertices,edges,names,parents,deform,segments,head,seam,faces):
 from . import _bvt_solver as rt
 from .rig_map_public import assemble as rig_map
 from .face_split import corner_plane,categories
 from .split_ownership import select
 v=np.asarray(vertices,'f4')[:,:3];e=np.asarray(edges,'i8')[:,:2];s=np.asarray(segments,'f4')[:,:2,:3]
 n=len(names);s=s.reshape((n,2,3));h=s[:,0];t=s[:,1]
 mapping=rig_map(names,parents,deform,s,None,'verified',head,None)
 if faces is not None and len(faces) and np.asarray(faces).shape[1]<3:raise rt.PanicException('ndarray: index out of bounds')
 groups={}
 for _,kind,_,side,_,ids,_,_ in mapping['elements']:groups.setdefault(kind,[]).extend(ids)
 up=groups.get('lip_top',[]);lo=groups.get('lip_bot',[]);corners=groups.get('lip_corner',[])
 if not up or not lo:return {}
 lipids=sorted(set(up+lo+corners));mins=np.full(3,np.inf,'f4');maxs=np.full(3,-np.inf,'f4')
 for i in lipids:mins=np.fmin(np.fmin(mins,h[i]),t[i]);maxs=np.fmax(np.fmax(maxs,h[i]),t[i])
 span=max(F(0),*(maxs-mins));reach=F(span)
 record=dict(upper=up,lower=lo,reach=reach,radius=F(reach*F(.75)),passes=3,corner_radius=F(reach*F(.1)),ambiguity=F(.001 if seam is None else seam),corner_plane=corner_plane(h[up],h[lo],h[corners]),corner_points=h[corners])
 lidids=sorted(set(groups.get('eyelid_top',[])+groups.get('eyelid_bot',[])+groups.get('eyelid_corner_inner',[])+groups.get('eyelid_corner_outer',[])))
 pairs=lambda ids:np.asarray([pt for i in ids for pt in (h[i],t[i])],'f4').reshape(-1,3)
 allowed=select(v,e,pairs(lidids),pairs(up+lo),F(reach*F(.25)),allow_nonfinite_vertices=True,bounded=False)
 if allowed is None:allowed=np.ones(len(v),'u1')
 graph=native_graph(v,e)
 f=np.empty((0,3),'i4') if faces is None else np.asarray(faces,'i4')[:,:3]
 return {'mouth':categories(rt,v,f,h,t,graph,record,allowed,allow_nonfinite_vertices=True,bounded=False).tolist()}


def native_graph(v,e):
 from .mouth_project import distance
 adj=[[] for _ in v]
 for a,b in e:
  if a!=b and 0<=a<len(v) and 0<=b<len(v):
   w=distance(v[a],v[b]);adj[a].append((int(b),w));adj[b].append((int(a),w))
 r=[0];c=[];w=[]
 for near in adj:
  for j,d in near:c.append(j);w.append(d)
  r.append(len(c))
 return dict(rows=r,columns=c,weights=w)
