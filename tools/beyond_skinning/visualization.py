"""Independent bounded cloud/seed/field display output. No original runtime imports.

Recovered display formulas: sub_9d854 and cube emitter sub_5b79c.
Seed skeleton geometry is supplied by the independent seed_preview module.
"""
import numpy as np
F=np.float32

def hsv(h,s,v):
 h=F(h*F(6));i=int(h);f=F(h-F(i));p=F(v*F(F(1)-s));q=F(v*F(F(1)-F(f*s)));t=F(v*F(F(1)-F(F(F(1)-f)*s)))
 return ((v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q))[i%6]

def display(mode,points,levels,voxel_size,multiplier,scale,hit,field=None,bone=None,label_map=None,seed_labels=None,skeleton=None):
 inv=F(F(1)/F(scale));p=np.asarray(points,'f4')*inv;voxel_size=F(F(voxel_size)*inv);level=np.asarray(levels);count=len(p)
 total=count
 if mode=='seeds':
  labels=np.asarray(seed_labels).copy();valid=labels!=0xffffffff
  p=p[valid];level=level[valid];labels=labels[valid];count=len(p)
 colors=np.empty((count,3),'f4');wmax=0.;shown=0
 if mode=='cloud':
  maximum=int(level.max(initial=0))
  for i,l in enumerate(level):colors[i]=hsv(F(F(F(1)-F(F(l)/F(maximum) if maximum else 0))*F(.66)),F(.85),F(.95))
 elif mode=='seeds':
  import ctypes,os
  powf=ctypes.CDLL('ucrtbase.dll' if os.name=='nt' else None).powf;powf.argtypes=(ctypes.c_float,ctypes.c_float);powf.restype=ctypes.c_float
  if label_map is not None:labels=np.asarray(label_map)[labels]
  shown=len(set(labels.tolist()))
  for i,b in enumerate(labels):
   hue=F(F(b)*F(.618034));rgb=hsv(F(hue-F(int(hue))),F(.88),F(1))
   colors[i]=[F(c/F(12.92)) if c<=F(.04045) else F(powf(F(F(c+F(.055))/F(1.055)),F(2.4))) for c in rgb]
 else:
  labels=np.argmax(field,axis=1);valid=np.max(field,axis=1)>0;labels=np.where(valid,labels,-1)
  if label_map is not None:labels[valid]=np.asarray(label_map)[labels[valid]]
  shown=len(set(labels[valid].tolist()))
  if bone is None:
   for i,b in enumerate(labels):
    hue=F(F(b)*F(.618034))
    colors[i]=hsv(F(hue-F(int(hue))),F(.88),F(1)) if b>=0 else (F(.25),)*3
  else:
   ramp=np.array([[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]],'f4')
   vals=field[:,bone];wmax=float(vals.max(initial=0))
   for i,w in enumerate(vals):
    x=F(np.clip(w,0,1)*F(4));j=min(int(x),3);f=F(x-F(j));colors[i]=ramp[j+1]*f+ramp[j]*F(F(1)-f)
 # Keep source operation order, including float32 reciprocal and world scaling.
 factor=F(F(1)/F(max(multiplier,1)))
 half=np.array([F(F(F(F(voxel_size)*F(float(factor)**int(l)))*F(.5))*F(.92)) for l in level],'f4')
 lo=p-half[:,None];hi=p+half[:,None]
 verts=np.empty((count,8,3),'f4')
 for i in range(8):verts[:,i]=np.stack([hi[:,j] if i&(1<<j) else lo[:,j] for j in range(3)],axis=1)
 verts=verts.reshape(-1,3)
 pattern=np.array([0,1,3,2,4,6,7,5,0,2,6,4,1,5,7,3,0,4,5,1,2,3,7,6],'i4')
 corners=(np.arange(count,dtype='i4')[:,None]*8+pattern).reshape(-1)
 rgba=np.ones((count,4),'f4');rgba[:,:3]=colors
 cubes=dict(verts=verts,corners=corners,loop_starts=np.arange(count*6,dtype='i4')*4,colours=np.repeat(rgba,24,axis=0) if count else None)
 return dict(cubes=cubes,skeleton=skeleton,voxels=total,per_level=[(int(l),int(np.count_nonzero(level==l))) for l in np.unique(level)],bones_shown=shown,wmax=wmax,cloud_hit=bool(hit))
