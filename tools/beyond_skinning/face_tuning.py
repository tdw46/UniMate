"""Bounded independent mouth-region selection and stable neighbor smoothing."""
import numpy as np
from .mouth_project import F,distance
from .mouth_surface_path import surface_distances

def mouth_mask(vertices,heads,tails,lip_ids,graph,margin=.03,radius=.6,*,allow_nonfinite_vertices=False,bounded=True):
    v=np.asarray(vertices,'f4');h=np.asarray(heads,'f4');t=np.asarray(tails,'f4')
    if not lip_ids:return np.zeros(len(v),'f4')
    points=np.stack([h[lip_ids],t[lip_ids]],axis=1).reshape(-1,3)
    width=F(np.ptp(points,axis=0).max());reach=F(F(radius)*width);fade=F(F(margin)*width)
    seeds=[]
    for point in points:
        ds=[distance(point,x) for x in v];i=int(np.argmin(np.where(np.isnan(ds),np.inf,ds)));seeds.append((i,ds[i]))
    dist=surface_distances(graph['rows'],graph['columns'],graph['weights'],seeds,F(reach+fade),allow_nonfinite_edges=allow_nonfinite_vertices,bounded=bounded)
    mask=np.zeros(len(v),'f4')
    for i,d in enumerate(dist):
        if not np.isfinite(d):continue
        if d<=reach:mask[i]=1
        elif fade>0:
            q=F(F(1)-F(F(d-reach)/fade));mask[i]=F(F(q*q)*F(F(3)-F(q+q)))
    return mask

def smooth_surface(weights,bones,graph,mask,*,factor=.5,passes=3,bounded=True):
    w=np.asarray(weights,'f4');ids=np.asarray(bones,'i4');m=np.asarray(mask,'f4');n,k=w.shape
    if w.shape!=ids.shape or len(m)!=n or passes<0 or (bounded and passes>16) or k>(16 if bounded else 32) or (bounded and n>10000):raise ValueError('face smoothing research bounds')
    r,c=graph['rows'],graph['columns'];factor=F(np.clip(factor,0,1))
    if not bounded:
        from .face_memory import require
        require(n,len(c),slots=k,fields=1)
    w=w.copy();ids=ids.copy()
    for _ in range(passes):
        prior=w.copy();groups=ids.copy()
        for i in range(n):
            alpha=F(factor*F(np.clip(m[i],0,1)));near=c[r[i]:r[i+1]]
            if alpha<=0 or not len(near):continue
            pairs=[]
            def add(bone,value):
                if bone<0 or value<=0:return
                item=next((p for p in pairs if p[0]==bone),None)
                if item is None:pairs.append([int(bone),value])
                else:item[1]=F(item[1]+value)
            for bone,value in zip(groups[i],prior[i]):add(bone,F(F(F(1)-alpha)*value))
            for j in near:
                for bone,value in zip(groups[j],prior[j]):add(bone,F(F(alpha/F(len(near)))*value))
            pairs.sort(key=lambda p:-float(p[1]));pairs=pairs[:k];total=F(-0.)
            for _,value in pairs:total=F(total+value)
            if total<=0:continue
            w[i]=0;ids[i]=-1
            for col,(bone,value) in enumerate(pairs):ids[i,col]=bone;w[i,col]=F(value/total)
    return w,ids

def face_weight_mask(weights,bones,roles):
    """Select positive influences owned by native face role IDs (bounded input).

    This recovers the selection contract, not the separate hash-map smoother.
    `roles` contains (bone index, numeric native role); no name guessing here.
    """
    w=np.asarray(weights,'f4');ids=np.asarray(bones,'i4')
    if w.ndim!=2 or w.shape!=ids.shape or not np.isfinite(w).all():
        raise ValueError('face weight selection requires finite matching arrays')
    owners={bone for bone,role in roles if 0<=role<23 and (0x7f3c60 & (1<<role))}
    return np.any((w>0)&(ids>=0)&np.isin(ids,list(owners)),axis=1).astype('u1')
