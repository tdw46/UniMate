"""Independent native face split component selector (0x10b304 / 0x10656c).

Bounded research implementation. Selection counts eyelid endpoint votes, not mesh
vertices. A component must also have a vertex within the mouth seed threshold.
"""
import numpy as np
from .mouth_project import F

def components(count,edges):
    adj=[[] for _ in range(count)]
    for a,b in np.asarray(edges,'i8').reshape(-1,2):
        if 0<=a<count and 0<=b<count:adj[a].append(int(b));adj[b].append(int(a))
    labels=np.full(count,-1,'i4');component=0
    for i in range(count):
        if labels[i]>=0:continue
        labels[i]=component;stack=[i]
        while stack:
            for j in adj[stack.pop()]:
                if labels[j]<0:labels[j]=component;stack.append(j)
        component+=1
    return labels

def distances(points,seed):
    d=np.asarray(points-seed,'f4');s=d*d
    return np.sqrt(np.asarray(np.asarray(s[:,0]+s[:,1],'f4')+s[:,2],'f4'))

def select(vertices,edges,eye_seeds,shape_seeds,threshold,*,allow_nonfinite_vertices=False,bounded=True):
    v=np.asarray(vertices,'f4').reshape(-1,3);eyes=np.asarray(eye_seeds,'f4').reshape(-1,3);shape=np.asarray(shape_seeds,'f4').reshape(-1,3);threshold=F(threshold)
    if not threshold>0 or not len(v) or not len(eyes) or not len(shape):return None
    if (bounded and (len(v)>4096 or not all(np.isfinite(x).all() for x in (eyes,shape)))) or (not allow_nonfinite_vertices and not np.isfinite(v).all()):raise ValueError('ownership research bounds')
    labels=components(len(v),edges);votes={}
    for pt in eyes:
        d=distances(v,pt)
        if not np.any(np.isfinite(d)):continue
        owner=int(labels[int(np.argmin(np.where(np.isnan(d),np.inf,d)))]);votes[owner]=votes.get(owner,0)+1
    eligible=[]
    for owner,count in votes.items():
        points=v[labels==owner]
        if any(np.fmin.reduce(distances(points,pt),initial=np.float32(np.inf))<=threshold for pt in shape):eligible.append((owner,count))
    if not eligible:return None
    owner=min(eligible,key=lambda pair:(-pair[1],pair[0]))[0]
    return np.asarray(labels==owner,'u1')

def mouth_inputs(runtime,names,heads,tails,parents,record):
    """Default named mouth route only; arbitrary shape collections remain open."""
    h=np.asarray(heads,'f4');t=np.asarray(tails,'f4');roles,_=runtime.detect_face(names,list(map(int,parents)),None,-1)
    lids=[i for i,r,_ in roles if r in ('eyelid_top','eyelid_bot','eyelid_corner')]
    pairs=lambda ids:np.asarray([pt for i in ids for pt in (h[i],t[i])],'f4').reshape(-1,3)
    return pairs(lids),pairs(record['upper']+record['lower']),F(record['reach']*F(.25))
