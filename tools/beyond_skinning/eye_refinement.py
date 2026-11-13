"""Bounded automatic eye refinement regions from the active solve skeleton."""
import numpy as np


def plan(roles,heads,tails,origin,bounds_max,coarse_cell,*,level,multiplier=2,bounded=True):
    if bounded and (not 0<=level<=3 or not 1<=multiplier<=4):raise ValueError('eye refinement level outside research bounds')
    if level==0:return []
    F=np.float32;h=np.asarray(heads,'f4');t=np.asarray(tails,'f4');lo=np.asarray(origin,'f4');hi=np.asarray(bounds_max,'f4')
    if h.shape!=t.shape or h.ndim!=2 or h.shape[1]!=3 or (bounded and len(h)>256) or not np.isfinite(h).all() or not np.isfinite(t).all():raise ValueError('invalid eye geometry')
    cell=F(F(coarse_cell)/F(multiplier**level))
    if lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite([*lo,*hi,cell]).all() or cell<=0 or np.any(hi<lo):raise ValueError('invalid eye grid')
    dims=np.ceil((hi-lo)/cell).astype('i8');boxes=[]
    for side in ('l','r'):
        ids=sorted(i for i,role,s in roles if s==side and role in ('eye','eyelid_top','eyelid_bot','eyelid_corner'))
        if len(ids)<2:continue
        points=np.stack([h[ids],t[ids]],axis=1).reshape(-1,3)
        low=points.min(axis=0);high=points.max(axis=0)
        # Reference padding deliberately uses X/Z span, not the eye's Y depth.
        margin=F(max(F(0),F(high[0]-low[0]),F(high[2]-low[2]))*F(.25))
        a=np.maximum(0,np.floor(((low-margin)-lo)/cell).astype('i8'))
        b=np.maximum(a,np.minimum(dims-1,np.floor(((high+margin)-lo)/cell).astype('i8')))
        boxes.append([level,*a.tolist(),*b.tolist()])
    return boxes
