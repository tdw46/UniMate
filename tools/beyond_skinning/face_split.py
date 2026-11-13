"""Bounded independent geometric split planning; no original binary dependency."""
from collections import deque
import numpy as np
from .mouth_project import F,dot,distance

def mean(points):
    out=np.zeros(3,'f4')
    for p in points:out=np.asarray(out+p/F(len(points)),'f4')
    return out

def normalize(v):
    v=np.asarray(v,'f4');length=F(np.sqrt(dot(v,v)))
    return np.zeros(3,'f4') if length<=F(1e-12) else v/length

def endpoints(heads,tails,ids,direction,upper):
    h=np.asarray(heads,'f4');t=np.asarray(tails,'f4');d=np.asarray(direction,'f4')
    # Native AArch64 FCMP/LE is true for unordered values as well as <=.
    return np.array([t[i] if bool(not dot(h[i],d)>dot(t[i],d))==upper else h[i] for i in ids],'f4').reshape(-1,3)

def median_depth(bvh,points,direction,reach):
    d=np.asarray(direction,'f4');depths=[]
    for p in np.asarray(points,'f4'):
        p=p.copy();total=F(0)
        for _ in range(12):
            hit=bvh._raycast((p+d*F(1e-5)).tolist(),d.tolist())
            if hit is None or F(total+F(hit[0]))>F(F(reach)*F(6)):break
            step=F(F(hit[0])+F(1e-5));total=F(total+step);p=np.asarray(p+d*step,'f4')
        depths.append(total)
    return sorted(depths)[len(depths)//2] if depths else F(0)

def project_seeds(bvh,vertices,faces,points,center,axis,direction,reach,allowed=None):
    v=np.asarray(vertices,'f4');f=np.asarray(faces,'i4');p=np.asarray(points,'f4');d=np.asarray(direction,'f4');axis=np.asarray(axis,'f4');center=np.asarray(center,'f4');reach=F(reach)
    if not len(p):return []
    if len(p)>1:
        target=mean(p);target=np.asarray(target-axis*dot(target-center,axis),'f4')
        anchor=p[min(range(len(p)),key=lambda i:distance(p[i],target))]
    else:anchor=p[0]
    mask=None if allowed is None or len(allowed)==0 else np.asarray(allowed,'u1')
    extent=F(reach*F(4));tol=F(reach*F(.25));pos=np.asarray(anchor+d*extent,'f4');total=F(0);best=None;best_distance=F(np.inf)
    for _ in range(16):
        hit=bvh._raycast(pos.tolist(),(-d).tolist())
        if hit is None:break
        step=F(F(hit[0])+F(1e-5));total=F(total+step)
        if F(total-extent)>tol:break
        tri=f[hit[1]];normal=normalize(np.cross(v[tri[1]]-v[tri[0]],v[tri[2]]-v[tri[0]]))
        delta=F(abs(F(total-extent)))
        if (mask is None or all(mask[i]==1 for i in tri)) and delta<=tol and dot(d,normal)>0 and delta<best_distance:
            best=list(map(int,tri));best_distance=delta
        pos=np.asarray(pos-d*step,'f4')
    if best is not None:return best
    eligible=[i for i in range(len(v)) if mask is None or mask[i]==1]
    best=None;cost=F(np.inf)
    for i in eligible:
        d=distance(anchor,v[i])
        if d<cost:best=i;cost=d
    return [] if best is None else [best]

def plan(bvh,vertices,faces,heads,tails,upper,lower,reach,allowed=None):
    v=np.asarray(vertices,'f4');reach=F(reach)
    up=endpoints(heads,tails,upper,[0,0,1],True);lo=endpoints(heads,tails,lower,[0,0,1],False)
    axis_up=normalize(mean(up)-mean(lo))
    if not np.any(axis_up):axis_up=np.array([0,0,1],'f4')
    up=endpoints(heads,tails,upper,axis_up,True);lo=endpoints(heads,tails,lower,axis_up,False)
    points=np.concatenate([up,lo]);center=mean(points);total=np.zeros(2,'f4');count=0
    for p in v:
        if distance(p,center)<F(reach*F(3)):total=np.asarray(total+p[:2],'f4');count+=1
    direction=np.array([*(center[:2]-total/F(count)),0],'f4') if count else np.array([0,-1,0],'f4')
    direction=normalize(direction)
    if not np.any(direction):direction=np.array([0,-1,0],'f4')
    forward=median_depth(bvh,points,direction,reach);back=median_depth(bvh,points,-direction,reach)
    if back<forward:direction=-direction
    axis=normalize(np.cross(direction,axis_up))
    seeds=[project_seeds(bvh,v,faces,p,center,axis,direction,reach,allowed) for p in (up,lo)]
    return dict(upper=up,lower=lo,center=center,axis=axis,direction=direction,seeds=seeds)

def propagate(vertices,graph,seeds,radius,passes=3,corner_plane=None,corner_points=None):
    """Native geometric branch: FIFO ownership then synchronous unambiguous growth."""
    v=np.asarray(vertices,'f4');r,c=graph['rows'],graph['columns'];tags=np.zeros(len(v),'u1');queue=deque()
    all_seeds=seeds[0]+seeds[1]
    if not all_seeds:return tags
    near=np.array([min(distance(p,v[j]) for j in all_seeds)<=F(radius) for p in v])
    sides=np.zeros(len(v),'i1')
    if corner_plane is not None and corner_points is not None and len(corner_points):
        center,normal,side=np.asarray(corner_plane,'f4').reshape(3,3)
        width=max(abs(dot(np.asarray(p,'f4')-center,normal)) for p in corner_points)
        for i,p in enumerate(v):
            if abs(dot(p-center,normal))>width:sides[i]=1 if dot(p-center,side)>0 else -1
    for tag,indices in enumerate(seeds,1):
        for i in indices:
            if not tags[i]:tags[i]=tag;queue.append(i)
    while queue:
        i=queue.popleft()
        if not near[i]:continue
        sign=1 if tags[i]==1 else -1
        for j in c[r[i]:r[i+1]]:
            if not tags[j] and sides[j] in (0,sign):tags[j]=tags[i];queue.append(j)
    for _ in range(passes):
        edits=[]
        for i in range(len(v)):
            if tags[i]:continue
            ns={int(tags[j]) for j in c[r[i]:r[i+1]] if tags[j] in (1,2)}
            if len(ns)==1:edits.append((i,ns.pop()))
        if not edits:break
        for i,tag in edits:tags[i]=tag
    return tags

def corner_plane(upper_points,lower_points,corners):
    up=np.asarray(upper_points,'f4');lo=np.asarray(lower_points,'f4');c=np.asarray(corners,'f4')
    if not len(up) or not len(lo) or len(c)<2:return None
    pairs=[(i,j) for i in range(len(c)) for j in range(i+1,len(c))]
    i,j=max(pairs,key=lambda pair:distance(c[pair[0]],c[pair[1]]));length=distance(c[i],c[j])
    if length<=0:return None
    axis=(c[j]-c[i])/length;delta=mean(up)-mean(lo)
    side=np.array([F(F(axis[1]*F(0))-axis[2]),F(F(axis[2]*F(0))+F(axis[0]*F(-0.))),F(axis[0]+F(axis[1]*F(-0.)))],'f4')
    norm=F(np.sqrt(F(side[2]*side[2]+F(side[0]*side[0]+side[1]*side[1]))))
    if norm<F(.1):side=delta-axis*dot(delta,axis);norm=F(np.sqrt(dot(side,side)))
    if norm<=F(1e-9):return None
    side=side/norm
    if dot(delta,side)<0:side=-side
    return np.array([(c[i]+c[j])*F(.5),axis,side],'f4')

def mouth_record(runtime,names,heads,tails,parents):
    """Default named mouth record. Exclusion/geometry availability is separate."""
    h=np.asarray(heads,'f4');t=np.asarray(tails,'f4');roles,_=runtime.detect_face(names,list(map(int,parents)),None,-1)
    upper=[i for i,r,_ in roles if r=='lip_top'];lower=[i for i,r,_ in roles if r=='lip_bot'];corners=[i for i,r,_ in roles if r=='lip_corner']
    if not upper or not lower:raise ValueError('split mouth requires upper and lower roles')
    lips=upper+lower+corners;lo=np.minimum(h[lips].min(0),t[lips].min(0));hi=t[lips].max(0);reach=F(max(F(0),*(hi-lo)))
    if not np.isfinite(reach) or reach<=0:raise ValueError('degenerate split mouth')
    return dict(upper=upper,lower=lower,reach=reach,radius=F(reach*F(.75)),passes=3,corner_radius=F(reach*F(.1)),ambiguity=F(.001),corner_plane=corner_plane(h[upper],h[lower],h[corners]),corner_points=h[corners])

def categories(runtime,vertices,faces,heads,tails,graph,record,allowed,*,allow_nonfinite_vertices=False,bounded=True):
    """Consume an explicit ownership mask, prepared or independently selected."""
    v=np.ascontiguousarray(vertices,'f4');f=np.ascontiguousarray(faces,'i4');mask=np.asarray(allowed,'u1')
    if (bounded and len(v)>4096) or mask.shape!=(len(v),) or (not allow_nonfinite_vertices and not np.isfinite(v).all()) or np.any(mask>1):raise ValueError('split research bounds')
    from .split_fallback import categories as fallback
    if not len(f) or not record['upper'] or not record['lower'] or record['radius']<=0:
        return fallback(v,heads,tails,graph,record,bounded=bounded)
    bvh=runtime.MeshBvh(v,f)
    prepared=plan(bvh,v,f,heads,tails,record['upper'],record['lower'],record['reach'],mask)
    if any(not x for x in prepared['seeds']):return fallback(v,heads,tails,graph,record,bounded=bounded)
    return propagate(v,graph,prepared['seeds'],record['radius'],record['passes'],record.get('corner_plane'),record.get('corner_points'))

def constrain_weights(vertices,heads,record,tags,weights,bones):
    """Final split enforcement (0x10c0f0) runs after both smoothing passes."""
    v=np.asarray(vertices,'f4');h=np.asarray(heads,'f4');w=np.asarray(weights,'f4').copy();ids=np.asarray(bones,'i4').copy()
    maps={}
    for label,source,target in [(1,record['lower'],record['upper']),(2,record['upper'],record['lower'])]:
        maps[label]={i:min(target,key=lambda j:distance(h[i],h[j])) for i in source} if target else {}
    corners=record['corner_points'];fade=F(record['reach']*F(.2))
    for i,tag in enumerate(tags):
        if tag not in maps:continue
        amount=F(1)
        if fade>0 and len(corners):amount=F(np.clip(F(min(distance(v[i],p) for p in corners)/fade),0,1))
        if amount<=0:continue
        for slot in range(w.shape[1]):
            bone=int(ids[i,slot]);weight=w[i,slot]
            if weight<=0 or bone not in maps[tag]:continue
            target=maps[tag][bone];moved=F(amount*weight);remaining=F(weight-moved)
            existing=next((j for j in range(w.shape[1]) if j!=slot and ids[i,j]==target),None)
            if existing is not None:
                w[i,existing]=F(w[i,existing]+moved);w[i,slot]=remaining
                if remaining<=0:ids[i,slot]=-1
            elif remaining<=0:ids[i,slot]=target
            else:
                empty=next((j for j in range(w.shape[1]) if j!=slot and (ids[i,j]<0 or w[i,j]<=0)),None)
                if empty is None:ids[i,slot]=target
                else:ids[i,empty]=target;w[i,empty]=moved;w[i,slot]=remaining
    return w,ids
