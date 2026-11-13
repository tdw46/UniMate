"""Independent bounded distance and corner-plane split fallback."""
import heapq
import numpy as np
from .mouth_project import F,distance

def nearest_endpoints(vertices,heads,tails,ids):
    v=np.asarray(vertices,'f4')
    if not len(v):return []
    result=set()
    for i in ids:
        for pt in (heads[i],tails[i]):
            best=None;cost=F(np.inf)
            for j,q in enumerate(v):
                d=distance(pt,q)
                if d<cost:best=j;cost=d
            if best is not None:result.add(best)
    return sorted(result)

def distances(graph,seeds,limit,blocked):
    r,c,w=[graph[k] for k in ('rows','columns','weights')];n=len(r)-1
    d=np.full(n,np.inf,'f4');heap=[];limit=F(limit)
    for i in seeds:
        if not blocked[i]:d[i]=0;heapq.heappush(heap,(0.,i))
    while heap:
        value,i=heapq.heappop(heap);value=F(value)
        if value>d[i]:continue
        for k in range(r[i],r[i+1]):
            j=c[k];candidate=F(value+F(w[k]))
            if not blocked[j] and candidate<=limit and candidate<d[j]:d[j]=candidate;heapq.heappush(heap,(float(candidate),j))
    return d

def grow(tags,graph,passes,eligible=None):
    r,c=graph['rows'],graph['columns'];tags=tags.copy()
    for _ in range(passes):
        edits=[]
        for i,tag in enumerate(tags):
            if tag or (eligible is not None and not eligible[i]):continue
            near={int(tags[j]) for j in c[r[i]:r[i+1]] if tags[j] in (1,2)}
            if len(near)==1:edits.append((i,near.pop()))
        if not edits:break
        for i,tag in edits:tags[i]=tag
    return tags

def categories(vertices,heads,tails,graph,record,*,bounded=True):
    v=np.asarray(vertices,'f4');h=np.asarray(heads,'f4');t=np.asarray(tails,'f4');radius=F(record['radius']);tags=np.zeros(len(v),'u1')
    if bounded and (len(v)>4096 or not 0<=record['passes']<=16):raise ValueError('fallback research bounds')
    upper=nearest_endpoints(v,h,t,record['upper']);lower=nearest_endpoints(v,h,t,record['lower'])
    if not upper or not lower or radius<=0:return tags
    if record.get('corner_plane') is not None:
        cr=F(record.get('corner_radius',F(record['reach'])*F(.1)))
        blocked=corner_barrier(v,h,graph,record,bounded=bounded)['blocked'] if cr>0 else np.zeros(len(v),'u1')
        union=distances(graph,sorted(set(upper+lower)),radius,np.zeros(len(v),bool))
        a=plane_region(v,graph,record['corner_plane'],union,blocked,upper,True)
        b=plane_region(v,graph,record['corner_plane'],union,blocked,lower,False)
        tags=np.where(a,1,b*2).astype('u1')
        passes=record['passes'] if record['passes'] else len(v)
        tags=grow(tags,graph,passes,np.isfinite(union))
        tags[(tags==0)&np.isfinite(union)]=3
        return tags
    corners=record['corner_points'];cr=F(record.get('corner_radius',F(record['reach'])*F(.1)))
    blocked=np.array([cr>0 and any(distance(pt,p)<cr for p in corners) for pt in v],bool)
    a=distances(graph,upper,radius,blocked);b=distances(graph,lower,radius,blocked)
    margin=F(radius*F(record.get('ambiguity',.001)))
    for i,(x,y) in enumerate(zip(a,b)):
        if not np.isfinite(x) and not np.isfinite(y):continue
        tolerance=F(F(min(x,y)*F(.1))+margin)
        tags[i]=1 if F(x+tolerance)<y else 2 if F(y+tolerance)<x else 3
    return grow(tags,graph,record['passes'])

def plane_region(vertices,graph,plane,dist,blocked,seeds,upper):
    """Native 0x8e93c; strict signed side, seed votes, last-wins ties."""
    from .mouth_project import dot
    if not len(seeds):raise ValueError('plane region requires nonempty seeds')
    v=np.asarray(vertices,'f4');plane=np.asarray(plane,'f4');r,c=graph['rows'],graph['columns'];n=len(v)
    if any(not 0<=i<n for i in seeds):raise ValueError('invalid plane seed')
    sign=np.array([dot(pt-plane[0],plane[2]) for pt in v],'f4')
    eligible=np.isfinite(dist)&(~np.asarray(blocked,bool))&((sign>0) if upper else (sign<0))
    labels=np.full(n,-1,'i4');sizes=[]
    for i in range(n):
        if not eligible[i] or labels[i]>=0:continue
        label=len(sizes);labels[i]=label;stack=[i];size=0
        while stack:
            j=stack.pop();size+=1
            for k in c[r[j]:r[j+1]]:
                if eligible[k] and labels[k]<0:labels[k]=label;stack.append(k)
        sizes.append(size)
    if not sizes:return np.zeros(n,'u1')
    votes=[0]*len(sizes)
    for i in seeds:
        if labels[i]>=0:votes[labels[i]]+=1
    scores=votes if any(votes) else sizes
    winner=max(range(len(scores)),key=lambda i:(scores[i],i))
    return np.asarray(labels==winner,'u1')


def corner_barrier(vertices,heads,graph,record,*,bounded=True):
    """Native two-path corner construction within 0x106f38; 0x8f4f8 anchors."""
    from .mouth_project import dot
    from .mouth_surface_path import shortest_path
    v=np.asarray(vertices,'f4');h=np.asarray(heads,'f4');plane=np.asarray(record['corner_plane'],'f4')
    center,axis,side=plane;reach=F(max(F(1e-6),F(record['reach'])))
    width=F(reach*F(.05));half=F(reach*F(.3));cap=F(reach*F(8))
    axial=np.array([dot(p-center,axis) for p in v],'f4');signed=np.array([dot(p-center,side) for p in v],'f4')
    blocked=(np.abs(signed)<width)&(np.abs(axial)<half)
    def anchor(ids):
        if not ids or not len(v):return None
        bone=min(ids,key=lambda i:abs(dot(h[i]-center,axis)))
        return min(range(len(v)),key=lambda j:distance(h[bone],v[j]))
    start=anchor(record['lower']);end=anchor(record['upper']);paths=[];indices=[]
    if start is not None and end is not None:
        for _ in range(2):
            path=shortest_path(graph['rows'],graph['columns'],graph['weights'],start,end,cap,blocked,bounded=bounded)
            if not path:break
            paths.append(path)
            corner=max(enumerate(path),key=lambda x:(abs(axial[x[1]]),x[0]))[1];indices.append(corner)
            sign=F(1) if axial[corner]>0 else F(-1)
            blocked|=np.asarray(axial*sign,'f4')>width
    centers=v[indices] if indices else np.asarray(record['corner_points'],'f4')
    cr=F(record.get('corner_radius',F(record['reach'])*F(.1)))
    mask=np.array([any(distance(p,q)<cr for q in centers) for p in v],'u1')
    return dict(blocked=mask,indices=indices,paths=paths,centers=centers)
