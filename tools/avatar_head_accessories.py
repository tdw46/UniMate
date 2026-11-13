"""Compact sheet-like head attachments, inferred from rest mesh geometry only."""
import heapq
import numpy as np
from mathutils import Vector
from avatar_virtual_topology import connected_graph, protected_extremities


def plan_head_accessories(rig, obj, points, excluded, height, prefix, start_index=0):
    head = rig.data.bones.get('Head')
    if head is None or height <= 0:
        return [], [], dict(components=0)
    allowed = set(range(len(points))) - set(excluded)
    allowed -= set(np.flatnonzero(protected_extremities(rig, points, height)))
    graph, groups, _ = connected_graph(obj.data, points, allowed, height*.0001)
    a, b = np.array(head.head_local), np.array(head.tail_local)
    axis = b-a; length2 = float(axis@axis)
    if length2 < 1e-12:
        return [], [], dict(components=0)
    neighbors = {i: [] for i in allowed}
    for edge in graph.edges:
        i,j = edge.vertices; distance = float(np.linalg.norm(points[i]-points[j]))
        neighbors[i].append((j,distance)); neighbors[j].append((i,distance))
    chains, regions = [], []
    for ids in groups:
        p = points[ids]; span = np.ptp(p,axis=0)
        if len(ids)<8 or not height*.025<max(span)<height*.10:
            continue
        # Rear/lateral upper-head attachments, not facial surfaces or earrings.
        if p[:,2].min()<a[2]+(b[2]-a[2])*.65 or p[:,2].max()>b[2]+height*.10:
            continue
        if np.median(p[:,1])<max(a[1],b[1]):
            continue
        eigen = np.linalg.eigvalsh(np.cov(p.T)).clip(0)
        if eigen[-1]<=0 or eigen[0]>eigen[-1]*.30**2 or eigen[1]<eigen[-1]*.20**2:
            continue
        t = np.clip((p-a)@axis/length2,0,1)
        distance = np.linalg.norm(p-a-t[:,None]*axis,axis=1)
        if distance.min()>height*.065:
            continue
        anchor = ids[int(np.argmin(distance))]
        valid=set(ids); dist={anchor:0.}; queue=[(0.,anchor)]
        while queue:
            d,i=heapq.heappop(queue)
            if d!=dist[i]:continue
            for j,w in neighbors[i]:
                nd=d+w
                if j in valid and nd<dist.get(j,float('inf')):
                    dist[j]=nd;heapq.heappush(queue,(nd,j))
        reach=max(dist.values())
        if reach<height*.025:continue
        # Geodesic bands follow the attachment's resting direction, including
        # upward bow loops; no vertical/gravity-oriented hair projection.
        samples=np.array([dist[i]/reach for i in ids])
        line=[Vector(points[anchor])]
        for level in (.5,1.):
            band=p[np.abs(samples-level)<=.16]
            if not len(band):band=p[[int(np.argmin(abs(samples-level)))]]
            line.append(Vector(np.mean(band,axis=0)))
        if any((y-x).length<height*.002 for x,y in zip(line,line[1:])):continue
        label=f'{prefix}Ornament_{start_index+len(chains):02d}'
        names=[label+f'_{i:02d}' for i in range(3)]
        values={}
        for i,t in zip(ids,samples):
            root=float(np.clip(t/.2,0,1));root=root*root*(3-2*root)
            tip=float(np.clip((t-.35)/.5,0,1));tip=tip*tip*(3-2*tip)
            values[i]={'Head':1-root,names[0]:root*(1-tip),names[1]:root*tip}
        chains.append(dict(role='ornament',names=names,line=line,parent='Head',physics_start=0))
        regions.append(dict(obj=obj,role='ornament',ids=ids,values=values))
    return chains,regions,dict(components=len(chains))
