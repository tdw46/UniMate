"""Fit shared finger knuckles to closed cross-sections of the base mesh."""
from itertools import product

import numpy as np


def contours(triangles, origin, axis, tolerance):
    """Return closed transverse contours, joining duplicated OBJ/UV seams."""
    signed = (triangles-origin) @ axis
    active = (signed.min(1) < 0.) & (signed.max(1) >= 0.)
    tris, d = triangles[active], signed[active]
    ends = [[] for _ in tris]
    for a,b in ((0,1),(1,2),(2,0)):
        ids = np.flatnonzero((d[:,a] < 0.) != (d[:,b] < 0.))
        values = tris[ids,a] + (tris[ids,b]-tris[ids,a])*(d[ids,a]/(d[ids,a]-d[ids,b]))[:,None]
        for i,p in zip(ids,values):
            ends[i].append(p)
    graph,points = {},{}
    for seg in ends:
        if len(seg)!=2:
            continue
        a,b = [tuple(np.rint((p-origin)/tolerance).astype(int)) for p in seg]
        if a==b:
            continue
        points[a],points[b]=seg
        graph.setdefault(a,set()).add(b)
        graph.setdefault(b,set()).add(a)
    remaining=set(graph)
    result=[]
    while remaining:
        a=remaining.pop(); stack=[a]; component=[a]
        while stack:
            for b in graph[stack.pop()] & remaining:
                remaining.remove(b);stack.append(b);component.append(b)
        if len(component)<3 or any(len(graph[k])!=2 for k in component):
            continue
        loop=[a];previous=None
        while True:
            following=next(k for k in sorted(graph[loop[-1]]) if k != previous)
            if following==a:
                break
            previous=loop[-1];loop.append(following)
        result.append(np.array([points[k] for k in loop]))
    return result


def _sections(tris, point, axis, scale):
    helper=np.eye(3)[np.abs(axis).argmin()]
    u=np.cross(axis,helper);u/=np.linalg.norm(u)
    v=np.cross(axis,u)
    result=[]
    for loop in contours(tris,point,axis,scale*.00002):
        flat=(loop-point)@np.array([u,v]).T
        nxt=np.roll(flat,-1,axis=0)
        cross=flat[:,0]*nxt[:,1]-flat[:,1]*nxt[:,0]
        area2=cross.sum()
        if abs(area2)<scale**2*.00008:
            continue
        center=((flat+nxt)*cross[:,None]).sum(0)/(3*area2)
        span=np.ptp(flat,axis=0)
        if span.max() > scale*.45 or span.min()<scale*.015:
            continue
        # Reject a concave loop whose centroid falls outside its own surface.
        q=flat-center; nxt=np.roll(q,-1,axis=0)
        cuts=(q[:,1]>0)!=(nxt[:,1]>0)
        if np.count_nonzero((q[cuts,0]-q[cuts,1]*(nxt[cuts,0]-q[cuts,0])/(nxt[cuts,1]-q[cuts,1]))>0)%2!=1:
            continue
        result.append((point+center[0]*u+center[1]*v,np.sqrt(abs(area2)*.5/np.pi)))
    return result


def _outside_samples(tree, chain, scale):
    """Test the interior of the limb shell using winding-independent ray parity."""
    from mathutils import Vector
    misses = 0
    for a,b in zip(chain,chain[1:]):
        for t in (0.,.25,.5,.75,1.):
            point = Vector(a*(1.-t)+b*t)
            votes = 0
            for dr in ((.25,.37,.89),(-.27,.11,.94),(.71,-.58,.19)):
                ray = Vector(dr).normalized()
                pos = point
                crossings = 0
                for _ in range(48):
                    hit = tree.ray_cast(pos,ray)[0]
                    if hit is None:
                        break
                    pos = hit+ray*(scale*.00001)
                    crossings += 1
                votes += crossings%2
            misses += votes<2
    return misses


def fit_digits(vertices, faces, joints):
    """Follow each digit from its predicted middle to its tip and back to the web."""
    from mathutils.bvhtree import BVHTree
    tree = BVHTree.FromPolygons(vertices.tolist(), faces.tolist(), all_triangles=True)
    report={}
    for side in ('Left','Right'):
        digits=('Thumb','Index','Middle','Ring','Pinky')
        if any(side+'Hand'+d+str(i) not in joints for d in digits for i in (1,2,3)):
            continue
        hand=joints[side+'Hand']
        scale=np.linalg.norm(np.subtract(hand['tail'],hand['head']))
        if scale<1e-7:
            continue
        chains=[]
        for d in digits:
            bs=[joints[f'{side}Hand{d}{i}'] for i in (1,2,3)]
            chains.append(np.array([bs[0]['head'],np.mean((bs[0]['tail'],bs[1]['head']),0),
                                    np.mean((bs[1]['tail'],bs[2]['head']),0),bs[2]['tail']]))
        # No source bone, weights, object names or material labels enter this crop.
        low=np.min(chains,axis=(0,1))-scale*.65
        high=np.max(chains,axis=(0,1))+scale*.65
        mask=np.all((vertices>=low)&(vertices<=high),axis=1)
        tris=vertices[faces[mask[faces].all(axis=1)]].astype(float)
        anchors=[]
        for d,c in zip(digits,chains):
            axis=c[-1]-c[0]; length=np.linalg.norm(axis)
            if length < scale*.15:
                anchors.append([]);continue
            axis/=length
            stations=(c-c[0])@axis
            samples=np.linspace(stations[0]-.16*length,stations[-1]+.42*length,49)
            slices=[_sections(tris,c[0]+axis*t,axis,scale) for t in samples]
            # Near mid-shaft the palm has already branched into separate tubes.
            candidates = []
            # A seam or the thumb web can leave individual slice planes open.
            # Try three middle stations and bridge only a short gap on the
            # same smooth trajectory, without jumping to an adjacent finger.
            for fraction in (.34,.46,.60,.73):
                mid = int(np.argmin(abs(samples-fraction*length)))
                predicted = c[0]+axis*samples[mid]
                if not slices[mid]:
                    continue
                nearby = sorted(slices[mid],key=lambda pair:np.linalg.norm(pair[0]-predicted))[:3]
                for center,radius in nearby:
                    offset = np.linalg.norm(center-predicted)
                    if offset > scale*.48:
                        continue
                    path = {mid:(center,radius)}
                    for step in (-1,1):
                        prev,rad,last = center,radius,mid
                        velocity = axis
                        for i in range(mid+step, len(samples) if step==1 else -1, step):
                            if abs(last-i)>4:
                                break
                            interval = samples[i]-samples[last]
                            expected = prev+velocity*interval
                            choices = [(p,r) for p,r in slices[i]
                                       if np.linalg.norm(p-expected)<max(rad*.95,scale*.034)
                                       and .24*rad<r<2.*rad and r<radius*2.2]
                            if choices:
                                nxt,rad = min(choices,key=lambda item:np.linalg.norm(item[0]-expected))
                                velocity = .5*axis+.5*(nxt-prev)/interval
                                prev,last = nxt,i
                                path[i] = (prev,rad)
                    ids = sorted(path)
                    if len(ids)>=8:
                        candidates.append((min(length,samples[ids[-1]]-samples[ids[0]])*.40 - offset,ids,path))
            if not candidates:
                anchors.append([]);continue
            alternatives = []
            for score,ids,path in sorted(candidates,key=lambda item:-item[0]):
                track=np.array([path[i][0] for i in ids]);t=samples[ids]
                fitted_end=t[-1]-.015*(t[-1]-t[0])
                targets=fitted_end*np.clip(stations/length,0.,1.)
                out=np.array([np.interp(targets,t,track[:,k]) for k in range(3)]).T
                # Blend back to the predicted palm mount beyond the closed web.
                for n,target in enumerate(targets):
                    if target<t[0]:
                        fixed = out[n]+axis*(target-t[0])
                        mix = min(1.,(t[0]-target)/(.4*length))
                        out[n] = fixed*(1.-mix)+c[n]*mix
                if fitted_end < length*.43 or any(np.linalg.norm(out[-1]-q[-1])<scale*.06 for _,q in alternatives):
                    continue
                alternatives.append((score,out))
            anchors.append(alternatives[:3])
        # Allocate distinct tubes to all five digits together. A greedy
        # nearest choice can send the index and middle down the same tube.
        best_key = (-1, -float('inf'))
        assigned = (None,)*5
        for option in product(*[items+[(0.,None)] for items in anchors]):
            lines = [q for _,q in option if q is not None]
            if any(np.linalg.norm(q[-1]-p[-1])<scale*.08 for i,q in enumerate(lines) for p in lines[i+1:]):
                continue
            quality = len(lines),sum(score for score,q in option)
            if quality > best_key:
                best_key = quality
                assigned = [q for _,q in option]
        for i,(d,out) in enumerate(zip(digits,assigned)):
            if out is not None:
                original_errors = _outside_samples(tree, chains[i], scale)
                # Keep an already-contained chain once its shared knuckles are
                # reconciled. An uncertain fit must not lose surface coverage.
                if original_errors == 0 or _outside_samples(tree,out,scale) >= original_errors:
                    out = None
            used=out is not None
            out=out if used else chains[i]
            for n,(a,b) in enumerate(zip(out,out[1:]),1):
                joints[f'{side}Hand{d}{n}']=dict(head=a.tolist(),tail=b.tolist())
            report[side+d]=used
    return report
