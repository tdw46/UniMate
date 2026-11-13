"""Recovered CPU isolation component with bounded public geometry scratch.

Original boundaries: 0x198664 (geometry ownership), 0x1438b4 (threshold,
foreign-finger pruning, normalization). No original module is imported.
"""
import numpy as np

def geometry_owners(vertices, normals, heads, tails, chains, *, bounded=True):
    v,n,h,t=(np.asarray(x,dtype='f4') for x in (vertices,normals,heads,tails))
    if v.ndim!=2 or v.shape[1]!=3 or n.shape!=v.shape or h.ndim!=2 or h.shape[1]!=3 or t.shape!=h.shape:
        raise ValueError('Expected matching three-component geometry arrays')
    if not all(np.isfinite(x[start:start+4096]).all()
               for x in (v,n,h,t) for start in range(0,len(x),4096)):
        raise ValueError('Non-finite geometry is not qualified')
    if not len(chains) or (bounded and (len(v)>65536 or len(h)>256 or len(chains)>32)):
        raise ValueError('Isolation component resource bounds exceeded')
    if any(any(i<0 or i>=len(h) for i in c) for c in chains):
        raise ValueError('Invalid chain bone')
    if bounded:
        return _geometry_owners_block(v,n,h,t,chains)
    # Every ownership decision uses only this vertex/normal and the complete
    # ordered chain set. Keep f32/f64 operations and first-owner ties intact;
    # never split/reduce the rig or merge decisions across blocks.
    owners=np.empty(len(v),'i8')
    block=max(1,min(4096,131072//len(chains)))
    for start in range(0,len(v),block):
        owners[start:start+block]=_geometry_owners_block(v[start:start+block],n[start:start+block],h,t,chains)
    return owners


def _geometry_owners_block(v,n,h,t,chains):
    distances=[];closest=[];p=v.astype('f8')
    for chain in chains:
        best=np.full(len(v),np.inf,'f4');qbest=np.zeros_like(v)
        for bone in chain:
            # Subtractions occur in f32 before the source widens to f64.
            a=h[bone];d=(t[bone]-a).astype('f8');ap=(v-a).astype('f8')
            length=(d[0]*d[0]+d[1]*d[1])+d[2]*d[2]
            u=((ap[:,0]*d[0]+ap[:,1]*d[1])+ap[:,2]*d[2])/(length if length else 1.)
            u=np.clip(u,0.,1.);q=a.astype('f8')+u[:,None]*d
            delta=p-q
            distance=np.sqrt(delta[:,2]*delta[:,2]+(delta[:,0]*delta[:,0]+delta[:,1]*delta[:,1])).astype('f4')
            take=distance<best;best[take]=distance[take];qbest[take]=q[take].astype('f4')
        distances.append(best);closest.append(qbest)
    distances=np.stack(distances,1);owners=np.argmin(distances,axis=1).astype('i8')
    near=distances[np.arange(len(v)),owners]*np.float32(2.5)
    nl=np.sqrt((n[:,0]*n[:,0]+n[:,1]*n[:,1])+n[:,2]*n[:,2])
    valid=nl>=np.float32(1e-9);den=np.where(valid,nl,np.float32(1));inward=-n/den[:,None]
    scores=np.full(len(v),-np.inf,'f4');preferred=owners.copy()
    for ci,q in enumerate(closest):
        d=q-v;dl=np.sqrt(d[:,2]*d[:,2]+(d[:,1]*d[:,1]+d[:,0]*d[:,0]))
        active=valid&(distances[:,ci]<=near)&(dl>=np.float32(1e-9))
        score=(inward[:,2]*d[:,2]+(inward[:,1]*d[:,1]+inward[:,0]*d[:,0]))/np.where(active,dl,np.float32(1))
        take=active&(score>scores);scores[take]=score[take];preferred[take]=ci
    use=scores>=np.float32(.25);owners[use]=preferred[use]
    return owners

def prune_weights(weights,bones,chains,owners,threshold,*,bounded=True):
    w=np.asarray(weights,dtype='f4');b=np.asarray(bones,dtype='i4');owners=np.asarray(owners,dtype='i8')
    if w.ndim!=2 or b.shape!=w.shape or owners.shape!=(len(w),) or not 1<=w.shape[1]<=(16 if bounded else 32):
        raise ValueError('Invalid bounded sparse weight layout')
    if not np.isfinite(w).all() or (bounded and not np.isfinite(threshold)) or (w<0).any():
        raise ValueError('Invalid finite weight/threshold input')
    if not len(chains) or (bounded and len(chains)>32) or (owners<0).any() or (owners>=len(chains)).any():
        raise ValueError('Invalid chain ownership')
    if any(any(i<0 or (bounded and i>=256) for i in c) for c in chains):raise ValueError('Invalid bounded chain bone')
    # Sparse IDs avoid allocating through the maximum bone index. Later chains
    # retain the original last-write ownership for a shared bone.
    mapping={int(bone):ci for ci,chain in enumerate(chains) for bone in chain}
    keys=np.array(sorted(mapping),'i8');values=np.array([mapping[int(i)] for i in keys],'i8')
    if bounded:return _prune_block(w,b,chains,owners,threshold,keys,values)
    out=np.empty_like(w);ids=np.empty_like(b);active=np.empty(len(w),bool)
    block=max(1,min(4096,131072//len(chains)))
    for start in range(0,len(w),block):
        end=start+block
        out[start:end],ids[start:end],active[start:end]=_prune_block(w[start:end],b[start:end],chains,owners[start:end],threshold,keys,values)
    return out,ids,active


def _prune_block(w,b,chains,owners,threshold,keys,values):
    groups=np.full(b.shape,-1,'i8')
    if len(keys):
        index=np.searchsorted(keys,b);valid=index<len(keys)
        index=np.minimum(index,len(keys)-1);known=valid&(keys[index]==b)
        groups[known]=values[index[known]]
    sums=np.zeros((len(w),len(chains)),'f4');rows=np.arange(len(w))
    for slot in range(w.shape[1]):
        valid=groups[:,slot]>=0
        sums[rows[valid],groups[valid,slot]]+=w[valid,slot]
    active=np.max(sums,axis=1)>np.float32(threshold)
    result=w.copy();result[active[:,None]&(groups>=0)&(groups!=owners[:,None])]=0
    total=np.zeros(len(w),'f4')
    for slot in range(w.shape[1]):total+=result[:,slot]
    normalize=active&(total>np.float32(1e-8))
    result[normalize]/=total[normalize,None]
    return result,b.copy(),active

def surface_inputs(vertices,faces):
    v=np.asarray(vertices,'f4');f=np.asarray(faces,'i8')
    if f.ndim!=2 or f.shape[1]!=3 or (f<0).any() or (f>=len(v)).any():raise ValueError('Invalid triangles')
    normals=np.zeros_like(v)
    cross=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    np.add.at(normals,f.ravel(),np.repeat(cross,3,axis=0))
    edges=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]))
    return normals,edges

def smooth_weights(weights,bones,active,edges,iterations,*,bounded=True):
    """Fixed exterior neighbors; duplicated triangle edges keep their multiplicity."""
    w=np.asarray(weights,'f4');b=np.asarray(bones,'i4');active=np.asarray(active,bool);e=np.asarray(edges,'i8')
    if iterations<0 or (bounded and iterations>8):raise ValueError('Isolation smoothing count outside research bound')
    if e.ndim!=2 or e.shape[1]!=2 or (e<0).any() or (e>=len(w)).any():raise ValueError('Invalid smoothing edges')
    if not iterations or not active.any() or not len(e):return w.copy(),b.copy()
    e=e[np.any(active[e],axis=1)]
    if not len(e):return w.copy(),b.copy()
    region=np.unique(e);labels=np.unique(b[region]);labels=labels[labels>=0]
    if bounded and len(region)*len(labels)>1_000_000:raise ValueError('Isolation smoothing field budget exceeded')
    if not bounded:_smoothing_memory(len(w),w.shape[1],len(region),len(labels),len(e))
    dense=np.zeros((len(region),len(labels)),'f4')
    local_edges=np.searchsorted(region,e)
    # Internal stage uses assignment in sparse-slot order rather than summing duplicates.
    for slot in range(w.shape[1]):
        good=b[region,slot]>=0;rr=region[good]
        dense[np.searchsorted(region,rr),np.searchsorted(labels,b[rr,slot])]=w[rr,slot]
    degree=np.zeros(len(region),'f4');np.add.at(degree,local_edges[:,0],np.float32(1));np.add.at(degree,local_edges[:,1],np.float32(1))
    rows=np.flatnonzero(active[region]&(degree>0))
    block=max(1,min(4096,65536//max(1,len(labels))))
    for _ in range(iterations):
        sums=np.zeros_like(dense)
        if bounded:
            np.add.at(sums,local_edges[:,0],dense[local_edges[:,1]]);np.add.at(sums,local_edges[:,1],dense[local_edges[:,0]])
            dense[rows]=dense[rows]*np.float32(.5)+(sums[rows]/degree[rows,None])*np.float32(.5)
        else:
            # Preserve two separate ordered sweeps. Interleaving directions per
            # chunk changes f32 accumulation at shared vertices.
            for dst,src in ((0,1),(1,0)):
                for start in range(0,len(local_edges),block):
                    chunk=local_edges[start:start+block]
                    np.add.at(sums,chunk[:,dst],dense[chunk[:,src]])
            for start in range(0,len(rows),block):
                rr=rows[start:start+block]
                dense[rr]=dense[rr]*np.float32(.5)+(sums[rr]/degree[rr,None])*np.float32(.5)
    del sums
    out=w.copy();ids=b.copy();rows=np.flatnonzero(active[region])
    for start in range(0,len(rows),max(1,len(rows)) if bounded else block):
        rr=rows[start:start+(max(1,len(rows)) if bounded else block)]
        order=np.argsort(-dense[rr],axis=1,kind='stable')[:,:w.shape[1]]
        values=np.take_along_axis(dense[rr],order,axis=1);totals=np.zeros(len(rr),'f4')
        for slot in range(values.shape[1]):totals+=values[:,slot]
        totals=np.where(totals<np.float32(1e-8),np.float32(1),totals)
        destination=region[rr];out[destination]=0;ids[destination]=0
        out[destination,:values.shape[1]]=values/totals[:,None];ids[destination,:values.shape[1]]=labels[order]
    return out,ids


def isolate_named(vertices,faces,heads,tails,chains,weights,bones,threshold=.15,smooth=2,*,bounded=True):
    normals,edges=surface_inputs(vertices,faces)
    owners=geometry_owners(vertices,normals,heads,tails,chains,bounded=bounded)
    weights,bones,active=prune_weights(weights,bones,chains,owners,threshold,bounded=bounded)
    weights,bones=smooth_weights(weights,bones,active,edges,smooth,bounded=bounded)
    return weights,bones,dict(chains=chains,active_vertices=int(active.sum()))


def _smoothing_memory(vertices,slots,region,labels,edges,*,budget_bytes=None):
    """Conservative finishing phase estimate; not native allocator-error parity."""
    from .field_memory import bind_phase_budget
    import operator
    v,k,r,l,e=map(operator.index,(vertices,slots,region,labels,edges))
    if min(v,k,r,l,e)<0:raise ValueError('Negative isolation smoothing dimension')
    budget=bind_phase_budget.get() if budget_bytes is None else operator.index(budget_bytes)
    # Dense field/sums, sparse input/output, graph/index/selection and bounded
    # gather/sort scratch. Source mesh/cache admission remains separate.
    raw=32*r*l+24*v*k+128*e+128*v+4*1024*1024
    estimate=(raw*5+3)//4
    if budget is not None and estimate>budget:raise ValueError(f'Isolation smoothing working-set research budget exceeded: {estimate} bytes > {budget} bytes')
    return estimate
