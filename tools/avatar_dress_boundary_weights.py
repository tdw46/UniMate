"""Bounded neighborhood averaging for upper and lower free skirt weights."""
import itertools
import math
import numpy as np
from mathutils import Vector


def project_chain_moments(values, bones, movable):
    """Preserve smoothed chain shares and joint coordinate in four slots.

    Never prune the smallest weights independently: that creates alternating
    chain/joint choices between neighboring vertices. Solve feasible local
    four-column bases and retain the closest to the averaged distribution.
    """
    chains = [n.rsplit('_',1)[0] for n in bones]
    joints = np.array([int(n.rsplit('_',1)[1]) for n in bones])
    result = values.copy();cache={};error=0.;count=0
    for i in np.flatnonzero(movable):
        columns=tuple(np.flatnonzero(values[i]>0))
        if len(columns)<=4:continue
        if columns not in cache:
            names=sorted({chains[k] for k in columns})
            if len(names)!=3 or len(columns)>6:raise ValueError('Nonlocal skirt patch')
            matrix=np.array([[float(chains[k]==n) for k in columns] for n in names]+[[float(joints[k]) for k in columns]])
            bases=[]
            for subset in itertools.combinations(range(len(columns)),4):
                square=matrix[:,subset]
                if np.linalg.matrix_rank(square)==4:bases.append((np.array(subset),np.linalg.inv(square)))
            cache[columns]=(matrix,bases)
        matrix,bases=cache[columns];dense=values[i,list(columns)];target=matrix@dense
        best=None;cost=float('inf')
        for subset,inverse in bases:
            weights=inverse@target
            if weights.min() < -1e-9:continue
            candidate=np.zeros(len(columns));candidate[subset]=np.maximum(weights,0.)
            delta=float(np.sum((candidate-dense)**2))
            if delta<cost:cost=delta;best=candidate
        if best is None:raise ValueError('No valid four-weight patch')
        err=float(np.max(np.abs(matrix@best-target)))
        if err>1e-7:raise ValueError('Projection changed skirt blend coordinates')
        result[i]=0.;result[i,list(columns)]=best;error=max(error,err);count+=1
    return result,dict(projected_rows=count,coordinate_error=error)


def average_local_weights(values,bones,edges,movable,allowed,iterations=20,factor=.08):
    """Twenty gentle Jacobi averages on the virtual garment graph.

    Only immediate same-leg chains and two adjacent longitudinal joints are
    eligible. The averaging strength is identical above and below the knee.
    Mixed attachments are pinned. Filtering never consumes the posed mesh.
    """
    seed=np.asarray(values,dtype=float)
    movable=np.asarray(movable,dtype=bool);allowed=np.asarray(allowed,dtype=bool)
    if seed.ndim!=2 or seed.shape[1]!=len(bones) or movable.shape!=(len(seed),) or allowed.shape!=seed.shape:
        raise ValueError('Invalid averaging dimensions')
    if not np.isfinite(seed).all() or (seed<0).any() or not 0<=factor<=1 or iterations<0:
        raise ValueError('Invalid averaging input')
    if not len(seed):return seed.copy(),dict(projected_rows=0,coordinate_error=0.)
    dense=seed.copy()
    e=np.asarray(edges,dtype=int).reshape(-1,2)
    if e.size and (e.min()<0 or e.max()>=len(seed)):raise ValueError('Edge outside garment')
    targets=np.r_[np.arange(len(seed)),e[:,0],e[:,1]]
    sources=np.r_[np.arange(len(seed)),e[:,1],e[:,0]]
    degree=np.bincount(targets,minlength=len(seed))[:,None]
    for _ in range(iterations):
        avg=np.zeros_like(dense);np.add.at(avg,targets,dense[sources])
        dense=(1-factor)*dense+factor*avg/degree
        dense*=allowed;dense/=np.maximum(dense.sum(axis=1,keepdims=True),1e-30)
        dense[~movable]=seed[~movable]
    result,report=project_chain_moments(dense,bones,movable)
    return result,report


def smooth_boundaries(rig,bones,body,values,edges,iterations=20):
    from avatar_dress import full_chain_names
    from avatar_directional_contacts import skirt_owner_leg
    w=np.asarray(values,dtype=float)
    if not len(w) or not iterations:return w,dict(changed=0)
    hum=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    thighs=[getattr(hum,s+'_upper_leg').node.bone_name for s in ('left','right')]
    chains={}
    for spring in rig.data.vrm_addon_extension.spring_bone1.springs:
        names=full_chain_names(rig,spring)
        if names and any(n in bones for n in names):chains[spring.vrm_name]=(skirt_owner_leg(rig,spring,thighs),rig.data.bones[names[0]].head_local.copy())
    if len(chains)<3:return w,dict(changed=0)
    center=sum((v[1] for v in chains.values()),Vector())/len(chains)
    order=sorted(chains,key=lambda n:math.atan2(chains[n][1].y-center.y,chains[n][1].x-center.x))
    adjacent={n:{order[(i-1)%len(order)],n,order[(i+1)%len(order)]} for i,n in enumerate(order)}
    cs=[n.rsplit('_',1)[0] for n in bones];joints=np.array([int(n.rsplit('_',1)[1]) for n in bones]);mass=w.sum(axis=1)
    shares=np.stack([w[:,[i for i,c in enumerate(cs) if c==n]].sum(axis=1) for n in order],axis=1)
    ids=[i for i in range(len(w)) if not body[i] and mass[i]>=1.-1e-6]
    if not ids:return w,dict(changed=0)
    lookup={i:j for j,i in enumerate(ids)};local=w[ids].copy();allowed=local>0;movable=np.zeros(len(ids),dtype=bool)
    for j,i in enumerate(ids):
        dominant=order[int(np.argmax(shares[i]))];owner=chains[dominant][0]
        if any(chains[c][0]!=owner for c,s in zip(order,shares[i]) if s>1e-6):continue
        neighbors={c for c in adjacent[dominant] if chains[c][0]==owner}
        q=float(w[i]@joints)/mass[i];low=math.floor(q+1e-8);high=math.ceil(q-1e-8)
        mask=np.array([c in neighbors and low<=k<=high for c,k in zip(cs,joints)])
        if np.any((local[j]>1e-8)&~mask):continue
        allowed[j]=mask;movable[j]=True
    e=[(lookup[a],lookup[b]) for a,b in edges if a in lookup and b in lookup]
    result,report=average_local_weights(local,bones,e,movable,allowed,iterations=iterations)
    output=w.copy();output[ids]=result
    return output,dict(changed=int((np.max(np.abs(output-w),axis=1)>1e-6).sum()),max_chain_neighborhood=3,max_influences=4,method='bounded_neighbor_average',iterations=iterations,factor=.08,**report)
