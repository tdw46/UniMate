"""Coherent local skirt smoothing with at most four skin influences.

Diffuse longitudinal coordinates within the existing circumferential chain
pair; reconstruct a two-chain/two-joint patch. RWT-style removal-mask limiting
is retained only for over-budget mixed waistband rows. Free skirt patches never use global bone-column diffusion. Upper and lower free skirt regions use the same gentle neighbor average.
Coincident garment vertices share one virtual node throughout the filters.
"""
import numpy as np


def smooth_attachment_mass(seed,edges,iterations=20,factor=.2):
    """Smooth only the scalar body/spring handoff with both ends pinned."""
    ids=sorted(seed);lookup={i:j for j,i in enumerate(ids)}
    initial=np.array([seed[i] for i in ids]);mass=initial.copy()
    edges=np.asarray([(lookup[a],lookup[b]) for a,b in edges],dtype=int).reshape((-1,2))
    targets=np.concatenate((np.arange(len(ids)),edges[:,0],edges[:,1]))
    sources=np.concatenate((np.arange(len(ids)),edges[:,1],edges[:,0]))
    degrees=np.bincount(targets,minlength=len(ids))
    movable=(initial>0)&(initial<1)
    for _ in range(iterations):
        average=np.bincount(targets,weights=mass[sources],minlength=len(ids))/degrees
        mass[movable]=(1-factor)*mass[movable]+factor*average[movable]
    return dict(zip(ids,map(float,mass)))


def smooth_limited_weights(values, edges, budgets, movable, iterations=20,
                           factor=.2, mask_iterations=5, allowed=None):
    """Smooth rows, preserve pinned rows, and enforce a per-row influence budget.

    Input/output rows are normalized spring distributions. The caller restores
    each vertex's original spring mass, keeping its body attachment separate.
    Edges must already be restricted to one garment; never use posed proximity.
    """
    w=np.asarray(values,dtype=float).copy()
    budgets=np.asarray(budgets,dtype=int);movable=np.asarray(movable,dtype=bool)
    if w.ndim!=2 or len(budgets)!=len(w) or len(movable)!=len(w):raise ValueError('Invalid weight dimensions')
    if not np.isfinite(w).all() or (w<0).any():raise ValueError('Invalid skin weights')
    if iterations<0 or mask_iterations<0 or not 0<=factor<=1:raise ValueError('Invalid smoothing settings')
    if (budgets<1).any():raise ValueError('No free influence slot for skirt springs')
    if not len(w):return w
    edges=np.asarray(edges,dtype=int).reshape((-1,2))
    if edges.size and (edges.min()<0 or edges.max()>=len(w)):raise ValueError('Edge outside garment')
    targets=np.concatenate((np.arange(len(w)),edges[:,0],edges[:,1]))
    sources=np.concatenate((np.arange(len(w)),edges[:,1],edges[:,0]))
    degrees=np.bincount(targets,minlength=len(w))[:,None]
    def average(rows):
        out=np.zeros_like(rows);np.add.at(out,targets,rows[sources])
        return out/degrees
    w/=np.maximum(w.sum(axis=1,keepdims=True),1e-30)
    pinned=w.copy()
    if allowed is not None:
        allowed=np.asarray(allowed,dtype=bool)
        if allowed.shape!=w.shape:raise ValueError("Invalid local support mask")
    for _ in range(iterations):
        w=(1-factor)*w+factor*average(w)
        if allowed is not None:
            w*=allowed
            w/=np.maximum(w.sum(axis=1,keepdims=True),1e-30)
        w[~movable]=pinned[~movable]
    # Stable ordering makes equal influences deterministic on symmetric rigs.
    def removal(rows):
        order=np.argsort(-rows,axis=1,kind='stable')
        rank=np.empty_like(order)
        rank[np.arange(len(rows))[:,None],order]=np.arange(rows.shape[1])
        return (rank>=budgets[:,None]) & ((rows>1e-4).sum(axis=1)>budgets)[:,None]
    w[w<=1e-4]=0.
    seeds=removal(w).astype(float)
    mask=seeds.copy()
    for _ in range(mask_iterations):mask=np.maximum(mask,average(mask))
    limited=w*(1-mask)
    # Pinned attachment distributions participate as a boundary, but should
    # only change if they themselves exceed the four-weight budget.
    limited[~movable]=w[~movable]*(1-seeds[~movable])
    limited[limited<=1e-4]=0.
    empty=limited.sum(axis=1)<=1e-12
    limited[empty]=w[empty]
    limited[removal(limited)]=0.
    totals=limited.sum(axis=1,keepdims=True)
    if (totals<=0).any():raise ValueError('Smoothing produced an empty skin assignment')
    limited/=totals
    assert np.all((limited>0).sum(axis=1)<=budgets)
    return limited


def smooth_local_coordinates(values, edges, bones, movable, iterations=20, factor=.2):
    """Diffuse longitudinal bend coordinates, retaining the original chain pair.

    Reconstruct a continuous two-chain by two-adjacent-joint patch instead of
    independently truncating diffused bone columns. This avoids third-chain
    leakage and mismatched upper/lower blend fractions at the same vertex.
    """
    w=np.asarray(values,dtype=float)
    chains=[n.rsplit('_',1)[0] for n in bones]
    joints=np.array([int(n.rsplit('_',1)[1]) for n in bones])
    if iterations<0 or not 0<=factor<=1:raise ValueError('Invalid smoothing settings')
    q=w@joints
    pinned=q.copy()
    edges=np.asarray(edges,dtype=int).reshape((-1,2))
    targets=np.concatenate((np.arange(len(w)),edges[:,0],edges[:,1]))
    sources=np.concatenate((np.arange(len(w)),edges[:,1],edges[:,0]))
    degrees=np.bincount(targets,minlength=len(w))
    for _ in range(iterations):
        averaged=np.bincount(targets,weights=q[sources],minlength=len(w))/degrees
        q=(1-factor)*q+factor*averaged
        q[~movable]=pinned[~movable]
    result=np.zeros_like(w)
    for chain in sorted(set(chains)):
        columns=[i for i,n in enumerate(chains) if n==chain]
        share=w[:,columns].sum(axis=1)
        ordered=sorted(columns,key=lambda i:joints[i])
        indices=joints[ordered]
        if not np.array_equal(indices,np.arange(indices[0],indices[-1]+1)):
            raise ValueError('Skirt smoothing requires contiguous generated joint indices')
        coordinate=np.clip(q,indices[0],indices[-1])
        for col in ordered:
            result[:,col]=share*np.maximum(0.,1.-np.abs(coordinate-joints[col]))
    assert np.all((result>1e-8).sum(axis=1)<=4)
    return result


def smooth_body_attachment(rig,obj,ids,body,mass,edges,iterations,factor):
    """Smooth body fractions and collapse them continuously onto a Hips star.

    Hips is always the anchor. Competition between secondary body influences
    fades toward Hips, softening the discontinuous top-two selection.
    Spring weights and their circumferential ownership are independent.
    """
    from avatar_pelvis_binding import landmarks,pelvis_owned_weights
    hips,spine,_,thighs,_,_,_,_=landmarks(rig)
    names=sorted({n for row in body for n in row}|{hips.name})
    total=np.array([sum(row.values()) for row in body])
    original=np.array([[row.get(n,0.) for n in names] for row in body])
    normalized=original/np.maximum(total[:,None],1e-30)
    # Only neighbors that themselves carry body weights are source samples.
    e=np.asarray([(a,b) for a,b in edges if total[a]>1e-8 and total[b]>1e-8],dtype=int).reshape((-1,2))
    targets=np.concatenate((np.arange(len(ids)),e[:,0],e[:,1]))
    sources=np.concatenate((np.arange(len(ids)),e[:,1],e[:,0]))
    degrees=np.bincount(targets,minlength=len(ids))[:,None]
    field=normalized.copy()
    for _ in range(iterations):
        averaged=np.zeros_like(field);np.add.at(averaged,targets,field[sources])
        field=(1-factor)*field+factor*averaged/degrees
    # Match the unchanged pure-body boundary continuously.
    blend=np.clip(mass/.25,0.,1.);blend=blend*blend*(3.-2.*blend)
    field=normalized*(1-blend[:,None])+field*blend[:,None]
    matrix=rig.matrix_world.inverted()@obj.matrix_world
    result=[]
    for j,index in enumerate(ids):
        if total[j]<=1e-8:result.append({});continue
        point=matrix@obj.data.vertices[index].co
        values=pelvis_owned_weights(dict(zip(names,field[j])),point,hips,spine,thighs)
        for bone in thighs:
            # The spring root already follows this leg. A second, direct
            # thigh skin path competes with that partial/physical transform.
            values[hips.name]=values.get(hips.name,0.)+values.pop(bone.name,0.)
        others=sorted(((v,n) for n,v in values.items() if n!=hips.name and v>1e-10),reverse=True)
        keep=0.;name=None
        if others:
            first,name=others[0];second=others[1][0] if len(others)>1 else 0.
            coherent=first*(max(0.,1.-second/first)**2)
            # Retain the sampled distribution near the pure-body boundary;
            # use the Hips anchor more strongly inside the overlap.
            keep=first*(1.-blend[j])+coherent*blend[j]
        row={hips.name:float(total[j]*(1.-keep))}
        if name and keep>1e-10:row[name]=float(total[j]*keep)
        result.append(row)
    return result

def smooth_skirt_weights(rig, meshes, domains, iterations=20, limit=4, factor=.2):
    """Write only known skirt domains; preserve body mass and all other meshes."""
    from avatar_apparel_weights import weights,assign
    results=[]
    for obj in meshes:
        for family,indices in domains.get(obj.name,{}).items():
            original={i:weights(obj,i) for i in indices}
            rows={i:w for i,w in original.items() if sum(v for n,v in w.items() if n.startswith(family+'_'))>1e-6}
            if not rows:continue
            from avatar_weight_seams import coincident_groups
            all_ids=sorted(rows)
            matrix=rig.matrix_world.inverted()@obj.matrix_world
            clusters=coincident_groups([matrix@obj.data.vertices[i].co for i in all_ids])
            groups=[[all_ids[k] for k in group] for group in clusters]
            ids=[group[0] for group in groups]
            lookup={index:j for j,group in enumerate(groups) for index in group}
            from avatar_dress import full_chain_names
            bones=set()
            for spring in rig.data.vrm_addon_extension.spring_bone1.springs:
                if spring.vrm_name.rsplit('_',1)[0]!=family:continue
                names=full_chain_names(rig,spring)
                if names:bones.update(names[:-1])
            bones=sorted(bones | {n for w in rows.values() for n in w if n.startswith(family+'_')})
            w=np.array([[rows[i].get(n,0.) for n in bones] for i in ids])
            mass=w.sum(axis=1)
            body=[{n:v for n,v in rows[i].items() if n not in bones and v>0} for i in ids]
            edges=sorted({tuple(sorted(lookup[i] for i in e.vertices)) for e in obj.data.edges
                          if all(i in lookup for i in e.vertices) and lookup[e.vertices[0]]!=lookup[e.vertices[1]]})
            body_relimited=sum(len(row)>2 for row in body)
            body=smooth_body_attachment(rig,obj,ids,body,mass,edges,iterations,factor)
            budgets=np.array([limit-len(v) for v in body])
            normalized=w/mass[:,None]
            smoothed=smooth_local_coordinates(normalized,edges,bones,mass>=1-1e-6,iterations=iterations,factor=factor)
            # Only mixed waistband rows can exceed four including their body
            # attachment. Limit those locally; never erode the free skirt's
            # coherent four-weight patches with a propagated removal mask.
            overflow=(smoothed>1e-8).sum(axis=1)>budgets
            if overflow.any():
                limited=smooth_limited_weights(smoothed,edges,budgets,np.zeros(len(ids),dtype=bool),iterations=0)
                smoothed[overflow]=limited[overflow]
            result=smoothed*mass[:,None]
            chain_columns={n.rsplit('_',1)[0]:[] for n in bones}
            for col,n in enumerate(bones):chain_columns[n.rsplit('_',1)[0]].append(col)
            active_chains=np.stack([result[:,cols].sum(axis=1)>1e-8 for cols in chain_columns.values()],axis=1)
            max_chains=int(active_chains.sum(axis=1).max())
            if max_chains>2:raise RuntimeError('Smoothing leaked beyond the local skirt chain pair')
            free=mass>=1-1e-6
            share_error=max(float(np.max(np.abs(result[free][:,cols].sum(axis=1)-w[free][:,cols].sum(axis=1)))) if free.any() else 0. for cols in chain_columns.values())
            if share_error>1e-6:raise RuntimeError('Smoothing changed circumferential chain ownership')
            from avatar_skirt_support import support_attachment, blend_hip_handoff
            body,result=support_attachment(rig,body,mass,bones,result,edges)
            body,result=blend_hip_handoff(rig,obj,ids,body,bones,result,edges)
            from avatar_dress_boundary_weights import smooth_boundaries
            result,boundary_smoothing=smooth_boundaries(rig,bones,body,result,edges,iterations=iterations)
            changed=0
            for j,index in enumerate(ids):
                values=dict(body[j]);values.update({n:float(v) for n,v in zip(bones,result[j]) if v>0})
                if max(abs(rows[index].get(n,0)-values.get(n,0)) for n in rows[index].keys()|values.keys())>1e-6:changed+=1
                assign(obj,groups[j],values)
            def edge_energy(a):
                return float(np.mean([np.sum((a[x]-a[y])**2) for x,y in edges])) if edges else 0.
            results.append(dict(object=obj.name,family=family,vertices=len(all_ids),virtual_vertices=len(ids),
                seam_groups=sum(len(g)>1 for g in groups),changed_vertices=changed,
                max_influences=max(len(b)+int(np.count_nonzero(v)) for b,v in zip(body,result)),
                waistband_rows_relimited=body_relimited,max_chains=max_chains,chain_share_error=share_error,
                lower_chain_boundaries=boundary_smoothing,
                final_max_chains=int(np.stack([result[:,cols].sum(axis=1)>1e-8 for cols in chain_columns.values()],axis=1).sum(axis=1).max()),
                edge_energy_before=edge_energy(w),edge_energy_after=edge_energy(result)))
    return dict(iterations=iterations,factor=factor,limit=limit,method='local_chain_coordinates',domains=results)
