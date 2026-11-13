"""Final skirt-only filtering. Anatomical weights are immutable constraints."""
import numpy as np


def is_cloth(name):
    return 'Secondary_Skirt_' in name


def preserve_attachment(original, distribution):
    """Spend only the remaining slots and original skirt mass; never prune body."""
    body = {n: w for n, w in original.items() if not is_cloth(n)}
    mass = sum(w for n, w in original.items() if is_cloth(n))
    budget = 4 - len(body)
    if mass <= 1e-8 or budget <= 0:
        return dict(original)
    pairs = sorted(((n, w) for n, w in distribution.items() if is_cloth(n) and w > 1e-8),
                   key=lambda v: (-v[1], v[0]))[:budget]
    total = sum(w for _, w in pairs)
    if total <= 1e-12:
        return dict(original)
    return dict(body, **{n: mass*w/total for n, w in pairs})


def chain_profile(names, stations, coordinate, budget=4):
    """Broad smooth joint kernels, using no more than the available slots.

    Cubic B-splines replace the old piecewise linear two-joint hinges. At
    anatomical attachment rows, quadratic or two-joint smoothstep kernels
    respect the smaller budget without discarding a body influence.
    """
    if not names or budget <= 0:
        return {}
    k = int(np.clip(np.searchsorted(stations, coordinate, side='right')-1, 0, len(names)-2)) if len(names)>1 else 0
    t = float(np.clip((coordinate-stations[k])/(stations[k+1]-stations[k]), 0, 1)) if len(names)>1 else 0.
    if budget >= 4:
        pairs = [(k-1, (1-t)**3/6), (k, (3*t**3-6*t*t+4)/6),
                 (k+1, (-3*t**3+3*t*t+3*t+1)/6), (k+2, t**3/6)]
    elif budget == 3:
        q=k+t; center=int(np.floor(q+.5)); d=q-center
        pairs=[(center-1, .5*(.5-d)**2), (center, .75-d*d), (center+1, .5*(.5+d)**2)]
    elif budget == 2:
        t=t*t*(3-2*t); pairs=[(k,1-t),(k+1,t)]
    else:
        pairs=[(k+int(t>=.5),1.)]
    row={}
    for index,w in pairs:
        name=names[min(len(names)-1,max(0,index))]
        row[name]=row.get(name,0.)+w
    return {n:w for n,w in row.items() if w>1e-8}


def smooth_skirt_distribution(rig, meshes, plan, *, protected=None):
    """Strong seam-aware filtering inside admitted skirt topology only."""
    from avatar_apparel_weights import weights, assign
    from avatar_virtual_topology import connected_graph
    from avatar_weight_seams import coincident_groups
    from avatar_weight_smoothing import smooth_limited_weights
    reports=[]
    for obj in meshes:
        ids={i for r in plan['regions'] if r['obj']==obj and r['role']=='skirt' for i in r['ids']}
        if not ids: continue
        pinned={i for r in (protected or {}).get('components',[]) if r['object']==obj.name
                for i in r['ids']+r['support_vertices']}
        points=plan['points'][obj.name]
        graph,components,_=connected_graph(obj.data,points,ids,plan['height']*.0001)
        all_edges=[tuple(e.vertices) for e in graph.edges]
        for component in components:
            original={i:weights(obj,i) for i in component}
            active=[i for i in component if any(is_cloth(n) for n in original[i]) and
                    sum(not is_cloth(n) for n in original[i])<4]
            if not active: continue
            groups=[[active[j] for j in g] for g in coincident_groups(points[active])]
            node={i:j for j,g in enumerate(groups) for i in g}
            names=sorted({n for i in active for n in original[i] if is_cloth(n)})
            field=np.asarray([[sum(original[i].get(n,0.) for i in g)/len(g) for n in names] for g in groups])
            field/=field.sum(axis=1,keepdims=True)
            budgets=np.asarray([min(4-sum(not is_cloth(n) for n in original[i]) for i in g) for g in groups])
            edges=np.asarray(sorted({tuple(sorted((node[a],node[b]))) for a,b in all_edges
                                     if a in node and b in node and node[a]!=node[b]}),dtype=int).reshape((-1,2))
            movable=np.asarray([not any(i in pinned for i in g) for g in groups])
            filtered=smooth_limited_weights(field,edges,budgets,movable,
                                            iterations=64,factor=.65,mask_iterations=12)
            def energy(w):return float(np.mean(np.sum((w[edges[:,0]]-w[edges[:,1]])**2,axis=1))) if len(edges) else 0.
            maximum=0
            for j,g in enumerate(groups):
                for i in g:
                    if i in pinned:continue
                    row=preserve_attachment(original[i],dict(zip(names,filtered[j])))
                    assert all(row.get(n)==w for n,w in original[i].items() if not is_cloth(n))
                    assert len(row)<=4
                    maximum=max(maximum,len(row));assign(obj,[i],row)
            reports.append(dict(object=obj.name,vertices=len(active),iterations=64,factor=.65,
                                edge_energy_before=energy(field),edge_energy_after=energy(filtered),
                                max_influences=maximum,body_weights_preserved=True))
    return reports
