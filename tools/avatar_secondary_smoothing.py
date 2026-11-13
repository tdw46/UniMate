"""Rest-space secondary skin smoothing with welded seams and four influences.

Only admitted generated hair/accessory domains are writable. No original rig,
material label or posed-neighbor search participates. Duplicate seam rows share
one node through diffusion and limiting, so pruning cannot reopen the seam.
"""
import numpy as np
from avatar_weight_seams import coincident_groups
from avatar_weight_smoothing import smooth_limited_weights


def smooth_domain(points, edges, rows, *, iterations=24, factor=.5,
                  tolerance=1e-4, isolated_families=False):
    if not rows:return {},dict(vertices=0)
    ids=sorted(rows);p=np.asarray([points[i] for i in ids],dtype=float)
    extent=max(float(np.linalg.norm(np.ptp(p,axis=0))),1e-9)
    groups=[[ids[j] for j in group] for group in coincident_groups(p,tolerance/extent)]
    node={i:j for j,group in enumerate(groups) for i in group}
    names=sorted({n for row in rows.values() for n,w in row.items() if w>1e-8})
    w=np.array([[sum(rows[i].get(n,0.) for i in group)/len(group) for n in names] for group in groups])
    e=sorted({tuple(sorted((node[a],node[b]))) for a,b in edges
              if a in node and b in node and node[a]!=node[b]})
    e=np.asarray(e,dtype=int).reshape((-1,2))
    # Pure humanoid attachment rows remain pinned. Seam averaging precedes
    # pinning, including where one duplicate was accidentally left on Head.
    secondary=np.array(['Secondary_' in n for n in names])
    movable=w[:,secondary].sum(axis=1)>1e-8
    allowed=None
    if isolated_families:
        allowed=np.ones(w.shape,dtype=bool)
        for j,n in enumerate(names):
            if not secondary[j]:continue
            family=n.rsplit('_',1)[0]+'_'
            columns=[k for k,name in enumerate(names) if name.startswith(family)]
            allowed[:,j]=w[:,columns].sum(axis=1)>1e-8
    def energy(a):return float(np.mean(np.sum((a[e[:,0]]-a[e[:,1]])**2,axis=1))) if len(e) else 0.
    before=energy(w)
    filtered=smooth_limited_weights(w,e,np.full(len(w),4),movable,
                                   iterations=iterations,factor=factor,
                                   mask_iterations=8,allowed=allowed)
    result={i:{n:float(v) for n,v in zip(names,filtered[j]) if v>0}
            for j,group in enumerate(groups) for i in group}
    return result,dict(vertices=len(ids),virtual_nodes=len(groups),seam_duplicates=len(ids)-len(groups),
                       max_influences=max(len(row) for row in result.values()),
                       edge_energy_before=before,edge_energy_after=energy(filtered))


def smooth_secondary_weights(rig, meshes, regions=None):
    from avatar_apparel_weights import weights,assign
    reports=[]
    for obj in meshes:
        matrix=rig.matrix_world.inverted_safe()@obj.matrix_world
        points=np.asarray([matrix@v.co for v in obj.data.vertices])
        if not len(points):continue
        height=float(np.ptp(points[:,2]));edges=[tuple(e.vertices) for e in obj.data.edges]
        for role,token in (('hair','Secondary_Hair_'),('accessory','Secondary_Accessory_')):
            if regions is not None:
                ids={i for r in regions if r['obj']==obj and r['role']==role for i in r['ids']}
            else:
                ids={v.index for v in obj.data.vertices if any(token in obj.vertex_groups[g.group].name and g.weight>1e-8 for g in v.groups)}
            if not ids:continue
            rows={i:weights(obj,i) for i in ids}
            result,report=smooth_domain(points,edges,rows,iterations=8 if role=='hair' else 32,
                                        factor=.25 if role=='hair' else .6,tolerance=height*.0001,
                                        isolated_families=role=='hair')
            for i,row in result.items():assign(obj,[i],row)
            reports.append(dict(object=obj.name,role=role,**report))
    return reports
