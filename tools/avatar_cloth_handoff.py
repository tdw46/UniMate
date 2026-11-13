"""Rest-space cloth density and final body attachment, shared by both hosts."""
import math


def cloth_segments(length, height, existing, *, short_jacket=False):
    """Preserve short skirts; space long cloth near 4.5% of body height."""
    if height <= 0 or length <= 0:
        raise ValueError('Invalid cloth dimensions')
    ratio = length / height
    if short_jacket and ratio < .14:
        return 2
    if ratio > .20:
        return max(existing, min(16, math.ceil(ratio / .045)))
    return existing


def free_mass(depth):
    """Normalized attachment depth: 0/100, 10/90 at 20%, then 100/0."""
    t = max(0., min(1., depth))
    def smooth(v):
        return v*v*(3.-2.*v)
    return .1*smooth(t/.2) if t <= .2 else .1+.9*smooth((t-.2)/.8)


def blend_four(body, cloth, amount):
    """Reserve two influences per family, preserving the requested mass ratio."""
    def limited(row, count):
        pairs = sorted(((n,w) for n,w in row.items() if w > 1e-10), key=lambda p:(-p[1],p[0]))[:count]
        total = sum(w for _,w in pairs)
        return {n:w/total for n,w in pairs} if total else {}
    if amount <= 0: return limited(body,4)
    if amount >= 1: return limited(cloth,4)
    a,b = limited(body,2),limited(cloth,2)
    if not a: return limited(cloth,4)
    if not b: return limited(body,4)
    row = {n:w*(1.-amount) for n,w in a.items()}
    for n,w in b.items(): row[n] = row.get(n,0.) + amount*w
    return row


def smooth_cross_waist(rig, meshes, plan):
    """Diffuse compact transverse cloth strips, including disconnected seams.

    Work only inside admitted skirt/material domains, never by proximity to
    fingers, skin, hair or equipment. Long hanging panels retain strip binding.
    """
    import numpy as np
    from avatar_virtual_topology import connected_graph
    from avatar_weight_seams import coincident_groups
    from avatar_weight_smoothing import smooth_limited_weights
    from avatar_apparel_weights import weights, assign
    reports=[];height=plan['height'];center=rig.data.bones['Hips'].head_local.x
    for obj in meshes:
        ids={i for r in plan['regions'] if r['obj']==obj and r['role']=='skirt' for i in r['ids']}
        if not ids:continue
        points=plan['points'][obj.name]
        graph,groups,_=connected_graph(obj.data,points,ids,height*.0001)
        edges=[tuple(e.vertices) for e in graph.edges]
        for group in groups:
            p=points[group];span=np.ptp(p,axis=0)
            if not (8<=len(group)<=512 and p[:,0].min()<center<p[:,0].max()
                    and span[0]>height*.08 and span[2]<height*.12
                    and max(span[:2])>span[2]*1.5):continue
            rows={i:weights(obj,i) for i in group}
            # Keep the sampled body field and its exact mass. Diffuse only
            # the cloth distribution, with the remaining influence budget.
            seams=coincident_groups(points[group])
            clusters=[[group[j] for j in seam] for seam in seams]
            active=[g for g in clusters if any('Secondary_' in n for n in rows[g[0]])]
            if not active:continue
            node={i:j for j,g in enumerate(active) for i in g}
            names=sorted({n for g in active for n in rows[g[0]] if 'Secondary_' in n})
            field=np.array([[rows[g[0]].get(n,0.) for n in names] for g in active])
            mass=field.sum(axis=1)
            body=[{n:w for n,w in rows[g[0]].items() if 'Secondary_' not in n} for g in active]
            budgets=np.array([4-len(row) for row in body])
            e=np.asarray(sorted({tuple(sorted((node[a],node[b]))) for a,b in edges
                                 if a in node and b in node and node[a]!=node[b]}),dtype=int).reshape((-1,2))
            def energy(w):return float(np.mean(np.sum((w[e[:,0]]-w[e[:,1]])**2,axis=1))) if len(e) else 0.
            filtered=smooth_limited_weights(field/mass[:,None],e,budgets,np.ones(len(active),bool),
                                            iterations=24,factor=.5,mask_iterations=8)*mass[:,None]
            result={}
            for j,g in enumerate(active):
                row=dict(body[j]);row.update({n:float(w) for n,w in zip(names,filtered[j]) if w>1e-8})
                for i in g:result[i]=row
            report=dict(vertices=len(result),max_influences=max(map(len,result.values())),
                        edge_energy_before=energy(field),edge_energy_after=energy(filtered),body_mass_preserved=True)
            for i,row in result.items():assign(obj,[i],row)
            reports.append(dict(object=obj.name,ids=group,**report))
    return reports


def finish_waistbands(rig, meshes, plan):
    """Finalize the top band after support/hip passes, before mounted fittings.

    Sample the newly generated body only. No imported rig or skinning enters
    this pass. The lower garment field remains untouched outside the blend.
    Pants without a secondary chain retain their generated leg field below it.
    """
    from avatar_waist_attachment import WaistAttachment
    from avatar_apparel_weights import weights, assign
    from avatar_material_roles import semantic_label
    from avatar_weight_seams import coincident_groups
    from mathutils import Vector
    attachment = WaistAttachment(rig, meshes)
    height = plan['height']
    # The default sampler's bone-length bound can be too short for a loose
    # waistband. Its donor BVH already excludes sleeves, hands and clothing.
    attachment.distance = max(attachment.distance, height*.08)
    changed = 0; bands = []; misses = 0
    for obj in meshes:
        points = plan['points'][obj.name]
        domains = []
        for region in plan['regions']:
            if region['obj'] != obj or region['role'] != 'skirt': continue
            domains.append((set(region['ids']),float(region['hi']),float(region['lo']),True))
        owned = set().union(*(r[0] for r in domains)) if domains else set()
        # A shared material slot is a domain, never permission to touch skin,
        # hands or unrelated equipment. Only the top of anatomical bottomwear.
        for slot,material in enumerate(obj.data.materials):
            label = semantic_label(material.name) if material else ''
            if not any(t in label for t in ('pants','trouser','shorts','ズボン','裤','바지')): continue
            ids = {i for f in obj.data.polygons if f.material_index==slot for i in f.vertices}-owned
            ids = {i for i in ids if points[i,2] < rig.data.bones['Hips'].tail_local.z+height*.04}
            if len(ids)<8: continue
            top=max(float(points[i,2]) for i in ids);bottom=min(float(points[i,2]) for i in ids)
            if top-bottom < height*.025: continue
            domains.append((ids,top,bottom,False));owned.update(ids)
        for ids,top,bottom,secondary in domains:
            span=min(height*.05,(top-bottom)*.20)
            active=sorted(i for i in ids if points[i,2]>=top-span)
            # Coincident seam duplicates get exactly one sample and assignment.
            groups=coincident_groups([Vector(points[i]) for i in active])
            count=0
            for group in groups:
                members=[active[j] for j in group];i=members[0];p=Vector(points[i])
                amount=free_mass((top-p.z)/max(span,1e-8))
                old=weights(obj,i)
                cloth={n:w for n,w in old.items() if n not in attachment.allowed} if secondary else old
                if amount>0 and not cloth:
                    # The old fixed band can have no spring weights. Borrow
                    # its own nearest generated chain root in rest space.
                    candidates=[c for c in plan['chains'] if c['role']=='skirt'
                                and abs(c.get('garment_top',top)-top)<height*.005]
                    near=sorted(candidates,key=lambda c:(c['line'][0]-p).length_squared)[:2]
                    if near:
                        ds=[max((c['line'][0]-p).length, height*1e-5) for c in near]
                        cloth={c['names'][0]:1./d for c,d in zip(near,ds)}
                prior=attachment.fallback
                body=attachment.sample(p,side_filter=False)
                if attachment.fallback!=prior:
                    misses+=len(members)
                    # A covered body provides no surface proof. Preserve its
                    # generated anatomical field rather than inventing a fit.
                    body={n:w for n,w in old.items() if n in attachment.allowed} or body
                row=blend_four(body,cloth,amount)
                if row:
                    assign(obj,members,row);count+=len(members)
            changed+=count
            bands.append(dict(object=obj.name,vertices=count,top=top,depth=span,secondary=secondary))
    return dict(vertices=changed,bands=bands,missing_surface_vertices=misses,max_influences=4,
                profile='body100_at_top_body90_at_one_fifth_then_smooth_cloth100')
