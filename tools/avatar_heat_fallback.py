"""Retry complete combined heat failures on independent material surfaces.

Only geometry and the newly generated skeleton participate; source weights are
never read. Shared vertices average their incident domain solutions, so material
boundaries cannot depend on iteration order. Normal combined heat is unchanged.
"""
import bpy
from avatar_apparel_weights import weights, assign


def seed_material_heat(proxy, rig, domains, missing_indices=None):
    from autorig_bust import repaired_heat
    accumulated = {}
    reports = []
    for label, faces in domains:
        ids = sorted({i for face in faces for i in face})
        if not ids or (missing_indices is not None and not missing_indices.intersection(ids)):
            continue
        lookup = {i: j for j, i in enumerate(ids)}
        data = bpy.data.meshes.new('Temporary isolated heat surface')
        data.from_pydata([proxy.data.vertices[i].co for i in ids], [],
                         [[lookup[i] for i in face] for face in faces])
        data.update()
        obj = bpy.data.objects.new('Temporary isolated heat surface', data)
        bpy.context.scene.collection.objects.link(obj)
        try:
            report = repaired_heat([obj], rig, allow_unweighted=True)
            seeded = 0
            for local, global_id in enumerate(ids):
                if missing_indices is not None and global_id not in missing_indices:continue
                row = weights(obj, local)
                if not row:
                    continue
                seeded += 1
                target = accumulated.setdefault(global_id, {})
                for name, value in row.items():
                    target[name] = target.get(name, 0.0) + value
            reports.append(dict(domain=label, vertices=len(ids), seeded=seeded,
                                heat=report))
        finally:
            bpy.data.objects.remove(obj, do_unlink=True)
            if not data.users:
                bpy.data.meshes.remove(data)
    for index, row in accumulated.items():
        total = sum(row.values())
        assign(proxy, [index], {name: value / total for name, value in row.items()})
    return dict(method='material-isolated fresh heat for unseeded vertices',
                seeded_vertices=len(accumulated), domains=reports,
                source_weights_used=False)


def distal_weights(point, rig):
    """Bounded footwear fallback when heat has no nearby seeded surface.

    Only generated foot/toe segments qualify. This cannot assign a remote
    torso, hand, opposite foot or arbitrary floating prop to the skeleton.
    """
    candidates=[]
    for side in ('L','R'):
        foot=rig.data.bones.get('Foot.'+side);toe=rig.data.bones.get('Toe.'+side)
        if foot is None or toe is None:continue
        scale=max(foot.length,toe.length)
        if point.z>max(foot.head_local.z,foot.tail_local.z)+scale*.15:continue
        distances=[]
        for b in (foot,toe):
            delta=b.tail_local-b.head_local
            t=max(0.,min(1.,(point-b.head_local).dot(delta)/max(delta.length_squared,1e-20)))
            distances.append((point-b.head_local-delta*t).length)
        # Include the heel directly beneath the ankle, not just a narrow
        # tube around the sloping ankle-to-toe segment (high heels need this).
        forward=toe.tail_local-foot.head_local;forward.z=0
        if forward.length_squared<1e-12:continue
        forward.normalize();offset=point-foot.head_local
        along=offset.dot(forward);lateral=abs(offset.x*forward.y-offset.y*forward.x)
        if lateral>scale*.55 or along < -scale*.45 or along>forward.dot(toe.tail_local-foot.head_local)+scale*.25:continue
        if point.z<min(foot.tail_local.z,toe.tail_local.z)-scale*.2 or min(distances)>scale:continue
        candidates.append((min(distances),foot,toe,distances,scale))
    if not candidates:return None
    _,foot,toe,distances,scale=min(candidates,key=lambda c:c[0])
    forward=toe.tail_local-foot.head_local;forward.z=0
    if (point-foot.head_local).dot(forward)<=0:return {foot.name:1.}
    row={b.name:1./max(d,scale*.15)**2 for b,d in zip((foot,toe),distances)}
    total=sum(row.values())
    return {n:w/total for n,w in row.items()}

def complete_domain_weights(mesh, rows, fresh, allowed):
    """Fill singular heat holes from this generated rig's domain, over edges.

    Heat rows stay pinned. Restricted fresh body weights seed disconnected
    islands; remaining holes interpolate only along their own mesh topology.
    No original rig input, cross-limb spatial transfer or vertex motion.
    """
    rows=[dict(row) for row in rows]
    adjacency=[set() for _ in mesh.vertices]
    for edge in mesh.edges:
        a,b=edge.vertices;adjacency[a].add(b);adjacency[b].add(a)
    missing={i for i,row in enumerate(rows) if not row};initial=len(missing)
    for i in list(missing):
        value={n:w for n,w in fresh[i].items() if n in allowed and w>1e-8}
        total=sum(value.values())
        if total>1e-8:rows[i]={n:w/total for n,w in value.items()};missing.remove(i)
    diffuse=set(missing)
    while missing:
        updates={}
        for i in missing:
            value={}
            for j in adjacency[i]:
                if not rows[j]:continue
                scale=1/max((mesh.vertices[i].co-mesh.vertices[j].co).length,1e-8)
                for n,w in rows[j].items():value[n]=value.get(n,0)+w*scale
            total=sum(value.values())
            if total:updates[i]={n:w/total for n,w in value.items()}
        if not updates:raise ValueError('Disconnected anatomical surface has no fresh weight seeds')
        for i,value in updates.items():rows[i]=value;missing.remove(i)
    # Smooth the reconstructed hole with pinned solved boundary values.
    for _ in range(16):
        updates={}
        for i in diffuse:
            value={}
            for j in adjacency[i]:
                scale=1/max((mesh.vertices[i].co-mesh.vertices[j].co).length,1e-8)
                for n,w in rows[j].items():value[n]=value.get(n,0)+w*scale
            total=sum(value.values())
            if total:updates[i]={n:w/total for n,w in value.items()}
        for i,value in updates.items():rows[i]=value
    return rows,dict(recovered=initial,topology_interpolated=len(diffuse),source_weights_used=False)
