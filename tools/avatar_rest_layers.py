"""Offline rest-shell repair: coherent radial profiles and complete contact audit.

Profiles move complete welded garment pieces, not independent fitted vertices.
Material names are never interpreted: layer order comes from source geometry.
"""
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def tree(points,faces):
    return BVHTree.FromPolygons([Vector(p) for p in points],np.asarray(faces).tolist(),all_triangles=True)


def crossings(points,faces,focus):
    """Nonadjacent triangle pairs, excluding welded/UV boundary contact."""
    bvh=tree(points,faces);pairs=[]
    for a,b in bvh.overlap(bvh):
        if a>=b or not (focus[a] or focus[b]):continue
        if set(faces[a])&set(faces[b]):continue
        if any(np.linalg.norm(points[i]-points[j])<1e-6 for i in faces[a] for j in faces[b]):continue
        pairs.append((a,b))
    return pairs


def outer_radius(bvh,p,center):
    origin=Vector((center[0],center[1],p[2]));r=Vector((p[0]-center[0],p[1]-center[1],0.))
    if r.length<1e-8:return 0.
    direction=r.normalized();start=origin.copy();outer=0.
    for _ in range(48):
        hit,_,_,_=bvh.ray_cast(start,direction,2.)
        if hit is None:break
        outer=max(outer,(hit-origin).dot(direction));start=hit+direction*1e-5
    return outer


def repair_layers(source,fitted,faces,materials,seeds,center,clearance=.002,smoothing=.04,budget=.10):
    """Return a rest correction without changing topology, weights or physics.

    Source is the pre-optimization surface. Seed faces are identified by their
    spring weights. Grow over welded edges within the same material, so rigid
    waist vertices and UV islands participate while unrelated material reuse
    remains untouched. Heights and budget are in armature coordinates.
    """
    if not np.isfinite(budget) or budget<=0:raise ValueError('Rest-layer budget must be positive and finite')
    source=np.asarray(source);faces=np.asarray(faces);materials=np.asarray(materials)
    radial=source[:,:2]-np.asarray(center)[:2];radius=np.linalg.norm(radial,axis=1)
    direction=radial/np.maximum(radius[:,None],1e-8)
    # Weld only for identifying shell membership; preserve original topology.
    keys=[tuple(np.round(p,6)) for p in source];owners={}
    adjacency=[set() for _ in faces]
    for i,face in enumerate(faces):
        for v in face:
            key=(int(materials[i]),keys[v])
            for j in owners.get(key,()):adjacency[i].add(j);adjacency[j].add(i)
            owners.setdefault(key,[]).append(i)
    selected=set(np.flatnonzero(seeds).tolist());pending=list(selected)
    while pending:
        i=pending.pop()
        for j in adjacency[i]-selected:selected.add(j);pending.append(j)
    focus=np.array([i in selected for i in range(len(faces))]);ids=np.unique(faces[focus])
    base_tree=tree(source,faces[focus]);shell_materials=set(materials[focus])
    lo,hi=source[ids,2].min(),source[ids,2].max()
    inner=[];outer=[];layer_report=[]
    for material in sorted(set(materials)-shell_materials):
        vertices=np.unique(faces[materials==material]);signed=[]
        for i in vertices:
            if not lo<=source[i,2]<=hi:continue
            nearest=base_tree.find_nearest(Vector(source[i]),smoothing*3)
            if nearest[0] is None:continue
            r=outer_radius(base_tree,source[i],center)
            if r:signed.append(radius[i]-r)
        if not signed:continue
        offset=float(np.median(signed))
        (inner if offset<0 else outer).append(material)
        layer_report.append(dict(material=int(material),signed_radius=offset))
    inner_tree=tree(source,faces[np.isin(materials,inner)])
    # A waist cap is a fan whose pole lies deep inside its surrounding ring.
    # Such an internal closure cannot clear a body passing through the waist.
    # Preserve vertices/shape-key indices, and remove only its incident faces.
    cap_faces=set()
    for pole in ids:
        incident=np.flatnonzero(focus & np.any(faces==pole,axis=1))
        if len(incident)<8:continue
        neighbors=np.unique(faces[incident]);neighbors=neighbors[neighbors!=pole]
        if radius[pole]>=.35*np.median(radius[neighbors]):continue
        if source[pole,2]<np.median(source[neighbors,2]):continue
        probes=np.concatenate([source[neighbors]*t+source[pole]*(1-t) for t in (.25,.5,.75,.9)])
        if not any(outer_radius(inner_tree,q,center)>np.linalg.norm(q[:2]-np.asarray(center)[:2]) for q in probes):continue
        normals=np.cross(source[faces[incident,1]]-source[faces[incident,0]],source[faces[incident,2]]-source[faces[incident,0]])
        vertical=np.abs(normals[:,2])/np.maximum(np.linalg.norm(normals,axis=1),1e-12)
        if np.median(vertical)<.4:continue
        cap_faces.update(incident.tolist())
    live_faces=np.array([i not in cap_faces for i in range(len(faces))])
    focus &= live_faces
    ids=np.unique(faces[focus])
    anchor_z=source[ids,2].tolist()
    anchor_d=np.linalg.norm(fitted[ids]-source[ids],axis=1).tolist()
    def profile(z,zs,ds):
        return np.max(np.asarray(ds)[None,:]*np.exp(-.5*((np.asarray(z)[:,None]-np.asarray(zs))/smoothing)**2),axis=1)
    def pose(zs,ds,extra=0.):
        result=source.copy();amount=profile(source[:,2],zs,ds)
        result[ids,:2]+=direction[ids]*amount[ids,None]
        return result,amount
    # Actual unposed rest geometry is a hard requirement, including face interiors.
    samples=[(i,) for i in ids]+[tuple(t) for t in faces[focus]]
    samples+=list({tuple(sorted((int(t[i]),int(t[(i+1)%3])))) for t in faces[focus] for i in range(3)})
    for iteration in range(12):
        result,amount=pose(anchor_z,anchor_d);updates=[]
        for members in samples:
            p=result[list(members)].mean(axis=0);r=np.linalg.norm(p[:2]-np.asarray(center)[:2])
            surface=outer_radius(inner_tree,p,center)
            deficit=surface+clearance-r if surface else 0.
            if deficit<1e-5:continue
            n=(p[:2]-np.asarray(center)[:2])/max(r,1e-8)
            projection=min(direction[i]@n for i in members)
            if projection<.15:continue
            for i in members:updates.append((source[i,2],amount[i]+deficit/projection))
        if not updates:break
        anchor_z.extend(z for z,d in updates);anchor_d.extend(d for z,d in updates)
        if max(anchor_d)>budget:raise ValueError('Rest clearance exceeds the allowed shape budget')
        # Heights repeat at welded boundaries; max bounds suffice.
        compact={}
        for z,d in zip(anchor_z,anchor_d):compact[z]=max(compact.get(z,0.),d)
        anchor_z=list(compact);anchor_d=list(compact.values())
    result,amount=pose(anchor_z,anchor_d)
    skirt_tree=tree(result,faces[focus])
    outer_faces=np.isin(materials,outer);outer_ids=np.unique(faces[outer_faces]);outer_amount=amount.copy()
    # Smoothly carry nearby outer clothing/accessories with the skirt envelope.
    # Remote reused-material pieces receive no displacement.
    active=[]
    for i in outer_ids:
        nearest=base_tree.find_nearest(Vector(source[i]),smoothing*3)
        if nearest[0] is not None:active.append(i)
    outer_ids=np.array(active,dtype=int)
    extra_z=[lo,hi];extra_d=[0.,0.]
    for iteration in range(12):
        extra=profile(source[:,2],extra_z,extra_d)
        result[outer_ids,:2]=source[outer_ids,:2]+direction[outer_ids]*(amount[outer_ids]+extra[outer_ids])[:,None]
        updates=[]
        # Include midpoints/centers; vertex-only clearance misses crossing faces.
        active_set=set(outer_ids.tolist())
        outer_samples=[(i,) for i in outer_ids]+[tuple(t) for t in faces[outer_faces] if all(int(i) in active_set for i in t)]
        for members in outer_samples:
            p=result[list(members)].mean(axis=0);r=np.linalg.norm(p[:2]-np.asarray(center)[:2])
            surface=outer_radius(skirt_tree,p,center)
            deficit=surface+clearance-r if surface else 0.
            if deficit<1e-5:continue
            n=(p[:2]-np.asarray(center)[:2])/max(r,1e-8)
            projection=min(direction[i]@n for i in members)
            if projection<.15:continue
            for i in members:updates.append((source[i,2],extra[i]+deficit/projection))
        if not updates:break
        extra_z.extend(z for z,d in updates);extra_d.extend(d for z,d in updates)
        if max(extra_d)+max(anchor_d)>budget:raise ValueError('Outer layer clearance exceeds the allowed shape budget')
    remaining=crossings(result,faces[live_faces],focus[live_faces])
    return result,dict(rest_pairs=len(remaining),pairs=remaining,layers=layer_report,
        clearance=clearance,smoothing=smoothing,budget=budget,
        removed_cap_triangles=sorted(cap_faces),skirt_faces=int(focus.sum()),skirt_vertices=len(ids),outer_vertices=len(outer_ids),
        maximum_displacement=float(np.linalg.norm(result-source,axis=1).max())),focus


def remove_faces(mesh, indices):
    raise RuntimeError('Disabled: final mesh geometry and existing shape keys must remain unchanged')
    """Remove cap polygons without deleting/reindexing shape-key vertices."""
    if not indices:return
    import bmesh
    count=len(mesh.vertices)
    bm=bmesh.new()
    try:
        bm.from_mesh(mesh);bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm,geom=[bm.faces[i] for i in indices],context='FACES_ONLY')
        bm.to_mesh(mesh)
    finally:bm.free()
    if len(mesh.vertices)!=count:raise RuntimeError('Cap removal changed vertex indices')
    mesh.update()


def repair_rig_layers(rig, source, budget=.10):
    raise RuntimeError('Disabled: final mesh geometry and existing shape keys must remain unchanged')
    """Apply in an isolated worker, then validate through the usual fit bundle."""
    import bpy
    from avatar_apparel_weights import weights
    if not bpy.app.background:raise RuntimeError('Rest-layer fitting requires an isolated background scene')
    original=[];current=[];faces=[];materials=[];seeds=[];records=[];polygons=[]
    material_ids={}
    for name,before in source['meshes'].items():
        obj=bpy.data.objects[name];matrix=rig.matrix_world.inverted()@obj.matrix_world
        base=len(original);mesh=obj.data;mesh.calc_loop_triangles()
        original.extend(matrix@Vector(p) for p in before['positions'])
        current.extend(matrix@v.co for v in mesh.vertices)
        records.extend((name,i) for i in range(len(mesh.vertices)))
        selected={v.index for v in mesh.vertices if any(n.startswith('Secondary_Skirt_') and w>.05 for n,w in weights(obj,v.index).items())}
        for t in mesh.loop_triangles:
            material=mesh.materials[t.material_index] if t.material_index<len(mesh.materials) else None
            key=material.as_pointer() if material else (name,t.material_index)
            material_ids.setdefault(key,len(material_ids));materials.append(material_ids[key])
            faces.append(tuple(base+i for i in t.vertices));seeds.append(all(i in selected for i in t.vertices))
            polygons.append((name,t.polygon_index))
    hips=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones.hips.node.bone_name
    result,report,focus=repair_layers(np.array(original),np.array(current),np.array(faces),np.array(materials),
                                     np.array(seeds),rig.data.bones[hips].head_local,budget=budget)
    if report['rest_pairs']:raise ValueError(f"Rest fit still has {report['rest_pairs']} skirt intersections")
    removed={}
    for i in report['removed_cap_triangles']:
        name,polygon=polygons[i];removed.setdefault(name,set()).add(polygon)
    for (name,i),old,new in zip(records,current,result):
        obj=bpy.data.objects[name];delta=(obj.matrix_world.inverted()@rig.matrix_world).to_3x3()@Vector(new-np.array(old))
        if delta.length<1e-8:continue
        original_co=obj.data.vertices[i].co.copy()
        if obj.data.shape_keys:
            values=[(k,k.data[i].co.copy()) for k in obj.data.shape_keys.key_blocks]
            for key,p in values:key.data[i].co=p+delta
        obj.data.vertices[i].co=original_co+delta
    for name in source['meshes']:
        obj=bpy.data.objects[name];remove_faces(obj.data,sorted(removed.get(name,())))
        obj.data.update()
    report['removed_polygons']={n:sorted(v) for n,v in removed.items()}
    return report
