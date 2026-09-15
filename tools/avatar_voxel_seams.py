"""Voxel skin proxy for continuous weights across mesh and UV boundaries.

Original geometry/UVs stay intact. Inputs are baked in rig-local coordinates
and carry binding_surface=skin on objects or materials. Hair and garments are
excluded; no asset names or source weights participate.
"""
from collections import Counter, deque
from statistics import median

import bpy
import bmesh
from mathutils import Vector
from mathutils.kdtree import KDTree

from avatar_apparel_weights import BodySurface, assign, constrain, weights


def skin_faces(obj):
    explicit = obj.get('binding_surface')
    if explicit:
        return list(obj.data.polygons) if explicit == 'skin' else []
    # Boolean caps can inherit an unused skin material slot from a mixed mesh.
    if obj.get('binding_role') not in ('body','head'):
        return []
    slots={i for i,m in enumerate(obj.data.materials) if m and m.get('binding_surface')=='skin'}
    return [p for p in obj.data.polygons if p.material_index in slots]


def skin_indices(obj):
    return {i for p in skin_faces(obj) for i in p.vertices}


def normalized(values):
    values = dict(sorted(((n,w) for n,w in values.items() if w>1e-8),
                         key=lambda item: (-item[1],item[0]))[:4])
    total = sum(values.values())
    assert total>1e-8
    return {n:w/total for n,w in values.items()}


def seam_clusters(parts, tolerance):
    boundary=[]
    for obj,faces in parts:
        counts=Counter(tuple(sorted(edge)) for p in faces for edge in p.edge_keys)
        indices={i for edge,count in counts.items() if count==1 for i in edge}
        boundary.extend((obj,i,obj.data.vertices[i].co.copy()) for i in sorted(indices))
    tree=KDTree(len(boundary))
    for j,(_,_,point) in enumerate(boundary):tree.insert(point,j)
    tree.balance()
    parents=list(range(len(boundary)))
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]]
            i=parents[i]
        return i
    for j,(obj,index,point) in enumerate(boundary):
        for _,k,_ in tree.find_range(point,tolerance):
            if k>j:
                parents[root(k)]=root(j)
    groups={}
    for j,value in enumerate(boundary):groups.setdefault(root(j),[]).append(value)
    return [g for g in groups.values() if len(g)>1]


def seam_patch(parts, clusters, seeds, max_rings):
    """Edge-loop distance, crossing coincident UV splits at zero cost."""
    neighbors={}
    for obj,faces in parts:
        for p in faces:
            for a,b in p.edge_keys:
                neighbors.setdefault((obj,a),set()).add((obj,b))
                neighbors.setdefault((obj,b),set()).add((obj,a))
    duplicates={}
    for cluster in clusters:
        members=[(o,i) for o,i,_ in cluster]
        for key in members:duplicates[key]=members
    distance={key:0 for g in seeds for o,i,_ in g for key in [(o,i)]}
    queue=deque(distance)
    while queue:
        key=queue.popleft(); d=distance[key]
        for other in duplicates.get(key,()):
            if distance.get(other,max_rings+1)>d:
                distance[other]=d;queue.appendleft(other)
        if d==max_rings:continue
        for other in neighbors.get(key,()):
            if distance.get(other,max_rings+1)>d+1:
                distance[other]=d+1;queue.append(other)
    return distance


def correct_skin_seams(meshes,rig,proxy_path=None,max_rings=2):
    if not isinstance(max_rings,int) or not 0<=max_rings<=2:
        raise ValueError('Seam expansion must be zero, one, or two edge loops')
    neck=rig.data.bones['Neck']
    voxel_size=neck.length/16
    parts=[(obj,skin_faces(obj)) for obj in meshes]
    parts=[(obj,faces) for obj,faces in parts if faces]
    explicit_boundary={o.get('binding_role') for o,_ in parts}>={'head','body'}
    tolerance=voxel_size*.1 if explicit_boundary else neck.length*1e-5
    clusters=seam_clusters(parts,tolerance)
    neck_top=neck.tail_local.z
    radius=abs(rig.data.bones['UpperArm.L'].head_local.x-neck.head_local.x)*.85
    def at_neck(g):
        center=sum((p for _,_,p in g),Vector())/len(g)
        return (neck.head_local.z<=center.z<=neck_top+neck.length*.25
                and abs(center.x-neck.head_local.x)<radius)
    head_clusters=[g for g in clusters if {o.get('binding_role') for o,_,_ in g}>={'head','body'}]
    # A UV split is not evidence of a head/body attachment. Only an explicit
    # head/body boundary gets rigid Head weights. Already matching UV weights
    # need no correction, especially after heat on a virtually welded mesh.
    def discontinuous(g):
        values=[weights(o,i) for o,i,_ in g]
        names=set().union(*(v.keys() for v in values))
        return any(max(v.get(n,0.) for v in values)-min(v.get(n,0.) for v in values)>1e-5 for n in names)
    seeds=head_clusters or [g for g in clusters if at_neck(g) and discontinuous(g)]
    patch=seam_patch(parts,clusters,seeds,max_rings)
    head_patch=seam_patch(parts,clusters,head_clusters,max_rings)
    audit={'method':'neck seam patch voxel proxy, edge-loop-limited transfer',
           'voxel_size':voxel_size,'seam_tolerance':tolerance,
           'seam_clusters':len(seeds),'max_expansion_loops':max_rings,'applied':False}
    if not seeds:
        audit.update(reason='No head/body boundary or discontinuous neck UV weights',
                     max_changed_loop=0,outside_patch_weight_changes=0,vertices_reweighted=0)
        return audit
    audit['head_boundary_clusters']=len(head_clusters)
    original={(o,v.index):weights(o,v.index) for o in meshes for v in o.data.vertices}
    patch_parts=[(o,[p for p in faces if all((o,i) in patch for i in p.vertices)]) for o,faces in parts]
    lengths=[(o.data.vertices[a].co-o.data.vertices[b].co).length
             for o,faces in patch_parts for p in faces for a,b in p.edge_keys]
    lengths=[d for d in lengths if d>1e-10]
    if lengths:voxel_size=min(voxel_size,median(lengths)*.5)
    audit['voxel_size']=voxel_size
    points,polygons=[],[]
    for obj,faces in patch_parts:
        indices=sorted({i for p in faces for i in p.vertices})
        mapping={i:len(points)+j for j,i in enumerate(indices)}
        points.extend(obj.data.vertices[i].co.copy() for i in indices)
        polygons.extend(tuple(mapping[i] for i in p.vertices) for p in faces)
    if not polygons:
        # A seam with zero surrounding faces has no volume to voxelize.
        # Reconcile its duplicate weights directly, without extending it.
        for cluster in seeds:
            total={}
            for o,i,_ in cluster:
                for n,w in original[o,i].items():total[n]=total.get(n,0)+w
            values={'Head':1.} if cluster in head_clusters else normalized(total)
            for o,i,_ in cluster:assign(o,[i],values)
        audit.update(applied=True,proxy_source_vertices=0,proxy_source_faces=0,
                     max_changed_loop=0,outside_patch_weight_changes=0,
                     method='seam-only virtual weld; no surrounding faces to voxelize')
        return audit
    data=bpy.data.meshes.new('Temporary joined skin')
    data.from_pydata(points,[],polygons)
    bm=bmesh.new()
    bm.from_mesh(data)
    bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=voxel_size*.1)
    # Skin-only extraction leaves eye/mouth openings. Close them on the proxy
    # so the voxel union represents a solid volume rather than thin fragments.
    holes=bmesh.ops.holes_fill(bm,edges=[e for e in bm.edges if e.is_boundary],sides=0)
    audit['proxy_holes_filled']=len(holes['faces'])
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    bm.to_mesh(data)
    bm.free()
    proxy=bpy.data.objects.new('Temporary voxel binding proxy',data)
    bpy.context.scene.collection.objects.link(proxy)
    selected=list(bpy.context.selected_objects)
    active=bpy.context.view_layer.objects.active
    try:
        modifier=proxy.modifiers.new('Voxel union across skin seams','REMESH')
        modes={e.identifier for e in modifier.bl_rna.properties['mode'].enum_items}
        if 'VOXEL' not in modes or not hasattr(modifier,'voxel_size'):
            raise RuntimeError('Voxel skin binding requires a runtime with Remesh VOXEL support')
        modifier.mode='VOXEL'
        modifier.voxel_size=voxel_size
        modifier.adaptivity=0
        bpy.context.view_layer.update()
        deps=bpy.context.evaluated_depsgraph_get()
        remeshed=bpy.data.meshes.new_from_object(proxy.evaluated_get(deps),depsgraph=deps)
        proxy.modifiers.clear()
        proxy.data=remeshed
        bpy.data.meshes.remove(data)
        data=remeshed
        assert data.polygons, 'Voxel skin proxy is empty'
        audit['proxy_vertices']=len(data.vertices)
        audit['proxy_faces']=len(data.polygons)
        bpy.ops.object.select_all(action='DESELECT')
        proxy.select_set(True)
        rig.select_set(True)
        bpy.context.view_layer.objects.active=rig
        assert bpy.ops.object.parent_set(type='ARMATURE_AUTO')=={'FINISHED'}
        repaired=0
        for v in data.vertices:
            values=weights(proxy,v.index)
            if not values:
                repaired+=1
                values={}
                for bone in rig.data.bones:
                    a,b=bone.head_local,bone.tail_local
                    t=max(0.,min(1.,(v.co-a).dot(b-a)/(b-a).length_squared))
                    values[bone.name]=1/max((v.co-a-(b-a)*t).length,neck.length*.07)**4
            assign(proxy,[v.index],constrain(rig,v.co,normalized(values)))
        surface=BodySurface(proxy,range(len(data.vertices)))
        changes={}
        for obj,faces in parts:
            for i in {i for p in faces for i in p.vertices if (obj,i) in patch}:
                point=obj.data.vertices[i].co
                factor=1-patch[obj,i]/(max_rings+1)
                if factor<=1e-8:
                    continue
                if obj.get('binding_role')=='head':
                    changes[obj,i]={'Head':1.}
                    continue
                old=weights(obj,i)
                new=constrain(rig,point,surface.sample(point,rig))
                if (obj,i) in head_patch:new={'Head':1.}
                values={n:(1-factor)*old.get(n,0)+factor*new.get(n,0) for n in old.keys()|new.keys()}
                changes[obj,i]=normalized(values)
        # Virtual welding makes UV duplicates and small pre-existing boundary
        # offsets use exactly the same weights, without moving any vertex.
        for cluster in clusters:
            if not all((o,i) in patch for o,i,_ in cluster):continue
            total={}
            for o,i,_ in cluster:
                for n,w in changes[o,i].items():total[n]=total.get(n,0)+w
            values=({'Head':1.} if cluster in head_clusters else normalized(total))
            for obj,index,_ in cluster:changes[obj,index]=values
        for (obj,index),values in changes.items():assign(obj,[index],values)
        outside=[key for key,old in original.items() if key not in patch and weights(key[0],key[1])!=old]
        assert not outside, 'Seam correction changed weights beyond the loop limit'
        audit.update(applied=True,vertices_reweighted=len(changes),
                     max_changed_loop=max((patch[key] for key in changes),default=0),
                     outside_patch_weight_changes=len(outside),
                     proxy_source_vertices=len(points),proxy_source_faces=len(polygons),
                     proxy_unweighted_fallback_vertices=repaired,
                     seam_vertices=sum(len(g) for g in seeds),
                     geometry_and_uvs_preserved=True)
        if proxy_path:
            bpy.data.libraries.write(str(proxy_path),{proxy},fake_user=True)
    finally:
        bpy.data.objects.remove(proxy,do_unlink=True)
        if data.users==0:bpy.data.meshes.remove(data)
        bpy.ops.object.select_all(action='DESELECT')
        for obj in selected:obj.select_set(True)
        bpy.context.view_layer.objects.active=active
    return audit
