"""Voxel skin proxy for continuous weights across mesh and UV boundaries.

Original geometry/UVs stay intact. Inputs are baked in rig-local coordinates
and carry binding_surface=skin on objects or materials. Hair and garments are
excluded; no asset names or source weights participate.
"""
from collections import Counter

import bpy
import bmesh
from mathutils import Vector
from mathutils.kdtree import KDTree

from avatar_apparel_weights import BodySurface, assign, constrain, smooth, weights


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


def correct_skin_seams(meshes,rig,proxy_path=None):
    neck=rig.data.bones['Neck']
    voxel_size=neck.length/16
    parts=[(obj,skin_faces(obj)) for obj in meshes]
    parts=[(obj,faces) for obj,faces in parts if faces]
    clusters=seam_clusters(parts,voxel_size*.1)
    audit={'method':'joined voxel skin proxy, fresh heat weights, local surface transfer',
           'voxel_size':voxel_size,'seam_tolerance':voxel_size*.1,
           'seam_clusters':len(clusters),'applied':False}
    if not clusters:
        return audit
    centers=[sum((p for _,_,p in g),Vector())/len(g) for g in clusters]
    head_seams=[center for center,g in zip(centers,clusters)
                if {obj.get('binding_role') for obj,_,_ in g} >= {'head','body'}]
    head_tree=KDTree(len(head_seams)) if head_seams else None
    if head_tree:
        for i,p in enumerate(head_seams):head_tree.insert(p,i)
        head_tree.balance()
    audit['head_boundary_clusters']=len(head_seams)
    def anchored(point,values):
        if not head_tree:return values
        distance=head_tree.find(point)[2]
        floor=(1-smooth(0,neck.length*.8,distance))*smooth(
            neck.head_local.z-neck.length*.35,neck.head_local.z,point.z)
        h=max(values.get('Head',0),floor)
        others={n:w for n,w in values.items() if n!='Head'}
        total=sum(others.values())
        if h>=1-1e-8 or total<1e-8:return {'Head':1.}
        return normalized({'Head':h,**{n:w/total*(1-h) for n,w in others.items()}})
    seed_tree=KDTree(len(centers))
    for i,p in enumerate(centers):seed_tree.insert(p,i)
    seed_tree.balance()
    points,polygons=[],[]
    for obj,faces in parts:
        indices=sorted({i for p in faces for i in p.vertices})
        mapping={i:len(points)+j for j,i in enumerate(indices)}
        points.extend(obj.data.vertices[i].co.copy() for i in indices)
        polygons.extend(tuple(mapping[i] for i in p.vertices) for p in faces)
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
            assign(proxy,[v.index],anchored(v.co,constrain(rig,v.co,normalized(values))))
        surface=BodySurface(proxy,range(len(data.vertices)))
        changes={}
        for obj,faces in parts:
            for i in {i for p in faces for i in p.vertices}:
                point=obj.data.vertices[i].co
                distance=seed_tree.find(point)[2]
                factor=1-smooth(0,neck.length,distance)
                if factor<=1e-8:
                    continue
                if obj.get('binding_role')=='head':
                    changes[obj,i]={'Head':1.}
                    continue
                old=weights(obj,i)
                new=anchored(point,constrain(rig,point,surface.sample(point,rig)))
                values={n:(1-factor)*old.get(n,0)+factor*new.get(n,0) for n in old.keys()|new.keys()}
                changes[obj,i]=normalized(values)
        # Virtual welding makes UV duplicates and small pre-existing boundary
        # offsets use exactly the same weights, without moving any vertex.
        for center,cluster in zip(centers,clusters):
            values=({'Head':1.} if any(obj.get('binding_role')=='head' for obj,_,_ in cluster)
                    else anchored(center,constrain(rig,center,surface.sample(center,rig))))
            for obj,index,_ in cluster:changes[obj,index]=values
        for (obj,index),values in changes.items():assign(obj,[index],values)
        audit.update(applied=True,vertices_reweighted=len(changes),
                     proxy_unweighted_fallback_vertices=repaired,
                     seam_vertices=sum(len(g) for g in clusters),
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
