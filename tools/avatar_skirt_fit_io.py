"""Portable pose-fit bundles with guarded, reversible live installation.

Offline fitting emits weights and native VRM capsule parameters.
Final rest geometry and existing shape keys are immutable. Applying a bundle adds no simulation callbacks or export dependency.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import bpy
import math
from avatar_mesh_invariant import snapshot, verify, signature
from avatar_apparel_weights import weights
from avatar_contact_colliders import snapshot_contact_colliders, replace_contact_colliders
from avatar_physics_preview import suspended


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def mesh_signature(obj, rig):
    """Expression values and current pose may change while the fit is running."""
    mesh=obj.data
    return digest(dict(vertices=[list(v.co) for v in mesh.vertices],
        polygons=[(list(p.vertices),p.material_index) for p in mesh.polygons],
        uv=[(u.name,[list(d.uv) for d in u.data]) for u in mesh.uv_layers],
        keys=[(k.name,k.relative_key.name,k.vertex_group,[list(v.co) for v in k.data])
              for k in mesh.shape_keys.key_blocks] if mesh.shape_keys else [],
        weights=[weights(obj,v.index) for v in mesh.vertices],
        transform=[list(row) for row in rig.matrix_world.inverted()@obj.matrix_world]))


def rig_signature(rig):
    sb=rig.data.vrm_addon_extension.spring_bone1
    # Live VRM UI may reorder independent springs; joint order remains significant.
    return digest(dict(bones=[(b.name,b.parent.name if b.parent else None,
        [list(row) for row in b.matrix_local],b.length) for b in rig.data.bones],
        springs=sorted((s.vrm_name,s.center.bone_name,[(j.node.bone_name,j.hit_radius,j.stiffness,
            j.drag_force,j.gravity_power,list(j.gravity_dir)) for j in s.joints]) for s in sb.springs)))


def source_record(rig, meshes):
    return dict(rig=rig.name,rig_signature=rig_signature(rig),
        collider_signature=digest(snapshot_contact_colliders(rig)),
        meshes={o.name:dict(signature=mesh_signature(o,rig),geometry_signature=signature(o),positions=[list(v.co) for v in o.data.vertices],
            weights=[weights(o,v.index) for v in o.data.vertices],polygons=[list(p.vertices) for p in o.data.polygons]) for o in meshes})


def write_bundle(rig, source, report, path):
    changes={}
    for name,before in source['meshes'].items():
        obj=bpy.data.objects[name];rows=[]
        if before.get('geometry_signature') and signature(obj)!=before['geometry_signature']:
            raise ValueError(f'{name}: fitting changed geometry or existing shape keys')
        if [list(v.co) for v in obj.data.vertices]!=before['positions'] or [list(p.vertices) for p in obj.data.polygons]!=before['polygons']:
            raise ValueError(f'{name}: fitting changed mesh coordinates or topology')
        for i,v in enumerate(obj.data.vertices):
            w=weights(obj,i)
            if w!=before['weights'][i]:rows.append(dict(index=i,delta=[0.,0.,0.],weights=w))
        if rows:changes[name]=rows
    payload=dict(version=1,source=source,changes=changes,removed_faces={},colliders=snapshot_contact_colliders(rig),report=report)
    payload['source']=dict(source,meshes={n:dict(signature=s['signature']) for n,s in source['meshes'].items()})
    Path(path).write_text(json.dumps(payload,separators=(',',':')))
    return payload


@contextmanager
def rest_edit(rig):
    """Scope official VRM setters without changing selection, pose or BVT state."""
    layer=bpy.context.view_layer;active=layer.objects.active
    selected=[o.name for o in bpy.context.selected_objects];mode=active.mode if active else 'OBJECT'
    active_name=active.name if active else None
    previous=rig.data.pose_position
    with suspended():
        try:
            if active and active.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
            layer.objects.active=rig
            rig.data.pose_position='REST';layer.update()
            yield
        finally:
            rig.data.pose_position=previous
            for o in bpy.context.selected_objects:o.select_set(False)
            for name in selected:
                obj=layer.objects.get(name)
                if obj:obj.select_set(True)
            restored=layer.objects.get(active_name) if active_name else None
            layer.objects.active=restored
            if restored and mode!='OBJECT':bpy.ops.object.mode_set(mode=mode)
            layer.update()


def apply_bundle(rig, path):
    payload=json.loads(Path(path).read_text())
    if payload.get('version')!=1:raise ValueError('Unsupported pose-fit bundle version')
    if payload.get('removed_faces') or any(any(abs(v)>0 for v in row.get('delta',())) for rows in payload['changes'].values() for row in rows):
        raise ValueError('This legacy fit alters the mesh. Only geometry-preserving rig fits can be applied')
    if payload.get('report',{}).get('maximum_direct_leg_weight',0)>0:
        raise ValueError('Direct leg-weight skirt fitting is retired; use spring contact generation')
    bundle_id=digest(payload)
    if rig.get('hallway_pose_fit_bundle')==bundle_id:
        return dict(already_applied=True)
    source=payload['source'];report=payload['report']
    if report.get('repair_only'):
        raise ValueError('Legacy geometry repairs are no longer supported')
    if sum(report['training_after'])>=sum(report['training_before']) or sum(report['validation_after'])>sum(report['validation_before']):
        raise ValueError('Fit did not improve the training set without regressing held-out poses')
    if rig_signature(rig)!=source['rig_signature']:
        raise ValueError('Bones or spring settings changed since sampling; run a new fit')
    if digest(snapshot_contact_colliders(rig))!=source['collider_signature']:
        raise ValueError('Skirt colliders changed since sampling; run a new fit')
    for name,record in source['meshes'].items():
        obj=bpy.data.objects.get(name)
        if not obj or mesh_signature(obj,rig)!=record['signature']:
            raise ValueError(f'{name}: source geometry, weights or transform changed; run a new fit')
    for name,rows in payload['changes'].items():
        if name not in source['meshes']:raise ValueError('Fit contains an unverified mesh')
        obj=bpy.data.objects[name]
        if obj.data.users!=1:raise ValueError(f'{name}: shared mesh data must be made single-user before fitting')
        seen=set()
        for row in rows:
            index=row['index'];values=row['weights']
            if not isinstance(index,int) or index in seen or not (0<=index<len(obj.data.vertices)):
                raise ValueError('Invalid or duplicate fitted vertex')
            seen.add(index)
            old=weights(obj,index)
            humanoid=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
            legs={getattr(humanoid,side+'_'+part).node.bone_name for side in ('left','right') for part in ('upper_leg','lower_leg')}
            if any(n.startswith('Secondary_Skirt_') and w>1e-6 for n,w in old.items()) and any(values.get(n,0)>old.get(n,0)+1e-6 for n in legs):
                raise ValueError('Skirts must remain spring-bound: direct leg-weight attachment fits are not supported')
            if set(values)-set(obj.vertex_groups.keys())-set(rig.data.bones.keys()):
                raise ValueError('Invalid fitted bone group')
            if len(row.get('delta',()))!=3 or not all(math.isfinite(x) for x in (*row['delta'],*values.values())):
                raise ValueError('Non-finite fitted vertex or weight')
            if any(v<0 or v>1 for v in values.values()) or abs(sum(values.values())-1)>1e-5:
                raise ValueError('Fitted weights must be nonnegative and normalized')
    previous=snapshot_contact_colliders(rig)
    collider_change=digest(previous)!=digest(payload['colliders'])
    meshes=[bpy.data.objects[n] for n in source['meshes']]
    geometry=snapshot(meshes)
    backups={n:([dict(index=row['index'],weights=weights(bpy.data.objects[n],row['index'])) for row in rows],
                set(bpy.data.objects[n].vertex_groups.keys())) for n,rows in payload['changes'].items()}
    def write_weights(obj,rows):
        indices=[row['index'] for row in rows]
        for group in obj.vertex_groups:group.remove(indices)
        batches={}
        for row in rows:
            for group,value in row['weights'].items():
                if value>0:batches.setdefault((group,value),[]).append(row['index'])
        for (name,value),indices in batches.items():
            group=obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
            group.add(indices,value,'REPLACE')
        obj.data.update()
    with rest_edit(rig):
        try:
            if collider_change:replace_contact_colliders(rig,payload['colliders'])
            for name,rows in payload['changes'].items():write_weights(bpy.data.objects[name],rows)
            verify(meshes,geometry)
        except Exception:
            for name,(rows,groups) in backups.items():
                obj=bpy.data.objects[name];write_weights(obj,rows)
                for group in list(obj.vertex_groups):
                    if group.name not in groups:obj.vertex_groups.remove(group)
            if collider_change:replace_contact_colliders(rig,previous)
            raise
    rig['hallway_pose_fit']=1
    rig['hallway_pose_fit_bundle']=bundle_id
    rig['hallway_pose_fit_path']=str(Path(path).resolve())
    rig['hallway_pose_fit_report']=json.dumps(report)
    from properties_hallway_rig import apply_thickness
    apply_thickness(rig)
    rig['hallway_pose_fit_settings_changed']=rig.hallway_rig.skirt_thickness!=1.
    return dict(already_applied=False,changed_vertices=sum(len(v) for v in payload['changes'].values()),
                skirt_colliders=len(payload['colliders']['colliders']))
