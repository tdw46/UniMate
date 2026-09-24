"""Fit VRM capsules against both body volume and spring rest clearance.

Uses the official VRM collider API; no collider meshes or simulator code.
All calculations use rig-local REST coordinates; shape metadata is written
only after the complete plan passes its contact checks.
"""
import math
import numpy as np
import bpy
from mathutils import Vector
from avatar_apparel_weights import weights

PREFIX='Secondary_'


def segment_distance(point, a, b):
    delta=b-a
    t=max(0.,min(1.,(point-a).dot(delta)/delta.length_squared)) if delta.length_squared>1e-16 else 0.
    return (point-(a+delta*t)).length


def spring_samples(rig, role):
    """The collision radius belongs to the parent joint, not its tail joint."""
    result=[]
    for spring in rig.data.vrm_addon_extension.spring_bone1.springs:
        if not spring.vrm_name.startswith(PREFIX+role.title()+'_'):continue
        for head,tail in zip(spring.joints,spring.joints[1:]):
            bone=rig.data.bones.get(tail.node.bone_name)
            if bone:result.append((bone.head_local.copy(),head.hit_radius,tail.node.bone_name))
    return result


def plan_colliders(rig, meshes, material_roles=None):
    ext=rig.data.vrm_addon_extension
    humanoid=ext.vrm1.humanoid.human_bones
    def bone_name(role):return getattr(humanoid,role).node.bone_name
    points={o.name:[rig.matrix_world.inverted()@o.matrix_world@v.co for v in o.data.vertices] for o in meshes}
    all_points=[p for values in points.values() for p in values]
    height=max(p.z for p in all_points)-min(p.z for p in all_points)
    margin=height*.0015
    skin=[]
    roles=material_roles or {}
    for obj in meshes:
        slots={i for i,m in enumerate(obj.data.materials) if m and
               (roles.get(m.name)=='skin' or (m.name not in roles and
                (m.get('binding_surface')=='skin' or any(t in m.name.lower() for t in ('skin','face')))))}
        ids={i for f in obj.data.polygons if f.material_index in slots for i in f.vertices}
        for i in ids:skin.append((points[obj.name][i],weights(obj,i)))
    role_nodes={'skirt':('left_upper_leg','right_upper_leg','left_lower_leg','right_lower_leg'),
                'hair':('head','neck','chest','left_upper_arm','right_upper_arm')}
    candidates=[];omitted=[]
    for role,nodes in role_nodes.items():
        samples=spring_samples(rig,role)
        if not samples:continue
        # Mesh surface clearance supplements the actual simulated endpoint
        # tests, preventing a capsule from starting outside its garment.
        garment=[]
        for obj in meshes:
            for i,p in enumerate(points[obj.name]):
                if any(n.startswith(PREFIX+role.title()+'_') and w>.05 for n,w in weights(obj,i).items()):
                    garment.append((p,0.,'surface'))
        for semantic in nodes:
            name=bone_name(semantic)
            bone=rig.data.bones.get(name)
            if not bone:continue
            local=[bone.matrix_local.inverted()@p for p,w in skin if w.get(name,0)>.4]
            sections=3 if 'leg' in semantic else 1
            for section in range(sections):
                lo=section/sections*bone.length;hi=(section+1)/sections*bone.length
                band=np.array([tuple(p) for p in local if lo<=p.y<=hi])
                center=np.median(band[:,(0,2)],axis=0) if len(band)>=4 else np.zeros(2)
                radius=float(np.quantile(np.linalg.norm(band[:,(0,2)]-center,axis=1),.85)) if len(band)>=4 else bone.length*.12
                radius=min(radius,(bone.length if role=='skirt' else hi-lo)*.45)
                # Overlap adjacent leg sections instead of shrinking every
                # capsule to a tiny disconnected sphere. Rounded ends remain
                # inside the whole bone's longitudinal extent.
                a_y=max(radius,lo) if role=='skirt' else lo+radius
                b_y=min(bone.length-radius,hi) if role=='skirt' else hi-radius
                a=Vector((center[0],a_y,center[1]));b=Vector((center[0],max(a_y,b_y),center[1]))
                world_a=bone.matrix_local@a;world_b=bone.matrix_local@b
                limit=min(segment_distance(p,world_a,world_b)-hit-margin for p,hit,_ in samples+garment)
                fitted=min(radius,limit)
                if fitted<=height*.001:
                    omitted.append(dict(role=role,bone=name,section=section,reason='No positive rest-clear capsule',clearance_limit=limit))
                    continue
                clearance=min(segment_distance(p,world_a,world_b)-hit-fitted for p,hit,_ in samples)
                assert clearance>=margin-1e-7
                candidates.append(dict(role=role,bone=name,offset=list(a),tail=list(b),radius=fitted,
                                       body_radius=radius,minimum_rest_clearance=clearance,section=section))
    return dict(height=height,margin=margin,collider_details=candidates,omitted=omitted)


def rebuild_colliders(rig, meshes, material_roles=None, collider_roles=None):
    """Replace owned VRM collider groups, preserving pose and artist groups."""
    from avatar_physics_preview import suspended
    from avatar_vrm_colliders import (add_capsule, add_group, call_operator,
                                     organize_colliders, cleanup_orphan_displays)
    if rig.data.vrm_addon_extension.spec_version != '1.0':
        raise ValueError('Generated capsule fitting requires VRM 1')
    plan=plan_colliders(rig,meshes,material_roles)
    roles=set(collider_roles or ('skirt','hair'))
    plan['collider_details']=[s for s in plan['collider_details'] if s['role'] in roles]
    plan['omitted']=[s for s in plan['omitted'] if s['role'] in roles]
    names={PREFIX+role.title()+'Body' for role in roles}
    sb=rig.data.vrm_addon_extension.spring_bone1
    owned_groups={g.uuid for g in sb.collider_groups if g.vrm_name in names}
    # A group shared by artist springs is not ours to replace.
    shared={ref.collider_group_uuid for spring in sb.springs
            if not spring.vrm_name.startswith(PREFIX) for ref in spring.collider_groups}
    if owned_groups & shared:
        raise ValueError('Generated collider groups are shared with artist springs')
    owned_colliders={r.collider_uuid for g in sb.collider_groups if g.uuid in owned_groups for r in g.colliders}
    external_colliders={r.collider_uuid for g in sb.collider_groups if g.uuid not in owned_groups for r in g.colliders}
    previous=rig.data.pose_position
    selected={obj.name:obj.as_pointer() for obj in bpy.context.selected_objects}
    active=bpy.context.view_layer.objects.active
    active_name=active.name if active else None
    groups={}
    with suspended():
        try:
            rig.data.pose_position='REST';bpy.context.view_layer.update()
            organize_colliders(rig)
            cleanup_orphan_displays(rig)
            # Recover generated records left ungrouped by an interrupted build.
            referenced={r.collider_uuid for g in sb.collider_groups for r in g.colliders}
            owned_colliders.update(c.uuid for c in sb.colliders
                if c.uuid not in referenced and c.bpy_object
                and c.bpy_object.get('hallway_collider_owner')==rig)
            for i in reversed(range(len(sb.collider_groups))):
                if sb.collider_groups[i].uuid in owned_groups:
                    call_operator('remove_spring_bone1_collider_group',rig,collider_group_index=i)
            for i in reversed(range(len(sb.colliders))):
                if sb.colliders[i].uuid in owned_colliders-external_colliders:
                    call_operator('remove_spring_bone1_collider',rig,collider_index=i)
            for spec in plan['collider_details']:
                role=spec['role']
                if role not in groups:groups[role]=add_group(rig,PREFIX+role.title()+'Body')
                c=add_capsule(rig,spec)
                ref=groups[role].colliders.add();ref.collider_uuid=c.uuid
            for spring in sb.springs:
                for role,group in groups.items():
                    if spring.vrm_name.startswith(PREFIX+role.title()+'_'):
                        ref=spring.collider_groups.add();ref.collider_group_uuid=group.uuid
            cleanup_orphan_displays(rig)
            rig.data.pose_position=previous;bpy.context.view_layer.update()
        finally:
            rig.data.pose_position=previous
            # VRM setters select their empties; restore the caller's selection.
            for obj in bpy.context.selected_objects:obj.select_set(False)
            for name,pointer in selected.items():
                obj=bpy.context.view_layer.objects.get(name)
                if obj and obj.as_pointer()==pointer:obj.select_set(True)
            bpy.context.view_layer.objects.active=bpy.context.view_layer.objects.get(active_name) if active_name else None
            bpy.context.view_layer.update()
    plan.update(colliders=len(plan['collider_details']),collider_groups=len(groups))
    rig['unimate_collider_generator']=5
    return plan


def rest_contacts(rig):
    """Audit real stored collider properties against every associated tail."""
    sb=rig.data.vrm_addon_extension.spring_bone1
    colliders={c.uuid:c for c in sb.colliders if c.bpy_object}
    groups={g.uuid:[colliders[r.collider_uuid] for r in g.colliders if r.collider_uuid in colliders] for g in sb.collider_groups}
    contacts=[];minimum=float('inf');pairs=0
    for spring in sb.springs:
        if not spring.vrm_name.startswith(PREFIX):continue
        for head,tail in zip(spring.joints,spring.joints[1:]):
            point=rig.data.bones[tail.node.bone_name].head_local
            for ref in spring.collider_groups:
                for c in groups.get(ref.collider_group_uuid,[]):
                    bone=rig.data.bones[c.node.bone_name];shape=c.shape.capsule
                    distance=segment_distance(point,bone.matrix_local@Vector(shape.offset),bone.matrix_local@Vector(shape.tail))-shape.radius-head.hit_radius
                    minimum=min(minimum,distance);pairs+=1
                    if distance<0:contacts.append(dict(joint=tail.node.bone_name,collider=c.node.bone_name,penetration=-distance))
    return dict(pairs=pairs,contacts=contacts,minimum_clearance=minimum if pairs else None)
