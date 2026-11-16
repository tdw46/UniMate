"""Fit VRM capsules against both body volume and spring rest clearance.

Uses the official VRM collider API; no collider meshes or simulator code.
All calculations use rig-local REST coordinates; shape metadata is written
only after the complete plan passes its contact checks.
"""
from avatar_material_roles import semantic_label

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


def _panel_contacts(bone, role, leg, samples, garment, height, margin, close_panel=False, fitted_top=False):
    """One-sided thigh, knee and calf supports for a midline sheet.

    A cylindrical leg capsule rolls the middle rail into the crotch. Three
    flatter capsules follow the leg profile, but present a broad surface
    across adjacent cloth rails. Each guard belongs only to the facing sheet.
    """
    inv = bone.matrix_local.inverted()
    # Angle each leg's contact face slightly outward on a narrow front sheet.
    # This limits forward kick where the two broad knee guards overlap.
    outward = .15 * np.sign(bone.head_local.x) if close_panel else 0.
    normal = inv.to_3x3() @ Vector((outward, -1. if role=='frontpanel' else 1., 0.))
    normal.y = 0.
    if normal.length < 1e-5 or bone.length < 1e-6:
        return []
    normal.normalize()
    n = np.asarray(normal)
    refs = samples + garment
    points = np.asarray([tuple(inv@point) for point,_,_ in refs])
    hit = np.asarray([radius for _,radius,_ in refs])
    leg = np.asarray([tuple(p) for p in leg]).reshape(-1,3)
    radius_factor, allowance = (8., .014) if close_panel else (14., .026)
    if fitted_top:
        radius_factor, allowance = 3.5, .008
    sections = []
    for i in range(3):
        lo,hi = i*bone.length/3.,(i+1)*bone.length/3.
        band = leg[(leg[:,1]>=lo)&(leg[:,1]<=hi)]
        depth = np.quantile(band@n,.96) if len(band)>=4 else .16*bone.length
        depth = float(np.clip(depth,.035*bone.length,.4*bone.length))
        # A tight front sheet needs a rounder contact face so the shin on
        # backswing can pass its center. Broad panels retain the flatter face.
        radius = min(2.4*bone.length, radius_factor * depth)
        radial = points.copy()
        radial[:,1] -= np.clip(radial[:,1],lo,hi)
        def clearance(bound):
            return np.linalg.norm(radial-(bound-radius)*n,axis=1)-radius-hit
        # Move the curved face outwards only until its measured skin limit
        # plus clearance allowance, stopping before any rest contact.
        low,high = -2.*radius,depth+allowance*height
        if clearance(low).min() < margin:
            return []  # Fall back to the ordinary limb fit for this bone.
        for _ in range(18):
            mid = (low+high)*.5
            if clearance(mid).min()>=margin:low=mid
            else:high=mid
        off = normal*(low-radius)
        sections.append(dict(role=role,bone=bone.name,offset=list(off+Vector((0.,lo,0.))),
            tail=list(off+Vector((0.,hi,0.))),radius=radius,body_radius=depth,
            minimum_rest_clearance=float(clearance(low)[:len(samples)].min()),section=i))
    return sections


def plan_colliders(rig, meshes, material_roles=None):
    ext=rig.data.vrm_addon_extension
    humanoid=ext.vrm1.humanoid.human_bones
    def bone_name(role):return getattr(humanoid,role).node.bone_name
    points={o.name:[rig.matrix_world.inverted()@o.matrix_world@v.co for v in o.data.vertices] for o in meshes}
    all_points=[p for values in points.values() for p in values]
    height=max(p.z for p in all_points)-min(p.z for p in all_points)
    margin=height*.0015
    skin=[]
    torso=[]
    chest_name=bone_name('chest')
    roles=material_roles or {}
    for obj in meshes:
        slots={i for i,m in enumerate(obj.data.materials) if m and
               (roles.get(m.name)=='skin' or (m.name not in roles and
                (m.get('binding_surface')=='skin' or any(t in semantic_label(m.name) for t in ('skin','face')))))}
        ids={i for f in obj.data.polygons if f.material_index in slots for i in f.vertices}
        for i in ids:skin.append((points[obj.name][i],weights(obj,i)))
        # Exposed skin may be just one shoulder on a clothed avatar. Fitting
        # a torso capsule from that patch offsets it into the lower hair. Use
        # the freshly skinned torso surface, including its clothing, while
        # excluding spring-driven accessories. Other anatomical fits retain
        # their existing skin-only references.
        for i,p in enumerate(points[obj.name]):
            values=weights(obj,i)
            if values.get(chest_name,0)>.4 and not any(n.startswith(PREFIX) and w>1e-6 for n,w in values.items()):
                torso.append((p,values))
    role_nodes={'breast':('chest','upper_chest'),
                'accessory':('hips','spine','chest','left_upper_leg','right_upper_leg','left_upper_arm','right_upper_arm'),
                'skirt':('left_upper_leg','right_upper_leg','left_lower_leg','right_lower_leg'),
                'hair':('head','neck','chest','left_upper_arm','right_upper_arm')}
    for role in ('frontpanel', 'backpanel'):
        role_nodes[role] = ('left_upper_leg','right_upper_leg','left_lower_leg','right_lower_leg')
    for role in ('flowinghair', 'bowribbon', 'bowloop'):
        role_nodes[role] = role_nodes['hair']
    role_nodes['tail'] = ('hips',)
    for side in ('left', 'right'):
        role_nodes['armstrap_' + side[0]] = ('spine', 'chest', 'upper_chest',
                                              side + '_shoulder', side + '_upper_arm')
    candidates=[];omitted=[]
    for role,nodes in role_nodes.items():
        samples=spring_samples(rig,role)
        top_samples = []
        if role == 'frontpanel':
            top_samples = spring_samples(rig, 'fittedfront')
            samples = samples + top_samples
        if not samples:continue
        # Mesh surface clearance supplements the actual simulated endpoint
        # tests, preventing a capsule from starting outside its garment.
        garment=[]
        for obj in meshes:
            for i,p in enumerate(points[obj.name]):
                if any(n.startswith(PREFIX+role.title()+'_') and w>.05 for n,w in weights(obj,i).items()):
                    garment.append((p,0.,'surface'))
        close_panel = role == 'frontpanel' and all(
            name in rig.data.bones and rig.data.bones[name].get('close_front_panel')
            for _, _, name in samples)
        for semantic in nodes:
            name=bone_name(semantic)
            bone=rig.data.bones.get(name)
            if not bone:continue
            reference=torso if role=='hair' and semantic=='chest' and torso else skin
            local=[bone.matrix_local.inverted()@p for p,w in reference if w.get(name,0)>.4]
            if role in ('frontpanel', 'backpanel') and garment:
                sections = _panel_contacts(bone,role,local,samples,garment,height,margin,close_panel)
                if sections:
                    if top_samples:
                        fitted = _panel_contacts(bone, role, local, samples, garment,
                                                 height, margin, True, fitted_top=True)
                        candidates.extend(dict(spec, role='fittedfront') for spec in fitted or sections)
                    candidates.extend(sections)
                    continue
            # Leg contact needs a continuous, straight surface. Independent
            # offset sections create changes in the contact normal that let a
            # low-follow spring slide between volumes after length projection.
            # Use the actual bone head/tail for the capsule center segment.
            sections=1
            for section in range(sections):
                lo=section/sections*bone.length;hi=(section+1)/sections*bone.length
                band=np.array([tuple(p) for p in local if lo<=p.y<=hi])
                center=np.median(band[:,(0,2)],axis=0) if role!='skirt' and len(band)>=4 else np.zeros(2)
                radius=float(np.quantile(np.linalg.norm(band[:,(0,2)]-center,axis=1),.85)) if len(band)>=4 else bone.length*.12
                radius=min(radius,(bone.length if role=='skirt' else hi-lo)*.45)
                a_y=0. if role=='skirt' else lo+radius
                b_y=bone.length if role=='skirt' else hi-radius
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
    if any(s.vrm_name.startswith(PREFIX + 'Fittedfront_') for s in rig.data.vrm_addon_extension.spring_bone1.springs):
        fitted_bones = {spec['bone'] for spec in candidates if spec['role'] == 'fittedfront'}
        candidates.extend(dict(spec, role='fittedfront') for spec in list(candidates)
                          if spec['role'] == 'frontpanel' and spec['bone'] not in fitted_bones)
    return dict(height=height,margin=margin,collider_details=candidates,omitted=omitted)


def rebuild_colliders(rig, meshes, material_roles=None, collider_roles=None):
    """Replace owned VRM collider groups, preserving pose and artist groups."""
    from avatar_physics_preview import suspended
    from avatar_vrm_colliders import (add_capsule, add_group, call_operator,
                                     organize_colliders, cleanup_orphan_displays)
    if rig.data.vrm_addon_extension.spec_version != '1.0':
        raise ValueError('Generated capsule fitting requires VRM 1')
    if rig.get('hallway_contact_colliders') and collider_roles and set(collider_roles)=={'skirt'}:
        from avatar_contact_colliders import install_contact_colliders
        from avatar_skirt_fit_io import rest_edit
        with rest_edit(rig):
            result=install_contact_colliders(rig,meshes)
        rig['hallway_pose_fit_settings_changed']=True
        return dict(result,colliders=result['unique_skirt_colliders'])
    plan=plan_colliders(rig,meshes,material_roles)
    roles=set(collider_roles or ('skirt','hair','accessory','ornament','tail','armstrap_l','armstrap_r',
                                'frontpanel','backpanel','flowinghair','bowribbon','bowloop'))
    if 'frontpanel' in roles:
        roles.add('fittedfront')
    plan['collider_details']=[s for s in plan['collider_details'] if s['role'] in roles]
    plan['omitted']=[s for s in plan['omitted'] if s['role'] in roles]
    names={PREFIX+role.title()+'Body' for role in roles}
    sb=rig.data.vrm_addon_extension.spring_bone1
    from avatar_contact_colliders import owned_group
    owned_groups={g.uuid for g in sb.collider_groups if g.vrm_name in names
                  or ('accessory' in roles and g.vrm_name.startswith(PREFIX+'StrapLeg_'))
                  or ('skirt' in roles and owned_group(g.vrm_name))}
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
                if role not in groups:
                    add_group(rig, PREFIX + role.title() + 'Body')
                    # Adding another group may relocate the RNA collection.
                    # Keep a stable index during this append-only build.
                    groups[role] = len(sb.collider_groups) - 1
                c = add_capsule(rig, spec)
                group = sb.collider_groups[groups[role]]
                group.colliders.add().collider_uuid = c.uuid
            for spring in sb.springs:
                for role, index in groups.items():
                    # The transition shares the fitted leg contacts. Its
                    # endpoints sit inside the already fitted garment surface.
                    transition = role == 'fittedfront' and spring.vrm_name.startswith(PREFIX + 'Paneltransition_')
                    if transition or spring.vrm_name.startswith(PREFIX + role.title() + '_'):
                        group_uuid = sb.collider_groups[index].uuid
                        ref = spring.collider_groups.add()
                        ref.collider_group_uuid = group_uuid
                        if not group_uuid or ref.collider_group_uuid != group_uuid:
                            raise RuntimeError('Could not attach fitted VRM collider group to ' + spring.vrm_name)
            if 'accessory' in roles:
                from avatar_strap_contacts import install_strap_contacts
                install_strap_contacts(rig, meshes, PREFIX, plan['height'])
                organize_colliders(rig)
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
    if 'skirt' in roles:
        rig['hallway_contact_colliders']=0
        if rig.get('hallway_pose_fit'):rig['hallway_pose_fit_settings_changed']=True
    rig['unimate_collider_generator']=6
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
