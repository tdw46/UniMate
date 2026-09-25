"""Deduplicated, per-chain skirt coverage using only standard VRM capsules."""
import bpy
from mathutils import Vector
from avatar_colliders import plan_colliders,segment_distance
from avatar_vrm_colliders import add_capsule,add_group,call_operator,organize_colliders,cleanup_orphan_displays

GROUP_PREFIX='Secondary_SkirtContact_'


def install_contact_colliders(rig, meshes):
    plan=plan_colliders(rig,meshes)
    sb=rig.data.vrm_addon_extension.spring_bone1
    body=[s for s in plan['collider_details'] if s['role']=='skirt']
    hum=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    thighs={hum.left_upper_leg.node.bone_name,hum.right_upper_leg.node.bone_name}
    groups={};specs={}
    def add(spec):
        key=(spec['bone'],*map(float,spec['offset']),*map(float,spec['tail']),float(spec['radius']))
        if key not in specs:specs[key]=dict(spec)
        return key
    fallback=[add(s) for s in body]
    for spring in sb.springs:
        if not spring.vrm_name.startswith('Secondary_Skirt_'):continue
        points=[(rig.data.bones[t.node.bone_name].head_local,h.hit_radius) for h,t in zip(spring.joints,spring.joints[1:])]
        keys=list(fallback)
        for original in body:
            bone=rig.data.bones[original['bone']]
            sections=6 if bone.name in thighs else 1
            for index in range(sections):
                a=Vector((0,(index+.5)/sections*bone.length if sections>1 else 0,0))
                b=a.copy() if sections>1 else Vector((0,bone.length,0))
                limit=min(segment_distance(p,bone.matrix_local@a,bone.matrix_local@b)-r-plan['margin'] for p,r in points)
                radius=min(original['body_radius'],limit)
                if radius<=plan['height']*.001:continue
                keys.append(add(dict(bone=bone.name,offset=list(a),tail=list(b),radius=radius,
                                     minimum_rest_clearance=plan['margin'])))
        groups[spring.vrm_name]=list(dict.fromkeys(keys))
    keys=list(specs)
    payload=dict(colliders=[dict(specs[k],contact=True) for k in keys], groups=[
        dict(name=GROUP_PREFIX+name, springs=[name], colliders=[keys.index(k) for k in refs])
        for name,refs in groups.items()])
    replace_contact_colliders(rig,payload)
    return dict(unique_skirt_colliders=len(specs),groups=len(groups),references=sum(map(len,groups.values())),
                deduplicated_references=sum(map(len,groups.values()))-len(specs))


def owned_group(name):
    return name=='Secondary_SkirtBody' or name.startswith(GROUP_PREFIX)


def snapshot_contact_colliders(rig):
    """Read native VRM data, preserving exact fitted shapes and group scope."""
    sb=rig.data.vrm_addon_extension.spring_bone1
    groups=[g for g in sb.collider_groups if owned_group(g.vrm_name)]
    ids={r.collider_uuid for g in groups for r in g.colliders}
    colliders=[c for c in sb.colliders if c.uuid in ids]
    if any(c.shape_type!='Capsule' for c in colliders):
        raise ValueError('Owned skirt group contains a non-capsule collider')
    lookup={c.uuid:i for i,c in enumerate(colliders)}
    return dict(colliders=[dict(bone=c.node.bone_name,offset=list(c.shape.capsule.offset),
        tail=list(c.shape.capsule.tail),radius=c.shape.capsule.radius,
        contact=bool(c.bpy_object.get('hallway_contact_collider')),
        directional=bool(c.bpy_object.get('hallway_directional_guard')),
        base=c.bpy_object.get('hallway_base_radius',c.shape.capsule.radius),
        limit=c.bpy_object.get('hallway_radius_limit',c.shape.capsule.radius)) for c in colliders],
        groups=[dict(name=g.vrm_name,colliders=[lookup[r.collider_uuid] for r in g.colliders],
        springs=[s.vrm_name for s in sb.springs if any(r.collider_group_uuid==g.uuid for r in s.collider_groups)]) for g in groups])


def replace_contact_colliders(rig,payload):
    """Caller owns pose/BVT state. Only replace our groups through VRM APIs."""
    sb=rig.data.vrm_addon_extension.spring_bone1
    spring_names={s.vrm_name for s in sb.springs}
    for spec in payload['colliders']:
        if spec['bone'] not in rig.data.bones or spec['radius']<0:
            raise ValueError('Invalid capsule fit')
    for group in payload['groups']:
        if not owned_group(group['name']) or set(group['springs'])-spring_names:
            raise ValueError('Invalid skirt group fit')
        if any(i<0 or i>=len(payload['colliders']) for i in group['colliders']):
            raise ValueError('Invalid collider reference')
    owned={g.uuid for g in sb.collider_groups if g.vrm_name=='Secondary_SkirtBody' or g.vrm_name.startswith(GROUP_PREFIX)}
    foreign={r.collider_group_uuid for s in sb.springs if not s.vrm_name.startswith('Secondary_Skirt_') for r in s.collider_groups}
    if owned & foreign:raise ValueError('Skirt contact groups are shared with artist springs')
    ids={r.collider_uuid for g in sb.collider_groups if g.uuid in owned for r in g.colliders}
    other={r.collider_uuid for g in sb.collider_groups if g.uuid not in owned for r in g.colliders}
    for i in reversed(range(len(sb.collider_groups))):
        if sb.collider_groups[i].uuid in owned:call_operator('remove_spring_bone1_collider_group',rig,collider_group_index=i)
    for i in reversed(range(len(sb.colliders))):
        if sb.colliders[i].uuid in ids-other:call_operator('remove_spring_bone1_collider',rig,collider_index=i)
    uuids=[]
    for spec in payload['colliders']:
        collider=add_capsule(rig,spec);uuids.append(collider.uuid)
        collider.bpy_object['hallway_contact_collider']=spec.get('contact',True)
        collider.bpy_object['hallway_directional_guard']=spec.get('directional',False)
        collider.bpy_object['hallway_base_radius']=spec.get('base',spec['radius'])
        collider.bpy_object['hallway_radius_limit']=spec.get('limit',spec['radius'])
    for spec in payload['groups']:
        group=add_group(rig,spec['name'])
        for index in spec['colliders']:group.colliders.add().collider_uuid=uuids[index]
        for name in spec['springs']:
            spring=next(s for s in sb.springs if s.vrm_name==name)
            spring.collider_groups.add().collider_group_uuid=group.uuid
    organize_colliders(rig);cleanup_orphan_displays(rig)
    rig['hallway_contact_colliders']=int(any(s.get('contact',True) for s in payload['colliders']))
