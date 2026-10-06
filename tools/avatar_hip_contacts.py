"""Chain-local upper-skirt contact patches made only from native VRM capsules.

Two finite capsule strips approximate the lateral and front/back faces of a
rounded corner. A short hip spring cannot slide around the single radial guard
used by longer dress chains. These shapes never belong to hair or lower-skirt
groups; existing long-chain contacts and weights are independent.
"""
from mathutils import Vector

GROUP_PREFIX = 'Secondary_HipContact_'
LEGACY_GROUP_PREFIX = 'Secondary_SkirtContact_Hip_'
SPRING_PREFIX = 'Secondary_HipSkirt_'


def plans(rig):
    from avatar_directional_contacts import skirt_owner_leg
    from avatar_colliders import segment_distance
    sb = rig.data.vrm_addon_extension.spring_bone1
    human = rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    thighs = [getattr(human, side + '_upper_leg').node.bone_name for side in ('left', 'right')]
    if not any(s.vrm_name.startswith(SPRING_PREFIX) for s in sb.springs):
        return []
    lateral = rig.data.bones[thighs[0]].head_local - rig.data.bones[thighs[1]].head_local
    lateral.z = 0.
    if lateral.length < 1e-8:
        raise ValueError('Upper-skirt contacts require distinct hip sockets')
    lateral.normalize()
    depth = Vector((0., 0., 1.)).cross(lateral).normalized()
    center = (rig.data.bones[thighs[0]].head_local + rig.data.bones[thighs[1]].head_local) * .5
    springs = {s.vrm_name: s for s in sb.springs}
    groups = {g.uuid: g for g in sb.collider_groups}
    colliders = {c.uuid: c for c in sb.colliders}
    result = []
    for spring in sb.springs:
        if not spring.vrm_name.startswith(SPRING_PREFIX):
            continue
        root = rig.data.bones[spring.joints[0].node.bone_name]
        original = springs[root['hallway_support_chain'].rsplit('_', 1)[0]]
        leg = rig.data.bones[skirt_owner_leg(rig, original, thighs)]
        inverse = leg.matrix_local.inverted()
        tip = rig.data.bones[spring.joints[-1].node.bone_name].head_local
        local = inverse @ tip
        hit = spring.joints[0].hit_radius
        margin = max(root.length * .015, 1e-5)
        # Bounded by the upper leg and the short spring's reachable region.
        # Broad backs flatten the face; no other spring can see their volume.
        radius = min(leg.length, root.length * 4.)
        specs = []
        for axis_index, axis in enumerate((lateral, depth)):
            axis = axis * (1. if (tip-center).dot(axis) >= 0 else -1.)
            normal = inverse.to_3x3() @ axis
            normal.y = 0.
            if normal.length < 1e-8:
                raise ValueError('Degenerate upper-skirt contact normal')
            normal.normalize()
            tangent = Vector((0., 1., 0.)).cross(normal).normalized()
            for row, shift in enumerate((-.5, 0., .5)):
                anchor = local - normal * (radius + hit + margin) + tangent * (shift * radius)
                anchor.y = 0.
                a = anchor + Vector((0., -root.length * .75, 0.))
                b = anchor + Vector((0., max(leg.length * .6, root.length * 2.), 0.))
                clearance = segment_distance(local, a, b) - radius - hit
                if clearance < margin - 1e-6:
                    raise ValueError('Upper-skirt patch intersects its spring at rest')
                specs.append(dict(key=f'{spring.vrm_name}:{axis_index}:{row}', bone=leg.name,
                                  offset=list(a), tail=list(b), radius=radius, clearance=clearance))
        # Only rest-clear hips support/roof and the owning thigh's centerline
        # capsule remain as fallback. Do not inherit other directional guards.
        fallback = []
        for ref in original.collider_groups:
            for item in groups[ref.collider_group_uuid].colliders:
                c = colliders[item.collider_uuid]
                if c.shape_type != 'Capsule':
                    continue
                shape = c.shape.capsule
                bone = rig.data.bones.get(c.node.bone_name)
                if not bone:
                    continue
                own_centerline = (bone.name == leg.name
                    and (Vector(shape.tail)-Vector(shape.offset)).length >= leg.length*.9
                    and Vector((shape.offset[0], 0., shape.offset[2])).length < leg.length*.025)
                if bone.name != human.hips.node.bone_name and not own_centerline:
                    continue
                if segment_distance(tip, bone.matrix_local @ Vector(shape.offset),
                                    bone.matrix_local @ Vector(shape.tail))-shape.radius-hit >= margin:
                    fallback.append(c.uuid)
        result.append(dict(spring=spring.vrm_name, name=GROUP_PREFIX+spring.vrm_name,
                           specs=specs, fallback=list(dict.fromkeys(fallback))))
    return result


def refresh(rig):
    """Fit/update owned native shapes in place; prune only unused owned shapes.

    Caller suspends simulation and supplies the rest pose for VRM RNA setters.
    Repeated fitting preserves UUIDs, including group references in exports.
    """
    from avatar_vrm_colliders import add_capsule, add_group, call_operator, organize_colliders, cleanup_orphan_displays
    sb = rig.data.vrm_addon_extension.spring_bone1
    plan = plans(rig)
    owned = {g.uuid for g in sb.collider_groups if g.vrm_name.startswith((GROUP_PREFIX, LEGACY_GROUP_PREFIX))}
    if any(ref.collider_group_uuid in owned for s in sb.springs
           if not s.vrm_name.startswith(SPRING_PREFIX) for ref in s.collider_groups):
        raise ValueError('An unrelated spring references generated upper-skirt contacts')
    previous = {ref.collider_uuid for g in sb.collider_groups if g.uuid in owned for ref in g.colliders}
    native = {}
    duplicates = set()
    for c in sb.colliders:
        key = c.bpy_object.get('hallway_hip_guard') if c.bpy_object else None
        if key:
            if key in native:
                duplicates.add(c.uuid)
            else:
                native[key] = c
            previous.add(c.uuid)
    groups = {g.vrm_name: g for g in sb.collider_groups}
    springs = {s.vrm_name: s for s in sb.springs}
    used_names = set()
    for row in plan:
        ids = list(row['fallback'])
        for spec in row['specs']:
            collider = native.get(spec['key'])
            if collider:
                collider.node.bone_name = spec['bone']
                for field in ('offset', 'tail', 'radius'):
                    current = getattr(collider.shape.capsule, field)
                    expected = spec[field]
                    same = abs(current-expected) < 1e-7 if field == 'radius' else (Vector(current)-Vector(expected)).length < 1e-7
                    if not same:
                        setattr(collider.shape.capsule, field, expected)
            else:
                collider = add_capsule(rig, spec, organize=False)
            obj = collider.bpy_object
            obj['hallway_hip_guard'] = spec['key']
            obj['hallway_contact_collider'] = True
            obj['hallway_base_radius'] = spec['radius']
            obj['hallway_radius_limit'] = spec['radius']
            ids.append(collider.uuid)
        group = groups.get(row['name']) or add_group(rig, row['name'])
        used_names.add(row['name'])
        if [r.collider_uuid for r in group.colliders] != ids:
            group.colliders.clear()
            for uuid in ids:
                group.colliders.add().collider_uuid = uuid
        spring = springs[row['spring']]
        if [r.collider_group_uuid for r in spring.collider_groups] != [group.uuid]:
            spring.collider_groups.clear()
            spring.collider_groups.add().collider_group_uuid = group.uuid
    for i in reversed(range(len(sb.collider_groups))):
        group = sb.collider_groups[i]
        if group.uuid in owned and group.vrm_name not in used_names:
            call_operator('remove_spring_bone1_collider_group', rig, collider_group_index=i)
    referenced = {r.collider_uuid for g in sb.collider_groups for r in g.colliders}
    # Rebuilding the lower dress can retain an old fallback because this group
    # still referenced it. Once reassigned, remove that unused generated shape.
    for i in reversed(range(len(sb.colliders))):
        c = sb.colliders[i]
        if c.uuid in previous | duplicates and c.uuid not in referenced and c.bpy_object and c.bpy_object.get('unimate_generated_collider'):
            call_operator('remove_spring_bone1_collider', rig, collider_index=i)
    organize_colliders(rig)
    cleanup_orphan_displays(rig)
    return [dict(spring=r['spring'], patches=len(r['specs']), fallback=len(r['fallback']),
                 minimum_rest_clearance=min(s['clearance'] for s in r['specs'])) for r in plan]
