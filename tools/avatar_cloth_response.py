"""Generation-only cloth contact tuning using standard VRM spring data.

No source rig, weights, model identifiers or simulation changes are used. Run
once after profile settings and fitted contacts; never as a load-time migration.
"""
import json
from mathutils import Vector
from avatar_colliders import segment_distance
from avatar_contact_colliders import snapshot_contact_colliders, replace_contact_colliders


def stabilize_cloth_contacts(rig, group):
    """Preserve anatomical coverage and rest positions while softening impacts."""
    key = 'hallway_cloth_contact_response'
    if key in rig:
        return json.loads(rig[key])
    sb = rig.data.vrm_addon_extension.spring_bone1
    springs = {s.vrm_name: s for s in sb.springs}
    cloth = [s for s in sb.springs if s.vrm_name.startswith('Secondary_Skirt_')]
    if not cloth or group is None:
        return dict(merged=0, reanchored=0, tuned=0)
    payload = snapshot_contact_colliders(rig)
    merged, rejected, anchored = [], [], []
    for contact_group in payload['groups']:
        samples = [(rig.data.bones[t.node.bone_name].head_local, h.hit_radius)
                   for name in contact_group['springs']
                   for h, t in zip(springs[name].joints, springs[name].joints[1:])]
        if not samples:
            continue
        for pos, index in enumerate(list(contact_group['colliders'])):
            a = payload['colliders'][index]
            if not a.get('directional'):
                continue
            axis = Vector(a['tail']) - Vector(a['offset'])
            bone = rig.data.bones[a['bone']]
            inverse = bone.matrix_local.inverted()
            choices = [payload['colliders'][i] for i in contact_group['colliders']
                       if not payload['colliders'][i].get('directional')
                       and payload['colliders'][i]['bone'] == a['bone']
                       and (Vector(payload['colliders'][i]['tail']) -
                            Vector(payload['colliders'][i]['offset']) - axis).length < 1e-6]
            if not choices:
                continue
            body = max(choices, key=lambda c: c['radius'])
            delta = Vector(body['offset']) - Vector(a['offset'])
            distance = delta.length
            if distance + min(a['radius'], body['radius']) <= max(a['radius'], body['radius']) + 1e-6:
                continue
            radius = (distance + a['radius'] + body['radius']) / 2
            offset = Vector(a['offset']) + delta * ((radius - a['radius']) / distance)
            tail = offset + axis
            clearance = min(segment_distance(inverse @ p, offset, tail) - r - radius for p, r in samples)
            if clearance < 1e-5:
                rejected.append(dict(bone=bone.name, clearance=clearance))
                continue
            contact_group['colliders'][pos] = len(payload['colliders'])
            payload['colliders'].append(dict(a, offset=list(offset), tail=list(tail),
                                             radius=radius, base=radius, limit=radius))
            merged.append(dict(bone=bone.name, clearance=clearance))
    used = sorted({i for g in payload['groups'] for i in g['colliders']})
    remap = {old: i for i, old in enumerate(used)}
    payload['colliders'] = [payload['colliders'][i] for i in used]
    for g in payload['groups']:
        g['colliders'] = [remap[i] for i in g['colliders']]
    # Preserve the qualified VRM serialization boundary before changing frames.
    if merged:
        replace_contact_colliders(rig, payload)
        payload = snapshot_contact_colliders(rig)
    human = rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    hips = rig.data.bones.get(human.hips.node.bone_name)
    lower_legs = {getattr(human, side + '_lower_leg').node.bone_name for side in ('left', 'right')}
    if hips is not None:
        for spec in payload['colliders']:
            if not spec.get('directional') or spec['bone'] not in lower_legs:
                continue
            old = rig.data.bones[spec['bone']]
            matrix = hips.matrix_local.inverted() @ old.matrix_local
            spec.update(bone=hips.name, offset=list(matrix @ Vector(spec['offset'])),
                        tail=list(matrix @ Vector(spec['tail'])))
            anchored.append(dict(from_bone=old.name, to_bone=hips.name))
    # Anatomical leg capsules retain their original bones and radii. Only the
    # extra generated support follows the pelvis, with identical rest endpoints.
    if anchored:
        replace_contact_colliders(rig, payload)
    tuned = []
    for spring in cloth:
        root = rig.data.bones[spring.joints[0].node.bone_name]
        owner = springs.get(root.get('hallway_dress_upper_spring'), spring)
        full = json.loads(rig.data.bones[owner.joints[0].node.bone_name].get('hallway_dress_full_chain', 'null'))
        if full is None:
            full = [j.node.bone_name for j in owner.joints]
        lengths = [(rig.data.bones[b].head_local - rig.data.bones[a].head_local).length for a, b in zip(full, full[1:])]
        total = sum(lengths)
        ends = {n: sum(lengths[:i+1]) / total for i, n in enumerate(full[:-1])} if total > 1e-8 else {}
        base = owner.joints[0].stiffness
        for index, joint in enumerate(spring.joints):
            joint.drag_force = min(.95, joint.drag_force + .25)
            if group.drag > 1e-8:
                joint['hallway_generated_drag_ratio'] = joint.drag_force / group.drag
            if index == len(spring.joints) - 1:
                continue  # VRM's terminal joint is a tail, not another solver segment.
            t = max(0., min(1., (ends.get(joint.node.bone_name, 0.) - .65) / .35))
            t = t * t * (3 - 2 * t)
            before = joint.stiffness
            joint.stiffness = max(before, base * .35 * t)
            if group.stiffness > 1e-8:
                joint['hallway_stiffness_ratio'] = joint.stiffness / group.stiffness
            if joint.stiffness != before:
                tuned.append(joint.node.bone_name)
    result = dict(version=1, merged=len(merged), reanchored=len(anchored), tuned=len(tuned),
                  merge_checks=merged, rejected_merges=rejected, anchors=anchored, tuned_bones=tuned)
    rig[key] = json.dumps(result)
    return result
