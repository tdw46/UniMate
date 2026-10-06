"""Hips-mounted, outward-sloped capsule ceilings for generated skirt springs.

Only collider metadata is changed. Each roof lies outside its resting collision
samples and is scoped to its own chain. The outward slope releases folds that a
horizontal lid can trap above the leg. Hair and body weights stay intact.
"""
import math
from mathutils import Vector
from avatar_vrm_colliders import add_capsule, add_group, call_operator, cleanup_orphan_displays, organize_colliders

GROUP_PREFIX = 'Secondary_SkirtCeiling_'
OUTWARD_SLOPE = .7


def plan_ceiling(rig, outward_slope=OUTWARD_SLOPE):
    sb = rig.data.vrm_addon_extension.spring_bone1
    hips = rig.data.vrm_addon_extension.vrm1.humanoid.human_bones.hips.node.bone_name
    if hips not in rig.data.bones:
        raise ValueError('A mapped hips bone is required for skirt ceiling contacts')
    height = max(b.head_local.z for b in rig.data.bones)-min(b.head_local.z for b in rig.data.bones)
    margin = max(height*.0015, 1e-5)
    inverse = rig.data.bones[hips].matrix_local.inverted()
    families = {}
    for spring in sb.springs:
        if spring.vrm_name.startswith('Secondary_Skirt_') and len(spring.joints) > 1:
            root=rig.data.bones[spring.joints[0].node.bone_name]
            family=root.get('hallway_dress_upper_spring',spring.vrm_name)
            families.setdefault(family if outward_slope else family.rsplit('_', 1)[0], []).append(spring)
    plans = []
    for family, springs in families.items():
        attachments=[s for s in springs if not rig.data.bones[s.joints[0].node.bone_name].get('hallway_dress_lower')]
        roots = [rig.data.bones[s.joints[0].node.bone_name].head_local for s in attachments]
        samples = [(rig.data.bones[t.node.bone_name].head_local, h.hit_radius)
                   for s in springs for h,t in zip(s.joints,s.joints[1:])]
        center = rig.data.bones[hips].head_local
        radial = sum((p-center for p in roots),Vector())/len(roots)
        radial.z = 0.
        if radial.length_squared < 1e-12:
            radial = Vector((1,0,0))
        else:
            radial.normalize()
        normal = (radial*outward_slope+Vector((0,0,-1))).normalized()
        bound = min(p.dot(normal)-r for p,r in samples)-margin
        anchor = sum(roots,Vector())/len(roots)
        anchor += normal*(bound-anchor.dot(normal))
        # A roof fitted only to simulated tips can sit below the waistband
        # and squeeze a raised thigh against the skirt. Keep its face above
        # the garment attachment instead. Older rigs without garment bounds
        # use their chain roots. Only translate upward: retain the slope,
        # radial coverage, slab depth and the existing rest-clear envelope.
        garment_top = max(max(float(rig.data.bones[s.joints[0].node.bone_name].get(
            'hallway_garment_top', root.z)), root.z) for root,s in zip(roots,attachments))
        minimum_height = garment_top + max(r for p,r in samples) + margin
        lift = max(0., minimum_height-anchor.z)
        anchor.z += lift
        tangent = Vector((-radial.y,radial.x,0)) if outward_slope else Vector((1,0,0))
        across = normal.cross(tangent).normalized()
        z = anchor.z
        # Cover a conservative reachable disk around the chain attachment.
        from avatar_dress import full_chain_names
        lengths=[sum(rig.data.bones[n].length for n in full_chain_names(rig,s)[:-1]) for s in attachments]
        reach = max((p-center).length+length for p,length in zip(roots,lengths)) + max(r for p,r in samples) + margin
        # Row spacing <= radius leaves a slab at least sqrt(3)*radius
        # thick everywhere. Size the blocked side for the whole chain,
        # not just one segment: a folded chain can cross a thinner slab.
        # Increasing depth leaves the supporting face and rest clearance put.
        longest = max(rig.data.bones[j.node.bone_name].length for s in springs for j in s.joints[:-1])
        chain_length = max(lengths)
        radius = max(height*.035, reach/24., longest*.75+margin, chain_length*.85+max(r for p,r in samples))
        count = max(2, math.ceil(2*reach/radius)+1)
        specs = []
        for i in range(count):
            row = anchor+across*(-reach+2*reach*i/(count-1))-normal*radius
            a = row-tangent*reach
            b = row+tangent*reach
            specs.append(dict(bone=hips, offset=list(inverse@a), tail=list(inverse@b), radius=radius))
        plans.append(dict(name=GROUP_PREFIX+family[len('Secondary_Skirt_'):], springs=[s.vrm_name for s in springs],
                          colliders=specs, height=z, normal=list(inverse.to_3x3()@normal),
                          offset=list(inverse@anchor), margin=margin,
                          garment_top=garment_top, vertical_lift=lift))
    return plans


def install_skirt_ceiling(rig, outward_slope=OUTWARD_SLOPE):
    """Idempotent native VRM capsule roof; caller owns rest/solver state."""
    plans = plan_ceiling(rig, outward_slope)
    sb = rig.data.vrm_addon_extension.spring_bone1
    owned = {g.uuid for g in sb.collider_groups if g.vrm_name.startswith(GROUP_PREFIX)}
    if any(ref.collider_group_uuid in owned for s in sb.springs if not s.vrm_name.startswith('Secondary_Skirt_') for ref in s.collider_groups):
        raise ValueError('An artist spring references generated skirt ceiling contacts')
    ids = {r.collider_uuid for g in sb.collider_groups if g.uuid in owned for r in g.colliders}
    foreign = {r.collider_uuid for g in sb.collider_groups if g.uuid not in owned for r in g.colliders}
    for i in reversed(range(len(sb.collider_groups))):
        if sb.collider_groups[i].uuid in owned:call_operator('remove_spring_bone1_collider_group', rig, collider_group_index=i)
    for i in reversed(range(len(sb.colliders))):
        if sb.colliders[i].uuid in ids-foreign:call_operator('remove_spring_bone1_collider', rig, collider_index=i)
    lookup = {s.vrm_name:s for s in sb.springs}
    for plan in plans:
        uuids = []
        for spec in plan['colliders']:
            c = add_capsule(rig, spec, organize=False)
            c.bpy_object['hallway_skirt_ceiling'] = True
            uuids.append(c.uuid)
        group = add_group(rig, plan['name'])
        for uuid in uuids:group.colliders.add().collider_uuid=uuid
        for name in plan['springs']:lookup[name].collider_groups.add().collider_group_uuid=group.uuid
    organize_colliders(rig)
    cleanup_orphan_displays(rig)
    rig['hallway_skirt_ceiling']=2
    return dict(outward_slope=outward_slope,groups=len(plans),capsules=sum(len(p['colliders']) for p in plans),
                underside_heights=[p['height'] for p in plans],portable=True)
