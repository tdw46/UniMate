"""Continuous torso-to-sleeve weights in canonical, baked rig coordinates."""
from mathutils import Vector
from avatar_apparel_weights import smooth, constrain, torso_distances


def loose_front_strength(region, rig, surface):
    """Area-weighted clearance distinguishes a loose panel from fitted cloth."""
    if region['role'] != 'torso':
        return 0., 0.
    neck = rig.data.bones['Neck']
    if not (region['low'].x < rig.data.bones['UpperArm.R'].head_local.x
            and region['high'].x > rig.data.bones['UpperArm.L'].head_local.x):
        return 0., 0.
    obj = region['object']; ids = set(region['indices'])
    obj.data.calc_loop_triangles()
    samples = []
    for triangle in obj.data.loop_triangles:
        if not all(i in ids for i in triangle.vertices): continue
        a,b,c = (obj.data.vertices[i].co for i in triangle.vertices)
        p = (a+b+c)/3
        arm = rig.data.bones['UpperArm.'+('L' if p.x >= neck.head_local.x else 'R')]
        lateral = (abs(p.x-neck.head_local.x)-abs(arm.head_local.x-neck.head_local.x))/arm.length
        if not (0 < lateral < .4 and p.y < neck.head_local.y-neck.length*.25
                and neck.head_local.z-neck.length*2.5 < p.z < neck.head_local.z):
            continue
        hit = surface.tree.find_nearest(p)
        if hit[0] is not None:
            samples.append((hit[3]/neck.length, (b-a).cross(c-a).length*.5))
    total = sum(area for _,area in samples)
    if total <= 1e-12: return 0., 0.
    cumulative = 0.
    for clearance, area in sorted(samples):
        cumulative += area
        if cumulative >= total*.9: break
    return smooth(.25, .65, clearance), clearance


def front_torso_transition(region, rig, source, strength=1.):
    """Broaden front torso attachment on body and sleeve shells together.

    Lateral progress and height below the humerus keep low chest cloth attached
    to the torso. A radial angular mask excludes the back, top and underside;
    it does not lock every corner of a triangle touching those surfaces. The
    caller shares clearance-derived strength across body and clothing layers.
    """
    neck = rig.data.bones['Neck']
    if not (region['role'] in ('torso', 'body')
            and region['low'].x < rig.data.bones['UpperArm.R'].head_local.x
            and region['high'].x > rig.data.bones['UpperArm.L'].head_local.x
            and region['low'].z < neck.head_local.z-neck.length*.35):
        return source, {}
    result = {}; changed = {}
    for i in region['indices']:
        p = region['object'].data.vertices[i].co
        original = source[i]
        side = 'L' if p.x >= neck.head_local.x else 'R'
        arm = rig.data.bones['UpperArm.'+side]
        axis = (arm.tail_local-arm.head_local).normalized()
        delta = p-arm.head_local
        radial = delta-axis*delta.dot(axis)
        lateral = (abs(p.x-neck.head_local.x)-abs(arm.head_local.x-neck.head_local.x))/arm.length
        front = -radial.y/max(radial.length, 1e-12)
        mask = strength*smooth(.15, .5, front)
        mask *= 1-smooth(.9, 1.2, lateral)
        mask *= smooth(-.4, -.05, lateral)
        mask *= smooth(0., neck.length*.2, p.z-region['low'].z)
        mask *= 1-smooth(neck.head_local.z-neck.length*.2, neck.head_local.z+neck.length*.2, p.z)
        if mask <= 1e-8:
            result[i] = original
            continue
        limb = {n:w for n,w in original.items() if n in
                ('UpperArm.'+side, 'Forearm.'+side, 'Hand.'+side)}
        torso = {n:w for n,w in original.items() if n in
                 ('Root', 'Spine', 'Chest', 'Clavicle.L', 'Clavicle.R')}
        if sum(torso.values()) < 1e-6:
            medial = p.copy(); medial.x = neck.head_local.x
            torso = {n:w for n,w in torso_distances(rig, medial).items()
                     if n in ('Root', 'Spine', 'Chest')}
        if not limb:
            limb = {'UpperArm.'+side: 1.}
        up = Vector((0,0,1))-axis*axis.z
        height = delta.dot(up.normalized())/arm.length if up.length > 1e-8 else 0.
        target = smooth(.15, .7, lateral+height*.8)
        reserved = {n:w for n,w in original.items() if n in ('Head', 'Neck')}
        available = 1-sum(reserved.values())
        fitted = dict(reserved)
        for group, mass in ((limb, target), (torso, 1-target)):
            total = sum(group.values())
            fitted.update({n: w/total*mass*available for n,w in group.items()})
        values = {n: original.get(n,0)*(1-mask)+fitted.get(n,0)*mask
                  for n in original.keys() | fitted.keys()}
        result[i] = constrain(rig, p, values, clothing=region['role']=='torso')
        changed[i] = mask
    return result, changed
