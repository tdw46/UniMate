"""Rest-clear, chain-scoped leg coverage for hanging garment panels.

Fit the newly skinned leg surfaces rather than requiring a skin material. Round
full-length capsules cover yaw; a compact offset capsule on the facing leg adds
broad support like the dress contacts. All shapes are standard VRM capsules.
"""
import numpy as np
from mathutils import Vector
from avatar_apparel_weights import weights


def _clearance(points, hits, a, b):
    axis = b-a
    t = np.clip((points-a)@axis / max(float(axis@axis), 1e-20), 0., 1.)
    return float((np.linalg.norm(points-a-t[:, None]*axis, axis=1)-hits).min())


def fit_directional_capsule(bone, points, hits, normal, body_radius, margin, spread):
    """Fit a finite curved contact face without touching its cloth in rest."""
    a = np.zeros(3)
    b = np.array((0., bone.length, 0.))
    radius = (1. + 1.5*spread) * body_radius
    low, high = -radius, body_radius

    def clearance(face):
        offset = normal*(face-radius)
        return _clearance(points, hits, a+offset, b+offset)-radius

    if clearance(low) < margin:
        return None
    for _ in range(20):
        mid = (low+high)*.5
        if clearance(mid) >= margin:
            low = mid
        else:
            high = mid
    if low < body_radius*.35:
        return None
    offset = normal*(low-radius)
    return dict(bone=bone.name, offset=(a+offset).tolist(), tail=(b+offset).tolist(),
                radius=radius, minimum_rest_clearance=clearance(low), kind='directional')


def plan_leg_contacts(rig, meshes, chains, height, prefix):
    hum = rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    legs = {}
    for side in ('left', 'right'):
        for part in ('upper_leg', 'lower_leg'):
            name = getattr(hum, side+'_'+part).node.bone_name
            if name in rig.data.bones:
                legs[name] = dict(side=side, points=[])
    calf_names = {side: getattr(hum, side+'_lower_leg').node.bone_name for side in ('left', 'right')}
    inverses = {name: rig.data.bones[name].matrix_local.inverted_safe() for name in legs}
    springs = {s.vrm_name: s for s in rig.data.vrm_addon_extension.spring_bone1.springs}
    garments = {c.get('spring_name', c['names'][0].rsplit('_', 1)[0]): c
                for c in chains if c.get('garment')}
    joint_chain = {n: name for name, c in garments.items() for n in c['names']}
    surfaces = {name: [] for name in garments}
    for obj in meshes:
        transform = rig.matrix_world.inverted_safe()@obj.matrix_world
        for vertex in obj.data.vertices:
            p = transform@vertex.co
            row = weights(obj, vertex.index)
            for name in {joint_chain[n] for n,w in row.items() if w>.05 and n in joint_chain}:
                surfaces[name].append(tuple(p))
            if any(n.startswith(prefix) and w>.05 for n,w in row.items()):
                continue
            for name, data in legs.items():
                if row.get(name, 0.)>.4:
                    data['points'].append(tuple(inverses[name]@p))
    for name, data in legs.items():
        bone = rig.data.bones[name]
        points = np.asarray(data.pop('points'), dtype=float).reshape(-1,3)
        band = points[(points[:,1]>.12*bone.length)&(points[:,1]<.88*bone.length)]
        # Torso clothing and shoes cannot enlarge the mid-limb estimate.
        radius = float(np.quantile(np.linalg.norm(band[:,(0,2)],axis=1),.90)) if len(band)>=8 else .16*bone.length
        data.update(radius=min(radius,.45*bone.length), measured_vertices=len(band))
    result = {}; margin = height*.0015
    thigh_names = [n for n in (hum.left_upper_leg.node.bone_name, hum.right_upper_leg.node.bone_name)
                   if n in legs]
    hips = rig.data.bones.get(hum.hips.node.bone_name) or rig.data.bones.get('Hips')
    if hips and len(thigh_names) == 2:
        lateral = rig.data.bones[thigh_names[0]].head_local - rig.data.bones[thigh_names[1]].head_local
        lateral.z = 0.
        lateral.normalize()
        forward = lateral.cross(Vector((0., 0., 1.)))
    else:
        forward = Vector()
    for name, chain in garments.items():
        spring = springs.get(name)
        if spring is None or chain.get('coat_torso'):
            continue
        samples = [(rig.data.bones[t.node.bone_name].head_local.copy(), h.hit_radius)
                   for h,t in zip(spring.joints,spring.joints[1:])]
        if not samples:
            continue
        points = np.asarray([tuple(p) for p,r in samples]+surfaces[name])
        hits = np.asarray([r for p,r in samples]+[0.]*len(surfaces[name]))
        center = np.mean([tuple(p) for p,r in samples], axis=0)
        owner = min(thigh_names,key=lambda n:abs(center[0]-rig.data.bones[n].head_local.x)) if thigh_names else None
        front = 0.
        if hips:
            radial = Vector(center) - hips.head_local
            radial.z = 0.
            front = radial.normalized().dot(forward)
        # Compact front faces let the hem slide over a lifted knee while
        # keeping support between the chains. Side and rear retain wider faces.
        hem = chain.get('coat_hem', min(p.z for p, _ in samples))
        calf = rig.data.bones.get(calf_names[legs[owner]['side']]) if owner else None
        reaches_calf = calf and hem < (calf.head_local.z + calf.tail_local.z)*.5
        knee_length_coat = chain.get('coat_lower_root') and not reaches_calf
        taper = 0. if reaches_calf else min(1., max(0., (front-.30)/.45))
        directional = 1. - .65*taper*taper*(3.-2.*taper)
        specs = []
        for bone_name, data in legs.items():
            bone = rig.data.bones[bone_name]; length = bone.length
            if min(p.z for p,r in samples)>max(bone.head_local.z,bone.tail_local.z)+data['radius']:
                continue
            if all((p-bone.head_local).length>length+data['radius']+r for p,r in samples):
                continue
            inv = bone.matrix_local.inverted_safe()
            local = np.asarray([tuple(inv@Vector(p)) for p in points])
            a=np.zeros(3);b=np.array((0.,length,0.))
            radius=min(data['radius'],_clearance(local,hits,a,b)-margin)
            if radius>height*.001:
                specs.append(dict(bone=bone_name,offset=a.tolist(),tail=b.tolist(),radius=radius,
                                  minimum_rest_clearance=_clearance(local,hits,a,b)-radius,kind='round'))
            if owner is None or data['side']!=legs[owner]['side']:
                continue
            # A broad shin capsule reaches back over the knee as it bends.
            # A coat that ends above mid-calf needs the rounded fallback there.
            if knee_length_coat and bone_name in calf_names.values():
                continue
            mid=inv@Vector(center);normal=np.array((mid.x,0.,mid.z));norm=float(np.linalg.norm(normal))
            if norm<height*.001:
                continue
            normal/=norm
            spec = fit_directional_capsule(bone, local, hits, normal, data['radius'], margin, directional)
            if spec:
                specs.append(spec)
        result[name]=specs
    return result, legs
