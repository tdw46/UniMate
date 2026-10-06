"""Rest-fit-dependent contact policy for narrow, full-length dresses.

Keep native VRM capsules and chain ownership. A broad rounded outward face
supports the cloth near the calf; its back remains inside the contact volume.
Opposite-leg and fixed opposite-pelvis guards compete with that face on return
and are excluded for tight bilateral dress sections.
"""
from mathutils import Vector
from avatar_colliders import segment_distance


def family_fit(rig, meshes, body_details):
    from avatar_dress_fit import measure_dress_fit
    report=measure_dress_fit(rig,meshes,body_details)
    return {row['family']:{section:values['tightness'] for section,values in row['sections'].items()}
            for row in report['families']}


def fit_for_spring(rig, spring, fits):
    root=rig.data.bones[spring.joints[0].node.bone_name]
    name=root.get('hallway_dress_upper_spring',spring.vrm_name)
    return fits.get(name.rsplit('_',1)[0],{})


def is_tight(fit):
    # A close waist alone must not classify a flared gown as a narrow tube.
    return fit.get('lower',0.) >= .5


def corner_normals(center):
    """Orthogonal outward faces in the owning leg's local transverse plane."""
    return [Vector((1. if center.x>=0 else -1.,0.,0.)),
            Vector((0.,0.,1. if center.z>=0 else -1.))]


def prune_competing_contacts(payload, group, allowed):
    """Prune references only, then let the existing owner cleanup remove orphans."""
    group['colliders']=[i for i in group['colliders'] if payload['colliders'][i]['bone'] in allowed]
    # A contained capsule adds no coverage, but a sequential spring solver can
    # project into it again after enforcing segment length. Keep the outer one.
    rows=list(dict.fromkeys(group['colliders']))
    removed=set()
    for index in rows:
        a=payload['colliders'][index]
        for other in rows:
            if other==index or other in removed:continue
            b=payload['colliders'][other]
            if a['bone']!=b['bone'] or a['radius']>b['radius']+1e-8:continue
            distance=max(segment_distance(Vector(p),Vector(b['offset']),Vector(b['tail']))
                         for p in (a['offset'],a['tail']))
            # Scale tolerance to the shapes, preserving tiny avatars too.
            tolerance=max(a['radius'],b['radius'])*1e-5
            if distance+a['radius']<=b['radius']+tolerance:
                removed.add(index);break
    group['colliders']=[i for i in rows if i not in removed]
    return len(rows)-len(group['colliders'])
