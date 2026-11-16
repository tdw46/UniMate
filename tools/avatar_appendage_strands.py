"""Fit free strands to isolated rest surfaces and their anatomical mounts."""
import numpy as np
from mathutils import Vector

from avatar_garment_straps import _centerline, _free_weights
from avatar_virtual_topology import connected_graph


def _mount(points, hips, arms, height):
    low, high = points.min(axis=0), points.max(axis=0)
    near_back = points[points[:, 1] <= low[1] + .05 * height]
    nearest = points[np.linalg.norm(points - hips, axis=1).argmin()]
    # A tail leaves the middle of the posterior pelvis. Require the free
    # portion to project beyond the rump; a panel hanging from the waist
    # or from the back of the head has a different attachment.
    if (abs(nearest[0] - hips[0]) < .07 * height
            and 0 < nearest[1] - hips[1] < .13 * height
            and abs(nearest[2] - hips[2]) < .105 * height
            and near_back[:, 2].max() < hips[2] + .13 * height
            and low[1] > hips[1] - .012 * height
            and high[1] - nearest[1] > .18 * height):
        distal = points[np.linalg.norm(points - nearest, axis=1).argmax()]
        axis = nearest - distal
        length = np.linalg.norm(axis)
        # A broad cape or apron can share the tail's mount and length.
        if high[0] - low[0] < .75 * length:
            return 'tail', 'Hips', axis / length, nearest

    # An arm ribbon is mounted along one upper arm, with a narrow free
    # length extending below the elbow. Reject sleeves and broad coat panels.
    for role, name, shoulder, elbow in arms:
        if not (elbow[2] - low[2] > .09 * height
                and abs(high[2] - shoulder[2]) < .085 * height):
            continue
        delta = elbow - shoulder
        t = np.clip((points - shoulder) @ delta / max(delta @ delta, 1e-16), 0, 1)
        distance = np.linalg.norm(points - (shoulder + t[:, None] * delta), axis=1)
        if distance.min() > .045 * height:
            continue
        run = high[2] - low[2]
        for fraction in (.2, .4, .6):
            band = points[abs(points[:, 2] - (low[2] + fraction * run)) < .075 * run]
            if len(band) < 2 or np.linalg.norm(np.ptp(band[:, :2], axis=0)) > .12 * height:
                break
        else:
            # Start the free web below the attachment wrap, as for a strap
            # lying against the top of a garment.
            mount_z = min(shoulder[2], high[2] - .1 * run)
            return role, name, np.array([0., 0., 1.]), np.array([0., 0., mount_z])
    return None


def plan_strands(rig, obj, points, height, prefix, start, excluded=frozenset()):
    if len(points) < 8 or np.max(np.ptp(points, axis=0)) < .17 * height:
        return [], []
    bones = rig.data.bones
    hips = np.array(bones['Hips'].head_local)
    mapped = rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    arms = []
    for side, suffix in (('left', 'L'), ('right', 'R')):
        name = getattr(mapped, side + '_upper_arm').node.bone_name or 'UpperArm.' + suffix
        bone = bones.get(name)
        if bone:
            arms.append(('armstrap_' + suffix.lower(), name, np.array(bone.head_local),
                         np.array(bone.tail_local)))
    # Use the same seam connectivity as hanging hair. Only accept whole
    # islands, so a ribbon touching a sleeve cannot cut through the sleeve.
    _, islands, _ = connected_graph(obj.data, points, range(len(points)), height * .0001)
    chains, regions = [], []
    triangles = None
    for ids in islands:
        if len(ids) < 8 or not excluded.isdisjoint(ids):
            continue
        sample = points[ids]
        if np.max(np.ptp(sample, axis=0)) < .17 * height:
            continue
        mount = _mount(sample, hips, arms, height)
        if mount is None:
            continue
        role, parent, axis, root = mount
        lateral = np.cross(axis, (1., 0., 0.))
        if np.linalg.norm(lateral) < 1.e-8:
            lateral = np.cross(axis, (0., 1., 0.))
        lateral /= np.linalg.norm(lateral)
        basis = np.stack((np.cross(lateral, axis), lateral, axis))
        along = sample @ axis
        top = min(along.max(), root @ axis)
        bottom = along.min()
        if top - bottom < .12 * height:
            continue
        if triangles is None:
            obj.data.calc_loop_triangles()
            triangles = np.array([tuple(t.vertices) for t in obj.data.loop_triangles], dtype=int).reshape(-1, 3)
        faces = triangles[np.isin(triangles[:, 0], ids)]
        count = int(np.clip(np.ceil((top - bottom) / (height * (.026 if role == 'tail' else .031))), 4, 40))
        levels = np.linspace(top, bottom, count + 1)
        sections = _centerline(points[faces] @ basis.T, levels)
        if not sections:
            continue
        line = [Vector(np.asarray(section) @ basis) for section in sections]
        names = [f'{prefix}{role.title()}_{start+len(chains):02d}_{i:02d}' for i in range(len(line))]
        controls = [parent, *names[:-1]]
        degree = 3
        spans = len(controls) - degree
        knots = [0] * degree + list(range(spans + 1)) + [spans] * degree
        values = {i: _free_weights(controls, degree, spans, knots,
                                   float(np.clip((top - p) / (top - bottom), 0, 1)))
                  for i, p in zip(ids, along)}
        chains.append(dict(role=role, names=names, line=line, parent=parent, free_appendage=True))
        regions.append(dict(obj=obj, role=role, ids=set(ids), values=values))
    from avatar_trailing_surfaces import plan_trailing_surfaces
    occupied = set(excluded)
    for region in regions:
        occupied.update(region['ids'])
    extra, bindings = plan_trailing_surfaces(rig, obj, points, height, prefix, start+len(chains),
                                             islands, occupied)
    chains.extend(extra)
    regions.extend(bindings)
    return chains, regions
