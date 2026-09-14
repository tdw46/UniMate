"""Small shoulder weight fairing in a skeleton-relative front/back sector.

Requires baked rig-local coordinates, as does the fresh apparel binding pass.
Only body and torso components participate; source geometry is never edited.
"""
import math

from mathutils.kdtree import KDTree

from avatar_apparel_weights import assign, classify_regions, constrain, smooth, weights


def shoulder_mask(rig, point):
    bones = rig.data.bones
    up = (bones['Neck'].tail_local-bones['Chest'].head_local).normalized()
    lateral = (bones['UpperArm.L'].head_local-bones['UpperArm.R'].head_local).normalized()
    front = lateral.cross(up).normalized()
    best = (0., 0.)
    for side in ('L', 'R'):
        arm = bones['UpperArm.'+side]
        axis = (arm.tail_local-arm.head_local).normalized()
        vertical = axis.cross(front).normalized()
        delta = point-arm.head_local
        along = delta.dot(axis)/arm.length
        depth, height = delta.dot(front)/arm.length, delta.dot(vertical)/arm.length
        radius = math.hypot(depth, height)
        # Front/back sectors only: exactly zero on top and underarm sectors.
        sector = smooth(.78, .96, abs(depth)/max(radius, 1e-12))
        axial = smooth(-.65, -.25, along)*(1-smooth(.25, .65, along))
        radial = smooth(.035, .10, radius)*(1-smooth(.45, .70, radius))
        shoulder_width = abs((arm.head_local-bones['Neck'].head_local).dot(lateral))
        lateral_distance = abs((point-bones['Neck'].head_local).dot(lateral))
        collar_exclusion = smooth(.60, .85, lateral_distance/max(shoulder_width, 1e-12))
        value = sector*axial*radial*collar_exclusion
        if value > best[0]:
            best = (value, arm.length)
    return best


def smooth_shoulders(meshes, rig, rigid_parts, strength=.45, iterations=3):
    """Area-weighted local averaging, bounded by fixed anatomical mask.

    Neighborhoods stay inside each connected surface and one facing side, so
    layered clothes, jewelry and opposing surfaces do not exchange weights.
    All coordinates, radii and angular exclusions come from the fresh rig.
    """
    _, regions = classify_regions(meshes, rig, rigid_parts)
    report = {'method': 'area-weighted shoulder front/back fairing',
              'strength': strength, 'iterations': iterations, 'regions': []}
    for region in regions:
        obj, ids, role = region['object'], region['indices'], region['role']
        if role not in ('body', 'torso') or obj.name in rigid_parts:
            continue
        points = {i: obj.data.vertices[i].co.copy() for i in ids}
        values = {i: weights(obj, i) for i in ids}
        masks = {i: shoulder_mask(rig, points[i]) for i in ids}
        eligible = {i for i in ids if values[i].get('Head', 0.)+values[i].get('Neck', 0.) < 1e-8}
        selected = [i for i in ids if masks[i][0] > 1e-6 and i in eligible]
        if not selected:
            continue
        before = {i: dict(v) for i, v in values.items()}
        areas = dict.fromkeys(ids, 0.)
        id_set = set(ids)
        for face in obj.data.polygons:
            if all(i in id_set for i in face.vertices):
                for i in face.vertices:
                    areas[i] += face.area/len(face.vertices)
        tree = KDTree(len(ids))
        for i in ids:
            tree.insert(points[i], i)
        tree.balance()
        neighbors = {}
        for i in selected:
            radius = masks[i][1]*.22
            normal = obj.data.vertices[i].normal
            near = [(j, areas[j]*math.exp(-4*(distance/radius)**2))
                    for _, j, distance in tree.find_range(points[i], radius)
                    if j in eligible and normal.dot(obj.data.vertices[j].normal) > .25 and areas[j] > 0]
            total = sum(w for _, w in near)
            neighbors[i] = [(j, w/total) for j, w in near] if total else [(i, 1.)]
        for _ in range(iterations):
            updated = dict(values)
            for i in selected:
                mean = {}
                for j, factor in neighbors[i]:
                    for bone, value in values[j].items():
                        mean[bone] = mean.get(bone, 0.)+value*factor
                blend = strength*masks[i][0]
                mixed = {b: values[i].get(b, 0.)*(1-blend)+mean.get(b, 0.)*blend
                         for b in values[i].keys() | mean.keys()}
                updated[i] = constrain(rig, points[i], mixed, clothing=role == 'torso')
            values = updated
        for i in selected:
            assign(obj, [i], values[i])
        delta = [max(abs(values[i].get(b, 0.)-before[i].get(b, 0.))
                     for b in values[i].keys() | before[i].keys()) for i in selected]
        report['regions'].append({'mesh': obj.name, 'role': role,
                                  'selected_vertices': len(selected),
                                  'changed_vertices': sum(d > 1e-6 for d in delta),
                                  'max_weight_change': max(delta),
                                  'mean_weight_change': sum(delta)/len(delta)})
    return report
