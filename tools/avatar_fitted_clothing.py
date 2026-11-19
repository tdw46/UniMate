"""Transfer generated torso weights to close-fitting clothing in rig rest space."""
import re
import sys

import numpy as np
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform
from mathutils import Vector

from avatar_apparel_weights import weights
from avatar_limb_domains import _write
from avatar_material_roles import semantic_label


def _role(name):
    words = set(re.findall(r'[^\W_]+', semantic_label(name)))
    if words & {'shirt', 'tshirt', 'undershirt', 'blouse', 'topwear'}:
        return 'shirt'
    if words & {'pants', 'trousers', 'jeans', 'shorts', 'slacks'}:
        return 'waist'
    if words & {'skin', 'body', 'torso'}:
        return 'body'
    return None


def _samples(points, vertices, faces):
    # Use the enabled Robust Weight Transfer implementation when available.
    # The standalone path needs only Blender and uses the same surface samples.
    for name, module in tuple(sys.modules.items()):
        if (name.rsplit('.', 2)[-2:] in (['robust-weight-transfer', 'weighttransfer'],
                                         ['robust_weight_transfer', 'weighttransfer'])
                and hasattr(module, 'find_closest_point_on_surface')):
            d2, face, _, bary = module.find_closest_point_on_surface(points, vertices, faces)
            return d2, face, bary
    tree = BVHTree.FromPolygons(vertices, faces, all_triangles=True)
    basis = tuple(Vector(row) for row in np.eye(3))
    distance, indices, coords = [], [], []
    for p in points:
        hit, _, i, dist = tree.find_nearest(p)
        tri = tuple(Vector(vertices[k]) for k in faces[i])
        if (tri[1]-tri[0]).cross(tri[2]-tri[0]).length_squared < 1e-24:
            bary = basis[min(range(3), key=lambda k: (tri[k]-hit).length_squared)]
        else:
            bary = barycentric_transform(hit, *tri, *basis)
        distance.append(dist * dist)
        indices.append(i)
        coords.append(tuple(bary))
    return np.asarray(distance), np.asarray(indices), np.asarray(coords)


def _transfer(targets, donors, allowed, radius):
    vertices, rows, triangles = [], [], []
    for obj, faces in donors:
        lookup = {}
        for tri in faces:
            corners = []
            for i in tri:
                if i not in lookup:
                    row = weights(obj, i)
                    lookup[i] = -1
                    if row and row.keys() <= allowed:
                        lookup[i] = len(rows)
                        vertices.append(tuple(obj.data.vertices[i].co))
                        rows.append(row)
                corners.append(lookup[i])
            if min(corners) >= 0:
                triangles.append(corners)
    if not triangles or not targets:
        return {}
    vertices, faces = np.array(vertices), np.array(triangles)
    counts = {}
    for obj, ids in targets:
        points = np.array([obj.data.vertices[i].co for i in ids])
        distances, indices, bary = _samples(points, vertices, faces)
        changes = 0
        for i, d2, face, factors in zip(ids, distances, indices, bary):
            if d2 >= radius**2:
                continue
            old = weights(obj, i)
            if not old or old.keys() - allowed:
                continue  # A spring, lower leg or accessory owns this vertex.
            u = min(1., 2. * (1. - np.sqrt(d2) / radius))
            mix = u * u * (3. - 2. * u)
            row = {n: w * (1. - mix) for n, w in old.items()}
            for source, fraction in zip(faces[face], np.clip(factors, 0., 1.)):
                for n, w in rows[source].items():
                    row[n] = row.get(n, 0.) + mix * fraction * w
            top = sorted(((n, w) for n, w in row.items() if w > 1e-8),
                         key=lambda v: (-v[1], v[0]))[:4]
            total = sum(w for _, w in top)
            if total:
                changes += _write(obj, i, old, {n: w / total for n, w in top})
        counts[obj.name] = changes
    return counts


def _waist_contact(rig, body, shirts, waists, allowed, scale):
    """Give touching layers the same pelvic frame at the top of fitted trousers."""
    if not body or not shirts or not waists:
        return {}
    center = rig.data.bones['Hips'].head_local
    counts = {}
    skin = [(obj, sorted({i for tri in faces for i in tri})) for obj, faces in body]
    for pants, ids in waists:
        top = max(pants.data.vertices[i].co.z for i in ids)
        if abs(top - center.z) > scale * .35:
            continue
        ring = [pants.data.vertices[i].co for i in ids
                if pants.data.vertices[i].co.z >= top - scale * .07]
        if not ring:
            continue
        xy = np.array([p[:2] for p in ring])
        lo, hi = xy.min(0) - scale * .04, xy.max(0) + scale * .04
        def at_waist(point):
            return (lo[0] <= point.x <= hi[0] and lo[1] <= point.y <= hi[1]
                    and abs(point.z - top) <= scale * .20)
        contacts = [(obj, selected) for obj, verts in shirts
                    if (selected := [i for i in verts if at_waist(obj.data.vertices[i].co)])]
        if not any(abs(obj.data.vertices[i].co.z - top) < scale * .07
                   for obj, verts in contacts for i in verts):
            continue
        for obj, verts in [*skin, *contacts, (pants, ids)]:
            changed = 0
            for i in verts:
                p = obj.data.vertices[i].co
                if not at_waist(p):
                    continue
                old = weights(obj, i)
                if not old or old.keys() - allowed:
                    continue
                t = min(1., max(0., (abs(p.z - top) / scale - .07) / .13))
                keep = t * t * (3. - 2. * t)
                row = {n: w * keep for n, w in old.items()}
                row['Hips'] = row.get('Hips', 0.) + 1. - keep
                top_four = sorted(((n, w) for n, w in row.items() if w > 1e-8),
                                  key=lambda item: (-item[1], item[0]))[:4]
                total = sum(w for _, w in top_four)
                changed += _write(obj, i, old, {n: w / total for n, w in top_four})
            counts[obj.name] = counts.get(obj.name, 0) + changed
    return counts


def transfer_fitted_clothing(rig, meshes):
    bones = rig.data.bones
    torso = {'Hips', 'Spine', 'UpperSpine', 'Chest', 'Neck',
             'Clavicle.L', 'Clavicle.R', 'UpperArm.L', 'UpperArm.R', 'Thigh.L', 'Thigh.R'}
    radius = (bones['Head'].head_local - bones['Hips'].head_local).length * .14
    if radius <= 1e-7:
        return {}
    body, pants, shirts, waists = [], [], [], []
    for obj in meshes:
        mesh = obj.data
        mesh.calc_loop_triangles()
        roles = [_role(m.name) if m else None for m in mesh.materials]
        # A more specific material (e.g. shoes on a pants object) owns its faces.
        if len(roles) <= 1:
            roles = [(roles[0] if roles else None) or _role(obj.name)]
        regions = {'body': [], 'shirt': [], 'waist': []}
        for tri in mesh.loop_triangles:
            role = roles[tri.material_index] if tri.material_index < len(roles) else None
            if role:
                regions[role].append(tuple(tri.vertices))
        if regions['body']:
            body.append((obj, regions['body']))
        for label, targets in (('waist', waists), ('shirt', shirts)):
            if regions[label]:
                targets.append((obj, sorted({i for tri in regions[label] for i in tri})))
        if regions['waist']:
            pants.append((obj, regions['waist']))
    counts = _transfer(waists + shirts, body, torso, radius)
    # A shirt touching the waistband has two interfaces. Sample the already
    # bound waistband locally at the hem, fading back to skin farther up.
    pelvis = {'Hips', 'Spine', 'UpperSpine', 'Chest', 'Thigh.L', 'Thigh.R'}
    for name, count in _transfer(shirts, pants, pelvis, radius * .44).items():
        counts[name] = counts.get(name, 0) + count
    # Return the shared field to close torso skin where a coarser shirt
    # triangle spans several body vertices. Exposed limbs retain their bind.
    shirt_faces = []
    for obj, ids in shirts:
        selected = set(ids)
        shirt_faces.append((obj, [tuple(t.vertices) for t in obj.data.loop_triangles
                                  if all(i in selected for i in t.vertices)]))
    covered = [(obj, sorted({i for tri in faces for i in tri})) for obj, faces in body]
    if covered and shirt_faces:
        for n, changed in _transfer(covered, shirt_faces, torso, radius * .45).items():
            counts[n] = counts.get(n, 0) + changed
    for n, changed in _waist_contact(rig, body, shirts, waists, pelvis, radius / .14).items():
        counts[n] = counts.get(n, 0) + changed
    return counts
