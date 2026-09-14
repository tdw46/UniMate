"""Garment-region binding from connected geometry and anatomical attachment.

No avatar identifiers, source vertex counts, or imported skin weights are used.
Coordinates must be baked into the new rig local space, with Z up and X lateral.
"""
from mathutils import Vector, Quaternion
import math
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform
import heapq
import numpy as np


def components(mesh):
    neighbors = [set() for _ in mesh.vertices]
    for edge in mesh.edges:
        a, b = edge.vertices
        neighbors[a].add(b)
        neighbors[b].add(a)
    remaining = set(range(len(neighbors)))
    result = []
    while remaining:
        seed = min(remaining)
        found, pending = {seed}, [seed]
        while pending:
            for vertex in neighbors[pending.pop()] - found:
                found.add(vertex)
                pending.append(vertex)
        remaining -= found
        result.append(sorted(found))
    return result


def weights(obj, index):
    return {obj.vertex_groups[g.group].name: g.weight
            for g in obj.data.vertices[index].groups if g.weight > 1e-8}


def assign(obj, indices, values):
    for group in obj.vertex_groups:
        group.remove(indices)
    for name, weight in values.items():
        if weight > 1e-8:
            group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
            group.add(indices, weight, 'REPLACE')


def smooth(low, high, value):
    t = max(0., min(1., (value-low)/(high-low)))
    return t*t*(3-2*t)


def torso_distances(rig, point):
    """Fallback against torso and the same-side arm, excluding Head and Neck."""
    side = 'L' if point.x >= rig.data.bones['Neck'].head_local.x else 'R'
    values = {}
    for bone in rig.data.bones:
        if bone.name in ('Head', 'Neck'):
            continue
        if '.' in bone.name and not bone.name.endswith('.'+side):
            continue
        a, b = bone.head_local, bone.tail_local
        t = max(0., min(1., (point-a).dot(b-a)/(b-a).length_squared))
        values[bone.name] = 1/max((point-a-(b-a)*t).length, rig.data.bones['Neck'].length*.07)**4
    total = sum(values.values())
    return {name: value/total for name, value in values.items()}


def constrain(rig, point, values, clothing=False):
    """Cap influences by anatomical region; renormalizing a lone Neck is unsafe."""
    neck = rig.data.bones['Neck']
    radius = abs(rig.data.bones['UpperArm.L'].head_local.x-neck.head_local.x)*.9
    neck_cap = smooth(neck.head_local.z-neck.length*.35,
                      neck.head_local.z+neck.length*.55, point.z)
    neck_cap *= 1-smooth(radius*.55, radius*1.25, abs(point.x-neck.head_local.x))
    head_cap = 0 if clothing else smooth(neck.head_local.z, neck.tail_local.z, point.z)
    head = min(values.get('Head', 0), head_cap)
    neck_weight = min(values.get('Neck', 0), neck_cap, 1-head)
    other = {name: value for name, value in values.items() if name not in ('Head', 'Neck')}
    total = sum(other.values())
    if total < 1e-8:
        other = torso_distances(rig, point)
        total = sum(other.values())
    # Preserve anatomical caps while pruning, instead of renormalizing a small
    # capped Neck influence upward when other influences are discarded.
    result = {name: value for name, value in {'Head':head, 'Neck':neck_weight}.items() if value > 1e-8}
    other = dict(sorted(other.items(), key=lambda item: item[1], reverse=True)[:4-len(result)])
    total = sum(other.values())
    remainder = 1-sum(result.values())
    result.update({name: value/total*remainder for name, value in other.items()
                   if value/total*remainder > 1e-8})
    return result


class BodySurface:
    """Snapshot of freshly generated weights; independent of mesh names/order."""
    def __init__(self, obj, indices):
        self.obj = obj
        self.indices = set(indices)
        obj.data.calc_loop_triangles()
        self.triangles = [tuple(t.vertices) for t in obj.data.loop_triangles
                          if all(i in self.indices for i in t.vertices)]
        self.points = [v.co.copy() for v in obj.data.vertices]
        self.tree = BVHTree.FromPolygons(self.points, self.triangles, all_triangles=True)
        self.values = {i: weights(obj, i) for i in self.indices}

    def sample(self, point, rig):
        hit, _, face, distance = self.tree.find_nearest(point)
        assert face is not None
        # Never project a detached hand onto a remote torso triangle.
        if distance > rig.data.bones['UpperArm.L'].length*.7:
            return torso_distances(rig, point)
        triangle = self.triangles[face]
        a, b, c = (self.points[i] for i in triangle)
        bary = barycentric_transform(hit, a, b, c,
                                     Vector((1,0,0)), Vector((0,1,0)), Vector((0,0,1)))
        values = {}
        for i, factor in zip(triangle, bary):
            for name, value in self.values[i].items():
                values[name] = values.get(name, 0) + max(0., factor)*value
        return values


def describe(obj, indices):
    points = [obj.data.vertices[i].co for i in indices]
    low = Vector(tuple(min(p[i] for p in points) for i in range(3)))
    high = Vector(tuple(max(p[i] for p in points) for i in range(3)))
    # Area-weighted surface moments avoid mistaking rotated hair tufts for
    # spherical beads, and remain stable when triangles are subdivided.
    obj.data.calc_loop_triangles()
    index_set = set(indices)
    triangles = np.array([[obj.data.vertices[i].co for i in t.vertices]
                          for t in obj.data.loop_triangles
                          if all(i in index_set for i in t.vertices)], dtype=float)
    stretch = float('inf')
    if len(triangles):
        area = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],
                                       triangles[:,2]-triangles[:,0]), axis=1)*.5
        if area.sum() > 1e-20:
            area /= area.sum()
            sums = triangles.sum(axis=1)
            mean = np.einsum('t,ti->i', area, sums)/3
            second = (np.einsum('t,tvi,tvj->ij', area, triangles, triangles)
                      + np.einsum('t,ti,tj->ij', area, sums, sums))/12
            eigenvalues = np.linalg.eigvalsh(second-np.outer(mean,mean))
            stretch = float(eigenvalues[-1]/max(eigenvalues[0], 1e-20))
    return {'object': obj, 'indices': indices, 'low': low, 'high': high,
            'size': high-low, 'center': (high+low)*.5, 'surface_stretch': stretch}


def classify_regions(meshes, rig, rigid_parts):
    """Infer attachment regions from connected geometry and skeleton proportions.

    Optional object property binding_role supplies explicit semantic information
    for ambiguous assets. Supported roles: body, head, neckwear, torso, hood,
    rigid. Mixed objects should be split into semantic regions by the caller
    when geometry alone cannot establish their intended attachment.
    """
    neck = rig.data.bones['Neck']
    shoulder = abs(rig.data.bones['UpperArm.L'].head_local.x-neck.head_local.x)
    regions = [describe(obj, part) for obj in meshes for part in components(obj.data)]
    base_z = min(r['low'].z for r in regions)
    torso_height = neck.head_local.z-base_z
    # The connected surface crossing the torso and both shoulders is the body
    # carrier. Score by spatial coverage, never tessellation density or identity.
    carriers = [r for r in regions if r['low'].z < base_z+torso_height*.25
                and r['size'].x > shoulder*3 and r['high'].z > base_z+torso_height*.7
                and r['object'].get('binding_role', 'auto') in ('auto','body')]
    explicit = [r for r in regions if r['object'].get('binding_role') == 'body']
    if not (explicit or carriers):
        raise ValueError('No body carrier found; specify binding_role=body on a body surface')
    carrier = max(explicit or carriers, key=lambda r: r['size'].x*r['size'].z)
    # Repeated compact parts arranged around the neck form a jewelry assembly.
    # The cluster requires matching physical size and left/right/front coverage;
    # vertex counts, mesh names, and face/hair labels do not participate.
    candidates = [r for r in regions if min(r['size']) > neck.length*.15
                  and max(r['size'])/min(r['size']) < 1.6
                  and r['surface_stretch'] < 3
                  and max(r['size']) < shoulder*.8
                  and abs(r['center'].z-neck.head_local.z) < neck.length*1.8
                  and abs(r['center'].x-neck.head_local.x) < shoulder*1.25]
    jewelry = set()
    for seed in candidates:
        group = [r for r in candidates if max(r['size'])/max(seed['size']) > .7
                 and max(r['size'])/max(seed['size']) < 1.4]
        if (len(group) >= 3
                and min(r['center'].x for r in group) < neck.head_local.x-shoulder*.4
                and max(r['center'].x for r in group) > neck.head_local.x+shoulder*.4
                and min(r['center'].y for r in group) < neck.head_local.y-neck.length*.5):
            jewelry.update(id(r) for r in group)
    for r in regions:
        obj = r['object']
        explicit_role = obj.get('binding_role', 'auto')
        if explicit_role not in ('auto','body','head','neckwear','torso','hood','rigid'):
            raise ValueError('Unknown binding_role: '+str(explicit_role))
        if explicit_role != 'auto':
            role, reason = explicit_role, 'explicit semantic role'
        elif r is carrier:
            role, reason = 'body', 'connected torso and bilateral shoulder coverage'
        elif id(r) in jewelry:
            role, reason = 'neckwear', 'repeated compact assembly around neck'
        elif r['low'].z <= base_z+neck.length*.001 and r['high'].z < base_z+torso_height*.2:
            role, reason = 'base', 'short remnant touching bust cut plane'
        else:
            wraps_neck = (r['low'].x < neck.head_local.x-shoulder*.6
                          and r['high'].x > neck.head_local.x+shoulder*.6
                          and r['low'].y < neck.head_local.y < r['high'].y
                          and r['low'].z < neck.head_local.z+neck.length*.15
                          and r['high'].z > neck.head_local.z)
            if wraps_neck and r['low'].z > neck.head_local.z-neck.length*1.2:
                if r['size'].z < max(r['size'].x, r['size'].y)*.85:
                    role, reason = 'neckwear', 'circumferential collar around neck'
                else:
                    role, reason = 'hood', 'continuous shell spans head and neck base'
            elif rigid_parts.get(obj.name) == 'Head' and r['high'].z > neck.tail_local.z:
                role, reason = 'head', 'head attachment extending into cranial region'
            elif rigid_parts.get(obj.name) not in (None, 'Head', 'Chest', 'Spine', 'Root'):
                role, reason = 'rigid', 'existing semantic limb attachment to new rig'
            else:
                role, reason = 'torso', 'body-adjacent shell or disconnected garment'
        r.update(role=role, reason=reason)
    return carrier, regions


def garment_heat_solution(region, rig, garment_heat):
    """Use a complete topology-based solve on a broad connected torso shell.

    Small detached ties, bows and jewelry keep their attachment transfer.
    An incomplete solve falls back as a whole region, avoiding new weight
    discontinuities at the boundary of failed heat vertices.
    """
    if region['role'] != 'torso' or garment_heat is None:
        return None
    neck = rig.data.bones['Neck']
    left = rig.data.bones['UpperArm.L'].head_local.x
    right = rig.data.bones['UpperArm.R'].head_local.x
    broad_shell = (region['low'].x < right and region['high'].x > left
                   and region['low'].z < neck.head_local.z-neck.length*.35
                   and region['high'].z > neck.head_local.z-neck.length)
    if not broad_shell:
        return None
    values = garment_heat.get(region['object'].name, {})
    # Blender's raw heat groups are not guaranteed to sum to one yet;
    # constrain() normalizes them while applying the anatomical caps.
    if not all(values.get(i) and sum(values[i].values()) > 1e-8
               for i in region['indices']):
        return None
    return values


def preserved_garment_vertices(region, rig, attachment_values=None):
    """Keep complete top/underarm faces and the applied bust-cut boundary.

    Face classification avoids changing a corner of an otherwise protected
    triangle. Source coordinates are baked with Z up by the preparation flow.
    """
    obj, ids = region['object'], set(region['indices'])
    neck = rig.data.bones['Neck']
    protected = set()
    protected.update(i for i in ids if obj.data.vertices[i].co.z < region['low'].z+neck.length*.05)
    for face in obj.data.polygons:
        if not all(i in ids for i in face.vertices):
            continue
        at_cut = all(abs(obj.data.vertices[i].co.z-region['low'].z) < neck.length*.001
                     for i in face.vertices)
        side = 'L' if face.center.x > neck.head_local.x else 'R'
        arm = rig.data.bones['UpperArm.'+side]
        axis = (arm.tail_local-arm.head_local).normalized()
        up = Vector((0, 0, 1))
        up = (up-axis*up.dot(axis)).normalized()
        delta = face.center-arm.head_local
        at_shoulder = (-.35 < delta.dot(axis)/arm.length < .9
                       and delta.length < arm.length*.85)
        at_collar = attachment_values is not None and any(
            attachment_values[i].get('Head', 0.)+attachment_values[i].get('Neck', 0.) > 1e-8
            for i in face.vertices)
        if at_cut or at_collar or (at_shoulder and abs(face.normal.dot(up)) > .7):
            protected.update(face.vertices)
    return protected


def garment_heat_blend(region, rig, protected):
    """Fade the weight source by surface distance from protected boundaries."""
    obj, ids = region['object'], set(region['indices'])
    neighbors = {i: [] for i in ids}
    for edge in obj.data.edges:
        a, b = edge.vertices
        if a in ids and b in ids:
            length = (obj.data.vertices[a].co-obj.data.vertices[b].co).length
            neighbors[a].append((b, length)); neighbors[b].append((a, length))
    distance = {i: float('inf') for i in ids}
    pending = [(0., i) for i in protected]
    for i in protected: distance[i] = 0.
    heapq.heapify(pending)
    while pending:
        d, i = heapq.heappop(pending)
        if d != distance[i]: continue
        for j, length in neighbors[i]:
            candidate = d+length
            if candidate < distance[j]:
                distance[j] = candidate; heapq.heappush(pending, (candidate, j))
    return {i: smooth(0., rig.data.bones['Neck'].length*.4, d) for i, d in distance.items()}


def garment_strain_score(region, rig, values):
    """Peak area-weighted strain in three anatomical arm/elbow probe poses.

    Pure array evaluation leaves the Blender pose and animation untouched.
    Rest coordinates use the preparation flow's canonical rig-local frame.
    """
    obj, ids = region['object'], set(region['indices'])
    obj.data.calc_loop_triangles()
    triangles = np.array([list(t.vertices) for t in obj.data.loop_triangles
                          if all(i in ids for i in t.vertices)], dtype=int)
    if not len(triangles): return float('inf')
    points = np.array([v.co[:] for v in obj.data.vertices])
    edges = points[triangles[:,1:]]-points[triangles[:,0,None]]
    area = np.linalg.norm(np.cross(edges[:,0], edges[:,1]), axis=1)*.5
    valid = area > rig.data.bones['Neck'].length**2*1e-8
    triangles, edges, area = triangles[valid], edges[valid], area[valid]
    if not len(triangles): return float('inf')
    inverse = np.linalg.pinv(edges.transpose(0,2,1))
    peak = 0.
    for arm_angle, elbow_angle in ((25,0), (-15,35), (-50,65)):
        posed = points.copy()
        for side, sign in (('L',1), ('R',-1)):
            arm = rig.data.bones['UpperArm.'+side]
            shoulder, elbow = np.array(arm.head_local), np.array(arm.tail_local)
            rotation = np.array(Quaternion((0,1,0), math.radians(sign*arm_angle)).to_matrix())
            flex = np.array(Quaternion((0,0,1), math.radians(-sign*elbow_angle)).to_matrix())
            upper = (points-shoulder) @ rotation.T+shoulder
            fore = (((points-elbow) @ flex.T+elbow)-shoulder) @ rotation.T+shoulder
            upper_weights = np.array([values.get(i,{}).get('UpperArm.'+side,0.) for i in range(len(points))])
            fore_weights = np.array([sum(values.get(i,{}).get(n+side,0.) for n in ('Forearm.','Hand.')) for i in range(len(points))])
            posed += (upper-points)*upper_weights[:,None]+(fore-points)*fore_weights[:,None]
        gradient = (posed[triangles[:,1:]]-posed[triangles[:,0,None]]).transpose(0,2,1) @ inverse
        stretch = np.linalg.svd(gradient, compute_uv=False)[:,:2]
        peak = max(peak, float(np.sqrt(np.average(np.sum((stretch-1)**2, axis=1), weights=area))))
    return peak


def correct_apparel(meshes, rig, rigid_parts, garment_heat=None):
    carrier, regions = classify_regions(meshes, rig, rigid_parts)
    neck = rig.data.bones['Neck']
    obj = carrier['object']
    for index in carrier['indices']:
        point = obj.data.vertices[index].co
        assign(obj, [index], constrain(rig, point, weights(obj, index)))
    surface = BodySurface(obj, carrier['indices'])
    audit = {'method': 'geometry regions, anatomical constraints, fresh body surface transfer',
             'carrier_mesh': obj.name, 'regions': []}
    for r in regions:
        obj, indices, role = r['object'], r['indices'], r['role']
        heat = garment_heat_solution(r, rig, garment_heat)
        original_values = ({i: constrain(rig, obj.data.vertices[i].co,
                                         surface.sample(obj.data.vertices[i].co, rig), clothing=True)
                            for i in indices} if role == 'torso' else {})
        protected = preserved_garment_vertices(r, rig, original_values) if heat is not None else set()
        heat_blend = garment_heat_blend(r, rig, protected) if heat is not None else {}
        strain_check = None
        if role == 'neckwear':
            assign(obj, indices, {'Neck': 1.})
        elif role == 'head':
            assign(obj, indices, {'Head': 1.})
        elif role == 'base':
            assign(obj, indices, {'Root': 1.})
        elif role == 'hood':
            for index in indices:
                point = obj.data.vertices[index].co
                h = smooth(neck.head_local.z, neck.tail_local.z+neck.length*.6, point.z)
                n = smooth(neck.head_local.z-neck.length, neck.tail_local.z, point.z)*(1-h)
                assign(obj, [index], {'Head':h, 'Neck':n, 'Chest':1-h-n})
        elif role == 'torso':
            candidate_values = {}
            for index in indices:
                point = obj.data.vertices[index].co
                values = original_values[index]
                if heat is not None and heat_blend[index] > 0:
                    original = original_values[index]
                    fitted = constrain(rig, point, heat[index], clothing=True)
                    blend = heat_blend[index]
                    values = {bone: original.get(bone, 0.)*(1-blend)+fitted.get(bone, 0.)*blend
                              for bone in original.keys() | fitted.keys()}
                    values = constrain(rig, point, values, clothing=True)
                candidate_values[index] = values
            if heat is not None:
                baseline_score = garment_strain_score(r, rig, original_values)
                candidate_score = garment_strain_score(r, rig, candidate_values)
                accepted = candidate_score < baseline_score*.999
                strain_check = {'body_transfer': baseline_score, 'garment_heat_blend': candidate_score,
                                'accepted': accepted}
                if not accepted:
                    candidate_values = original_values
                    heat = None
            for index in indices:
                assign(obj, [index], candidate_values[index])
        elif role == 'body' and r is not carrier:
            for index in indices:
                point = obj.data.vertices[index].co
                assign(obj, [index], constrain(rig, point, weights(obj, index)))
        audit['regions'].append({'mesh': obj.name, 'vertices': len(indices),
                                 'role': role, 'reason': r['reason'],
                                 'weight_source': 'fresh garment heat' if heat is not None else 'anatomical attachment or body transfer',
                                 'preserved_top_underarm_cut_vertices': len(protected),
                                 'strain_check': strain_check,
                                 'bounds_min': list(r['low']), 'bounds_max': list(r['high'])})
    # Fit the body and overlying cloth with the same continuous anatomical
    # field. Smoothing only cloth leaves the underlying rigid arm protruding.
    from avatar_garment_transition import front_torso_transition, loose_front_strength
    strengths = [loose_front_strength(r, rig, surface) for r in regions]
    body_strength = max((s[0] for s in strengths), default=0.)
    # The front attachment objective alone can displace strain into sleeves.
    # Bound its shared strength against each broad garment's whole-surface
    # deformation, keeping body and every clothing layer synchronized.
    strain_bounds = []
    for r in regions:
        if r['role'] != 'torso' or body_strength <= 0.: continue
        obj = r['object']; source = {i: weights(obj, i) for i in r['indices']}
        fitted, changed = front_torso_transition(r, rig, source, body_strength)
        if not changed: continue
        baseline = garment_strain_score(r, rig, source)
        limit = baseline*1.05+1e-8
        score = garment_strain_score(r, rig, fitted)
        if score > limit:
            low, high = 0., body_strength
            for _ in range(8):
                middle = (low+high)*.5
                trial, _ = front_torso_transition(r, rig, source, middle)
                if garment_strain_score(r, rig, trial) <= limit: low = middle
                else: high = middle
            body_strength = low
        strain_bounds.append({'mesh':obj.name, 'baseline_strain':baseline,
                              'maximum_strain':limit, 'allowed_strength':body_strength})
    audit['front_torso_strain_bounds'] = strain_bounds
    audit['front_torso_transition'] = []
    for r, (strength, clearance) in zip(regions, strengths):
        if r['role'] not in ('body', 'torso'):
            continue
        obj = r['object']
        source = {i: weights(obj, i) for i in r['indices']}
        # Nested broad clothing layers need the body's same field strength;
        # otherwise a fitted inner layer can pierce a slower outer panel.
        strength = body_strength
        fitted, changed = front_torso_transition(r, rig, source, strength)
        for i in changed:
            assign(obj, [i], fitted[i])
        if changed:
            audit['front_torso_transition'].append({
                'mesh': obj.name, 'role': r['role'], 'vertices': len(changed),
                'strength': strength, 'front_clearance_neck_lengths_p90': clearance,
                'method': 'clearance-adaptive front torso-to-sleeve field, shared with underlying body'})
    return audit
