"""Garment-region binding from connected geometry and anatomical attachment.

No avatar identifiers, source vertex counts, or imported skin weights are used.
Coordinates must be baked into the new rig local space, with Z up and X lateral.
"""
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform
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


def hair_root_length(height, strand_height):
    return min(height*.02, strand_height*.08)


def hair_free_top(points, height, neck_z, crown_z):
    """Find the movable portion of a hanging strand; scalp sheets stay rigid.

    Long hair keeps the established attachment rule. Short hair must extend
    below the crown and be vertically elongated. Its entire crown remains
    fixed, even when a connected strip includes both scalp and hanging hair.
    """
    span = np.ptp(points, axis=0)
    lo, hi = points[:, 2].min(), points[:, 2].max()
    root = hi-hair_root_length(height, span[2])
    if span[2] >= height*.22 and lo <= neck_z:
        return float(root)
    if span[2] < height*.04 or span[2] < min(span[:2])*1.2:
        return None
    top = min(root, crown_z)
    return float(top) if top-lo >= height*.035 else None


def head_cap_vertices(meshes, rig):
    """Find cranial skin and the fixed hair cap in the prepared rest pose.

    A head/face surface establishes the skull envelope. This also catches scalp
    islands in mixed body meshes without capturing nearby sleeves or hands.
    Hanging strands below the head attachment remain available for springs.
    """
    head_z = rig.data.bones['Head'].head_local.z
    head_xy = np.array(tuple(rig.data.bones['Head'].head_local))[:2]
    humanoid=getattr(getattr(rig.data,'vrm_addon_extension',None),'vrm1',None)
    arm_names=[]
    for semantic,legacy,mixamo in (('left_upper_arm','UpperArm.L','LeftArm'),('right_upper_arm','UpperArm.R','RightArm')):
        mapped=getattr(humanoid.humanoid.human_bones,semantic).node.bone_name if humanoid else ''
        arm_names.append(next(n for n in (mapped,legacy,mixamo) if n in rig.data.bones))
    cranial_radius = max(abs(rig.data.bones[n].head_local.x-head_xy[0]) for n in arm_names)*.9
    points, skin, hair, reference = {}, {}, {}, []
    for obj in meshes:
        tr = rig.matrix_world.inverted() @ obj.matrix_world
        points[obj.name] = np.array([tuple(tr @ v.co) for v in obj.data.vertices])
        skin[obj.name], hair[obj.name] = set(), set()
        for face in obj.data.polygons:
            mat = obj.data.materials[face.material_index] if face.material_index < len(obj.data.materials) else None
            label = mat.name.lower() if mat else ''
            if 'hair' in label:
                hair[obj.name].update(face.vertices)
            elif any(token in label for token in ('skin', 'face')):
                skin[obj.name].update(face.vertices)
        if obj.get('binding_role') == 'head':
            reference.extend(points[obj.name][i] for i in skin[obj.name]
                             if points[obj.name][i, 2] >= head_z)
        # A face reference does not extend around the back of the skull.
        # Include central cranial skin even when it is a separate neck/scalp
        # island in a mixed body object and was misbound by the heat solve.
        reference.extend(points[obj.name][i] for i in skin[obj.name]
                         if points[obj.name][i, 2] >= head_z
                         and np.linalg.norm(points[obj.name][i, :2]-head_xy) <= cranial_radius)
    bounds = None
    if reference:
        reference = np.array(reference)
        low, high = reference.min(axis=0), reference.max(axis=0)
        padding = np.maximum((high-low)*.05, rig.data.bones['Neck'].length*.05)
        bounds = low-padding, high+padding
    result = {}
    height = max(p[:, 2].max() for p in points.values())-min(p[:, 2].min() for p in points.values())
    for obj in meshes:
        p = points[obj.name]
        ids = {i for i in hair[obj.name] if p[i, 2] >= head_z}
        for component in components(obj.data):
            strand = sorted(set(component) & hair[obj.name])
            if len(strand) < 8:
                continue
            z = p[strand, 2]
            fixed_end = hair_free_top(p[strand], height,
                                      rig.data.bones['Neck'].head_local.z,
                                      rig.data.bones['Head'].tail_local.z)
            if fixed_end is not None:
                # Long connected strips attach at the cap, even when their
                # hanging section begins above the anatomical head joint.
                ids.difference_update(i for i in strand if p[i, 2] < fixed_end)
                ids.update(i for i in strand if p[i, 2] >= fixed_end)
        if bounds is not None:
            low, high = bounds
            ids.update(i for i in skin[obj.name] if p[i, 2] >= head_z
                       and np.all(p[i] >= low) and np.all(p[i] <= high))
        if ids:
            result[obj.name] = sorted(ids)
    return result


def enforce_head_cap(meshes, rig):
    cap = head_cap_vertices(meshes, rig)
    for obj in meshes:
        if obj.name in cap:
            assign(obj, cap[obj.name], {'Head': 1.})
    return cap


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
        elif 'Thigh.L' not in rig.data.bones and r['low'].z <= base_z+neck.length*.001 and r['high'].z < base_z+torso_height*.2:
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


def correct_apparel(meshes, rig, rigid_parts):
    """Keep ordinary heat binding; only explicit attachment semantics override it.

    Broad body and clothing regions are no longer projected onto a body carrier,
    re-solved against lateral arm bones, or replaced with a shoulder field.
    Detached neckline cloth retains the established anatomical attachment flow.
    """
    carrier, regions = classify_regions(meshes, rig, rigid_parts)
    surface = BodySurface(carrier['object'], carrier['indices'])
    audit = {'method': 'ordinary heat weights with explicit attachment semantics',
             'carrier_mesh': carrier['object'].name, 'regions': []}
    for r in regions:
        obj, ids, role = r['object'], r['indices'], r['role']
        original = {i:weights(obj,i) for i in ids}
        source = 'ordinary normalized heat'
        if role in ('head', 'neckwear', 'base'):
            assign(obj, ids, {dict(head='Head', neckwear='Neck', base='Root')[role]: 1.})
            source = 'explicit rigid attachment'
        elif role == 'torso':
            broad = (r['low'].x < rig.data.bones['UpperArm.R'].head_local.x
                     and r['high'].x > rig.data.bones['UpperArm.L'].head_local.x)
            if not broad and r['high'].z > rig.data.bones['Neck'].head_local.z-rig.data.bones['Neck'].length:
                for i in ids:
                    p = obj.data.vertices[i].co
                    assign(obj, [i], constrain(rig, p, surface.sample(p, rig), clothing=True))
                source = 'detached neckline cloth attachment'
            elif any(original[i].get('Head',0)>0 for i in ids):
                # Preserve the established clothing Head exclusion without
                # projecting the rest of its heat solution onto the body.
                for i in ids:
                    values=dict(original[i]);head=values.pop('Head',0.)
                    if head<=0:continue
                    torso={n:values.get(n,0.) for n in ('Root','Spine','Chest')}
                    total=sum(torso.values())
                    if total<1e-8:torso={'Chest':1.};total=1.
                    for n,w in torso.items():values[n]=values.get(n,0.)+head*w/total
                    assign(obj,[i],values)
                source = 'heat with clothing Head exclusion'

        if source == 'ordinary normalized heat':
            assert all(weights(obj,i)==original[i] for i in ids)
        audit['regions'].append({'mesh':obj.name, 'vertices':len(ids), 'role':role,
                                 'reason':r['reason'], 'weight_source':source,
                                 'bounds_min':list(r['low']), 'bounds_max':list(r['high'])})
    cap = enforce_head_cap(meshes, rig)
    audit['rigid_head_cap_vertices'] = {name:len(ids) for name, ids in cap.items()}
    return audit
