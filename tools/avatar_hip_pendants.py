"""Identify a suspended disk between an upper tether and a hanging loop with a tassel."""
import numpy as np
from mathutils import Vector


def _parts(points, islands, occupied, hips, height):
    for ids in islands:
        if len(ids) < 4 or not occupied.isdisjoint(ids):
            continue
        p = points[ids]
        lo, hi = p.min(axis=0), p.max(axis=0)
        span = hi - lo
        center = (lo + hi) * .5
        if (span.max() > .12 * height or hi[2] < hips[2] - .24 * height
                or lo[2] > hips[2] + .13 * height
                or np.linalg.norm(center[:2] - hips[:2]) > .20 * height):
            continue
        axes = np.linalg.svd(p - p.mean(axis=0), full_matrices=False)[2]
        projected = p @ axes.T
        extent = np.ptp(projected, axis=0)
        if extent.min() < 1e-9 and extent[1] < 1e-9:
            continue
        projected -= (projected.min(axis=0) + projected.max(axis=0)) * .5
        mid = projected[np.abs(projected[:, 0]) <= extent[0] * .23, 1]
        # A loop has a gap between both sides in its middle cross-section.
        loop = (extent[2] < .45 * extent[1] and len(mid) >= 4
                and mid.min() < -.2 * extent[1] and mid.max() > .2 * extent[1]
                and np.abs(mid).min() > .10 * extent[1])
        band = max(.08 * span[2], .0004 * height)
        yield dict(ids=ids, extent=extent, axis=axes[0], normal=axes[2], center=center,
                   length=span[2], loop=loop, points=p,
                   top=np.median(p[p[:, 2] >= hi[2] - band], axis=0),
                   bottom=np.median(p[p[:, 2] <= lo[2] + band], axis=0))


def _next_link(parts, joint, diameter, above, loop, used):
    matches = []
    for i, part in enumerate(parts):
        length, width, _ = part['extent']
        if (i in used or width > .85 * diameter or length < 1.55 * width
                or not .35 * diameter < length < 3.5 * diameter
                or abs(part['axis'][2]) < .62 or (loop and not part['loop'])):
            continue
        near, far = ('bottom', 'top') if above else ('top', 'bottom')
        if (part[far][2] - joint[2]) * (1 if above else -1) < .28 * diameter:
            continue
        distance = np.linalg.norm(part[near] - joint)
        if distance < .31 * diameter:
            matches.append((distance, i))
    return min(matches)[1] if matches else None


def plan_pendants(surfaces, hips, height, prefix, spine=None):
    parts = []
    for obj, points, islands, occupied in surfaces:
        for part in _parts(points, islands, occupied, hips, height):
            part['obj'] = obj
            parts.append(part)
    chains, regions, used = [], [], set()
    for disc_id, disc in enumerate(parts):
        diameter, breadth, thickness = disc['extent']
        if (disc_id in used or not .014 * height < diameter < .065 * height
                or breadth < .74 * diameter or thickness > .31 * breadth
                or abs(disc['normal'][2]) > .70
                or abs(disc['center'][2] - hips[2]) > .09 * height):
            continue
        tether = _next_link(parts, disc['top'], diameter, True, False, used)
        loop = _next_link(parts, disc['bottom'], diameter, False, True, used)
        if tether is None or loop is None:
            continue
        tassel = _next_link(parts, parts[loop]['bottom'], diameter, False, False, used | {loop})
        if tassel is None:
            continue
        order = [tether, disc_id, loop, tassel]
        pieces = [parts[i] for i in order]
        # Each pivot sits between the neighboring rest endpoints, including
        # any tiny detached clasp. The disk rotates as a single rigid link.
        pivots = [pieces[0]['top']]
        pivots.extend((a['bottom'] + b['top']) * .5 for a, b in zip(pieces, pieces[1:]))
        pivots.append(pieces[-1]['bottom'])
        clasp_centers = pivots[1:-1]
        # Seat the loop pivot just inside the hollow end of the ring. A
        # weighted clasp should turn with the disk before the cord bends.
        pivots[2] = pivots[2] + .14 * (pivots[3] - pivots[2])
        parent = 'Hips'
        if spine is not None and abs(pivots[0][2]-spine[2]) < abs(pivots[0][2]-hips[2]):
            parent = 'Spine'
        names = [f'{prefix}Pendant_{len(chains):02d}_{i:02d}' for i in range(5)]
        chains.append(dict(role='pendant', parent=parent, names=names,
                           line=list(map(Vector, pivots)), free_appendage=True))
        used.update(order)
        bindings = {}
        for k, part in enumerate(pieces):
            rows = bindings.setdefault(part['obj'], {})
            for vertex, point in zip(part['ids'], part['points']):
                if k in (1, 3):
                    rows[vertex] = {names[k]: 1.}
                else:
                    delta = pivots[k+1] - pivots[k]
                    t = float(np.clip((point - pivots[k]) @ delta / (delta @ delta), 0., 1.))
                    near, far = (parent if k == 0 else names[k-1]), names[k+1]
                    a = .5 * max(0., 1. - 3.*t)**2
                    b = .5 * max(0., 3.*t - 2.)**2
                    rows[vertex] = {n:w for n,w in ((near,a),(names[k],1.-a-b),(far,b)) if w>0.}
        for i, part in enumerate(parts):
            if i in used or part['extent'].max() > .34 * diameter:
                continue
            distances = [np.linalg.norm(part['center'] - pivot) for pivot in clasp_centers]
            joint = int(np.argmin(distances))
            if distances[joint] < .22 * diameter:
                # Capture the small end caps with the rigid adjoining piece.
                bone = names[1 if joint < 2 else 3]
                bindings.setdefault(part['obj'], {}).update({v:{bone:1.} for v in part['ids']})
                used.add(i)
        regions.extend(dict(obj=obj, ids=set(rows), role='pendant', values=rows)
                       for obj, rows in bindings.items())
    return chains, regions


def waist_contacts(rig, prefix, height):
    """Fit a transverse waist capsule to each suspended load in rest space."""
    springs = rig.data.vrm_addon_extension.spring_bone1.springs
    hips = rig.data.bones.get('Hips')
    if hips is None:
        return []
    contacts = []
    for spring in springs:
        if not spring.vrm_name.startswith(prefix + 'Pendant_') or len(spring.joints) < 2:
            continue
        root = rig.data.bones.get(spring.joints[0].node.bone_name)
        if root is None or root.parent is None:
            continue
        mount = root.head_local
        # Fit the side support to mount distance from the centerline and
        # tether length. A front or centered charm uses its radial direction.
        outward = mount - hips.head_local
        outward.z = 0.
        span = min(outward.length, .10 * height)
        if span < .008 * height:
            continue
        axis = outward.normalized()
        radius = min(.75 * span, 1.5 * root.length)
        outer = mount - Vector((0., 0., radius + .15 * root.length))
        inner = outer - axis * span
        if abs(axis.x) > abs(axis.y):
            # Across a flank the waist is wider than it is deep. Set a
            # transverse axis inside the clothing so contact lets the load
            # swing fore and aft without constantly pushing it backwards.
            inset = np.clip((hips.head_local.y - mount.y) / 3., -.2 * radius, .2 * radius)
            outer.y += float(inset)
            inner.y = outer.y
        inverse = root.parent.matrix_local.inverted()
        contacts.append(dict(role=spring.vrm_name[len(prefix):].lower(), bone=root.parent.name,
                             offset=list(inverse @ inner), tail=list(inverse @ outer),
                             radius=radius, body_radius=radius))
    return contacts


def arm_contacts(rig, arm_points, pendant_samples, height):
    """Fit shared upper arm, forearm and palm capsules to their weighted surface."""
    if not arm_points or not pendant_samples:
        return []
    points = np.array([tuple(p) for p, _, _ in pendant_samples])
    hits = np.array([r for _, r, _ in pendant_samples])
    # VRM checks joint endpoints. The rigid disk and terminal tassel can
    # extend well beyond them. Estimate the unsampled reach from the actual
    # skinned surface, capping it for unusually large ornaments.
    reach = np.full(len(points), np.inf)
    for point, _, label in pendant_samples:
        if label != 'surface':
            reach = np.minimum(reach, np.linalg.norm(points - np.asarray(point), axis=1))
    # A full ornament-width margin catches the tassel on ordinary arm swing.
    # Use only a fraction of the overhang and bound it by character size.
    margin = min(.30 * float(np.quantile(reach, .98)), .0085 * height)
    contacts = []
    for name, surface in arm_points.items():
        if len(surface) < 4:
            continue
        bone = rig.data.bones[name]
        inverse = bone.matrix_local.inverted_safe()
        local = np.array([tuple(inverse @ vertex) for vertex in surface])
        lo, hi = np.quantile(local[:, 1], (.03, .97))
        band = local[(local[:, 1] >= lo) & (local[:, 1] <= hi)]
        if len(band) < 4 or hi <= lo:
            continue
        center = np.median(band[:, (0, 2)], axis=0)
        distances = np.linalg.norm(band[:, (0, 2)] - center, axis=1)
        radius = min(float(np.quantile(distances, .85)), (hi - lo) * .48)
        # Round the ends inside the measured limb. Keep the entire pendant
        # clear at rest, including the edges of its rigid disk.
        a = Vector((center[0], lo + radius * .5, center[1]))
        b = Vector((center[0], hi - radius * .5, center[1]))
        start = np.array(bone.matrix_local @ a)
        axis = np.array(bone.matrix_local @ b) - start
        squared_length = float(axis @ axis)
        if squared_length <= 1e-16:
            continue
        t = np.clip((points - start) @ axis / squared_length, 0., 1.)
        clearance = float((np.linalg.norm(points - start - t[:, None]*axis, axis=1) - hits).min())
        fitted = min(radius + margin, clearance - .0015*height)
        if fitted > .001*height:
            contacts.append(dict(role='pendant_arms', bone=name, offset=list(a), tail=list(b),
                                 radius=fitted, body_radius=radius,
                                 minimum_rest_clearance=clearance-fitted))
    return contacts
