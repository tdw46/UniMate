"""Rest-surface ownership for trouser tubes and individual finger segments."""
import numpy as np

from avatar_apparel_weights import weights
from avatar_virtual_topology import connected_graph


def _write(obj, i, old, new):
    if all(abs(old.get(k, 0.) - new.get(k, 0.)) < 1e-6 for k in old.keys() | new.keys()):
        return False
    i = int(i)
    for name in old.keys() - new.keys():
        obj.vertex_groups[name].remove([i])
    for name, value in new.items():
        if abs(old.get(name, 0.) - value) >= 1e-6:
            group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
            group.add([i], value, 'REPLACE')
    return True


def _leg_tubes(rig, obj, points):
    legs = ('Thigh', 'Shin', 'Foot', 'Toe')
    sides = ('.L', '.R')
    bones = rig.data.bones
    if any(bones.get('Thigh'+s) is None for s in sides):
        return 0
    hips = np.array([bones['Thigh'+s].head_local for s in sides])
    knees = np.array([bones['Thigh'+s].tail_local for s in sides])
    span = np.mean(hips[:, 2] - knees[:, 2])
    if span <= 0:
        return 0
    names = {part+s for part in legs for s in sides}
    if not all(obj.vertex_groups.get('Thigh'+s) is not None for s in sides):
        return 0
    # Cut the connectivity below the crotch. A tube can cross the sagittal
    # plane without changing its owner. A connected skirt has no such pair.
    for fraction in (.90, .75, .55, .30):
        ceiling = knees[:, 2].mean() + fraction*span
        ids = np.flatnonzero(points[:, 2] < ceiling)
        if not len(ids):
            return 0
        _, components, _ = connected_graph(obj.data, points, ids, span*.00025)
        owners = []
        for component in components:
            p = points[component]
            if len(p) < 8 or p[:, 2].min() > knees[:, 2].mean() or p[:, 2].max() < ceiling-span*.25:
                continue
            d = np.linalg.norm(p[:, None, :2]-knees[None, :, :2], axis=2)
            votes = np.mean(d[:, 0] < d[:, 1])
            if votes > .85 or votes < .15:
                owners.append((component, int(votes < .15)))
        if len({side for _, side in owners}) < 2:
            continue
        count = 0
        for component, side in owners:
            for i in component:
                row = weights(obj, i)
                if not row or row.keys() - (names | {'Hips', 'Spine'}):
                    continue
                new = row.copy()
                for part in legs:
                    wrong, right = part+sides[1-side], part+sides[side]
                    if wrong in new and right in bones:
                        new[right] = new.get(right, 0.) + new.pop(wrong)
                count += _write(obj, i, row, new)
        return count
    return 0


def _digit_segments(rig, obj, points, side):
    """Bind finger shafts rigidly, with a narrow palm or adjacent-joint crease."""
    digits = ('Thumb', 'Index', 'Middle', 'Ring', 'Little')
    chains = [[f'{digit}{j}.{side}' for j in (1, 2, 3)] for digit in digits]
    names = {name for chain in chains for name in chain}
    if not all(n in rig.data.bones for n in names) or not any(g.name in names for g in obj.vertex_groups):
        return 0
    hand, wrist = 'Hand.' + side, 'Forearm.' + side
    base = np.array([[rig.data.bones[n].head_local for n in chain] for chain in chains])
    tip = np.array([[rig.data.bones[n].tail_local for n in chain] for chain in chains])
    length = np.linalg.norm(tip - base, axis=2)
    if length.min() < 1e-7:
        return 0
    axis = (tip - base) / length[:, :, None]
    lo = np.minimum(base, tip).min((0, 1)) - length.max()
    hi = np.maximum(base, tip).max((0, 1)) + length.max()
    allowed = names | {hand, wrist}
    ids, rows = [], []
    for i in np.flatnonzero(((points >= lo) & (points <= hi)).all(axis=1)):
        row = weights(obj, int(i))
        if row and row.keys() <= allowed and (wrist not in row or names.intersection(row)):
            ids.append(int(i))
            rows.append(row)
    if not ids:
        return 0
    delta = points[ids, None, None, :] - base
    along = np.einsum('ndjc,djc->ndj', delta, axis)
    residual = delta - np.clip(along, 0., length)[:, :, :, None] * axis
    distance = np.linalg.norm(residual, axis=3)
    digit_distance = distance.min(axis=2)
    # Separate the actual skin tubes at the web. A fingertip can be closer to
    # the palm through air than to its own bone. Only connected surface paths
    # can carry weights back from an identified digit.
    from heapq import heappop, heappush
    lookup = {i: j for j, i in enumerate(ids)}
    graph, _, _ = connected_graph(obj.data, points, ids, length.mean() * .002)
    edges = [[] for _ in ids]
    for edge in graph.edges:
        a, b = edge.vertices
        cost = float(np.linalg.norm(points[a] - points[b]))
        a, b = lookup[a], lookup[b]
        edges[a].append((b, cost))
        edges[b].append((a, cost))
    shares = np.zeros((len(ids), len(digits)))
    locked = np.full(len(ids), -1, dtype=int)
    approach = np.full(shares.shape, np.inf)
    owner = digit_distance.argmin(axis=1)
    for d in range(len(digits)):
        first = length[d, 0]
        station = along[:, d, 0] / first
        segment = distance[:, d].argmin(axis=1)
        for cut in (-.15, .15, .4, .65, .95, 1.3, 1.6):
            remaining = set(np.flatnonzero(station >= cut))
            tube = []
            reached_tip = False
            while remaining:
                seed = remaining.pop()
                component = [seed]
                for j in component:
                    for other, _ in edges[j]:
                        if other in remaining:
                            remaining.remove(other)
                            component.append(other)
                if (np.mean(owner[component] == d) < .85 or
                        np.quantile(digit_distance[component, d], .95) > first * .75):
                    continue
                tube.extend(component)
                reached_tip |= bool(np.any(segment[component] == 2) and np.any(station[component] < cut + .55))
            if reached_tip:
                break
        else:
            # A fused glove or folded rest digit may never separate at a
            # transverse cut. The local heat bind already owns that region.
            continue
        locked[tube] = d
        shares[tube, d] = 1.
        # Keep the palm side of the thumb web attached, even when it lies
        # beside the second or third thumb segment in Euclidean space.
        reach = first * max(.55, cut + .5)
        dist_to = np.full(len(ids), np.inf)
        queue = []
        tube_set = set(tube)
        for j in tube:
            if any(station[o] < cut for o, _ in edges[j]):
                dist_to[j] = 0.
                heappush(queue, (0., j))
        while queue:
            cost, j = heappop(queue)
            if cost != dist_to[j]:
                continue
            for other, step in edges[j]:
                new_cost = cost + step
                if other in tube_set or new_cost >= min(reach, dist_to[other]):
                    continue
                dist_to[other] = new_cost
                approach[other, d] = cut * first - new_cost
                x = 1. - new_cost / reach
                shares[other, d] = x*x*(3.-2.*x)
                heappush(queue, (new_cost, other))
    has_tube = shares.max(axis=0) > 0.
    if not np.any(has_tube):
        return 0
    for j in np.flatnonzero(locked >= 0):
        shares[j] = 0.
        shares[j, locked[j]] = 1.
    web = np.column_stack((shares, 1. - shares.max(axis=1)))
    bisector = axis[:, :-1] + axis[:, 1:]
    bisector /= np.maximum(np.linalg.norm(bisector, axis=2, keepdims=True), 1e-9)
    planes = np.concatenate((axis[:, :1], bisector), axis=1)
    widths = .14 * np.concatenate((length[:, :1], np.minimum(length[:, :-1], length[:, 1:])), axis=1)
    # Signed distances to joint planes give all seam copies the same crease.
    # Only the two joints bounding the closest segment can affect its skin.
    targets = [None] * len(ids)
    for j in range(len(ids)):
        if not has_tube[owner[j]] and locked[j] < 0:
            continue
        target = {}
        for d in np.flatnonzero(web[j]):
            share = web[j, d]
            if d == len(digits):
                target[hand] = target.get(hand, 0.) + share
                continue
            chain = chains[d]
            segment = distance[j, d].argmin()
            signed = np.einsum('jc,jc->j', delta[j, d], planes[d])
            if np.isfinite(approach[j, d]):
                # On the shared web, distance travelled back from the tube
                # bounds the longitudinal station. A palm point lying beside
                # a curled thumb cannot acquire its distal joint this way.
                signed = np.minimum(signed, approach[j,d] - np.r_[0., np.cumsum(length[d,:2])])
            lower = segment
            if np.isfinite(approach[j,d]):
                lower = int(np.count_nonzero(signed[1:] >= 0.))
                segment = lower
            if segment < 2 and signed[segment + 1] > -widths[d, segment + 1]:
                lower = segment + 1
            if lower == 0:
                # Ease slightly farther into the palm than into the free shaft.
                t = (signed[0] / length[d, 0] + .28) / .48
            else:
                t = .5 + .5 * signed[lower] / widths[d, lower]
            t = min(1., max(0., t))
            t *= t * (3. - 2. * t)
            parent = hand if lower == 0 else chain[lower-1]
            for n, w in ((parent, 1.-t), (chain[lower], t)):
                if w > 1e-8:
                    target[n] = target.get(n, 0.) + share*w
        mass = sum(target.values())
        targets[j] = {n: w/mass for n, w in target.items()}
    # Round only the hand-to-base attachment on the connected web. The pad on
    # the palm side of the pivot needs a little yield. PIP/DIP creases and the
    # interior of each free phalanx retain their analytical weights.
    root_names = [chain[0] for chain in chains]
    palm_names = {hand, *root_names}
    field = np.array([[target.get(name, 0.) for name in root_names] if target else np.zeros(len(digits))
                      for target in targets])
    radius = .48 * float(np.median(length[:, 0]))
    for j, target in enumerate(targets):
        if target is None or target.keys() - palm_names:
            continue
        own = locked[j]
        if own >= 0 and along[j, own, 0] >= .40 * length[own, 0]:
            continue
        # Completely homogeneous neighborhoods have no boundary to soften.
        if not field[j].any() and not any(shares[j]):
            continue
        total = 1.
        value = field[j].copy()
        visit = {j: 0.}
        queue = [(0., j)]
        while queue:
            cost, here = heappop(queue)
            if cost != visit[here]:
                continue
            if here != j:
                if targets[here] is None or targets[here].keys() - palm_names:
                    continue
                w = (1.-(cost/radius)**2)**2
                total += w
                value += w * field[here]
            for other, step in edges[here]:
                distance_to = cost + step
                if distance_to < min(radius, visit.get(other, radius)):
                    visit[other] = distance_to
                    heappush(queue, (distance_to, other))
        value /= total
        if own >= 0:
            value[np.arange(len(digits)) != own] = 0.
        target = {name: v for name, v in zip(root_names, value) if v > 1e-7}
        target[hand] = max(0., 1. - sum(target.values()))
        targets[j] = target
    count = 0
    for i, old, target in zip(ids, rows, targets):
        if target is None:
            continue
        wrist_w = old.get(wrist, 0.)
        top = sorted(target.items(), key=lambda item: -item[1])[:3 if wrist_w else 4]
        total = sum(w for n, w in top)
        target = {n: (1.-wrist_w)*w/total for n, w in top}
        if wrist_w:
            target[wrist] = wrist_w
        count += _write(obj, i, old, target)
    return count


def _sleeve_weights(rig, obj, points):
    """Keep a connected jacket sleeve on its arm, including loose cuff seams."""
    from avatar_material_roles import semantic_label
    jacket_terms = ('jacket', 'coat', 'vest', 'topwear', 'ジャケット', '上着', '外套', '재킷')
    slots = {i for i, mat in enumerate(obj.data.materials)
             if mat and any(term in semantic_label(mat.name) for term in jacket_terms)}
    if not slots:
        return 0
    ids = np.array(sorted({i for face in obj.data.polygons if face.material_index in slots
                           for i in face.vertices}), dtype=int)
    if not len(ids):
        return 0
    changed = 0
    for side in ('.L', '.R'):
        upper, fore, hand = (name + side for name in ('UpperArm', 'Forearm', 'Hand'))
        bones = rig.data.bones
        if not all(bones.get(n) for n in (upper, fore, hand)):
            continue
        shoulder, elbow, wrist = (np.asarray(bones[n].head_local) for n in (upper, fore, hand))
        axis = wrist - shoulder
        span = np.linalg.norm(axis)
        if span < 1e-6:
            continue
        axis /= span
        along = (points - shoulder) @ axis / span
        radial = np.linalg.norm(points - shoulder - along[:, None] * span * axis, axis=1) / span
        # Cut across the upper arm before grouping. A lowered hem or a hanging
        # ribbon alone cannot establish a tube that surrounds the forearm.
        candidates = ids[(along[ids] > .18) & (along[ids] < 1.28) & (radial[ids] < .39)]
        if not len(candidates):
            continue
        _, islands, _ = connected_graph(obj.data, points, candidates, span * .0004)
        elbow_t = float((elbow - shoulder) @ axis / span)
        for island in islands:
            section = points[island] - shoulder - along[island, None] * span * axis
            if len(island) < 6 or np.max(along[island]) < elbow_t:
                continue
            # Require sleeve surface on both sides of its axis in two directions.
            u = np.cross(axis, (0., 0., 1.))
            if np.linalg.norm(u) < .01:
                u = np.cross(axis, (0., 1., 0.))
            u /= np.linalg.norm(u)
            if any(min(section @ normal) > -.012 * span or max(section @ normal) < .012 * span
                   for normal in (u, np.cross(axis, u))):
                continue
            for i in island:
                old = weights(obj, i)
                if any('Secondary_' in n and 'Accessory_Jacket_' not in n for n in old):
                    continue
                # Preserve shoulder transfer. Replace cuff weights as a band so
                # a duplicated seam or a stray finger seed cannot split the ring.
                t = along[i]
                blend = float(np.clip((t - .18) / .19, 0., 1.))
                joint = float(np.clip(.5 + (t - elbow_t) / .25, 0., 1.))
                palm = float(np.clip((t - .93) / .26, 0., 1.))
                blend *= blend * (3. - 2. * blend)
                joint *= joint * (3. - 2. * joint)
                palm *= palm * (3. - 2. * palm)
                target = {upper: 1. - joint, fore: joint * (1. - palm), hand: joint * palm}
                merged = {n: old.get(n, 0.) * (1. - blend) + target.get(n, 0.) * blend
                          for n in old.keys() | target.keys()}
                top = sorted(((n, w) for n, w in merged.items() if w > 1e-8),
                             key=lambda item: (-item[1], item[0]))[:4]
                total = sum(w for _, w in top)
                changed += _write(obj, i, old, {n: w / total for n, w in top})
    return changed


def refine_limb_domains(rig, meshes):
    report = {}
    for obj in meshes:
        xf = rig.matrix_world.inverted_safe() @ obj.matrix_world
        points = np.array([xf @ v.co for v in obj.data.vertices])
        if not len(points):
            continue
        report[obj.name] = dict(sleeves=_sleeve_weights(rig, obj, points), legs=_leg_tubes(rig,obj,points),
                                fingers=sum(_digit_segments(rig,obj,points,s) for s in ('L','R')))
    return report
