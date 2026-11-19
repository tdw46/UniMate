"""Align the rest fingers of a generated hand before any animation is authored."""

import numpy as np
from mathutils import Vector


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def _hand(rig, side):
    chains = [
        [f"{d}{j}.{side}" for j in (1, 2, 3)]
        for d in ("Index", "Middle", "Ring", "Little", "Thumb")
    ]
    bones = rig.data.bones
    if any(n not in bones for chain in chains for n in chain):
        return
    heads = np.array([[bones[n].head_local for n in chain] for chain in chains])
    segments = (
        np.array([[bones[n].tail_local for n in chain] for chain in chains]) - heads
    )
    lengths = np.linalg.norm(segments, axis=2)
    if lengths.min() < 1e-7:
        return
    forward = _unit(np.mean(segments[:4, 0] / lengths[:4, 0, None], axis=0))
    across = heads[3, 0] - heads[0, 0]
    across = _unit(across - forward * np.dot(across, forward))
    palm = np.cross(forward, across) * (-1.0 if side == "L" else 1.0)
    if np.linalg.norm(palm) < 0.9:
        return
    thumb_forward = _unit(segments[4].sum(axis=0))
    # Direct the four curl planes toward the center of the palm with a
    # shallow fan. The pinky has further to travel than the index.
    lateral = (heads[1, 0] - heads[:4, 0]) @ across
    lean = np.clip(0.32 * lateral / lengths[:4].sum(axis=1), -0.25, 0.25)
    axes = [_unit(np.cross(forward - k * across, palm + k * across)) for k in lean]
    axes.append(_unit(np.cross(thumb_forward, 0.82 * across + 0.57 * palm)))
    if any(np.linalg.norm(axis) < 0.9 for axis in axes):
        return
    plans = {}
    for d, chain in enumerate(chains):
        members = _chain_plan(chain, heads[d], segments[d], axes[d])
        if not members:
            return
        plans.update(members)
    return chains, across, plans


def _chain_plan(chain, heads, segments, axis):
    members = {}
    origin = heads[0].copy()
    for name, head, old in zip(chain, heads, segments):
        tangent = old - axis * np.dot(old, axis)
        if np.linalg.norm(tangent) < 0.55 * np.linalg.norm(old):
            return {}
        target = _unit(tangent) * np.linalg.norm(old)
        # Swing the skin with the shaft. Rolling the animation basis itself
        # must never twist a neutral nail or fingertip.
        rotation = np.array(Vector(old).rotation_difference(Vector(target)).to_matrix())
        members[name] = (origin.copy(), target, axis, rotation, head)
        origin += target
    return members


def _skin_rows(rig, meshes, names):
    for obj in meshes:
        groups = {g.index: g.name for g in obj.vertex_groups if g.name in names}
        if not groups:
            continue
        rows = {}
        for vertex in obj.data.vertices:
            for weight in vertex.groups:
                name = groups.get(weight.group)
                if name and weight.weight > 0:
                    rows.setdefault(name, []).append((vertex.index, weight.weight))
        if rows:
            packed = {}
            for name, row in rows.items():
                indices, weights = zip(*row)
                packed[name] = (np.array(indices, dtype=int), np.array(weights))
            yield (
                obj,
                np.array(rig.matrix_world.inverted_safe() @ obj.matrix_world),
                packed,
            )


def _surface_patches(chains, skins):
    """Evaluate only finger-owned faces, at generation time, in rig coordinates."""
    pieces = [[] for _ in chains]
    for obj, matrix, rows in skins:
        mesh = obj.data
        mesh.calc_loop_triangles()
        faces = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", faces)
        faces = faces.reshape(-1, 3)
        for chain, parts in zip(chains, pieces):
            if any(n not in rows for n in chain):
                continue
            w = np.zeros((len(mesh.vertices), 3))
            for j, name in enumerate(chain):
                indices, weights = rows[name]
                w[indices, j] = weights
            owned = faces[(w.sum(axis=1)[faces] > 0.88).all(axis=1)]
            if len(owned):
                ids, remap = np.unique(owned, return_inverse=True)
                points = np.array([mesh.vertices[i].co[:] for i in ids])
                parts.append(
                    (
                        points @ matrix[:3, :3].T + matrix[:3, 3],
                        w[ids],
                        remap.reshape(-1, 3),
                    )
                )
    surfaces = []
    for chain, parts in zip(chains, pieces):
        if not parts:
            return []
        points = np.concatenate([p for p, _, _ in parts])
        w = np.concatenate([weights for _, weights, _ in parts])
        offsets = np.cumsum([0] + [len(p) for p, _, _ in parts[:-1]])
        faces = np.concatenate([f + off for (_, _, f), off in zip(parts, offsets)])
        surfaces.append((points, w, faces))
    return surfaces


def _fold(chain, plans, patch, poses):
    from mathutils import Matrix

    points, w, faces = patch
    rest = points.copy()
    for j, n in enumerate(chain):
        head, _, _, swing, old = plans[n]
        rest += w[:, j, None] * ((points - old) @ (swing - np.eye(3)).T + head - old)
    sweep = []
    for angles in poses:
        root = plans[chain[0]][0].copy()
        posed = rest.copy()
        for j, (n, total) in enumerate(zip(chain, np.cumsum(angles))):
            head, direction, axis, _, _ = plans[n]
            turn = np.array(Matrix.Rotation(np.deg2rad(total), 3, Vector(axis)))
            posed += w[:, j, None] * (
                (rest - head) @ (turn - np.eye(3)).T + root - head
            )
            root += turn @ direction
        sweep.append(posed)
    return sweep, faces


_CURL = [(a, a, a) for a in range(0, 91, 15)]
_OPPOSE = (18, 45, 60)


def _lane_shifts(hand, patches):
    from mathutils.bvhtree import BVHTree

    chains, across, plans = hand
    shifts = np.zeros(3)
    if not patches:
        return shifts
    surfaces = [_fold(c, plans, p, _CURL) for c, p in zip(chains[:4], patches)]
    for d in range(3):
        gap = (plans[chains[d + 1][0]][0] - plans[chains[d][0]][0]) @ across
        if gap <= 0:
            continue
        a, fa = surfaces[d]
        b, fb = surfaces[d + 1]
        fixed = [BVHTree.FromPolygons(p, fa, all_triangles=True) for p in a]

        def clear(offset):
            return all(
                not tree.overlap(
                    BVHTree.FromPolygons(p - offset * across, fb, all_triangles=True)
                )
                for tree, p in zip(fixed, b)
            )

        if not clear(0.0):
            continue
        low, high = 0.0, 0.45 * gap
        for _ in range(7):
            mid = (low + high) * 0.5
            if clear(mid):
                low = mid
            else:
                high = mid
        shifts[d] = max(0.0, low - 0.010 * gap)
    return shifts


def _opposition(rig, hand, patches):
    """Choose one plane for the entire thumb using the folded finger surfaces."""
    from mathutils import Matrix
    from mathutils.bvhtree import BVHTree

    if not patches:
        return
    chains, _, plans = hand
    chain = chains[4]
    bones = [rig.data.bones[n] for n in chain]
    heads = np.array([b.head_local for b in bones])
    segments = np.array([b.tail_local for b in bones]) - heads
    forward = _unit(segments.sum(axis=0))
    axis = plans[chain[0]][2]
    trees = []
    for digit, patch in zip(chains[:4], patches):
        sweep, faces = _fold(digit, plans, patch, _CURL)
        trees.append(
            [BVHTree.FromPolygons(p, faces, all_triangles=True) for p in sweep]
        )
    pads = np.flatnonzero(patches[4][1][:, 2] > 0.85)
    if not len(pads):
        return
    # Probe distal points only. Full surfaces decide the collision gate.
    pads = pads[:: max(1, len(pads) // 80)]
    span = np.linalg.norm(plans[chains[1][0]][0] - plans[chains[0][0]][0])
    handedness = 1 if chain[0].endswith(".L") else -1

    def trial(degrees):
        turn = np.array(
            Matrix.Rotation(np.deg2rad(handedness * degrees), 3, Vector(forward))
        )
        proposal = _chain_plan(chain, heads, segments, turn @ axis)
        if not proposal:
            return float("inf"), {}
        settle = [tuple(90 + t * (np.array(_OPPOSE) - 90)) for t in (0.33, 0.67, 1.0)]
        sweep, faces = _fold(chain, proposal, patches[4], _CURL + settle)
        # Check both equal-X flexion and a thumb closing across the pads.
        for pts, frame in zip(
            sweep, (*range(len(_CURL)), *([len(_CURL) - 1] * len(settle)))
        ):
            thumb = BVHTree.FromPolygons(pts, faces, all_triangles=True)
            if any(thumb.overlap(tree[frame]) for tree in trees):
                return float("inf"), {}
        distance = min(
            tree[-1].find_nearest(p)[3] for tree in trees[:2] for p in sweep[-1][pads]
        )
        return abs(distance / span - 0.035) + 0.0004 * abs(degrees), proposal

    return trial


def _fit_opposition(rig, hands, patches, symmetric):
    evaluators = [
        (hand[2], test)
        for hand, patch in zip(hands, patches)
        if (test := _opposition(rig, hand, patch))
    ]
    for group in [evaluators] if symmetric else [[e] for e in evaluators]:
        if not group:
            continue
        best_error, best_angle, best = float("inf"), 0, []

        def consider(angles):
            nonlocal best_error, best_angle, best
            for angle in angles:
                trials = [test(angle) for _, test in group]
                error = max(v for v, _ in trials)
                if error < best_error:
                    best_error, best_angle, best = error, angle, trials

        consider(range(-32, 33, 4))
        consider(range(best_angle - 3, best_angle + 4))
        for (plans, _), (_, proposal) in zip(group, best):
            plans.update(proposal)


def _carry_skin(obj, matrix, rows, plans):
    if obj.data.users > 1:
        obj.data = obj.data.copy()
    mesh = obj.data
    ids = np.unique(np.concatenate([row[0] for row in rows.values()]))
    linear = np.zeros((len(ids), 3, 3))
    offset = np.zeros((len(ids), 3))
    transform, origin = matrix[:3, :3], matrix[:3, 3]
    inverse = np.linalg.inv(transform)
    for name, (indices, weights) in rows.items():
        head, _, _, swing, old = plans[name]
        delta = swing - np.eye(3)
        local = np.searchsorted(ids, indices)
        linear[local] += weights[:, None, None] * (inverse @ delta @ transform)
        offset[local] += weights[:, None] * (
            inverse @ (delta @ (origin - old) + head - old)
        )

    def apply(points):
        coords = np.empty((len(points), 3), dtype=np.float32)
        points.foreach_get("co", coords.ravel())
        coords[ids] += np.einsum("nij,nj->ni", linear, coords[ids]) + offset
        points.foreach_set("co", coords.ravel())

    apply(mesh.vertices)
    if mesh.shape_keys:
        for key in mesh.shape_keys.key_blocks:
            apply(key.data)
    mesh.update()


def fit_fist_axes(rig, meshes, symmetric=False, *, correct_geometry=False):
    """Share one flexion axis down each digit. Only rebase owned output meshes."""
    import bpy

    hands = [hand for side in ("L", "R") if (hand := _hand(rig, side))]
    plans = {n: p for _, _, members in hands for n, p in members.items()}
    if not plans:
        return 0
    if correct_geometry:
        skins = list(_skin_rows(rig, meshes, plans))
        weighted = {n for _, _, rows in skins for n in rows}
        hands = [hand for hand in hands if hand[2].keys() <= weighted]
        plans = {n: p for _, _, members in hands for n, p in members.items()}
        if not plans:
            return 0
        skins = [
            (obj, matrix, {n: row for n, row in rows.items() if n in plans})
            for obj, matrix, rows in skins
            if plans.keys() & rows.keys()
        ]
        patches = [_surface_patches(hand[0], skins) for hand in hands]
        shifts = [_lane_shifts(hand, patch) for hand, patch in zip(hands, patches)]
        if symmetric and len(shifts) == 2:
            shifts = [np.minimum(*shifts)] * 2
        for (chains, across, _), gaps in zip(hands, shifts):
            moves = -np.r_[0.0, np.cumsum(gaps)]
            moves -= moves[1]
            for chain, dist in zip(chains, moves):
                for name in chain:
                    plans[name][0][:] += dist * across
        _fit_opposition(rig, hands, patches, symmetric)
        plans = {n: p for _, _, members in hands for n, p in members.items()}
        for obj, matrix, rows in skins:
            _carry_skin(obj, matrix, rows, plans)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        for name, (head, direction, axis, _, _) in plans.items():
            bone = rig.data.edit_bones[name]
            if correct_geometry:
                bone.head = Vector(head)
                bone.tail = Vector(head + direction)
            bone.align_roll(Vector(axis).cross(bone.tail - bone.head))
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    return len(plans)
