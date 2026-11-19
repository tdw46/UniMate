"""Bounded rest-surface clearance on the generated copy, in rig-local space."""
import numpy as np
from mathutils.bvhtree import BVHTree

from avatar_fitted_clothing import _role


def _surface(meshes, roles):
    points, triangles, owners = [], [], []
    for obj in meshes:
        mesh = obj.data
        labels = [_role(m.name) if m else None for m in mesh.materials]
        if len(labels) <= 1:
            labels = [(labels[0] if labels else None) or _role(obj.name)]
        if not roles.intersection(labels):
            continue
        mesh.calc_loop_triangles()
        corners = {}
        for tri in mesh.loop_triangles:
            if tri.material_index >= len(labels) or labels[tri.material_index] not in roles:
                continue
            face = []
            for i in tri.vertices:
                if i not in corners:
                    corners[i] = len(points)
                    points.append(tuple(mesh.vertices[i].co))
                    owners.append((obj, i))
                face.append(corners[i])
            triangles.append(face)
    return np.asarray(points), np.asarray(triangles, dtype=int), owners


def _weld(points, faces):
    # Coincident seam copies take precisely the same displacement; the input
    # topology, UV seams, vertex indices and relative shape deltas survive.
    _, ids, inverse = np.unique(points.round(7), axis=0, return_index=True, return_inverse=True)
    return points[ids], inverse[faces], inverse


def _pair_corrections(a, b, center, margin):
    # Triangle separating axes: both face normals and the nine edge crosses.
    ea, eb = np.roll(a, 1, axis=1) - a, np.roll(b, 1, axis=1) - b
    axes = np.concatenate((np.cross(ea[:, 0], ea[:, 1])[:, None],
                           np.cross(eb[:, 0], eb[:, 1])[:, None],
                           np.cross(ea[:, :, None], eb[:, None, :]).reshape(len(a), 9, 3)), axis=1)
    lengths = np.linalg.norm(axes, axis=2)
    axes /= np.maximum(lengths, 1e-15)[:, :, None]
    outward = a.mean(axis=1) - center
    outward[:, 2] = 0
    outward /= np.maximum(np.linalg.norm(outward, axis=1), 1e-12)[:, None]
    cosine = np.einsum('nic,nc->ni', axes, outward)
    axes *= np.where(cosine < 0, -1., 1.)[:, :, None]
    outer = np.einsum('nvc,nic->niv', a, axes)
    inner = np.einsum('nvc,nic->niv', b, axes)
    separation = inner.max(axis=2) - outer.min(axis=2)
    score = np.where((lengths > 1e-12) & (abs(cosine) > .12),
                     separation / (.3 + abs(cosine)), np.inf)
    row = np.arange(len(a))
    best = score.argmin(axis=1)
    valid = np.isfinite(score[row, best])
    move = axes[row, best] * (separation[row, best] + margin)[:, None]
    return move[valid], valid


def fit_clearance(rig, meshes):
    outer, faces_a, owner_a = _surface(meshes, {'shirt'})
    inner, faces_b, owner_b = _surface(meshes, {'waist', 'body'})
    if not len(faces_a) or not len(faces_b):
        return {}
    scale = (rig.data.bones['Head'].head_local - rig.data.bones['Hips'].head_local).length
    limit = scale * .009
    center = np.asarray(rig.data.bones['Hips'].head_local)
    extent = inner[faces_b]
    near = ((extent.max(axis=1) >= outer.min(axis=0) - 2*limit) &
            (extent.min(axis=1) <= outer.max(axis=0) + 2*limit)).all(axis=1)
    if not near.any() or limit <= 1e-8:
        return {}
    # Shared garment/body hem vertices can be intentional stitched boundaries.
    # Leave these stationary in both domains.
    shared = set(owner_a).intersection(owner_b)
    base_a, fa, rev_a = _weld(outer, faces_a)
    base_b, fb, rev_b = _weld(inner, faces_b[near])
    a, b = base_a.copy(), base_b.copy()
    frozen = []
    for base, rev, owners in ((a, rev_a, owner_a), (b, rev_b, owner_b)):
        mask = np.zeros(len(base), dtype=bool)
        for i, owner in enumerate(owners):
            if owner in shared:
                mask[rev[i]] = True
        frozen.append(mask)
    best_count = None
    for iteration in range(81):
        first = BVHTree.FromPolygons(a, fa, all_triangles=True)
        second = BVHTree.FromPolygons(b, fb, all_triangles=True)
        hits = first.overlap(second)
        if iteration == 0:
            initial = len(hits)
        if best_count is None or len(hits) < best_count:
            best_count = len(hits)
            best_a, best_b = a.copy(), b.copy()
        if not hits or iteration == 80:
            break
        # Bound scratch to the part of the garment that actually has contact.
        sums = [np.zeros_like(a), np.zeros_like(b)]
        counts = [np.zeros(len(a)), np.zeros(len(b))]
        for start in range(0, len(hits), 2048):
            h, j = np.asarray(hits[start:start+2048]).T
            ai, bi = fa[h], fb[j]
            delta, valid = _pair_corrections(a[ai], b[bi], center, limit*.04)
            for i, ids in enumerate((ai[valid], bi[valid])):
                correction = delta * (.5 if i == 0 else -.5)
                for column in ids.T:
                    np.add.at(sums[i], column, correction)
                    np.add.at(counts[i], column, 1.)
        changed = False
        for i, (points, base) in enumerate(((a, base_a), (b, base_b))):
            ids = np.flatnonzero((counts[i] > 0) & ~frozen[i])
            if not len(ids):
                continue
            off = points[ids] - base[ids] + sums[i][ids] / counts[i][ids, None]
            off *= np.minimum(1., limit / np.maximum(np.linalg.norm(off, axis=1), 1e-15))[:, None]
            updated = base[ids] + off
            changed |= bool(np.max(abs(updated - points[ids])) > limit * 1e-6)
            points[ids] = updated
        if not changed:
            break
    moved = {}
    for owners, rev, offset in ((owner_a, rev_a, best_a-base_a), (owner_b, rev_b, best_b-base_b)):
        for i in np.flatnonzero(np.linalg.norm(offset[rev], axis=1) > 1e-9):
            obj, vertex = owners[i]
            moved.setdefault(obj, {})[vertex] = offset[rev[i]]
    report = {}
    for obj, offsets in moved.items():
        idx = np.asarray(list(offsets))
        delta = np.asarray(list(offsets.values()))
        mesh = obj.data
        base = np.asarray([mesh.vertices[int(i)].co for i in idx])
        if mesh.shape_keys:
            for key in mesh.shape_keys.key_blocks:
                values = np.asarray([key.data[int(i)].co for i in idx]) + delta
                for i, point in zip(idx, values):
                    key.data[int(i)].co = point
        for i, point in zip(idx, base + delta):
            mesh.vertices[int(i)].co = point
        mesh.update()
        report[obj.name] = dict(vertices=len(idx), max_distance=float(np.linalg.norm(delta, axis=1).max()))
    return dict(initial_pairs=initial, remaining_pairs=best_count, limit=limit, meshes=report)
