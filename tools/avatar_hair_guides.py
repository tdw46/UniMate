"""Find hanging branches through mesh connectivity without joining hair gaps."""
import numpy as np
from mathutils import Vector


def hanging_branches(mesh, ids, points, free_top, root_length):
    """Cut only the guide graph at the cap, never the actual mesh.

    A scalp sheet may connect many bangs/locks. Below the attachment level each
    connected branch gets its own guide and weights. Crossing edges provide
    exact boundary samples so sparse topology cannot shift the root sideways.
    Sideways strands are retained: connectivity, not a bone angle, partitions
    them. The short fixed root follows that branch's own incoming edges.
    """
    ids = set(ids)
    free = {i for i in ids if points[i, 2] < free_top}
    neighbors = {i: set() for i in free}
    crossing = {}
    edges = []
    for edge in mesh.edges:
        a, b = edge.vertices
        if a not in ids or b not in ids:
            continue
        edges.append((a, b))
        if a in free and b in free:
            neighbors[a].add(b)
            neighbors[b].add(a)
        elif (a in free) != (b in free):
            low, high = (a, b) if a in free else (b, a)
            delta = points[high] - points[low]
            t = (free_top - points[low, 2]) / delta[2]
            crossing.setdefault(low, []).append((points[low] + t*delta, delta))
    remaining = set(free)
    result = []
    while remaining:
        seed = min(remaining)
        found, pending = {seed}, [seed]
        while pending:
            for i in neighbors[pending.pop()] - found:
                found.add(i)
                pending.append(i)
        remaining -= found
        boundary = [pair for i in sorted(found) for pair in crossing.get(i, [])]
        if len(found) < 2 or not boundary:
            continue
        branch_ids = sorted(found)
        samples = np.unique(np.array([p for p, _ in boundary]), axis=0)
        anchor = np.median(samples, axis=0)
        direction = np.median([d / max(np.linalg.norm(d), 1e-12) for _, d in boundary], axis=0)
        direction /= max(np.linalg.norm(direction), 1e-12)
        # Fixed attachment remains short even for almost horizontal branches.
        root = anchor + direction*root_length
        result.append(dict(ids=branch_ids, points=np.vstack((points[branch_ids], samples)),
                           anchor=Vector(anchor), root=Vector(root),
                           edges=np.array([[points[a], points[b]] for a, b in edges
                                           if a in found or b in found])))
    return result


def branch_line(branch, segments):
    """Use actual edge/height intersections instead of uneven vertex bands."""
    edges = branch['edges']
    a, b = edges[:, 0], edges[:, 1]
    delta = b-a
    nonhorizontal = np.abs(delta[:, 2]) > 1e-10
    result = [branch['anchor']]
    low = float(branch['points'][:, 2].min())
    for z in np.linspace(branch['anchor'].z, low, segments+1)[1:]:
        mask = nonhorizontal & (np.minimum(a[:, 2], b[:, 2]) <= z+1e-7) & (np.maximum(a[:, 2], b[:, 2]) >= z-1e-7)
        if mask.any():
            t = np.clip((z-a[mask, 2])/delta[mask, 2], 0., 1.)
            samples = a[mask]+delta[mask]*t[:, None]
        else:
            points = branch['points']
            nearest = np.abs(points[:, 2]-z)
            samples = points[nearest <= nearest.min()+1e-7]
        center = np.median(np.unique(samples, axis=0), axis=0)
        result.append(Vector((center[0], center[1], z)))
    return result
