"""Crease-aware hand/garment ownership in the isolated inference runtime.

Cut the face adjacency graph, not the vertex graph: a cheap concave edge then
means cutting *along* that crease. Proximity to predicted fingers alone cannot
distinguish a curled fingertip from the garment touching it.
"""
import json
import sys
import numpy as np


def surface_owners(vertices, faces, cuts, prediction):
    import trimesh
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import maximum_flow, breadth_first_order

    mesh = trimesh.Trimesh(vertices, faces, process=False)
    points = np.asarray(mesh.triangles_center)
    height = float(np.ptp(vertices[:, 2]))
    joints = prediction['joints']
    origin = joints['Hips']['head'][0]
    adjacent = mesh.face_adjacency
    edges = mesh.face_adjacency_edges
    length = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    concavity = np.where(mesh.face_adjacency_convex, 0., mesh.face_adjacency_angles)
    cost = np.maximum(1, 1000 * length / np.median(length) *
                      (.002 + np.exp(-30 * concavity))).astype(np.int64)
    count = len(faces)
    owners = np.zeros(count, dtype=np.int32)
    for cut in cuts:
        side = cut['side']
        sign = 1 if side == 'Left' else -1
        curve = np.asarray(cut['curve'])
        boundary = np.interp(points[:, 2], curve[::-1, 1], curve[::-1, 0])
        bottom = min(joint['tail'][2] for name, joint in joints.items()
                     if name.startswith(side + 'Hand'))
        top = curve[0, 1]
        # Only unequivocal outer arm surfaces are hard hand seeds. Let the
        # crease solve recover the thumb, cuff and curled fingers as a whole.
        arm = ((points[:, 0] - boundary) * sign > height * .025)
        arm &= (points[:, 2] < top - height * .015)
        arm &= points[:, 2] > bottom - height * .004
        span = abs(joints[side + 'Arm']['head'][0] - origin)
        body = (sign * (points[:, 0] - origin) < span * .70)
        body |= (points[:, 2] > top) | (points[:, 2] < bottom - height * .016)
        arm &= ~body
        source, sink = np.flatnonzero(arm), np.flatnonzero(body)
        if len(source) < 3 or len(sink) < 3:
            raise ValueError('Fused-hand separation lacks reliable surface seeds; keep the source and review the predicted joints')
        rows = np.r_[adjacent[:, 0], adjacent[:, 1], np.full(len(source), count), sink]
        cols = np.r_[adjacent[:, 1], adjacent[:, 0], source, np.full(len(sink), count+1)]
        capacity = coo_matrix((np.r_[cost, cost, np.full(len(source)+len(sink), 100000000)],
                              (rows, cols)), shape=(count+2, count+2)).tocsr()
        flow = maximum_flow(capacity, count, count+1)
        residual = capacity - flow.flow
        residual.data = (residual.data > 0).astype(np.int64)
        residual.eliminate_zeros()
        reachable = breadth_first_order(residual, count, directed=True, return_predecessors=False)
        owned = reachable[reachable < count]
        if np.any(owners[owned]):
            raise ValueError('Fused-hand surface regions overlap; review the predicted arm joints')
        owners[owned] = sign
    return owners


if __name__ == '__main__':
    with np.load(sys.argv[1], allow_pickle=False) as data:
        request = json.loads(str(data['request']))
        owners = surface_owners(data['vertices'], data['faces'], request['cuts'], request['prediction'])
    np.save(sys.argv[2], owners, allow_pickle=False)
