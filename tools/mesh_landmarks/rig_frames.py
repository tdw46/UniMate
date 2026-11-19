"""Geometry-derived rest frames for generated upright, -Y-facing characters."""
from copy import deepcopy
import numpy as np
from mathutils import Vector


def symmetric_prediction(prediction):
    """Average both sides around the fitted sagittal plane before binding."""
    result = deepcopy(prediction)
    joints = result['joints']
    pairs = [(left, 'Right' + left[4:]) for left in joints if left.startswith('Left')]
    offsets = [np.asarray(joints[a]['head']) - joints[b]['head'] for a, b in pairs]
    # Project onto the floor so the rest pose stays upright. Digit pairs give
    # a stable lateral axis even when shoulders or elbows are slightly uneven.
    normal = np.median([d[:2] / np.linalg.norm(d[:2]) for d in offsets
                        if np.linalg.norm(d[:2]) > 1e-8], axis=0)
    normal = np.array([*normal, 0.])
    normal /= np.linalg.norm(normal)
    center = float(np.median([(np.asarray(joints[a]['head']) + joints[b]['head']) @ normal * .5
                              for a, b in pairs]))
    def reflect(p):
        return p - 2 * (p @ normal - center) * normal
    for key in ('head', 'tail'):
        for a, b in pairs:
            point = (np.asarray(joints[a][key]) + reflect(np.asarray(joints[b][key]))) * .5
            joints[a][key], joints[b][key] = point.tolist(), reflect(point).tolist()
        for name, joint in joints.items():
            if not name.startswith(('Left', 'Right')):
                p = np.asarray(joint[key])
                joint[key] = ((p + reflect(p)) * .5).tolist()
    return result


def humanoid_rolls(bones):
    """Initialize VRM rest frames. Finger flexion axes are fitted after binding."""
    for bone in bones:
        bone.roll = 0.


def symmetrize_guides(chains, rig, height):
    """Average reciprocal bilateral guide paths. Preserve unique anatomy."""
    normal = np.array(rig.data.bones['Hand.L'].head_local - rig.data.bones['Hand.R'].head_local)
    normal[2] = 0.
    normal /= np.linalg.norm(normal)
    origin = np.array(rig.data.bones['Hips'].head_local)
    roles = [c['role'].removesuffix('_l').removesuffix('_r') for c in chains]
    paths = [np.asarray(c['line'], dtype=float) for c in chains]
    def mirror(path):
        return path - 2 * ((path-origin) @ normal)[..., None] * normal
    nearest = {}
    for i, path in enumerate(paths):
        side = (path.mean(axis=0)-origin) @ normal
        if abs(side) < .003 * height:
            continue
        reflected = mirror(path)
        for j, other in enumerate(paths):
            if path.shape != other.shape or ((other.mean(axis=0)-origin) @ normal)*side >= 0:
                continue
            if roles[i] != roles[j]:
                continue
            distance = np.linalg.norm(reflected-other, axis=1).max()
            if distance < .045 * height and distance < nearest.get(i, (float('inf'),))[0]:
                nearest[i] = distance, j
    for i, (_, j) in nearest.items():
        if i < j and nearest.get(j, (None,None))[1] == i:
            average = (paths[i] + mirror(paths[j])) * .5
            chains[i]['line'] = [Vector(p) for p in average]
            chains[j]['line'] = [Vector(p) for p in mirror(average)]
