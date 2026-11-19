"""Center elbow and wrist pivots on a continuous arm section inside loose sleeves."""
import numpy as np

from .finger_surface import _sections


def _joint_center(triangles, point, axis, length):
    # A wrist strap or an isolated fold can look like a small arm on one plane.
    # Require the same tube on both sides of the predicted hinge.
    offsets = np.array([-.075, -.038, 0., .038, .075]) * length
    nearby = np.linalg.norm(triangles - point, axis=2).min(axis=1) < length * .65
    local = triangles[nearby]
    slices = []
    for offset in offsets:
        station = point + axis * offset
        sections = _sections(local, station, axis, length * 2.)
        slices.append([(center - offset * axis, radius) for center, radius in sections
                       if .058 * length < radius < .45 * length
                       and np.linalg.norm(center - station) < .39 * length])
    candidates = []
    for center, radius in slices[2]:
        track = []
        for rings in slices:
            matches = [(p, r) for p, r in rings
                       if .62 * radius < r < 1.55 * radius
                       and np.linalg.norm(p - center) < radius * .58]
            if not matches:
                break
            track.append(min(matches, key=lambda pair: np.linalg.norm(pair[0]-center)))
        if len(track) == len(offsets):
            candidates.append((np.median([p for p, r in track], axis=0), radius))
    if not candidates:
        return None
    # Start at the best-supported outer outline, then descend through nested
    # contours. Never jump to a neighboring disconnected tube or piece of trim.
    outer, radius = min(candidates, key=lambda item: max(0., np.linalg.norm(item[0]-point)-item[1]))
    nested = [(p,r) for p,r in candidates if np.linalg.norm(p-outer)+r <= radius*1.10]
    return min(nested, key=lambda item: item[1])[0]


def fit_arm_pivots(vertices, faces, joints):
    triangles = vertices[faces]
    adjusted = {}
    for side in ('Left', 'Right'):
        names = [side + part for part in ('Arm', 'ForeArm', 'Hand')]
        if any(name not in joints for name in names):
            continue
        upper, lower, hand = (joints[name] for name in names)
        shoulder, elbow, wrist, palm = (np.asarray(p, dtype=float) for p in
                                        (upper['head'], lower['head'], hand['head'], hand['tail']))
        for name, parent, child, before, point, after in (
            ('elbow', upper, lower, shoulder, elbow, wrist),
            ('wrist', lower, hand, elbow, wrist, palm),
        ):
            incoming, outgoing = point-before, after-point
            length, next_length = np.linalg.norm(incoming), np.linalg.norm(outgoing)
            if min(length, next_length) < 1e-7:
                continue
            axis = incoming/length + outgoing/next_length
            norm = np.linalg.norm(axis)
            if norm < .5:
                continue
            center = _joint_center(triangles, point, axis/norm, length)
            if center is not None:
                parent['tail'] = center.tolist()
                child['head'] = center.tolist()
                adjusted[side+name] = float(np.linalg.norm(center-point))
    return adjusted
