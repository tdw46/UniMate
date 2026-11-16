"""Fit bilateral skirt guides to horizontal sections of the rest garment."""
from collections import OrderedDict, defaultdict
from bisect import bisect_right
import math

import numpy as np


def sector_angles(count, front_counts=None):
    # One outside seam per side. Keep front and back guides paired so each
    # remains attached to its own leg even on a continuous skirt.
    half = count // 2
    angles = []
    for side, front in zip((1, -1), front_counts or (half // 2, half // 2)):
        side_angles = [0.]
        for sign, steps in ((-1, front), (1, half - half // 2 - 1)):
            side_angles.extend(sign * math.pi / 2 * i / (steps + .5)
                               for i in range(1, steps + 1))
        angles.extend(a if side == 1 else math.pi - a for a in side_angles)
    return sorted(a % math.tau for a in angles)


def angle_weights(angles, angle):
    a = (bisect_right(angles, angle % math.tau) - 1) % len(angles)
    b = (a + 1) % len(angles)
    t = ((angle - angles[a]) % math.tau) / ((angles[b] - angles[a]) % math.tau)
    return ((a, 1 - t), (b, t))


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def garment_panel_indices(triangles):
    """Choose broad cloth panels independently of disconnected trim and straps."""
    points = triangles.reshape(-1, 3)
    tolerance = max(float(np.ptp(points, axis=0).max()) * 1e-6, 1e-12)
    _, welded = np.unique(np.rint((points - points.min(axis=0)) / tolerance).astype(np.int64),
                          axis=0, return_inverse=True)
    faces = welded.reshape(-1, 3)
    parent = list(range(len(faces)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, face in enumerate(faces):
        for v in face:
            other = owners.setdefault(v, i)
            parent[root(i)] = root(other)
    groups = defaultdict(list)
    for i in range(len(parent)):
        groups[root(i)].append(i)
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1)
    areas = {k: area[ids].sum() for k, ids in groups.items()}
    largest = max(areas.values())
    return [i for k, ids in groups.items() if areas[k] >= largest * .12 for i in ids]


def garment_panels(triangles):
    return triangles[garment_panel_indices(triangles)]


class Section:
    def __init__(self, segments):
        self.segments = segments
        self.gaps = []
        self.arc_samples = {}
        if not len(segments):
            return
        # Union the angular intervals occupied by triangles. UV splits and
        # separate front/back faces do not create gaps at coincident seams.
        angle = np.arctan2(segments[:, :, 0], -segments[:, :, 1])
        start = angle[:, 0]
        delta = (angle[:, 1] - start + math.pi) % math.tau - math.pi
        left = start + np.minimum(delta, 0)
        right = start + np.maximum(delta, 0)
        order = left.argsort()
        left, right = left[order], np.maximum.accumulate(right[order])
        for i in np.flatnonzero(left[1:] - right[:-1] > 1e-5):
            a, b = right[i], left[i + 1]
            if -math.pi + .02 < a < b < math.pi - .02:
                self.gaps.append((float(a), float(b)))

    def hit(self, angle, previous=None):
        if not len(self.segments):
            return None
        direction = np.array((math.cos(angle), math.sin(angle)))
        a, b = self.segments[:, 0], self.segments[:, 1]
        da, db = _cross(a, direction), _cross(b, direction)
        use = ((da <= 1e-9) & (db >= -1e-9)) | ((db <= 1e-9) & (da >= -1e-9))
        if not np.any(use):
            return None
        a, b, da, db = a[use], b[use], da[use], db[use]
        t = np.divide(da, da - db, out=np.zeros_like(da), where=abs(da - db) > 1e-14)
        hits = a + (b - a) * np.clip(t, 0, 1)[:, None]
        hits = hits[hits @ direction > 0]
        if not len(hits):
            return None
        if previous is None:
            return hits[np.argmax(hits @ direction)]
        return hits[np.argmin(np.sum((hits - previous) ** 2, axis=1))]

    def arc_angle(self, start, end, fraction):
        if fraction >= 1 - 1e-10:
            return end
        key = (start, end)
        if key not in self.arc_samples:
            angles = np.linspace(start, end, 13)
            hits = []
            for angle in angles:
                hit = self.hit(angle, hits[-1] if hits else None)
                if hit is None:
                    break
                hits.append(hit)
            self.arc_samples[key] = None
            if len(hits) == len(angles):
                lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(hits, axis=0), axis=1))]
                if lengths[-1] > 0:
                    self.arc_samples[key] = (lengths, angles)
        samples = self.arc_samples[key]
        if samples is None:
            return start + fraction*(end-start)
        lengths, angles = samples
        return float(np.interp(fraction*lengths[-1], lengths, angles))

    def nearby_edge(self, angle, limit):
        if not len(self.segments):
            return None
        ends = self.segments.reshape(-1, 2)
        angles = np.arctan2(ends[:, 1], ends[:, 0])
        distance = abs((angles-angle+math.pi) % math.tau - math.pi)
        j = distance.argmin()
        return ends[j] if distance[j] <= limit else None


class SkirtSurface:
    def __init__(self, triangles, center, top, count):
        self.center = np.asarray(center)
        triangles = np.asarray(triangles, dtype=float)
        self.face_indices = garment_panel_indices(triangles)
        self.triangles = triangles[self.face_indices]
        self.triangles[:, :, :2] -= self.center
        self.scale = np.ptp(self.triangles.reshape(-1, 3), axis=0).max()
        z = self.triangles[:, :, 2]
        self.low, self.high = z.min(axis=1), z.max(axis=1)
        self.bottom = float(self.low.min())
        self.top = float(top)
        self.eps = (self.high.max() - self.bottom) * 1e-7
        self.sections = OrderedDict()
        self.edge_fallback = set()
        self.angles = sector_angles(count)
        self.levels = np.linspace(self.top, self.bottom + self.eps, 49)
        # Track a persistent front opening. A tiny isolated hole in a pocket
        # or waistband should not change where the whole skirt is rigged.
        runs = []
        active = []
        for z in self.levels:
            following = []
            for gap in self.section(z).gaps:
                if abs(sum(gap) / 2) > .7 or max(abs(a) for a in gap) >= math.pi / 2:
                    continue
                matches = [(max(abs(a-b) for a, b in zip(row[-1][1], gap)), j)
                           for j, row in enumerate(active)]
                distance, j = min(matches, default=(math.inf, None))
                if distance < .28:
                    run = active.pop(j)
                else:
                    run = []
                run.append((float(z), gap))
                following.append(run)
            runs.extend(active)
            active = following
        runs.extend(active)
        self.opening = max(runs, key=len, default=[])
        if len(self.opening) < 3:
            self.opening = []
        else:
            # Allocate front guides by the fabric left on each bank. The
            # rear keeps its established spacing as the opening widens.
            gaps = np.array([gap for _, gap in self.opening])
            arcs = math.pi / 2 - np.median(np.abs(gaps), axis=0)
            half = count // 2
            front = [int(np.clip(math.ceil((half - .5) * arc / (arc + math.pi / 2)), 1, half - 2))
                     for arc in arcs[::-1]]
            self.angles = sector_angles(count, front)
        self.front = [min((i for i, a in enumerate(self.angles) if sign * math.cos(a) > 0),
                          key=lambda i: abs(self.angles[i] - 1.5 * math.pi))
                      for sign in (1, -1)]

    def section(self, z):
        if z in self.sections:
            self.sections.move_to_end(z)
        else:
            tri = self.triangles[(self.low <= z) & (self.high > z)]
            above = tri[:, :, 2] > z
            # Every triangle crossing this plane has one vertex on one side
            # and two on the other. Intersect the two incident edges.
            odd = np.where(above.sum(axis=1)[:, None] == 1, above, ~above).argmax(axis=1)
            row = np.arange(len(tri))
            a = tri[row, odd][:, None, :]
            b = tri[row[:, None], (odd[:, None] + [1, 2]) % 3]
            t = (z - a[:, :, 2]) / (b[:, :, 2] - a[:, :, 2])
            if len(self.sections) >= 96:
                self.sections.popitem(last=False)
            self.sections[z] = Section(a[:, :, :2] + (b[:, :, :2] - a[:, :, :2]) * t[:, :, None])
        return self.sections[z]

    def hit(self, i, z, previous=None):
        section = self.section(z)
        angle = self.angles[i]
        if self.opening and math.sin(angle) < -1e-8:
            _, nearby = min(self.opening, key=lambda sample: abs(sample[0] - z))
            gap = min(section.gaps, default=nearby,
                      key=lambda g: max(abs(a-b) for a, b in zip(nearby, g)))
            # Carry the same panel edge to the waistband above a slit. At
            # the bottom, stop at its hem instead of turning across the back.
            if (max(abs(a-b) for a, b in zip(nearby, gap)) > .25
                    or max(abs(a) for a in gap) >= math.pi / 2):
                gap = nearby
            side = 1 if math.cos(angle) > 0 else -1
            edge = gap[1 if side > 0 else 0]
            bank = next(j for j in self.front if side * math.cos(self.angles[j]) > 0)
            fraction = abs(math.asin(math.sin(angle)) / math.asin(math.sin(self.angles[bank])))
            start = -math.pi / 2 + side * math.pi / 2
            angle = section.arc_angle(start, edge - math.pi / 2, fraction)
        point = section.hit(angle, previous)
        if point is None and i in self.edge_fallback:
            return section.nearby_edge(angle, math.pi / len(self.angles))
        return point

    def interval(self, i):
        # An irregular hem can end at a different height on each strip. Stop
        # at the first loss of support below the waist, then refine the boundary.
        valid = [self.hit(i, z) is not None for z in self.levels]
        if not any(valid):
            # A slit can coincide exactly with a requested side seam. Keep a
            # guide on its closest bank when the center ray is empty throughout.
            if i in self.edge_fallback:
                return None
            self.edge_fallback.add(i)
            return self.interval(i)
        runs = []
        for j, present in enumerate(valid):
            if present:
                if j == 0 or not valid[j - 1]:
                    runs.append([j, j])
                runs[-1][1] = j
        start, end = max(runs, key=lambda r: r[1] - r[0])
        def boundary(inside, outside):
            for _ in range(14):
                mid = (inside + outside) / 2
                if self.hit(i, mid) is None:
                    outside = mid
                else:
                    inside = mid
            return float(inside)
        top = self.levels[start] if not start else boundary(self.levels[start], self.levels[start - 1])
        bottom = self.levels[end] if end + 1 == len(valid) else boundary(self.levels[end], self.levels[end + 1])
        if top - bottom <= self.eps * 10:
            return None
        return float(top), float(bottom)

    def guide(self, i, profile):
        heights = profile['heights']
        line = []
        # Trace from the free fabric into the waist. Choosing the outermost
        # hit at the waistband can put the first bone on an overlapping belt.
        trace_z = np.linspace(heights[-1], heights[0], max(len(heights)*4, 49))
        trace_xy = []
        previous = None
        for z in trace_z:
            previous = self.hit(i, z, previous)
            if previous is None:
                return None
            trace_xy.append(previous)
        trace_xy = np.asarray(trace_xy)
        def sample(z):
            guess = [np.interp(z, trace_z, trace_xy[:, axis]) for axis in (0, 1)]
            return self.hit(i, z, guess)
        for z in heights:
            xy = sample(z)
            if xy is None:
                return None
            line.append(np.r_[self.center + xy, z])

        tolerance = self.scale * .004
        def follow(a, b, depth):
            # A straight bone may cut a fold even when both joints lie on it.
            # Subdivide only where the measured surface deviates from the chord.
            if depth == 3 or a[2] - b[2] < 2*tolerance:
                return [a]
            samples = []
            for t in (.25, .5, .75):
                chord = a + t*(b-a)
                hit = sample(chord[2])
                if hit is not None:
                    p = np.r_[self.center + hit, chord[2]]
                    samples.append((np.linalg.norm(chord-p), p))
            if samples:
                error, point = max(samples, key=lambda row: row[0])
                if error > tolerance:
                    return follow(a, point, depth+1) + follow(point, b, depth+1)
            return [a]

        fitted = []
        for a, b in zip(line, line[1:]):
            fitted.extend(follow(a, b, 0))
        fitted.append(line[-1])
        # Adaptive subdivision can leave a very short bone beside a nominal
        # station. Merge it only when the same cloth tolerance is satisfied
        # across the combined span. Preserve anatomical knee and edge stations.
        knee = heights[profile['knee_index']] if profile['knee_index'] is not None else None
        nominal = (heights[0] - heights[-1])/(len(heights)-1)
        opening_bank = self.opening and math.sin(self.angles[i]) < -1e-8
        for j in range(len(fitted)-2 if opening_bank else 0, 0, -1):
            a, p, b = fitted[j-1:j+2]
            if p[2] == knee or a[2]-b[2] > nominal*1.5:
                continue
            if min(a[2]-p[2], p[2]-b[2]) > nominal*.48:
                continue
            for z in sorted({p[2], *np.linspace(a[2], b[2], 9)[1:-1]}):
                xy = sample(z)
                t = (z-a[2])/(b[2]-a[2])
                if xy is None or np.linalg.norm(self.center+xy - (a+t*(b-a))[:2]) > tolerance:
                    break
            else:
                fitted.pop(j)
        if profile['knee_index'] is not None:
            knee = heights[profile['knee_index']]
            profile['knee_index'] = next(j for j, p in enumerate(fitted) if p[2] == knee)
        profile['heights'] = [p[2] for p in fitted]
        return fitted
