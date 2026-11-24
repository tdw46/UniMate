"""Interpolate coat weights on the admitted rest surface."""
import math
from collections import defaultdict
from heapq import heapify, heappop, heappush

import numpy as np

from avatar_skirt_surface import angle_weights
from avatar_weight_seams import coincident_groups


def _smoothstep(t):
    return t*t*(3.-2.*t)


def _limit_four(row):
    """Fade a departing influence to zero at the change in support."""
    ranked = sorted(((n, w) for n, w in row.items() if w > 1e-9), key=lambda item: (-item[1], item[0]))
    cutoff = ranked[4][1] if len(ranked) > 4 else 0.
    limited = {n: w-cutoff for n, w in ranked[:4] if w-cutoff > 1e-9}
    if not limited:
        limited = dict(ranked[:4])
    total = sum(limited.values())
    return {n: w/total for n, w in limited.items()}


def close_enclosed_patches(points, edges, mass, repairable, height):
    """Recover bounded holes surrounded by the same coat surface."""
    coat = {i for i, amount in mass.items() if amount > 1e-6}
    adjacency = defaultdict(list)
    borders = set()
    for a, b in edges:
        if a not in coat:
            adjacency[a].append(b)
            if b in coat:
                borders.add(a)
        if b not in coat:
            adjacency[b].append(a)
            if a in coat:
                borders.add(b)
    seen = set()
    result = {}
    for start in sorted(borders):
        if start in seen or not repairable[start]:
            continue
        seen.add(start)
        connected = [start]
        rim = set()
        blocked = False
        for i in connected:
            for j in adjacency[i]:
                if j in coat:
                    rim.add(j)
                elif not repairable[j]:
                    blocked = True
                elif j not in seen:
                    seen.add(j)
                    connected.append(j)
        if blocked or len(rim) < 3 or np.ptp(points[connected], axis=0).max() > height*.052:
            continue
        # A neckline or a genuine fused body boundary remains attached.
        amount = float(np.median([mass[i] for i in rim]))
        if amount >= .65:
            result.update(dict.fromkeys(connected, amount))
    return result


def _fringe(points, edges, nodes, positions, width):
    """Measure the handoff on the inside of actual fused body edges."""
    neighbors = defaultdict(list)
    distance = {}
    for a, b in edges:
        i, j = nodes.get(a), nodes.get(b)
        if i == j:
            continue
        if i is None or j is None:
            owned = j if i is None else i
            length = float(np.linalg.norm(points[a]-points[b]))
            distance[owned] = min(length, distance.get(owned, width))
        else:
            length = float(np.linalg.norm(positions[i]-positions[j]))
            neighbors[i].append((j, length))
            neighbors[j].append((i, length))
    queue = [(length, i) for i, length in distance.items() if length < width]
    heapify(queue)
    while queue:
        length, i = heappop(queue)
        if length != distance[i]:
            continue
        for j, step in neighbors[i]:
            trial = length+step
            if trial < distance.get(j, width):
                distance[j] = trial
                heappush(queue, (trial, j))
    return distance


def interpolate_coat(points, edges, rows, body_at, guides, directions, center, height,
                     coat_mass, opening_top):
    """Use neighboring guides and weld the body handoff in the same field."""
    ids = sorted(i for i in rows if coat_mass[i] > 1e-6)
    if not ids:
        return rows
    positions = points[ids]
    span = max(float(np.linalg.norm(np.ptp(positions, axis=0))), 1e-9)
    groups = [[ids[j] for j in group] for group in coincident_groups(positions, height*.0001/span)]
    nodes = {i: j for j, group in enumerate(groups) for i in group}
    positions = np.array([points[group].mean(axis=0) for group in groups])
    width = height*.022
    fringe = _fringe(points, edges, nodes, positions, width)
    result = rows.copy()
    front = 1.5*math.pi
    for j, (group, point) in enumerate(zip(groups, positions)):
        z = point[2]
        radial = point[:2]-center
        angle = math.atan2(radial[1], radial[0]) % math.tau
        pair = angle_weights(directions, angle) if len(guides) > 1 else [(0, 1.)]
        if len(pair) == 2 and z <= opening_top:
            (a, _), (b, _) = pair
            if directions[a] < front < directions[b]:
                pair = [(a if angle < front else b, 1.)]
        amount = sum(coat_mass[i] for i in group)/len(group)
        amount *= _smoothstep(min(1., fringe.get(j, width)/width))
        row = {}
        for index, share in pair:
            heights, names = guides[index]
            level = max(0, min(len(names)-2, np.searchsorted(heights, z, side='right')-1))
            t = np.clip((z-heights[level])/(heights[level+1]-heights[level]), 0., 1.)
            mix = _smoothstep(t)
            for n, weight in ((names[level], 1.-mix), (names[level+1], mix)):
                row[n] = row.get(n, 0.) + share*amount*weight
        if amount < 1.-1e-8:
            for i in group:
                body = body_at(i) or {'Hips': 1.}
                scale = (1.-amount)/(sum(body.values())*len(group))
                for n, w in body.items():
                    row[n] = row.get(n, 0.) + scale*w
        row = _limit_four(row)
        for i in group:
            result[i] = row
    return result
