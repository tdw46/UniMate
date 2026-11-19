"""Trace slender upward branches emerging from broad hair caps."""
import heapq
import numpy as np
from mathutils import Vector


def crown_strands(graph, ids, points, height, crown):
    from avatar_hair_guides import hanging_branches
    if points[ids, 2].max() < crown + height * .015:
        return []
    ids_set = set(ids)
    edges = np.array([e.vertices for e in graph.edges if all(i in ids_set for i in e.vertices)], dtype=int)
    if not len(edges):
        return []
    a, b = points[edges[:, 0]], points[edges[:, 1]]
    delta = b-a
    top = float(points[ids, 2].max())
    previous = 0.
    # Search for the narrow neck above a broad scalp section. A smooth dome
    # tapers too late to leave a usable strand beyond the cut.
    for z in np.arange(max(crown-.025*height, top-.12*height), top-.018*height, .002*height):
        mask = (a[:, 2] < z) != (b[:, 2] < z)
        if not mask.any():
            continue
        section = a[mask] + delta[mask] * ((z-a[mask, 2])/delta[mask, 2])[:, None]
        width = np.linalg.norm(np.ptp(section[:, :2], axis=0))
        if previous > .028*height and width < .50*previous and width < .022*height:
            inverted = points.copy()
            inverted[:, 2] *= -1
            branches = hanging_branches(graph, ids, inverted, -z, height*.002)
            result = []
            for branch in branches:
                members = set(branch['ids'])
                anchor = np.asarray(branch['anchor']).copy()
                anchor[2] = z
                neighbors = {i: [] for i in members}
                distance = {}
                for u, v in edges:
                    if u in members and v in members:
                        length = float(np.linalg.norm(points[u]-points[v]))
                        neighbors[u].append((v,length)); neighbors[v].append((u,length))
                    elif (u in members) != (v in members):
                        i = u if u in members else v
                        distance[i] = float(np.linalg.norm(points[i]-anchor))
                queue = [(d,i) for i,d in distance.items()]
                heapq.heapify(queue)
                while queue:
                    d,i = heapq.heappop(queue)
                    if d != distance[i]: continue
                    for j, length in neighbors[i]:
                        if d+length < distance.get(j, float('inf')):
                            distance[j] = d+length
                            heapq.heappush(queue,(d+length,j))
                if not distance: continue
                tip = max(distance, key=distance.get)
                reach = distance[tip]
                if reach < height*.025 or points[tip,2]-z < .35*reach: continue
                # Arc-distance slices follow the tip even if it curves back down.
                count = int(np.clip(np.ceil(reach / (.014*height)), 3, 7))
                line = [Vector(anchor)]
                for t in np.linspace(0.,reach,count+1)[1:-1]:
                    samples = [points[i]+(points[j]-points[i])*((t-d)/(distance[j]-d))
                               for i,d in distance.items() if d <= t
                               for j,_ in neighbors[i] if distance[j] > t]
                    if samples: line.append(Vector(np.median(samples,axis=0)))
                line.append(Vector(points[tip]))
                result.append((line,{int(i):d/reach for i,d in distance.items()}))
            return result
        previous = width
    return []
