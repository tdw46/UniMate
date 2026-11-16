"""Fit free cloth panels and swept head strands from their rest mesh."""
from bisect import bisect_right

import numpy as np
from mathutils import Vector

from avatar_garment_straps import _free_weights
from avatar_virtual_topology import connected_graph


def _section_rails(points, triangles, axis, levels, fractions):
    """Intersect triangles at each station; bridge air gaps between ribbon islands."""
    tri = points[triangles]
    depth = tri @ axis
    rows = []
    stations = []
    for k, level in enumerate(levels):
        plane = np.clip(level, depth.min()+1e-8, depth.max()-1e-8)
        mask = (depth.min(1) <= plane) & (depth.max(1) > plane)
        if not mask.any():
            continue
        d, p = depth[mask], tri[mask]
        above = d > plane
        odd = np.where(above.sum(1)[:, None] == 1, above, ~above).argmax(1)
        i = np.arange(len(d))[:, None]
        a = p[i, odd[:, None]]
        b = p[i, (odd[:, None]+[1, 2]) % 3]
        hit = a + (b-a)*((plane-a@axis)/((b-a)@axis))[:, :, None]
        mids = hit.mean(1)
        lengths = np.linalg.norm(hit[:, 1]-hit[:, 0], axis=1)
        if lengths.sum() <= 1e-12:
            continue
        order = np.argsort(mids[:, 0])
        cdf = (np.cumsum(lengths[order])-.5*lengths[order])/lengths.sum()
        rows.append([np.average(mids, axis=0, weights=lengths) if len(fractions)==1 else
                     [np.interp(f, cdf, mids[order, dim]) for dim in range(3)] for f in fractions])
        stations.append(k)
    if len(rows) < 2:
        return []
    data = np.asarray(rows)
    return [np.column_stack([np.interp(np.arange(len(levels)), stations, rail[:, d])
                             for d in range(3)]) for rail in data.transpose(1, 0, 2)]


def _smooth_panel_row(controls, station):
    """Blend four deform bones on a regular grid along the free cloth."""
    last = len(controls) - 1
    cell = min(int(station), last - 1)
    f = min(1., station - cell)
    # These weights and their first derivatives agree across cell boundaries.
    # Ease in from the fitted row; keep the hem distributed at the other end.
    ease = min(1., station)
    ease = ease * ease * (3. - 2. * ease)
    outer_left = .25 * (1. - f) ** 2 * ease
    outer_right = .25 * f ** 2 * ease
    weights = (outer_left, 1. - f - 2. * outer_left + outer_right,
               f + outer_left - 2. * outer_right, outer_right)
    row = {}
    for i, weight in enumerate(weights, cell - 1):
        if weight > 0.:
            name = controls[min(last, max(0, i))]
            row[name] = row.get(name, 0.) + weight
    return row


def _smooth_fraction(value):
    value = max(0., min(1., value))
    return value * value * (3. - 2. * value)


def _four_weights(row):
    strongest = sorted(row.items(), key=lambda item: -item[1])[:4]
    total = sum(weight for _, weight in strongest)
    return {name: weight / total for name, weight in strongest if weight > 0.}


def _blend_panel_handoff(row, middle, cloth_names, station, cut):
    """Spread the change of rotation across the join in a fitted panel."""
    start = max(1, cut - 2)
    end = min(len(middle) - 1, cut + min(5, cut))
    if not start < station < end:
        return row

    t = (station - start) / (end - start)
    free_share = t + .5 * t * (1. - t) * (2. * t - 1.)
    edge = min(station - start, end - station)
    share = .85 * _smooth_fraction(edge / ((end - cut) * .5))

    body = {name: weight for name, weight in row.items() if name not in cloth_names}
    fabric_weight = sum(weight for name, weight in row.items() if name in cloth_names)
    if not fabric_weight:
        return row
    cloth = {name: weight * (1. - share) for name, weight in row.items() if name in cloth_names}
    for name, fraction in ((middle[start - 1], 1. - free_share), (middle[end - 1], free_share)):
        cloth[name] = cloth.get(name, 0.) + fabric_weight * share * fraction
    # Reserve the pelvis/thigh shares before choosing the remaining cloth slots.
    strongest = sorted(cloth, key=cloth.get, reverse=True)[:4 - len(body)]
    total = sum(cloth[name] for name in strongest)
    body.update({name: fabric_weight * cloth[name] / total for name in strongest if cloth[name] > 0.})
    return body


def _fit(obj, points, triangles, ids, axis, top, pitch, fractions, parent, role, prefix, start, leg_center=None):
    sample = points[ids]
    along = sample @ axis
    span = top - along.min()
    if span <= 1e-7:
        return [], []
    faces = triangles[np.isin(triangles, ids).all(1)]
    if not len(faces):
        return [], []
    width, depth = np.ptp(sample[:, :2], axis=0)
    close_panel = (role == 'frontpanel' and len(fractions) == 3 and
                   width < .32 * span and depth < .32 * width)
    if close_panel:
        # Thin sheets need shorter links to carry a bend through the hem.
        pitch *= .72
    count = int(np.clip(np.ceil(span/pitch), 2, 32))
    levels = np.linspace(top, along.min(), count+1)
    rails = _section_rails(points, faces, axis, levels, fractions)
    if not rails:
        return [], []
    if len(rails)>1:
        # A slanted hem may end on just one side. Keep the rail spacing at
        # the last full section instead of folding every rail onto that point.
        for rail in rails:
            rail[0,:2] = rail[1,:2]
            rail[-1,:2] = rail[-2,:2]
    chains = []
    for rail in rails:
        names = [f'{prefix}{role.title()}_{start+len(chains):02d}_{i:02d}' for i in range(len(rail))]
        chain = dict(names=names, line=list(map(Vector, rail)), role=role, parent=parent,
                     free_appendage=True, close_panel=close_panel)
        if close_panel and count >= 8:
            # The fitted upper quarter shares one support. A short first
            # spring would have to rotate nearly flat to clear a raised leg.
            chain['panel_mount'] = names[0] + '_Mount'
            chain['panel_free_start'] = round(count * .25)
            # Keep the waist anchored. Leave a skin link to carry the
            # transition's rotation up to the start of the loose panel.
            stop = max(2, chain['panel_free_start'] - 1)
            chain['panel_transition'] = names[max(1, stop - 2):stop]
            chain['panel_mount_head'] = Vector(rail[0] + axis * (.82 * span))
        chains.append(chain)
    values = {}
    controls = [parent, *chains[0]['names'][:-1]]
    degree = min(3, count)
    spans = len(controls)-degree
    knots = [0]*degree + list(range(spans+1)) + [spans]*degree
    free_controls = []
    if close_panel and count >= 8:
        # The lateral offset amplifies differences in rotation between short
        # links. Spread skinning across a distance proportional to panel width.
        free_start = int(np.ceil(.25 * count))
        stride = max(1, min(4, round(.6 * width / (span / count))))
        middle = chains[len(chains) // 2]['names']
        # Keep adjacent controls through the first blend-width of loose cloth:
        # these links still wrap the upper leg. Broaden only farther down.
        stations = [free_start]
        while stations[-1] < count:
            delta = 1 if stations[-1] - free_start < stride else stride
            stations.append(stations[-1] + delta)
        free_controls = [middle[min(n, count) - 1] for n in stations]
        # A Hermite map keeps the physical weight slope continuous where
        # control spacing changes. Endpoint slopes use their sole interval.
        slopes = [(min(j+1, len(stations)-1) - max(j-1, 0)) /
                  (stations[min(j+1, len(stations)-1)] - stations[max(j-1, 0)])
                  for j in range(len(stations))]
    depths = np.clip((top-along)/span, 0, 1)
    for i, t in zip(ids, depths):
        if len(rails)==1:
            values[i] = _free_weights(controls, degree, spans, knots, float(t))
            continue
        # Two longitudinal and two lateral influences make the fabric bend
        # across the panel while keeping the waistband exactly on the pelvis.
        u = t*count
        if free_controls and u >= free_start:
            cell = min(bisect_right(stations, u) - 1, len(stations) - 2)
            span_u = stations[cell+1] - stations[cell]
            t = (u - stations[cell]) / span_u
            f = t + t * (1.-t) * ((1.-t) * (span_u * slopes[cell] - 1.)
                                   + t * (1. - span_u * slopes[cell+1]))
            values[i] = _smooth_panel_row(free_controls, cell + f)
            continue
        j = min(int(u), count-1)
        blend = u-j
        centers = [(1-blend)*rail[j, 0]+blend*rail[j+1, 0] for rail in rails]
        lateral = np.interp(points[i, 0], centers, np.arange(len(rails)))
        if close_panel:
            # Keep the widthwise blend at the waistband. Below the upper
            # quarter a thin free sheet bends around a common center rail.
            fade = min(1., 4. * t)
            lateral += (1. - lateral) * fade * fade * (3. - 2. * fade)
        left = min(int(lateral), len(rails)-2)
        across = lateral-left
        row = {}
        for index, share in ((left, 1-across), (left+1, across)):
            names = chains[index]['names']
            for name, weight in ((names[j-1] if j else parent, 1-blend), (names[j], blend)):
                if share*weight>0:
                    row[name] = row.get(name, 0)+share*weight
        if close_panel and leg_center is not None and t < .23:
            # The stitched top follows the pelvis. Share the transition with
            # the underlying thighs so a raised leg can still clear the cloth.
            free = float(np.clip((t - .06) / .17, 0., 1.))
            free = free * free * (3. - 2. * free)
            supported = 1. - free
            thigh = .9 * min(1., t / .12)
            left_share = float(np.clip(.5 + (points[i, 0] - leg_center) / (.5 * width), 0., 1.))
            row = {name: weight * free for name, weight in row.items()}
            for name, share in ((parent, 1. - thigh), ('Thigh.L', thigh * left_share),
                                ('Thigh.R', thigh * (1. - left_share))):
                row[name] = row.get(name, 0.) + supported * share
            row = _four_weights(row)
        values[i] = row

    if free_controls and leg_center is not None and width > 0.:
        # Each side follows its nearer thigh under the fitted support. Fade
        # this extra influence out through the first section of loose fabric.
        for i, depth in zip(ids, depths):
            if not .04 < depth < .31:
                continue
            follow = (.55 * _smooth_fraction((depth - .04) / .12)
                      * _smooth_fraction((.31 - depth) / .155))
            left = _smooth_fraction(.5 + 2. * (points[i, 0] - leg_center) / width)
            row = {name: weight * (1. - follow) for name, weight in values[i].items()}
            row['Thigh.L'] = row.get('Thigh.L', 0.) + follow * left
            row['Thigh.R'] = row.get('Thigh.R', 0.) + follow * (1. - left)
            values[i] = _four_weights(row)
    if free_controls:
        cloth_names = {name for chain in chains for name in chain['names']}
        cut = chains[0]['panel_free_start']
        for vertex, depth in zip(ids, depths):
            values[vertex] = _blend_panel_handoff(values[vertex], middle, cloth_names, depth * count, cut)
    return chains, [dict(obj=obj, ids=set(ids), values=values, role=role)]


def plan_trailing_surfaces(rig, obj, points, height, prefix, start, islands, excluded):
    hips = np.asarray(rig.data.bones['Hips'].head_local)
    head = rig.data.bones['Head']
    crown = np.asarray(head.tail_local)
    head_base = np.asarray(head.head_local)
    chains, regions = [], []
    obj.data.calc_loop_triangles()
    triangles = np.array([tuple(t.vertices) for t in obj.data.loop_triangles], dtype=int).reshape(-1, 3)
    if not len(triangles):
        return [], []
    slots = np.asarray([t.material_index for t in obj.data.loop_triangles])
    leg_center = None
    if 'Thigh.L' in rig.data.bones and 'Thigh.R' in rig.data.bones:
        leg_center = .5 * (rig.data.bones['Thigh.L'].head_local.x + rig.data.bones['Thigh.R'].head_local.x)
    def add(ids, axis, top, pitch, fractions, parent, role):
        new, rows = _fit(obj, points, triangles, ids, axis, top, height*pitch, fractions,
                         parent, role, prefix, start+len(chains), leg_center)
        chains.extend(new)
        regions.extend(rows)
        return bool(new)

    # Cutting connectivity at the waist exposes separate panels even when
    # their top edges share vertices with the bodice, legs or each other.
    height_axis = np.array([0., 0., 1.])
    below = set(np.flatnonzero(points[:, 2]<=hips[2]).tolist()) - excluded
    if below:
        _, panels, _ = connected_graph(obj.data, points, below, height*.0001)
        for ids in panels:
            p = points[ids]-hips
            lo, hi = p.min(0), p.max(0)
            if (hi[2] < -.035*height or hi[2]-lo[2] < .22*height or
                    max(abs(lo[0]), abs(hi[0])) > .15*height or
                    max(abs(lo[1]), abs(hi[1])) > .13*height):
                continue
            # Legs enclose their own axis. A free apron stays on one side
            # of both legs and is thin in depth throughout its free length.
            if not (lo[1] > .012*height or hi[1] < -.012*height):
                continue
            if (hi[1]-lo[1] > .30*(hi[2]-lo[2]) or
                    abs((hi[0]+lo[0])*.5) > .42*(hi[0]-lo[0])):
                continue
            fractions = (.12,.5,.88) if hi[0]-lo[0] > .07*height else (.5,)
            role = 'frontpanel' if hi[1]<0 else 'backpanel'
            # The broad rear flap hangs as a sheet. Longer sections keep
            # the hem from overreacting to the small oscillations of a walk.
            pitch = .060 if role == 'backpanel' and len(fractions)>1 else .032
            add(ids, height_axis, hips[2]+hi[2], pitch,
                fractions, 'Hips', role)

    # A long swept lock or bow ribbon begins at the back of the head. Use
    # its oblique length, so a horizontal sweep can flex along its full span.
    vertex_slot = np.full(len(points), -1, dtype=int)
    vertex_slot[triangles.ravel()] = np.repeat(slots, 3)
    pieces, carriers = [], []
    for ids in islands:
        if len(ids)<4 or not excluded.isdisjoint(ids):
            continue
        p = points[ids]
        top, bottom = p[:, 2].max(), p[:, 2].min()
        startpoint = np.median(p[p[:,2] >= top-.009*height], axis=0)
        endpoint = np.median(p[p[:,2] <= bottom+.009*height], axis=0)
        material = frozenset(vertex_slot[ids])
        pieces.append((ids,startpoint,endpoint,material))
        if (top-bottom < .21*height or
                not head_base[2]+.025*height < top < crown[2]+.09*height or
                np.linalg.norm(startpoint[:2]-crown[:2]) > .15*height or
                startpoint[1] < head_base[1] or endpoint[1]-startpoint[1] < .07*height):
            continue
        carriers.append(len(pieces)-1)
    # Preserve the established twin-tail rig for isolated, mostly vertical
    # locks. A rearward fan needs independently articulated oblique guides.
    carriers = [i for i in carriers if
                pieces[i][2][1]-pieces[i][1][1] > .18*height or
                sum(np.linalg.norm(pieces[j][1]-pieces[i][1]) < .10*height
                    for j in carriers if pieces[j][3]==pieces[i][3]) >= 3]
    used = set(carriers)
    mounts = []
    for index in carriers:
        ids, root, tip, material = pieces[index]
        ids = list(ids)
        direction = tip-root
        direction /= np.linalg.norm(direction)
        extended = False
        while True:
            candidates = [(np.linalg.norm(a-tip), k) for k, (_,a,b,slots_) in enumerate(pieces)
                          if k not in used and slots_==material and
                          np.linalg.norm(a-tip)<.038*height and (b-tip)@direction>.006*height and
                          np.linalg.norm((b-a)-direction*((b-a)@direction))<.10*height]
            if not candidates:
                break
            _, k = min(candidates)
            used.add(k)
            child, _, tip, _ = pieces[k]
            ids.extend(child)
            extended = True
        axis = root-tip
        axis /= np.linalg.norm(axis)
        role = 'bowribbon' if extended else 'flowinghair'
        if add(ids, axis, root@axis-.012*height, .027, (.5,), 'Head', role) and extended:
            mounts.append((root, material))
    # Flex petals/loops at the knot. The compact central fastening follows
    # the head, and details touching the free ribbon followed its chain above.
    for k,(ids, _, _, material) in enumerate(pieces):
        if k in used:
            continue
        p = points[ids]
        roots = [r for r,m in mounts if m==material and np.linalg.norm(p.mean(0)-r)<.065*height]
        # A raised ribbon wing can be trimmed from different fabric than
        # the knot. Require a nearby verified ribbon mount and an upward,
        # outward free end behind the head; a horn or loose cap is not a wing.
        if mounts and p[:, 1].min() > head_base[1] and .04*height < np.ptp(p[:,0]) < .13*height:
            distances = np.min([np.linalg.norm(p-r,axis=1) for r,_ in mounts],axis=0)
            near = int(distances.argmin())
            distance = distances[near]
            base = p[near]
            tip = p[np.linalg.norm(p-base,axis=1).argmax()]
            delta = tip-base
            if (distance < .048*height and delta[2] > .024*height and
                    abs(tip[0])-abs(base[0]) > .035*height and
                    abs(delta[1]) < .65*abs(delta[0])):
                axis = -delta/np.linalg.norm(delta)
                span = base@axis-(p@axis).min()
                if add(ids,axis,base@axis,span/(1.98*height),(.5,),'Head','bowloop'):
                    continue
        if not roots or not .012*height<np.ptp(p,axis=0).max()<.10*height:
            continue
        root = min(roots, key=lambda r: np.linalg.norm(p.mean(0)-r))
        base = p[np.linalg.norm(p-root,axis=1).argmin()]
        axis = base-p[np.linalg.norm(p-base,axis=1).argmax()]
        axis /= np.linalg.norm(axis)
        add(ids,axis,base@axis,.018,(.5,),'Head','bowloop')
    return chains, regions
