"""Find independently hanging ribbons and their hardware from the rest surface."""
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from avatar_virtual_topology import connected_graph


def _tree(triangles):
    return BVHTree.FromPolygons(triangles.reshape(-1, 3).tolist(),
                                np.arange(triangles.size // 3).reshape(-1, 3).tolist(),
                                all_triangles=True)


def _centerline(tri, levels):
    # Intersect the ribbon itself, then average by section length. Mesh
    # tessellation on one edge must not pull the centerline toward that edge.
    line = []
    for z in levels:
        step = max(np.ptp(levels) * 1e-7, 1e-12)
        plane = np.clip(z, levels[-1] + step, levels[0] - step)
        cuts = tri[(tri[:, :, 2].min(axis=1) <= plane) & (tri[:, :, 2].max(axis=1) > plane)]
        above = cuts[:, :, 2] > plane
        odd = np.where(above.sum(axis=1)[:, None] == 1, above, ~above).argmax(axis=1)
        row = np.arange(len(cuts))
        a = cuts[row, odd][:, None, :]
        b = cuts[row[:, None], (odd[:, None] + [1, 2]) % 3]
        hit = a + (b-a) * ((plane-a[:, :, 2])/(b[:, :, 2]-a[:, :, 2]))[:, :, None]
        length = np.linalg.norm(hit[:, 1] - hit[:, 0], axis=1)
        if length.sum() == 0:
            return []
        xy = np.average(hit.mean(axis=1), axis=0, weights=length)
        line.append(Vector((xy[0], xy[1], z)))
    return line


def _mount_bands(mounts, max_gap, *, include_snap=True):
    intervals = []
    for ids, bottom, top, snap in mounts:
        if include_snap and snap is not None:
            bottom = min(bottom, snap)
            top = max(top, snap)
        intervals.append((top, bottom, ids))
    bands = []
    for top, bottom, ids in sorted(intervals, key=lambda interval: -interval[0]):
        if bands and bands[-1][1] - top <= max_gap:
            bands[-1][1] = min(bands[-1][1], bottom)
            bands[-1][2].extend(ids)
        else:
            bands.append([top, bottom, list(ids)])
    return bands


def _free_weights(names, degree, spans, knots, t):
    # Ease the first and last tenth of a span into rigid contact. Across the
    # remaining fabric a uniform parameter avoids concentrating shear midway.
    ramp = .1/spans
    if t < ramp:
        t = .5*t*t/ramp
    elif t > 1.-ramp:
        t = 1.-ramp-.5*(1.-t)**2/ramp
    else:
        t -= .5*ramp
    u = spans*t/(1.-ramp)
    knot = min(int(u)+degree, len(names)-1)
    basis = [1.]
    for order in range(1, degree+1):
        last = 0.
        for i in range(order):
            low, high = knots[knot+1-order+i], knots[knot+i+1]
            weight = basis[i]/(high-low)
            basis[i] = last+(high-u)*weight
            last = (u-low)*weight
        basis.append(last)
    return {name: weight for name, weight in zip(names[knot-degree:knot+1], basis) if weight > 0.}


def _strap_weights(points, carrier, bands, fitted=False):
    """Keep closely spaced fittings on one support and spread bends across free fabric."""
    names = carrier['names'][:-1]
    levels = carrier['levels']
    centers = (levels[1:] + levels[:-1])*.5
    anchors = []
    values = {}
    for top, bottom, ids in bands:
        mid = .5*(top+bottom)
        if fitted:
            j = int(np.clip(np.searchsorted(-levels, -mid, side='right')-1, 0, len(names)-1))
        else:
            j = int(np.argmin(abs(centers-mid)))
        anchors.append((top, bottom, j))
        values.update((i, {names[j]: 1.}) for i in ids)
    hardware = set(values)
    if not anchors or anchors[0][2] != 0:
        anchors.insert(0, (levels[0], levels[0], 0))
    if anchors[-1][2] != len(names)-1:
        anchors.append((levels[-1], levels[-1], len(names)-1))
    gaps = []
    for (_, bottom, j), (top, _, k) in zip(anchors, anchors[1:]):
        if top >= bottom:
            continue
        controls = names[j:k+1]
        degree = min(3, len(controls)-1)
        spans = len(controls)-degree
        knots = [0]*degree + list(range(spans+1)) + [spans]*degree
        gaps.append((bottom, top, controls, degree, spans, knots))
    for i in carrier['ids']:
        z = points[i, 2]
        for bottom, top, controls, degree, spans, knots in gaps:
            if top < z < bottom:
                t = (bottom-z)/(bottom-top)
                values[i] = _free_weights(controls, degree, spans, knots, t)
                break
        else:
            _, _, j = min(anchors, key=lambda a: max(a[1]-z, z-a[0], 0.))
            values[i] = {names[j]: 1.}
    return values, hardware


def _nearby_panel(line, guides, height):
    """Follow one continuous panel rail when the ribbon hangs alongside it."""
    stations = (line[1:] + line[:-1])*.5
    best = None
    for guide in guides:
        path = np.array(guide['line'])
        depths = -path[:, 2]
        nearest = np.column_stack([np.interp(-stations[:, 2], depths, path[:, axis])
                                   for axis in range(3)])
        distances = np.linalg.norm(nearest - stations, axis=1)
        score = float(np.mean(distances))
        if best is None or score < best[0]:
            best = (score, guide, distances)
    if best is None or np.quantile(best[2], .75) >= height*.09:
        return None
    return best[1]


def _load_layout(carrier, bands, pitch, guide, center_x):
    """Place strap joints between rigid fittings and along the loose end."""
    if not guide or not guide.get('leg'):
        return None
    levels = carrier['levels']
    line = np.asarray(carrier['line'])
    knee = guide['dress']['knee_height']
    for band in sorted(bands, key=lambda b: b[0]-b[1], reverse=True):
        top, bottom, _ = band
        if top-bottom <= pitch or (top+bottom)*.5 < knee:
            continue
        if levels[0]-top <= .15*pitch or bottom-levels[-1] <= .35*pitch:
            continue
        endpoints = np.interp([-top, -bottom], -levels, line[:, 0]) - center_x
        if min(endpoints * (guide['line'][0][0]-center_x)) <= 0.:
            continue
        # Separate fittings can rotate relative to each other. Put joints in
        # the exposed ribbon between their contact patches.
        mounts = [m for m in carrier['mounts'] if bottom <= m[1] and m[2] <= top]
        # A shared nearest vertex on a coarse ribbon can belong to the gap
        # between two distinct fittings. Use their actual contact footprints
        # to locate hinges. Keep a vertex snap if it leaves the neighboring
        # fitting a distinct gap on the fabric.
        sections = _mount_bands(mounts, pitch*.06, include_snap=False)
        snaps = {int(ids[0]): snap for ids, _, _, snap in mounts if snap is not None}
        snapped = []
        for high, low, ids in sections:
            witnesses = [snaps[i] for i in ids if i in snaps]
            snapped.append((max([high, *witnesses]), min([low, *witnesses])))
        for i, (upper_section, lower_section) in enumerate(zip(sections, sections[1:])):
            if snapped[i][1] - snapped[i+1][0] > pitch*.06:
                upper_section[1], lower_section[0] = snapped[i][1], snapped[i+1][0]
        sections[0][0] = top
        sections[-1][1] = bottom
        band_index = bands.index(band)
        fittings = bands[:band_index] + sections + bands[band_index+1:]
        pivots = [.5*(a[1]+b[0]) for a, b in zip(sections, sections[1:])]
        upper_steps = max(1, round((levels[0]-top)/pitch))
        lower_steps = int(np.clip(np.ceil((bottom-levels[-1])/(.5*pitch)), 2, 12))
        upper = np.linspace(levels[0], top, upper_steps+1)
        lower = np.linspace(bottom, levels[-1], lower_steps+1)
        fitted = np.r_[upper, pivots, lower]
        path = _centerline(carrier['tri'], fitted)
        if path:
            free_tail_start = len(upper) + len(pivots) if pivots else None
            return fitted, path, upper_steps, fittings, free_tail_start
    return None


def _panel_mount(carrier, points, stop, height, tree, faces):
    """Sample the supporting sheet only above the first independent fitting."""
    from avatar_garment_attachments import _barycentric
    top = carrier['levels'][0]
    span = top - stop
    if span <= 0.:
        return {}
    # Sample the middle of the ribbon: near a fold, its opposite edges may
    # lie closest to different sheets, twisting the fabric if sampled alone.
    levels = np.linspace(top, stop, 7)
    path = _centerline(carrier['tri'], levels)
    if not path:
        return {}
    samples = []
    for v in path:
        hit, _, face, distance = tree.find_nearest(v)
        if face is None:
            return {}
        tri = faces[face]
        samples.append((tri, _barycentric(np.asarray(hit), *points[tri]), distance))
    release = {}
    for i in carrier['ids']:
        h = (points[i, 2]-stop)/span
        if h <= 0.:
            continue
        u = np.clip(1.-h, 0., 1.)*(len(levels)-1)
        j = min(int(u), len(levels)-2)
        t = u-j
        gap = (1.-t)*samples[j][2]+t*samples[j+1][2]
        near = np.clip((.055*height-gap)/(.025*height), 0., 1.)
        # Match the mounted web to the supporting rail; release smoothly
        # before the first fitting so it keeps its independent pivot.
        end_fade = np.clip(h/.65, 0., 1.)
        amount = float(end_fade**2*(3.-2.*end_fade)*near**2*(3.-2.*near))
        if amount > 0.:
            release[i] = (j, float(t), amount)
    return dict(release=release, samples=samples) if release else {}


def bind_panel_mount(region):
    """Transfer finished waist and skirt weights only on the supported web."""
    from avatar_apparel_weights import assign, weights
    from avatar_surface_attachments import normalized
    obj = region['obj']
    contact = region['panel_mount']
    samples = {}
    rail = []
    for tri, bary, _ in contact['samples']:
        row = {}
        for v, factor in zip(tri, bary):
            if v not in samples:
                samples[v] = weights(obj, int(v))
            for n,w in samples[v].items():
                row[n] = row.get(n, 0.) + factor*w
        rail.append(row)
    for i, (j, t, amount) in contact['release'].items():
        row = {n:w*(1.-amount) for n,w in region['values'][i].items()}
        for donor,alpha in zip(rail[j:j+2], (1.-t,t)):
            for n,w in donor.items():
                row[n] = row.get(n, 0.) + w*amount*alpha
        if row:
            assign(obj,[i],normalized(row))


def plan_straps(obj, points, skirt, triangles, surface, height, prefix, skirt_index, excluded, guides):
    """Separate top-mounted ribbons with a demonstrably unsupported lower span."""
    _, pieces, _ = connected_graph(obj.data, points, skirt, height*.0001)
    shell = surface.triangles.copy()
    shell[:, :, :2] += surface.center
    shell_tree = _tree(shell)
    sheet_top = surface.top
    carriers = []
    mesh_faces = np.asarray(triangles)
    for ids in pieces:
        p = points[ids]
        size = np.ptp(p, axis=0)
        if size[2] < .15*height or size[2] < 3*max(size[:2]) or p[:, 2].max() < sheet_top:
            continue
        tri = points[mesh_faces[np.isin(mesh_faces, ids).all(axis=1)]]
        if not len(tri):
            continue
        z = p[:, 2]
        levels = np.linspace(z.max(), z.min(), 13)
        probe = _centerline(tri, levels)
        if not probe:
            continue
        gaps = np.array([shell_tree.find_nearest(v)[3] for v in probe])
        # Flush overlays and their free tips keep their cloth attachment.
        if gaps[:3].min() > height*.035 or np.median(gaps[5:]) < height*.025:
            continue
        count = int(np.clip(round(size[2]/(.085*height)), 3, 8))
        levels = np.linspace(z.max(), z.min(), count+1)
        line = _centerline(tri, levels)
        if not line:
            continue
        names = [f'{prefix}Accessory_Garment{skirt_index:02d}_{len(carriers):02d}_Strand_{i:02d}'
                 for i in range(count+1)]
        vertices = KDTree(len(ids))
        for i in ids:
            vertices.insert(points[i], i)
        vertices.balance()
        carriers.append(dict(vertices=vertices, ids=ids, tri=tri, line=line, names=names, tree=_tree(tri), levels=levels,
                             bounds=(p.min(axis=0), p.max(axis=0)), mounts=[]))
    chains, regions = [], []
    if not carriers:
        return chains, regions
    # A waistband can support the top of the web without being large enough
    # to define the hemline. Include broad, horizontally wrapping pieces at
    # waist height only for mount samples, never for guides or loose hardware.
    panels = mesh_faces[surface.face_indices]
    waistband = []
    panel_ids = set(panels.ravel())
    for ids in pieces:
        if panel_ids.intersection(ids):
            continue
        q = points[ids]
        extent = np.ptp(q, axis=0)
        if (max(extent[:2]) >= height*.12 and min(extent[:2]) >= height*.06
                and extent[2] < max(extent[:2])*.65
                and q[:, 2].min() <= sheet_top+height*.02
                and q[:, 2].max() >= sheet_top-height*.04):
            faces = mesh_faces[np.isin(mesh_faces, ids).all(axis=1)]
            if len(faces):
                waistband.append((faces, _tree(points[faces])))
    used = set(excluded).union(*(c['ids'] for c in carriers))
    _, groups, _ = connected_graph(obj.data, points, range(len(points)), height*.0001)
    for ids in groups:
        if used.intersection(ids):
            continue
        p = points[ids]
        size = np.ptp(p, axis=0)
        # Compact fittings only. Reject cloth panels, other hanging carriers
        # and pieces whose nearest support is the garment itself.
        if max(size) > height*.095 or size[2] > height*.065:
            continue
        mounts = []
        for k, c in enumerate(carriers):
            lo, hi = c['bounds']
            if (p.min(axis=0) > hi + .03*height).any() or (p.max(axis=0) < lo - .03*height).any():
                continue
            near = [c['tree'].find_nearest(v) for v in p]
            distances = np.array([hit[3] for hit in near])
            if distances.min() > height*.014 or np.median(distances) > height*.032:
                continue
            if shell_tree.find_nearest(p.mean(axis=0))[3] < np.median(distances):
                continue
            # A multipart fitting may touch the ribbon only along one edge.
            contact = [hit[0].z for hit in near if hit[3] <= distances.min()+height*.004]
            snap = None
            # A narrow transverse fitting also locks the nearest actual
            # ribbon vertex at its attachment, even on a coarse ribbon mesh.
            if size[2] < .35*max(size[:2]):
                vertex, _, separation = min((c['vertices'].find(v) for v in p), key=lambda hit: hit[2])
                if separation <= height*.02:
                    snap = vertex.z
            mounts.append((float(np.mean(distances)), k, min(contact), max(contact), snap))
        if mounts:
            _, k, *mount = min(mounts)
            carriers[k]['mounts'].append((ids, *mount))
    for c in carriers:
        levels = c['levels']
        pitch = (levels[0]-levels[-1])/(len(levels)-1)
        guide = _nearby_panel(np.array(c['line']), guides, height)
        bands = _mount_bands(c['mounts'], pitch*.5)
        layout = _load_layout(c, bands, pitch, guide, surface.center[0])
        if layout:
            c['levels'], c['line'], load_index, bands, free_tail_start = layout
            label = c['names'][0].rsplit('_', 1)[0]
            c['names'] = [f'{label}_{i:02d}' for i in range(len(c['levels']))]
        names = c['names'][:-1]
        values, hardware = _strap_weights(points, c, bands, fitted=bool(layout and free_tail_start is not None))
        chain = dict(role='accessory', names=c['names'], line=c['line'], parent='Hips',
                     physics_start=1, serial_accessory=True)
        if layout:
            chain['articulated_fittings'] = free_tail_start is not None
            if free_tail_start is not None:
                # The web starts at the lower edge of the clustered load. A
                # separate end cap moves rigidly with its own swinging link.
                chain['free_tail_start'] = free_tail_start
            chain['load_index'] = load_index
            if chain['articulated_fittings']:
                # Suspend the load from the cloth above its release. The web
                # can then lag behind that cloth when the supporting leg moves.
                levels = -np.asarray(guide['line'])[:, 2]
                release = -c['line'][load_index].z
                support = int(np.clip(np.searchsorted(levels, release, side='right')-2,
                                      0, len(guide['names'])-2))
                chain['load_parent'] = guide['names'][support]
            else:
                # Share skirt rotation, but turn about a pivot nearer the web
                # attachment. A longer mounted span warrants a shorter lever.
                chain['panel_follow'] = guide['follow']
                chain['load_parent'] = label + '_Mount'
                chain['strap_center'] = chain['load_parent']
                supported = c['levels'][0]-c['levels'][load_index]
                hanging = c['levels'][load_index]-c['levels'][-1]
                # The free span's distributed load acts at its midpoint.
                chain['mount_fraction'] = float(supported/(supported+.5*hanging))
            chain['physics_start'] = load_index
        elif guide:
            path = np.asarray(c['line'])
            stations = (path[1:] + path[:-1])*.5
            depths = -np.asarray(guide['line'])[:, 2]
            indices = np.clip(np.searchsorted(depths, -stations[:, 2])-1, 0, len(depths)-2)
            parents = [guide['names'][i] for i in indices]
            chain['mount_parents'] = ['Hips', *parents[1:], parents[-1]]
        panel_mount = {}
        if layout:
            # A band threaded through the ribbon at rest is not a second
            # backing sheet: switching between its faces folds the web.
            backing = [f for f, tree in waistband if not tree.overlap(c['tree'])]
            faces = np.concatenate([panels, *backing]) if backing else panels
            tree = _tree(points[faces]) if backing else shell_tree
            panel_mount = _panel_mount(c, points, c['levels'][load_index], height, tree, faces)
        chains.append(chain)
        regions.append(dict(role='accessory', obj=obj, ids=set(values), values=values,
                            garment_strap=True, panel_mount=panel_mount,
                            hardware=hardware, names=[names]))
    return chains, regions
