"""Conservative thin hanging-panel fallback for anonymous, fused GLB surfaces.

Semantic garments retain their existing route. This fallback admits only thin
opposed surfaces below the pelvis with a substantial connected vertical run.
Hands, feet and sleeves are protected geometrically. No vertices are moved.
"""
import math
import re
from collections import defaultdict
from heapq import heapify, heappop, heappush
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from avatar_virtual_topology import connected_graph, protected_extremities
from avatar_apparel_weights import weights
from avatar_surface_attachments import normalized


def _opening_mask(surface, points, center, height):
    """Trace a persistent free opening through a fused front layer."""
    run = sorted((z, left, right) for z, (left, right) in surface.opening
                 if left < -.08 and right > .08)
    if len(run) < 3 or run[-1][0]-run[0][0] < height*.055:
        return np.zeros(len(points), dtype=bool)
    end = run[-1][0]
    left, right = np.median(np.asarray(run[-5:])[:, 1:], axis=0)
    resumed = []
    for z in np.linspace(end, surface.top, 11)[1:]:
        for lo, hi in surface.section(z).gaps:
            if abs(lo-left) < .13 and abs(hi-right) < .13:
                resumed.append((z, lo, hi))
                break
    # An opening can stop where the front panels button together. Extend it
    # above the source band only if the edges persist or reappear above trim.
    continues = surface.top-end < height*.024 or (
        len(resumed) > 1 and resumed[-1][0] > surface.top-height*.024)
    samples = np.asarray(run + resumed)
    z = points[:, 2]
    angle = np.arctan2(points[:, 0]-center[0], center[1]-points[:, 1])
    lo = np.interp(z, samples[:, 0], samples[:, 1]) + .015
    hi = np.interp(z, samples[:, 0], samples[:, 2]) - .015
    return (angle > lo) & (angle < hi) & (continues | (z <= samples[-1, 0]))


def _cut_rim_distances(points, neighbors, rims, limit):
    distances = dict.fromkeys(rims, 0.)
    queue = [(0., i) for i in rims]
    heapify(queue)
    while queue:
        length, i = heappop(queue)
        if length != distances[i]:
            continue
        for j in neighbors[i]:
            trial = length + float(np.linalg.norm(points[i]-points[j]))
            if trial <= limit and trial < distances.get(j, math.inf):
                distances[j] = trial
                heappush(queue, (trial, j))
    return distances


def _close_coat_gaps(points, edges, source, coat, rows, admit, radius, fill=None, exposed=None):
    """Close bounded holes on the cloth surface without crossing an opening."""
    if not coat:
        return {}
    neighbors = defaultdict(list)
    rim = set()
    grow = []
    distance = {}
    for a, b in edges:
        if not (source[a] and source[b]):
            if fill is not None:
                if source[a] and fill[b]:
                    rim.add(a)
                if source[b] and fill[a]:
                    rim.add(b)
            continue
        neighbors[a].append(b)
        neighbors[b].append(a)
        if (a in coat) != (b in coat):
            if b in coat:
                a, b = b, a
            length = float(np.linalg.norm(points[a]-points[b]))
            grow.append((length, a, b))
    cut = _cut_rim_distances(points, neighbors, rim, radius*2.3)
    seeds, grow = grow, []
    for length, owner, i in seeds:
        trial = length, owner
        if length <= radius*(1.55 if i in cut else 1.) and trial < distance.get(i, (math.inf, -1)) and admit(i):
            distance[i] = trial
            heappush(grow, (*trial, i))
    while grow:
        length, owner, i = heappop(grow)
        if (length, owner) != distance[i]:
            continue
        for j in neighbors[i]:
            if j in coat:
                continue
            trial = (length+float(np.linalg.norm(points[i]-points[j])), owner)
            if trial[0] <= radius*(1.55 if j in cut else 1.) and trial < distance.get(j, (math.inf, -1)) and admit(j):
                distance[j] = trial
                heappush(grow, (*trial, j))
    # Erode from the exterior of the expanded mask. This leaves enclosed
    # holes and shallow cut-side recesses while rejecting a general spill.
    exterior = {}
    queue = []
    for i in distance:
        length = min((float(np.linalg.norm(points[i]-points[j])) for j in neighbors[i]
                      if j not in distance and j not in coat), default=math.inf)
        if length <= radius:
            exterior[i] = length
            heappush(queue, (length, i))
    while queue:
        length, i = heappop(queue)
        if length != exterior[i]:
            continue
        for j in neighbors[i]:
            if j not in distance:
                continue
            trial = length + float(np.linalg.norm(points[i]-points[j]))
            if trial <= radius and trial < exterior.get(j, math.inf):
                exterior[j] = trial
                heappush(queue, (trial, j))
    # The reconstructed slit can leave an exterior recess open at the cut.
    # Its original surface may still borrow from neighboring cloth in the
    # same rim band even though an open recess has no closed perimeter.
    return {i: rows[owner].copy() for i, (_, owner) in distance.items()
            if i not in exterior or (i in cut and exposed(i))}


def _bind_cut_fill(points, edges, fill, body_bank, rows, coat, height):
    """Carry proven rim ownership into the closure on the body bank."""
    if not fill.any() or not coat:
        return {}
    adjacent = {int(i): [] for i in np.flatnonzero(fill)}
    surface = defaultdict(list)
    rim_ids = set()
    nearest = {}
    queue = []
    for a, b in edges:
        if not (body_bank[a] and body_bank[b]):
            continue
        if b in adjacent:
            a, b = b, a
        if a not in adjacent:
            surface[a].append(b)
            surface[b].append(a)
            continue
        distance = float(np.linalg.norm(points[a]-points[b]))
        if b in adjacent:
            adjacent[a].append((b, distance))
            adjacent[b].append((a, distance))
        elif (distance, b) < nearest.get(a, (math.inf, -1)):
            nearest[a] = distance, b
            heappush(queue, (distance, b, a))
        if b not in adjacent:
            rim_ids.add(b)
    rim_rows = {}
    for rim in rim_ids - coat:
        # The slit can turn a piece of the outer face inward. Recover its
        # rim from the original skin edges before stepping into the new cap.
        distances = {rim: 0.}
        nearby = [(0., rim)]
        while nearby:
            length, i = heappop(nearby)
            if i in coat:
                rim_rows[rim] = rows[i].copy()
                break
            if length != distances[i]:
                continue
            for j in surface[i]:
                trial = length + float(np.linalg.norm(points[i]-points[j]))
                if trial <= height*.006 and trial < distances.get(j, math.inf):
                    distances[j] = trial
                    heappush(nearby, (trial, j))
    while queue:
        distance, rim, i = heappop(queue)
        if (distance, rim) != nearest[i]:
            continue
        for j, length in adjacent[i]:
            trial = distance+length, rim
            if trial < nearest.get(j, (math.inf, -1)):
                nearest[j] = trial
                heappush(queue, (*trial, j))
    return {**rim_rows, **{i: (rim_rows[rim] if rim in rim_rows else rows[rim]).copy()
                           for i, (_, rim) in nearest.items() if rim in coat or rim in rim_rows}}


def plan_panels(rig, obj, points, excluded, height, prefix, start_index=0):
    if any(m and not re.fullmatch(r'(?:material|default)(?:[ ._\d]*)',m.name,re.I)
           for m in obj.data.materials):
        return [], []
    hips_bone=rig.data.bones.get('Hips')
    feet=[rig.data.bones.get('Foot.'+s) for s in ('L','R')]
    feet=[bone for bone in feet if bone is not None]
    # Without pelvis and foot landmarks the conservative panel region is unknown.
    if hips_bone is None or not feet:
        return [], []
    hips=hips_bone.head_local
    floor=max(b.head_local.z for b in feet)+height*.055
    pelvis_limit = hips.z + height*.055
    anchors = [bone for name in ('Spine', 'UpperSpine')
               if (bone := rig.data.bones.get(name)) and bone.head_local.z > pelvis_limit]
    probe_limit = max((bone.head_local.z + height*.012 for bone in anchors), default=pelvis_limit)
    eligible=(points[:,2]<probe_limit)&(points[:,2]>floor)
    eligible &= ~protected_extremities(rig,points,height)
    for side in ('L','R'):
        for token in ('UpperArm.','Forearm.'):
            bone=rig.data.bones.get(token+side)
            if bone is None:continue
            a=np.array(bone.head_local);d=np.array(bone.tail_local)-a
            t=np.clip((points-a)@d/max(d@d,1e-16),0,1)
            eligible &= np.linalg.norm(points-(a+t[:,None]*d),axis=1)>height*.045
    bank=obj.data.attributes.get('hallway_arm_bank')
    body_bank = np.ones(len(points), dtype=bool)
    if bank:
        owned=np.empty(len(points),dtype=np.int32);bank.data.foreach_get('value',owned)
        body_bank = owned == 0
        eligible &= body_bank
    fill_attr = obj.data.attributes.get('hallway_generated_fill')
    fill = np.zeros(len(points), dtype=bool)
    if fill_attr:
        generated = np.empty(len(points), dtype=np.int32)
        fill_attr.data.foreach_get('value', generated)
        fill = (generated != 0) & body_bank
    cut_bank = body_bank.copy()
    cut_bank[list(excluded)] = False
    eligible &= cut_bank
    fill &= cut_bank
    obj.data.calc_loop_triangles()
    triangles=np.array([t.vertices[:] for t in obj.data.loop_triangles])
    coordinates = points.tolist()
    tree=BVHTree.FromPolygons(coordinates,triangles.tolist(),all_triangles=True)
    normal_matrix=(rig.matrix_world.inverted_safe()@obj.matrix_world).to_3x3().inverted_safe().transposed()
    seeds=set()
    for i in np.flatnonzero(eligible):
        n=(normal_matrix@obj.data.vertices[i].normal).normalized()
        hit,other,_,distance=tree.ray_cast(Vector(points[i])-n*height*1e-5,-n,height*.024)
        if hit is not None and height*.0005<distance and other.dot(n)<-.25:
            seeds.add(int(i))
    lower = eligible & (points[:,2] < pelvis_limit)
    lower_seeds = {i for i in seeds if lower[i]}
    if not lower_seeds:
        return [], []

    def thin_components(mask, roots):
        # Close only small tessellation gaps in the same thin surface.
        import heapq
        adjacency = {int(i): [] for i in np.flatnonzero(mask)}
        for edge in obj.data.edges:
            a, b = edge.vertices
            if a in adjacency and b in adjacency:
                adjacency[a].append(b)
                adjacency[b].append(a)
        distance = {i: 0. for i in roots}
        queue = [(0., i) for i in roots]
        heapq.heapify(queue)
        while queue:
            d, i = heapq.heappop(queue)
            if d != distance[i]:
                continue
            for j in adjacency[i]:
                trial = d + float(np.linalg.norm(points[j]-points[i]))
                if trial <= height*.012 and trial < distance.get(j, math.inf):
                    distance[j] = trial
                    heapq.heappush(queue, (trial, j))
        return connected_graph(obj.data, points, distance, height*.0001)[1]

    components = thin_components(lower, lower_seeds)
    # Fused coat torso panels can lose their inner wall where they meet the
    # body. Continue the admitted hem along its connected outer surface.
    use_torso = anchors and any(len(c) >= 200 and min(points[c, 2]) < hips.z-height*.15
                               for c in components)
    shell_components = connected_graph(obj.data, points, np.flatnonzero(eligible).tolist(),
                                       height*.0001)[1] if use_torso else []
    shell_lookup = {i: index for index, component in enumerate(shell_components) for i in component}
    chains=[];regions=[]
    from avatar_garment_straps import _centerline
    from avatar_springs import _vertical_weights
    from avatar_skirt_surface import SkirtSurface, angle_weights
    from avatar_cloth_handoff import blend_four
    from avatar_secondary_smoothing import smooth_domain
    cloth_visibility = None
    edges = [tuple(e.vertices) for e in obj.data.edges]
    for component in components:
        ids=set(component);p=points[component]
        if len(ids)<200 or np.ptp(p[:,2])<height*.065 or np.ptp(p[:,0])<height*.035:continue
        faces=triangles[np.all(np.isin(triangles,list(ids)),axis=1)]
        if len(faces)<100:continue
        panel_top=float(np.quantile(p[:,2],.99));bottom=float(p[:,2].min())
        center=np.array(hips)[:2];angles=np.arctan2(points[:,1]-center[1],points[:,0]-center[0])%math.tau
        guides=[];directions=[]
        for sector in range(16):
            angle=sector*math.tau/16
            delta=(angles[faces].mean(1)-angle+math.pi)%math.tau-math.pi
            # Circular means avoid a false gap at the wrap seam.
            face_angle=np.arctan2(np.sin(angles[faces]).mean(1),np.cos(angles[faces]).mean(1))
            delta=(face_angle-angle+math.pi)%math.tau-math.pi
            subset=points[faces[abs(delta)<math.tau/16*.75]]
            if not len(subset):continue
            lo=float(subset[:,:,2].min());hi=min(panel_top,float(subset[:,:,2].max()))
            if hi-lo<height*.055:continue
            levels=np.linspace(hi-height*.001,lo+height*.001,int(np.clip(math.ceil((hi-lo)/(height*.035)),3,10))+1)
            line=_centerline(subset,levels)
            if not line:continue
            names=[f'{prefix}Accessory_SurfacePanel_{start_index+len(regions):02d}_{sector:02d}_{j:02d}' for j in range(len(line))]
            chain=dict(role='accessory',names=names,line=line,parent='Hips',garment=True,physics_start=1,anonymous_panel=True)
            guides.append(chain);directions.append(angle)
        if not guides:continue
        # Grow the same angular guides over the waist. Sample the connected
        # shell at successive heights, so a missing inner wall is harmless,
        # but a gap or a jump to another layer ends that guide.
        extensions = {}
        if use_torso and bottom < hips.z-height*.15:
            shell_index = shell_lookup.get(component[0])
            if shell_index is not None:
                continuous = shell_components[shell_index]
                shell_faces = triangles[np.all((np.isin(triangles, continuous) & ~fill[triangles]), axis=1)]
                if len(shell_faces):
                    surface = SkirtSurface(points[shell_faces], center, probe_limit, 16)
                    level_step = height*.009
                    for index, (guide, angle) in enumerate(zip(guides, directions)):
                        root = guide['line'][0]
                        if guide['line'][-1].z >= hips.z-height*.15:
                            continue
                        radial = np.array(root[:2]) - center
                        root_angle = math.atan2(radial[1], radial[0])
                        angular_step = (angle-root_angle+math.pi) % math.tau - math.pi
                        previous = radial
                        slices = [(root.z, radial)]
                        for level in np.arange(root.z + level_step, probe_limit, level_step):
                            section = surface.section(float(level))
                            # A lower guide can sit off its sector's center on a
                            # lapel or fold. Start at the actual angle and ease
                            # to the evenly spaced sector above the waist.
                            t = min(1., (level-root.z)/(height*.065))
                            sample_angle = root_angle + angular_step*t*t*(3.-2.*t)
                            hit = section.hit(sample_angle, previous)
                            if hit is None:
                                hit = section.nearby_edge(sample_angle, math.tau/16*.22)
                            jump = height*(.043 if len(slices) == 1 else .027)
                            if (hit is None or np.linalg.norm(hit-previous) > jump
                                    or np.linalg.norm(hit) < .55*np.linalg.norm(radial)):
                                break
                            slices.append((float(level), hit))
                            previous = hit
                        anchor = max((b for b in anchors if root.z+height*.020 < b.head_local.z <= slices[-1][0]),
                                     key=lambda b: b.head_local.z, default=None)
                        if not anchor:
                            continue
                        top = anchor.head_local.z
                        count = max(2, int(math.ceil((top-root.z)/(height*.032))))
                        heights = np.linspace(top, root.z, count+1)
                        samples, coords = zip(*slices)
                        xy = np.column_stack([np.interp(heights, samples, np.asarray(coords)[:, axis])
                                              for axis in (0, 1)]) + center
                        line = [Vector((x, y, z)) for (x, y), z in zip(xy, heights)]
                        line[-1] = root.copy()
                        stem = guide['names'][0].rsplit('_',1)[0]
                        names = [f'{stem}_U{n:02d}' for n in range(count)]
                        extensions[index] = (names, line, anchor.name)
        values = {}
        coat_owned = set()
        coat_mass = {}
        # Each half meeting the old waist sample uses the same cloth joint.
        # Transfer to the body is measured only from the spine attachment on
        # extended guides. Short panels retain their original waist ramp.
        rails = {index: np.array([tuple(p) for p in reversed(line)])
                 for index, (_, line, _) in extensions.items()}
        lower_rails = {index: np.array([tuple(p) for p in reversed(guides[index]['line'])])
                       for index in rails}
        domain = ids
        shell_start = math.inf
        opening = None
        if extensions:
            opening = _opening_mask(SkirtSurface(points[faces], center, panel_top, 16),
                                    points, center, height)
            shell_start = min(r[0,2] for r in rails.values())-height*.040
            domain = ids | {i for i in continuous if points[i,2] > shell_start}
            if cloth_visibility is None:
                # The lowered sleeves occlude the coat in a radial probe.
                # The new cut faces also obscure its source surface.
                visible_faces = triangles[np.all((body_bank & ~fill)[triangles], axis=1)]
                cloth_visibility = BVHTree.FromPolygons(coordinates, visible_faces.tolist(), all_triangles=True)
        def exposed(i, depth):
            radial = points[i,:2]-center
            size = np.linalg.norm(radial)
            if size < 1e-8:
                return False
            direction = Vector((radial[0]/size, radial[1]/size, 0.))
            _, _, _, distance = cloth_visibility.ray_cast(
                Vector(points[i])+direction*height, -direction, height*1.0001)
            return distance is not None and height-distance <= depth
        for i in sorted(domain):
            z = points[i,2]
            original = i in ids
            if extensions and (fill[i] or opening[i]):
                continue
            row = {}
            support = 0.
            owned = 0.
            adjacent = angle_weights(directions, angles[i]) if len(guides)>1 else [(0,1.)]
            if z > shell_start:
                radius, angular_mass = 0., 0.
                for index, angular in adjacent:
                    rail = rails.get(index)
                    if rail is None or z > rail[-1,2]:
                        continue
                    if z < rail[0,2] and original:
                        rail = lower_rails[index]
                    elif z < rail[0,2]-height*.020:
                        continue
                    at = np.array([np.interp(z, rail[:,2], rail[:,ax]) for ax in (0,1)])
                    radius += angular*np.linalg.norm(at-center)
                    angular_mass += angular
                if not original and angular_mass <= 1e-8:
                    continue
                if angular_mass > 1e-8:
                    radial = points[i,:2]-center
                    size = np.linalg.norm(radial)
                    delta = abs(size-radius/angular_mass)
                    if delta > height*.025 or not exposed(i, height*.010):
                        continue
            for index, angular in adjacent:
                if angular <= 1e-8:
                    continue
                guide = guides[index]
                lower = guide['line']
                waist, hem = lower[0].z, lower[-1].z
                names = guide['names']
                parent = 'Hips'
                start, end = waist, hem
                if index in extensions:
                    upper_names, line, _ = extensions[index]
                    top = line[0].z
                    if not original and (z > top or z < waist-height*.020):
                        continue
                    if z > waist:
                        names = upper_names + [guide['names'][0]]
                        start, end = top, waist
                    else:
                        parent = upper_names[-1]
                    fade = float(np.clip((top-z)/(height*.047),0,1))
                    amount = fade*fade*(3.-2.*fade)
                    owned += angular
                elif original:
                    fade = float(np.clip((waist-z)/(height*.045),0,1))
                    amount = fade*fade*(3.-2.*fade)
                else:
                    continue
                factor = angular*amount
                if factor <= 1e-8:
                    continue
                support += factor
                for name, weight in _vertical_weights(names, parent, (start-z)/(start-end), True).items():
                    row[name] = row.get(name,0.)+factor*weight
            if original or support > 1e-8:
                body = normalized(weights(obj,i)) or {'Hips':1.}
                values[i] = blend_four(body, row, min(1.,support)) if owned else normalized({
                    n:body.get(n,0.)*(1.-support)+row.get(n,0.) for n in body.keys()|row.keys()
                })
                if owned > .5:
                    coat_owned.add(i)
                    coat_mass[i] = min(1., support)
        if extensions:
            # A nearby body island can't claim coat weights unless it is
            # reachable from the admitted lower cloth within the outer layer.
            _, surface_groups, _ = connected_graph(obj.data, points, values, height*.0001)
            included = {i for group in surface_groups if any(j in ids and points[j,2] <= shell_start for j in group)
                        for i in group}
            values = {i: row for i, row in values.items() if i in included}
            coat_owned.intersection_update(included)
            coat_mass = {i: amount for i, amount in coat_mass.items() if i in included}
            source = cut_bank & ~fill & ~opening
            admitted = {}
            def admit(i):
                if i not in admitted:
                    z = points[i,2]
                    admitted[i] = (not opening[i] and bottom <= z <= probe_limit
                                   and (bank is not None or eligible[i])
                                   and exposed(i, height*.014))
                return admitted[i]
            holes = _close_coat_gaps(points, edges, source, coat_owned, values, admit, height*.020,
                                    fill, lambda i: exposed(i, height*.009))
            values.update(holes)
            coat_owned.update(holes)
            cut = _bind_cut_fill(points, edges, fill, cut_bank & ~opening, values, coat_owned, height)
            values.update(cut)
            coat_owned.update(cut)
        if extensions:
            from avatar_coat_binding import close_enclosed_patches, interpolate_coat
            charts = []
            for index, guide in enumerate(guides):
                names, line = guide['names'], guide['line']
                if index in extensions:
                    upper_names, upper_line, _ = extensions[index]
                    names, line = upper_names+names, upper_line[:-1]+line
                charts.append((np.array([p.z for p in reversed(line)]), list(reversed([names[0]]+names[:-1]))))
            mass = {i: coat_mass.get(i, sum(w for n,w in row.items() if n.startswith(prefix)))
                    for i,row in values.items()}
            tops = max(chart[0][-1] for chart in charts)
            repairable = cut_bank & ~opening & (points[:,2] >= bottom) & (points[:,2] < tops)
            recovered = close_enclosed_patches(points, edges, mass, repairable, height)
            values.update({i: {} for i in recovered})
            mass.update(recovered)
            coat_owned.update(recovered)
            opening_top = max(points[opening,2], default=-math.inf)
            values = interpolate_coat(points, edges, values, lambda i: weights(obj,i),
                                      charts, directions, center, height, mass, opening_top)
        else:
            values,_ = smooth_domain(points, edges, values,
                                     iterations=16, factor=.5, tolerance=height*.0001)
        for index, (names, line, anchor) in extensions.items():
            guide = guides[index]
            guide['coat_spine'] = anchor
            guide['coat_lower_root'] = guide['names'][guide['physics_start']]
            guide['names'] = names + guide['names']
            guide['line'] = line[:-1] + guide['line']
        chains.extend(guides)
        regions.append(dict(obj=obj,role='garment',ids=sorted(values),values=values,anonymous_panel=True,coat_owned_ids=coat_owned))
    return chains,regions
