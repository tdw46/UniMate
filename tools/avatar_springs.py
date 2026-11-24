"""Geometry-derived secondary rigs and standard VRM 1 spring metadata.

Hallway owns generation, skinning and collider setup. Only the official VRM
schema is required; no simulator or BVT generation code is imported. Meshes
must be in the upright, aligned rest pose produced by our full-body pipeline.
Material roles can be supplied for assets without semantic material names.
"""
from avatar_material_roles import semantic_label

import math
import re

import bpy
import numpy as np
from mathutils import Vector

from avatar_apparel_weights import assign, components, weights, head_cap_vertices, hair_root_length, hair_free_top

from avatar_hair_guides import split_hanging_branches, branch_line
from avatar_waist_attachment import WAIST_FRACTION
from avatar_dress import dress_profile, install_lower_follow

from avatar_skirt_follow import INFLUENCE, set_rest_frame, set_constraint, preserve_export_rest_frame

PREFIX = 'Secondary_'


def set_generated_spring_center(rig, center=None):
    """Opt into translation compensation only for a deliberately named joint."""
    if center and center not in rig.data.bones:
        raise ValueError('Unknown explicitly requested spring center: '+center)
    changed=[]
    for spring in rig.data.vrm_addon_extension.spring_bone1.springs:
        if spring.vrm_name.startswith(PREFIX):
            spring.center.bone_name=center or ''
            changed.append(spring.vrm_name)
    return changed


def humanoid_roles():
    roles = {'hips':'Hips', 'spine':'Spine', 'chest':'Chest', 'neck':'Neck', 'head':'Head'}
    for side, suffix in (('left', 'L'), ('right', 'R')):
        for role, name in (('Shoulder','Clavicle'), ('UpperArm','UpperArm'), ('LowerArm','Forearm'),
                           ('Hand','Hand'), ('UpperLeg','Thigh'), ('LowerLeg','Shin'), ('Foot','Foot'), ('Toes','Toe')):
            roles[side+role] = name+'.'+suffix
        for digit in ('Thumb', 'Index', 'Middle', 'Ring', 'Little'):
            parts = ('Metacarpal', 'Proximal', 'Distal') if digit == 'Thumb' else ('Proximal', 'Intermediate', 'Distal')
            for i, part in enumerate(parts, 1):
                roles[side+digit+part] = f'{digit}{i}.{suffix}'
    return roles


def _points(rig, obj):
    transform = rig.matrix_world.inverted() @ obj.matrix_world
    return np.array([tuple(transform @ v.co) for v in obj.data.vertices])


def _material_region(obj, role, roles):
    tokens = {'hair': ('hair',), 'skirt': ('skirt', 'bottoms', 'hem'),
              'skin': ('skin', 'body')}[role]
    slots = {i for i, mat in enumerate(obj.data.materials) if mat and
             (roles.get(mat.name) == role or
              (mat.name not in roles and any(t in semantic_label(mat.name) for t in tokens)))}
    return {v for p in obj.data.polygons if p.material_index in slots for v in p.vertices}


def _line(points, segments, top=None):
    """Robust cross-section centers, including the actual root and tip rows."""
    lo, hi = points[:, 2].min(), points[:, 2].max()
    if top is not None:
        hi = min(hi, top)
    result = []
    for z in np.linspace(hi, lo, segments + 1):
        distances = np.abs(points[:, 2] - z)
        band = points[distances <= max((hi-lo)/segments*.3, np.sort(distances)[min(3, len(points)-1)])]
        result.append(Vector((*np.median(band[:, :2], axis=0), z)))
    return result


def _vertical_weights(names, parent, t, smooth_transition=False):
    # Parent occupies the attachment row; each segment takes over at its end.
    # This makes attachment exactly rigid and all transitions continuous.
    nodes = [parent] + names[:-1]
    t = float(np.clip(t, 0, 1)) * (len(nodes)-1)
    a = min(int(t), len(nodes)-2)
    blend = t-a
    if smooth_transition:
        blend = blend*blend*(3-2*blend)
    return {nodes[a]: 1-blend, nodes[a+1]: blend}


def _skirt_coverage(obj, points, ids, center, height):
    p=points[sorted(ids)];lo,hi=p[:,2].min(),p[:,2].max()
    if hi-lo<height*.10:return False,[]
    angles=np.arctan2(p[:,1]-center[1],p[:,0]-center[0])%math.tau
    coverage=[]
    for t in (.2,.5,.8):
        band=np.abs(p[:,2]-(hi-(hi-lo)*t))<(hi-lo)*.2
        coverage.append(len(set((angles[band]/math.tau*8).astype(int))))
    ids=set(ids)
    bridges=any(all(i in ids for i in face.vertices)
                and points[list(face.vertices),0].min()<center[0]<points[list(face.vertices),0].max()
                and points[list(face.vertices),2].mean()<lo+(hi-lo)*.35
                for face in obj.data.polygons)
    return min(coverage)>=7 and bridges,coverage


def plan_secondary(rig, meshes, material_roles=None, segments=4, skirt_sectors=12, skirt_segments=None, *, include_hair=True):
    """Detect long strands and circumferential skirts before mutating anything.

    Bottoms that do not surround both legs are rejected (e.g. separate trouser
    legs). Ambiguous garments should use explicit material roles instead of
    silently guessing. No source bones or original skin weights are consumed.
    """
    if segments < 2 or skirt_sectors < 6 or skirt_sectors % 2:
        raise ValueError('Need at least two segments and an even number of skirt sectors, at least six')
    roles = material_roles or {}
    from avatar_hair_regions import hair_images, extend_hair_region
    images = hair_images(meshes, roles) if include_hair else set()
    humanoid=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    def limb(side,part,legacy,mixamo):
        mapped=getattr(humanoid,side+'_'+part).node.bone_name
        return next((n for n in (mapped,legacy,mixamo) if n in rig.data.bones),None)
    thighs={'L':limb('left','upper_leg','Thigh.L','LeftUpLeg'),'R':limb('right','upper_leg','Thigh.R','RightUpLeg')}
    shins=[limb('left','lower_leg','Shin.L','LeftLeg'),limb('right','lower_leg','Shin.R','RightLeg')]
    if None in [*thighs.values(),*shins]:raise ValueError('Missing humanoid leg mapping')
    all_points = {o.name: _points(rig, o) for o in meshes}
    height = max(p[:, 2].max() for p in all_points.values()) - min(p[:, 2].min() for p in all_points.values())
    from avatar_surface_attachments import plan_mounted, plan_jackets
    chains, regions = plan_mounted(rig, meshes, all_points, height, PREFIX) if include_hair else ([], [])
    islands_by_mesh = {}
    hair_domains = {}
    pendants = []
    if include_hair:
        from avatar_hip_pendants import plan_pendants
        from avatar_virtual_topology import connected_graph
        hips = np.asarray(rig.data.bones['Hips'].head_local)
        surfaces = []
        for obj in meshes:
            points = all_points[obj.name]
            if not len(points) or points[:, 2].min() > hips[2] + .13 * height or points[:, 2].max() < hips[2] - .24 * height:
                continue
            _, islands, _ = connected_graph(obj.data, points, range(len(points)), height * .0001)
            islands_by_mesh[obj] = islands
            occupied = {i for region in regions if region['obj'] == obj for i in region['ids']}
            surfaces.append((obj, points, islands, occupied))
        spine = rig.data.bones.get('Spine')
        pendants, bindings = plan_pendants(surfaces, hips, height, PREFIX,
                                          spine.head_local if spine else None)
        regions.extend(bindings)
    rejected = []
    neck_z = rig.data.bones['Neck'].head_local.z
    center = np.array(tuple(rig.data.bones['Hips'].head_local))[:2]
    for obj in meshes:
        points = all_points[obj.name]
        if include_hair:
            from avatar_appendage_strands import plan_strands
            occupied = {i for r in regions if r['obj'] == obj for i in r['ids']}
            strands, bindings = plan_strands(rig, obj, points, height, PREFIX, len(chains), occupied, islands_by_mesh.get(obj))
            chains.extend(strands)
            regions.extend(bindings)
        hair = _material_region(obj, 'hair', roles) if include_hair else set()
        if include_hair:
            hair = extend_hair_region(obj, rig, points, hair, roles, images)
        from avatar_virtual_topology import connected_graph, protected_extremities
        hair -= set(np.flatnonzero(protected_extremities(rig, points, height)))
        hair_domains[obj] = hair.copy()
        hair -= {i for r in regions if r['obj']==obj for i in r['ids']}
        hair_graph, hair_components, hair_links = connected_graph(obj.data, points, hair, height*.0001)
        for component in hair_components:
            ids = sorted(set(component) & hair)
            if len(ids) < 8:
                continue
            from avatar_crown_strands import crown_strands
            for guide, fractions in crown_strands(hair_graph, ids, points, height, rig.data.bones['Head'].tail_local.z):
                label = f'{PREFIX}Hair_Crown_{len(chains):02d}'
                root = guide[0] - (guide[1]-guide[0]).normalized() * height * .002
                line = [root, *guide]
                names = [f'{label}_{i:02d}' for i in range(len(line))]
                chains.append(dict(role='hair', names=names, line=line, parent='Head', physics_start=1))
                values = {i: _vertical_weights(names[1:], 'Head', t, True) for i,t in fractions.items()}
                regions.append(dict(obj=obj, role='hair', ids=set(fractions), values=values))
            p = points[ids]
            free_top = hair_free_top(p, height, neck_z, rig.data.bones['Head'].tail_local.z)
            if free_top is None:
                continue
            branches = split_hanging_branches(hair_graph, ids, points, free_top,
                                         hair_root_length(height, np.ptp(p[:, 2])))
            for branch in branches:
                label = f'{PREFIX}Hair_{sum(c["role"] == "hair" for c in chains):02d}'
                free = branch_line(branch, segments)
                if free[0].z-free[-1].z < height*.008:
                    continue  # Too short below the cap for a meaningful spring.
                line = [branch['root']]+free
                names = [f'{label}_{i:02d}' for i in range(len(line))]
                chains.append(dict(role='hair', names=names, line=line, parent='Head', physics_start=1))
                regions.append(dict(obj=obj, ids=branch['ids'], role='hair', names=[names[1:]],
                                    lo=float(free[-1].z), hi=float(free[0].z)))
        if include_hair:
            from avatar_hair_satellites import attach_satellites
            satellites=attach_satellites(obj,points,hair,regions,height)
            if satellites:regions.append(satellites)
        skirt = sorted(_material_region(obj, 'skirt', roles)-{i for r in regions if r['obj']==obj for i in r['ids']})
        skirt = [i for i in skirt if points[i,2] < rig.data.bones['Hips'].tail_local.z+height*.04]
        if not skirt:
            # VRoid long dresses often label every layer as Tops, not Bottoms.
            # Restrict them anatomically before the circumferential tests below.
            waist = rig.data.bones['Hips'].tail_local.z
            knee = min(rig.data.bones[n].head_local.z for n in shins)
            slots = {i for i, mat in enumerate(obj.data.materials) if mat and
                     mat.name not in roles and 'cloth' in semantic_label(mat.name) and
                     not any(t in semantic_label(mat.name) for t in ('shoe', 'glove'))}
            # Validate each material independently. A dress must not lend its
            # circumferential coverage to an unrelated tail/belt accessory.
            accepted=set()
            for slot in slots:
                candidate={v for f in obj.data.polygons if f.material_index==slot for v in f.vertices
                           if points[v,2]<=waist}
                if candidate and min(points[i,2] for i in candidate)<knee:
                    valid,_=_skirt_coverage(obj,points,candidate,center,height)
                    if valid:accepted.update(candidate)
            skirt=sorted(accepted)
        if not skirt:
            continue
        p = points[skirt]
        lo, hi = p[:, 2].min(), p[:, 2].max()
        from avatar_skirt_surface import SkirtSurface
        obj.data.calc_loop_triangles()
        owned = set(skirt)
        triangles = [tuple(t.vertices) for t in obj.data.loop_triangles if owned.issuperset(t.vertices)]
        if not triangles or hi-lo <= height * 1e-6:
            continue
        surface = SkirtSurface(points[triangles], center, hi-(hi-lo)*WAIST_FRACTION, skirt_sectors)
        # Test the occupied cloth surface, including low-poly skirts whose
        # vertices all lie on the waistband and hem. Pants have no front or
        # back bridge between the legs in the lower skirt.
        span = surface.top - surface.bottom
        if span <= surface.eps:
            continue
        slices = [surface.section(surface.top - span*t) for t in (.2, .5, .75)]
        coverage = [sum(section.hit(j*math.tau/8) is not None for j in range(8))
                    for section in slices]
        bridge = any(hit is not None and np.linalg.norm(hit) > .12*np.ptp(p[:, 0])
                     for angle in (math.pi/2, -math.pi/2) for hit in [slices[-1].hit(angle)])
        slots = {f.material_index for f in obj.data.polygons if owned.issuperset(f.vertices)}
        materials = [obj.data.materials[i] for i in slots if i < len(obj.data.materials)]
        named_skirt = any(mat and (roles.get(mat.name) == 'skirt' or
                                   any(t in semantic_label(mat.name) for t in ('skirt', 'hem')))
                          for mat in materials)
        open_panel = named_skirt and surface.opening and any(
            slices[0].hit(angle) is not None for angle in (math.pi/2, -math.pi/2))
        if not (bridge or open_panel) or min(coverage) < 3 or max(coverage) < 5:
            rejected.append(dict(object=obj.name, role='skirt', reason='Not a continuous circumferential skirt', coverage=coverage))
            continue
        skirt_resolution = skirt_segments if skirt_segments is not None else segments
        names_by_sector = []
        guide_angles = []
        skirt_index = sum(r['role'] == 'skirt' for r in regions)
        skirt_chains = []
        for sector, angle in enumerate(surface.angles):
            interval = surface.interval(sector)
            if interval is None:
                continue
            top, sector_lo = interval
            side = 'L' if math.cos(angle) >= 0 else 'R'
            lower_leg = shins[0 if side == 'L' else 1]
            calf = rig.data.bones[lower_leg]
            from avatar_cloth_handoff import cloth_segments
            resolution = cloth_segments(top-sector_lo, height, skirt_resolution)
            profile = dress_profile(top, sector_lo, calf.head_local.z, calf.tail_local.z, resolution)
            coordinates = surface.guide(sector, profile)
            if coordinates is None:
                continue
            line = [Vector(point) for point in coordinates]
            guide_angles.append(angle)
            names = [f'{PREFIX}Skirt_{skirt_index:02d}_{sector:02d}_{i:02d}' for i in range(len(line))]
            skirt_chains.append(dict(role='skirt', names=names, line=line, parent='Hips',
                               follow=f'{PREFIX}SkirtFollow_{skirt_index:02d}_{sector:02d}',
                               leg=thighs[side], lower_leg=lower_leg, dress=profile,
                               follow_influence=INFLUENCE, opening_guide=bool(surface.opening and sector in surface.front),
                               garment_top=float(hi), garment_bottom=float(sector_lo)))
            names_by_sector.append(names)
        if len(names_by_sector) < 2:
            continue
        chains.extend(skirt_chains)
        if include_hair:
            from avatar_garment_straps import plan_straps
            excluded = {i for r in regions if r['obj'] == obj for i in r['ids']}
            straps, bindings = plan_straps(obj, points, skirt, triangles, surface, height,
                                          PREFIX, skirt_index, excluded, skirt_chains)
            chains.extend(straps)
            regions.extend(bindings)
            if bindings:
                mounted = set().union(*(r['ids'] for r in bindings))
                skirt = [i for i in skirt if i not in mounted]
        regions.append(dict(obj=obj, ids=skirt, role='skirt', names=names_by_sector,
                            lo=float(lo), hi=float(hi), center=center.tolist(), angles=guide_angles))
    if include_hair:
        for obj in meshes:
            excluded={i for r in regions if r['obj']==obj for i in r['ids']}
            extra,bindings=plan_jackets(rig,obj,all_points[obj.name],excluded,height,PREFIX,
                                       sum(c.get('garment',False) for c in chains))
            chains.extend(extra);regions.extend(bindings)
            if not extra:
                from avatar_anonymous_cloth import plan_panels
                extra,bindings=plan_panels(rig,obj,all_points[obj.name],excluded,height,PREFIX,start_index=len(chains))
                chains.extend(extra);regions.extend(bindings)
    accessory_reports=[]
    if include_hair:
        from avatar_accessories import plan_accessories
        for obj in meshes:
            excluded={i for r in regions if r['obj']==obj for i in r['ids']}
            # Exclude the complete hair domain, including its fixed scalp.
            named=_material_region(obj,'hair',roles)
            excluded.update(extend_hair_region(obj,rig,all_points[obj.name],named,roles,images))
            extra,bindings,audit=plan_accessories(rig,obj,all_points[obj.name],excluded,height,PREFIX,start_index=sum(a['components'] for a in accessory_reports))
            chains.extend(extra);regions.extend(bindings);accessory_reports.append(audit)
    if include_hair:
        from avatar_head_accessories import plan_head_accessories
        for obj in meshes:
            excluded={i for region in regions if region['obj']==obj for i in region['ids']}
            extra,bindings,_=plan_head_accessories(rig,obj,all_points[obj.name],excluded,height,PREFIX,
                                                   sum(c['role']=='ornament' for c in chains), hair_domain=hair_domains[obj])
            chains.extend(extra);regions.extend(bindings)
    cap = head_cap_vertices(meshes, rig, roles) if include_hair else {}
    regions.extend(dict(obj=obj, ids=set(cap[obj.name])-{i for r in regions if r['obj']==obj for i in r['ids']}, role='head_cap')
                   for obj in meshes if obj.name in cap)
    chains.extend(pendants)
    return dict(chains=chains, regions=regions, rejected=rejected, skirt_center=center.tolist(), height=float(height), points=all_points, accessories=accessory_reports)


def generate_secondary(rig, meshes, material_roles=None, segments=4, skirt_sectors=12, spring_center=None, skirt_segments=None, refine_cloth=True, symmetric_guides=False):
    """Add a secondary rig once, preserving all geometry and unrelated weights.

    Repeated calls reject before mutation rather than accumulating duplicate
    chains or deleting an existing artist-authored setup.
    """
    if bpy.context.mode != 'OBJECT':
        raise ValueError('Generate secondary rigs in Object mode')
    if spring_center and spring_center not in rig.data.bones:
        raise ValueError('Unknown explicitly requested spring center: '+spring_center)
    if any(b.name.startswith(PREFIX) for b in rig.data.bones):
        raise ValueError('This rig already has Hallway secondary chains')
    if not hasattr(rig.data, 'vrm_addon_extension'):
        raise RuntimeError('Enable the official VRM schema add-on before generation')
    for name in ('Head', 'Neck', 'Hips'):
        if name not in rig.data.bones:
            raise ValueError('Missing humanoid attachment bone: '+name)
    plan = plan_secondary(rig, meshes, material_roles, segments, skirt_sectors, skirt_segments)
    if symmetric_guides:
        from mesh_landmarks.rig_frames import symmetrize_guides
        symmetrize_guides(plan['chains'], rig, plan['height'])
    if not plan['chains']:
        raise ValueError('No supported long-hair or skirt regions detected')
    selected = list(bpy.context.selected_objects)
    active = bpy.context.view_layer.objects.active
    pose_position = rig.data.pose_position
    rig.data.pose_position = 'REST'
    try:
        from avatar_pelvis_binding import repair_pelvis_weights,transfer_garment_waist,confine_pelvis_weights
        pelvis_binding=repair_pelvis_weights(rig,meshes) if any(c['role']=='skirt' for c in plan['chains']) else dict(applied=False)
        if any(c['role']=='skirt' for c in plan['chains']):
            pelvis_binding['socket_falloff']=confine_pelvis_weights(rig,meshes)
        bpy.ops.object.select_all(action='DESELECT')
        rig.select_set(True)
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode='EDIT')
        for chain in plan['chains']:
            if chain['role'] == 'skirt':
                # A separate unweighted parent owns leg-follow. Spring bones
                # remain unconstrained, so the solver can add secondary motion.
                leg = rig.data.edit_bones[chain['leg']]
                follow = rig.data.edit_bones.new(chain['follow'])
                set_rest_frame(follow, leg)
                # The official VRM exporter uses glTF's deform-bones filter.
                # Keep the control exportable, but give it no vertex weights.
                follow.use_deform = True
                chain['parent'] = follow.name
            if chain.get('garment') and chain['parent'] in ('Spine', 'UpperSpine', 'Chest'):
                parent = rig.data.edit_bones[chain['parent']]
                source = parent
                rear = chain.get('rear_share', 0.)
                if rear > .2:
                    hem = chain['line'][-1].z
                    while source.parent and source.parent.name in ('Spine', 'UpperSpine', 'Chest'):
                        if abs(source.parent.head.z - hem) >= abs(source.head.z - hem):
                            break
                        source = source.parent
                mount = rig.data.edit_bones.new(chain['names'][0] + '_Mount')
                mount.head = chain['line'][0]
                mount.tail = mount.head + (source.tail - source.head)
                # Roll the mount half a turn about Y to reverse the two bend
                # axes using a single VRM local rotation constraint.
                mount.roll = source.roll + math.pi
                mount.parent = parent
                chain['jacket_follow'] = (mount.name, source.name, .18 + .17 * rear)
                chain['parent'] = mount.name
            if 'panel_follow' in chain:
                mount = rig.data.edit_bones.new(chain['load_parent'])
                driver = rig.data.edit_bones[chain['panel_follow']]
                set_rest_frame(mount, driver)
                shift = (chain['line'][chain['load_index']]-driver.head)*chain['mount_fraction']
                mount.head = driver.head+shift
                mount.tail = driver.tail+shift
                mount.roll = driver.roll
            if 'panel_mount' in chain:
                mount = rig.data.edit_bones.new(chain['panel_mount'])
                mount.head = chain['panel_mount_head']
                mount.tail = chain['line'][chain['panel_free_start']]
                mount.parent = rig.data.edit_bones[chain['parent']]
                mount.use_deform = True  # Export the spring node without skin weights.
                tip = rig.data.edit_bones.new(chain['panel_mount'] + 'Tip')
                tip.head = mount.tail
                tip.tail = tip.head + (chain['line'][chain['panel_free_start']+1] - tip.head) * .1
                tip.parent = mount
                tip.use_connect = True
                tip.use_deform = True
            for i, (name, point) in enumerate(zip(chain['names'], chain['line'])):
                bone = rig.data.edit_bones.new(name)
                bone.head = point
                bone.tail = chain['line'][i+1] if i < len(chain['line'])-1 else point+(point-chain['line'][i-1])*.1
                parent = chain['parent'] if i == 0 else chain['names'][i-1]
                starts_at_load = i == chain.get('load_index')
                if 'panel_mount' in chain and i in (0, chain['panel_free_start']):
                    # The fitted skin bones inherit the support rotation. The
                    # free spring starts at its tip on a direct child branch.
                    parent = chain['panel_mount'] if i == 0 else chain['panel_mount'] + 'Tip'
                elif starts_at_load:
                    parent = chain['load_parent']
                elif 'mount_parents' in chain:
                    parent = chain['mount_parents'][i]
                bone.parent = rig.data.edit_bones[parent]
                bone.use_connect = (i > 0 and not starts_at_load and 'mount_parents' not in chain
                                    and i != chain.get('panel_free_start'))
                if chain['role'] == 'skirt' and i == 0:
                    # Binding runs before the upper/lower spring split. Mark
                    # its knee now so fresh generation gets the same smoothing.
                    if chain['dress']['lower_follow']:
                        bone['hallway_dress_knee_index'] = chain['dress']['knee_index']
                    bone['hallway_bilateral_chain'] = True
                    if chain['opening_guide']:
                        bone['hallway_opening_guide'] = True
                    bone['hallway_skirt_surface_center'] = plan['skirt_center']
                    bone['hallway_garment_top'] = chain['garment_top']
                    bone['hallway_garment_bottom'] = chain['garment_bottom']
                if 'panel_transition' in chain and name == chain['panel_transition'][-1]:
                    # Terminate the transition on its own node. The loose
                    # panel still starts at the same point under its support.
                    terminal = rig.data.edit_bones.new(name + '_TransitionTip')
                    terminal.head = chain['line'][chain['panel_free_start']]
                    terminal.tail = terminal.head + (bone.tail - bone.head) * .1
                    terminal.parent = bone
                    terminal.use_connect = False
                    terminal.use_deform = True
                bone.use_deform = i < len(chain['line'])-1
                if chain['role'] == 'hair' and i == 0:
                    bone['unimate_fixed_hair_root'] = True
        bpy.ops.object.mode_set(mode='OBJECT')
        for chain in plan['chains']:
            if 'jacket_follow' in chain:
                mount, source, share = chain['jacket_follow']
                constraint = rig.pose.bones[mount].constraints.new('COPY_ROTATION')
                set_constraint(constraint, rig, source, share)
                constraint.name = 'Jacket spine follow'
            if 'panel_follow' in chain:
                constraint = rig.pose.bones[chain['load_parent']].constraints.new('COPY_ROTATION')
                set_constraint(constraint, rig, chain['panel_follow'], 1.)
            if chain['role'] != 'skirt':
                continue
            constraint = rig.pose.bones[chain['follow']].constraints.new('COPY_ROTATION')
            set_constraint(constraint, rig, chain['leg'], chain['follow_influence'])
        for region in plan['regions']:
            obj = region['obj']
            if region['role'] == 'head_cap':
                assign(obj, region['ids'], {'Head': 1.})
                continue
            if 'values' in region:
                for index, values in region['values'].items():assign(obj,[index],values)
                continue
            for index in region['ids']:
                p = plan['points'][obj.name][index]
                t = (region['hi']-p[2])/(region['hi']-region['lo'])
                if region['role'] == 'hair':
                    values = _vertical_weights(region['names'][0], 'Head', t, smooth_transition=True)
                else:
                    angle = math.atan2(p[1]-region['center'][1], p[0]-region['center'][0]) % math.tau
                    from avatar_skirt_surface import angle_weights
                    values = {}
                    for s, factor in angle_weights(region['angles'], angle):
                        for name, weight in _vertical_weights(region['names'][s], 'Hips', t).items():
                            values[name] = values.get(name, 0)+weight*factor
                assign(obj, [index], values)
        from avatar_secondary_smoothing import smooth_secondary_weights
        smoothing=smooth_secondary_weights(rig,meshes,plan['regions'])
        rig.data.pose_position = 'POSE'
        bpy.context.view_layer.update()
        metadata = _write_vrm(rig, meshes, plan, material_roles or {}, spring_center)
        metadata['pelvis_binding']=pelvis_binding
        metadata['secondary_smoothing']=smoothing
        if any(c['role']=='skirt' for c in plan['chains']):
            from avatar_contact_colliders import install_contact_colliders
            from avatar_skirt_binding import rebind_skirt_strips
            from avatar_directional_contacts import install_skirt_contact_rig
            metadata['contact_colliders']=install_contact_colliders(rig,meshes)
            metadata['strip_binding']=rebind_skirt_strips(rig,meshes,regions=[r for r in plan['regions'] if r['role']=='skirt'])
            metadata['waist_attachment']=transfer_garment_waist(rig,meshes)
            metadata['dress']=install_lower_follow(rig,plan['chains'])
            metadata['contact_rig']=install_skirt_contact_rig(rig,meshes)
            from avatar_skirt_hip_physics import extend_hip_physics
            metadata['hip_physics']=extend_hip_physics(rig)
        from avatar_cloth_handoff import finish_waistbands, smooth_cross_waist
        metadata['waistband_handoff']=finish_waistbands(rig,meshes,plan)
        from avatar_pelvis_anchor import strengthen_front_pelvis
        metadata['front_pelvis_attachment']=strengthen_front_pelvis(rig,meshes,plan)
        metadata['cross_waist_smoothing']=smooth_cross_waist(rig,meshes,plan)
        from avatar_garment_attachments import bind_garment_attachments
        rigid_mounts = {(r['obj'].name, i): row for r in plan['regions'] if r.get('garment_strap')
                       for i, row in r['values'].items() if i in r['hardware']}
        metadata['garment_attachments']=bind_garment_attachments(rig,meshes,rigid_mounts=rigid_mounts)
        from avatar_cloth_overlays import bind_cloth_overlays, refine_bound_overlays
        metadata['cloth_overlays']=bind_cloth_overlays(rig,meshes,refine=False)
        from avatar_cloth_smoothing import smooth_skirt_distribution
        metadata['skirt_distribution_smoothing']=smooth_skirt_distribution(
            rig,meshes,plan,protected=metadata['cloth_overlays'])
        # Generic nearby-cloth transfers can also reach a mounted fitting.
        # Its independent carrier has final ownership over that small domain.
        for region in plan['regions']:
            if region.get('garment_strap'):
                for index, row in region['values'].items():
                    assign(region['obj'], [index], row)
        # Re-sample only the cloth fraction; preserve the accepted body attachment.
        metadata['garment_attachments']=bind_garment_attachments(rig,meshes,preserve_anatomical=True,rigid_mounts=rigid_mounts)
        from avatar_garment_straps import bind_panel_mount
        for region in plan['regions']:
            if region.get('panel_mount'):
                bind_panel_mount(region)
        if refine_cloth:refine_bound_overlays(rig,meshes,metadata['cloth_overlays'])
        from avatar_cloth_contacts import install_overlay_contacts
        metadata['cloth_overlay_contacts']=install_overlay_contacts(rig,metadata['cloth_overlays'])
        from avatar_pelvis_cloth import bind_short_pelvis_cloth
        metadata['short_pelvis_cloth']=bind_short_pelvis_cloth(rig,meshes,plan)
        metadata['spring_names']=[s.vrm_name for s in rig.data.vrm_addon_extension.spring_bone1.springs if s.vrm_name.startswith(PREFIX)]
    finally:
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        rig.data.pose_position = pose_position
        bpy.ops.object.select_all(action='DESELECT')
        for obj in selected:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = active
    report = dict(height=plan['height'], chains=len(plan['chains']),
                  deform_bones=sum(len(c['names'])-1 for c in plan['chains']),
                  terminal_bones=len(plan['chains']), rejected=plan['rejected'], accessories=plan['accessories'],
                  fixed_hair_roots=[c['names'][0] for c in plan['chains'] if c['role'] == 'hair'],
                  hair_attachments=[dict(root=c['names'][0], physics_start=c['names'][1],
                                         cap_position=list(c['line'][0]), fixed_end=list(c['line'][1]))
                                    for c in plan['chains'] if c['role'] == 'hair'],
                  leg_follow=[dict(bone=c['follow'], target=c['leg'], influence=c['follow_influence'])
                              for c in plan['chains'] if c['role'] == 'skirt'],
                  regions=[dict(object=r['obj'].name, role=r['role'], vertices=len(r['ids'])) for r in plan['regions']],
                  **metadata)
    rig['unimate_secondary_generator'] = 4
    rig['hallway_bilateral_skirt_layout'] = 1
    from properties_hallway_rig import initialize
    from avatar_bone_collections import organize_bones
    initialize(rig)
    from avatar_dress_fit import configure_dress_fit
    report['dress_fit'] = configure_dress_fit(rig,meshes)
    from properties_hallway_rig import set_profile_defaults, apply_settings
    report['settings_profile'] = set_profile_defaults(rig)
    apply_settings(rig)
    from avatar_cloth_response import stabilize_cloth_contacts
    report['cloth_contact_response'] = stabilize_cloth_contacts(rig, rig.hallway_rig.spring_groups.get('Skirt'))
    for row in report['leg_follow']:
        row['influence'] = rig.pose.bones[row['bone']].constraints[0].influence
    # Fit the limb surfaces after the mesh has its final skin weights.
    suspended_roots = {c['names'][c['physics_start']] for c in plan['chains']
                       if c.get('articulated_fittings')}
    if suspended_roots:
        for spring in rig.data.vrm_addon_extension.spring_bone1.springs:
            if spring.joints and spring.joints[0].node.bone_name in suspended_roots:
                spring['hallway_suspended_strap'] = True
        from avatar_strap_contacts import install_strap_contacts
        from avatar_vrm_colliders import organize_colliders
        from avatar_skirt_fit_io import rest_edit
        with rest_edit(rig):
            install_strap_contacts(rig, meshes, PREFIX, plan['height'])
            organize_colliders(rig)
        spring_data = rig.data.vrm_addon_extension.spring_bone1
        report.update(colliders=len(spring_data.colliders), collider_groups=len(spring_data.collider_groups))
    report['bone_collections'] = organize_bones(rig)
    return report


def _fit_free_panel_radii(rig, details, height, spring_names):
    """Keep contact on the swinging hem without displacing it at rest."""
    from avatar_colliders import segment_distance
    capsules = []
    for spec in details:
        if spec['role'] == 'frontpanel':
            rest = rig.data.bones[spec['bone']].matrix_local
            capsules.append((rest @ Vector(spec['offset']), rest @ Vector(spec['tail']), spec['radius']))
    if not capsules:
        return
    for spring in rig.data.vrm_addon_extension.spring_bone1.springs:
        if spring.vrm_name not in spring_names:
            continue
        for head, tail in zip(spring.joints, spring.joints[1:]):
            point = rig.data.bones[tail.node.bone_name].head_local
            space = min(segment_distance(point, a, b) - r for a, b, r in capsules)
            head.hit_radius = max(0., min(.0025 * height, space - .0015 * height))


_FITTED_PANEL_SETTINGS = (1.2, .7, 1.6)  # stiffness, drag and gravity


def _write_panel_transition(sb, chain, center, free_spring):
    names = chain['panel_transition']
    first = free_spring.joints[0]
    target = (first.stiffness, first.drag_force, first.gravity_power)
    spring = sb.springs.add()
    spring.vrm_name = chain['names'][0].rsplit('_', 1)[0].replace('Frontpanel_', 'Paneltransition_')
    spring.center.bone_name = center
    for index, name in enumerate([*names, names[-1] + '_TransitionTip']):
        fraction = (index + 1) / (len(names) + 1)
        joint = spring.joints.add()
        joint.node.bone_name = name
        joint.stiffness, joint.drag_force, joint.gravity_power = (
            fitted + fraction * (loose - fitted)
            for fitted, loose in zip(_FITTED_PANEL_SETTINGS, target))
        joint.gravity_dir = (0, 0, -1)
        joint.hit_radius = 0.


def _write_fitted_panel_support(sb, mount, tip, center):
    """Give just the waistband support the narrow leg contacts."""
    spring = sb.springs.add()
    spring.vrm_name = mount.rsplit('_', 2)[0].replace('Frontpanel_', 'Fittedfront_')
    spring.center.bone_name = center
    for name in (mount, tip):
        joint = spring.joints.add()
        joint.node.bone_name = name
        joint.stiffness, joint.drag_force, joint.gravity_power = _FITTED_PANEL_SETTINGS
        joint.gravity_dir = (0, 0, -1)
        joint.hit_radius = 0.


def _write_vrm(rig, meshes, plan, roles, spring_center=None):
    ext = rig.data.vrm_addon_extension
    ext.spec_version = '1.0'
    preserve_export_rest_frame(rig)
    human_roles = humanoid_roles()
    if 'UpperSpine' in rig.data.bones:
        human_roles.update(chest='UpperSpine', upperChest='Chest')
    for role, name in human_roles.items():
        if name in rig.data.bones:
            key = re.sub(r'(?<!^)(?=[A-Z])', '_', role).lower()
            getattr(ext.vrm1.humanoid.human_bones, key).node.bone_name = name
    sb = ext.spring_bone1
    height = plan['height']
    from avatar_accessory_physics import physics_chains
    for chain in physics_chains([c for c in plan['chains'] if 'mount_parents' not in c]):
        # Honor an explicit rig-wide center. Floating loads otherwise swing
        # in world space while their attachment tracks the supporting garment.
        close_panel = chain.get('close_panel', False)
        center = spring_center or chain.get('strap_center') or (chain['parent'] if close_panel else '')
        start = chain.get('physics_start', 0)
        if 'panel_mount' in chain:
            tip = chain['panel_free_start']
            _write_fitted_panel_support(sb, chain['panel_mount'], chain['panel_mount'] + 'Tip', center)
            # Distinct terminal and free-root nodes share the same location.
            # Each spring owns its own joints and its own leg contacts.
            rig.data.bones[chain['panel_mount'] + 'Tip']['close_front_panel'] = True
            start = tip
        spring = sb.springs.add()
        spring.vrm_name = chain['names'][0].rsplit('_', 1)[0]
        spring.center.bone_name = center
        hair = chain['role'] == 'hair'
        # A terminal tassel inherits the last integrated link. Including the
        # mesh tip would integrate another spring and flick the tassel twice.
        stop = -1 if chain['role'] == 'pendant' else None
        for i, name in enumerate(chain['names'][start:stop]):
            joint = spring.joints.add()
            joint.node.bone_name = name
            if chain.get('free_appendage'):
                joint.stiffness = 1.
                joint.drag_force = .5
                joint.gravity_power = 0.
                joint.hit_radius = 0.
                if close_panel:
                    # Support the upper fabric against the leg. Ease the pull
                    # down the free length so the shorter hem links can curl.
                    along = (start + i) / max(1, len(chain['names']) - 1)
                    bend = min(1., max(0., (along - .26) / .42))
                    bend = bend * bend * (3. - 2. * bend)
                    joint.stiffness = height * (.266 + .325 * bend)
                    joint.drag_force = .58 - .08 * bend
                    joint.gravity_power = height * (1.655 - 1.39 * bend)
                    rig.data.bones[name]['close_front_panel'] = True
                joint.gravity_dir = (0, 0, -1)
                continue
            from avatar_rig_defaults import SPRINGS
            defaults = SPRINGS['Head Accessories' if chain['role']=='ornament' else 'Hanging Straps' if chain['role']=='accessory' and '_Strand_' in chain['names'][0] else chain['role'].title()]
            joint.stiffness = defaults['stiffness'] * (1. if chain['role']=='ornament' or chain.get('serial_accessory') else 1-.10*i)
            joint.drag_force = defaults['drag']
            joint.gravity_power = defaults['gravity']
            if chain.get('articulated_fittings'):
                # Short links between fittings flex more easily. Let gravity
                # carry the loaded web below its attachment as the leg recedes.
                segment = rig.data.bones[name].length
                joint.stiffness = .65 * min(1., segment / (.055 * height))
                joint.drag_force = .46
                joint.gravity_power = .23
                joint['hallway_stiffness_ratio'] = joint.stiffness / defaults['stiffness']
                joint['hallway_generated_drag_ratio'] = joint.drag_force / defaults['drag']
                joint['hallway_generated_gravity_ratio'] = joint.gravity_power / defaults['gravity']
            joint.gravity_dir = (0, 0, -1)
            # Bone-point collisions need a margin for the skinned cloth between
            # adjacent chains, which can lie inside their collision envelope.
            joint.hit_radius = 0. if chain['role']=='ornament' else height*(.012 if chain['role']=='skirt' else .004)
        if chain.get('mounted_pack'):
            # Geometry-generated pack behavior: fixed shoulder support, one
            # coherent pouch, and a separate front strap where present.
            for joint in spring.joints:
                joint.stiffness=1.;joint.drag_force=.5;joint.gravity_power=0.;joint.hit_radius=0.
                joint['hallway_stiffness_ratio']=1./SPRINGS['Accessory']['stiffness']
                joint['hallway_generated_drag_ratio']=.5/SPRINGS['Accessory']['drag']
                joint['hallway_generated_gravity_ratio']=0.
        if hair:
            from avatar_rig_defaults import configure_hair_spring
            configure_hair_spring(spring, rig, height)
        elif 'panel_mount' in chain:
            _write_panel_transition(sb, chain, center, spring)
    from avatar_garment_follow import install_garment_follow
    contact_chains, coat_follow = install_garment_follow(rig, plan['chains'])
    from avatar_colliders import rebuild_colliders
    metadata = rebuild_colliders(rig, meshes, roles)
    free_panels = {c['names'][0].rsplit('_', 1)[0] for c in plan['chains'] if 'panel_mount' in c}
    if free_panels:
        _fit_free_panel_radii(rig, metadata['collider_details'], height, free_panels)
    if any(c.get('garment') for c in plan['chains']):
        from avatar_surface_attachments import install_garment_contacts
        metadata['garment_contacts']=install_garment_contacts(
            rig, meshes, dict(plan, chains=contact_chains), PREFIX)
        metadata['coat_follow'] = coat_follow
    if any(c['role']=='hair' for c in plan['chains']):
        from avatar_hair_contacts import install_hair_contacts, fit_generated_hair_radii
        metadata['hair_contacts'] = install_hair_contacts(rig, meshes, roles)
        metadata['hair_contact_radii'] = fit_generated_hair_radii(rig, height)
        metadata['colliders'] = len(sb.colliders)
        metadata['collider_groups'] = len(sb.collider_groups)
    if any(c.get('garment') or c['role']=='hair' for c in plan['chains']):
        from avatar_surface_guards import fit_surface_guards
        metadata['surface_guards'] = fit_surface_guards(rig, meshes, height)
        metadata['colliders'] = len(sb.colliders)
        metadata['collider_groups'] = len(sb.collider_groups)
    metadata.pop('height', None)
    return dict(**metadata, spring_names=[s.vrm_name for s in sb.springs if s.vrm_name.startswith(PREFIX)])
