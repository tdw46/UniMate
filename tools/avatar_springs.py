"""Geometry-derived secondary rigs and standard VRM 1 spring metadata.

UniMate owns generation, skinning and collider setup. Only the official VRM
schema is required; no simulator or BVT generation code is imported. Meshes
must be in the upright, aligned rest pose produced by our full-body pipeline.
Material roles can be supplied for assets without semantic material names.
"""
import math
import re

import bpy
import numpy as np
from mathutils import Vector

from avatar_apparel_weights import assign, components, weights, head_cap_vertices, hair_root_length, hair_free_top

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
    tokens = {'hair': ('hair',), 'skirt': ('skirt', 'bottoms'),
              'skin': ('skin', 'body')}[role]
    slots = {i for i, mat in enumerate(obj.data.materials) if mat and
             (roles.get(mat.name) == role or
              (mat.name not in roles and any(t in mat.name.lower() for t in tokens)))}
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


def plan_secondary(rig, meshes, material_roles=None, segments=4, skirt_sectors=12, skirt_segments=None):
    """Detect long strands and circumferential skirts before mutating anything.

    Bottoms that do not surround both legs are rejected (e.g. separate trouser
    legs). Ambiguous garments should use explicit material roles instead of
    silently guessing. No source bones or original skin weights are consumed.
    """
    if segments < 2 or skirt_sectors < 6:
        raise ValueError('Need at least two segments and six skirt sectors')
    roles = material_roles or {}
    humanoid=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    def limb(side,part,legacy,mixamo):
        mapped=getattr(humanoid,side+'_'+part).node.bone_name
        return next((n for n in (mapped,legacy,mixamo) if n in rig.data.bones),None)
    thighs={'L':limb('left','upper_leg','Thigh.L','LeftUpLeg'),'R':limb('right','upper_leg','Thigh.R','RightUpLeg')}
    shins=[limb('left','lower_leg','Shin.L','LeftLeg'),limb('right','lower_leg','Shin.R','RightLeg')]
    if None in [*thighs.values(),*shins]:raise ValueError('Missing humanoid leg mapping')
    all_points = {o.name: _points(rig, o) for o in meshes}
    height = max(p[:, 2].max() for p in all_points.values()) - min(p[:, 2].min() for p in all_points.values())
    chains, regions, rejected = [], [], []
    neck_z = rig.data.bones['Neck'].head_local.z
    center = np.array(tuple(rig.data.bones['Hips'].head_local))[:2]
    for obj in meshes:
        points = all_points[obj.name]
        hair = _material_region(obj, 'hair', roles)
        for component in components(obj.data):
            ids = sorted(set(component) & hair)
            if len(ids) < 8:
                continue
            p = points[ids]
            free_top = hair_free_top(p, height, neck_z, rig.data.bones['Head'].tail_local.z)
            if free_top is None:
                continue
            label = f'{PREFIX}Hair_{sum(c["role"] == "hair" for c in chains):02d}'
            root = _line(p, segments, top=min(p[:, 2].max(), free_top+hair_root_length(height, np.ptp(p[:, 2]))))[0]
            free = _line(p, segments, top=free_top)
            line = [root]+free
            names = [f'{label}_{i:02d}' for i in range(len(line))]
            chains.append(dict(role='hair', names=names, line=line, parent='Head', physics_start=1))
            regions.append(dict(obj=obj, ids=ids, role='hair', names=[names[1:]],
                                lo=float(p[:, 2].min()), hi=float(free[0].z)))
        skirt = sorted(_material_region(obj, 'skirt', roles))
        if not skirt:
            # VRoid long dresses often label every layer as Tops, not Bottoms.
            # Restrict them anatomically before the circumferential tests below.
            waist = rig.data.bones['Hips'].tail_local.z
            knee = min(rig.data.bones[n].head_local.z for n in shins)
            slots = {i for i, mat in enumerate(obj.data.materials) if mat and
                     mat.name not in roles and 'cloth' in mat.name.lower() and
                     not any(t in mat.name.lower() for t in ('shoe', 'glove'))}
            candidate = {v for f in obj.data.polygons if f.material_index in slots for v in f.vertices}
            if candidate and min(points[i, 2] for i in candidate) < knee:
                skirt = sorted(i for i in candidate if points[i, 2] <= waist)
        if not skirt:
            continue
        p = points[skirt]
        lo, hi = p[:, 2].min(), p[:, 2].max()
        delta = p[:, :2]-center
        angles = np.arctan2(delta[:, 1], delta[:, 0]) % math.tau
        # A skirt must surround the pelvis at multiple heights. Require faces
        # crossing the sagittal plane below the waist; separated pants fail.
        skirt_set = set(skirt)
        bridges = sum(1 for face in obj.data.polygons
                      if all(i in skirt_set for i in face.vertices)
                      and points[list(face.vertices), 0].min() < center[0]
                      < points[list(face.vertices), 0].max()
                      and points[list(face.vertices), 2].mean() < lo+(hi-lo)*.35)
        coverage = []
        for t in (.2, .5, .8):
            band = np.abs(p[:, 2]-(hi-(hi-lo)*t)) < (hi-lo)*.2
            coverage.append(len(set((angles[band]/math.tau*8).astype(int))))
        if hi-lo < height*.10 or min(coverage) < 7 or not bridges:
            rejected.append(dict(object=obj.name, role='skirt', reason='Not a continuous circumferential skirt', coverage=coverage))
            continue
        skirt_resolution = skirt_segments if skirt_segments is not None else segments
        names_by_sector = []
        radii = np.linalg.norm(delta, axis=1)
        skirt_index = sum(r['role'] == 'skirt' for r in regions)
        for sector in range(skirt_sectors):
            angle = sector/skirt_sectors*math.tau
            angular_distance = np.abs((angles-angle+math.pi) % math.tau-math.pi)
            line = []
            for z in np.linspace(hi, lo, skirt_resolution+1):
                score = angular_distance/(math.tau/skirt_sectors) + np.abs(p[:, 2]-z)/((hi-lo)/skirt_resolution)
                near = np.argsort(score)[:max(4, len(p)//(skirt_sectors*skirt_resolution))]
                radius = float(np.median(radii[near]))
                line.append(Vector((* (center+radius*np.array((math.cos(angle), math.sin(angle)))), z)))
            names = [f'{PREFIX}Skirt_{skirt_index:02d}_{sector:02d}_{i:02d}' for i in range(skirt_resolution+1)]
            side = 'L' if math.cos(angle) >= 0 else 'R'
            chains.append(dict(role='skirt', names=names, line=line, parent='Hips',
                               follow=f'{PREFIX}SkirtFollow_{skirt_index:02d}_{sector:02d}',
                               leg=thighs[side], follow_influence=INFLUENCE))
            names_by_sector.append(names)
        regions.append(dict(obj=obj, ids=skirt, role='skirt', names=names_by_sector,
                            lo=float(lo), hi=float(hi), center=center.tolist()))
    cap = head_cap_vertices(meshes, rig)
    regions.extend(dict(obj=obj, ids=cap[obj.name], role='head_cap')
                   for obj in meshes if obj.name in cap)
    return dict(chains=chains, regions=regions, rejected=rejected, height=float(height), points=all_points)


def generate_secondary(rig, meshes, material_roles=None, segments=4, skirt_sectors=12, spring_center=None, skirt_segments=None):
    """Add a secondary rig once, preserving all geometry and unrelated weights.

    Repeated calls reject before mutation rather than accumulating duplicate
    chains or deleting an existing artist-authored setup.
    """
    if bpy.context.mode != 'OBJECT':
        raise ValueError('Generate secondary rigs in Object mode')
    if spring_center and spring_center not in rig.data.bones:
        raise ValueError('Unknown explicitly requested spring center: '+spring_center)
    if any(b.name.startswith(PREFIX) for b in rig.data.bones):
        raise ValueError('This rig already has UniMate secondary chains')
    if not hasattr(rig.data, 'vrm_addon_extension'):
        raise RuntimeError('Enable the official VRM schema add-on before generation')
    for name in ('Head', 'Neck', 'Hips'):
        if name not in rig.data.bones:
            raise ValueError('Missing humanoid attachment bone: '+name)
    plan = plan_secondary(rig, meshes, material_roles, segments, skirt_sectors, skirt_segments)
    if not plan['chains']:
        raise ValueError('No supported long-hair or skirt regions detected')
    selected = list(bpy.context.selected_objects)
    active = bpy.context.view_layer.objects.active
    pose_position = rig.data.pose_position
    rig.data.pose_position = 'REST'
    try:
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
            for i, (name, point) in enumerate(zip(chain['names'], chain['line'])):
                bone = rig.data.edit_bones.new(name)
                bone.head = point
                bone.tail = chain['line'][i+1] if i < len(chain['line'])-1 else point+(point-chain['line'][i-1])*.1
                bone.parent = rig.data.edit_bones[chain['parent'] if i == 0 else chain['names'][i-1]]
                bone.use_connect = i > 0
                bone.use_deform = i < len(chain['line'])-1
                if chain['role'] == 'hair' and i == 0:
                    bone['unimate_fixed_hair_root'] = True
        bpy.ops.object.mode_set(mode='OBJECT')
        for chain in plan['chains']:
            if chain['role'] != 'skirt':
                continue
            constraint = rig.pose.bones[chain['follow']].constraints.new('COPY_ROTATION')
            set_constraint(constraint, rig, chain['leg'], chain['follow_influence'])
        for region in plan['regions']:
            obj = region['obj']
            if region['role'] == 'head_cap':
                assign(obj, region['ids'], {'Head': 1.})
                continue
            for index in region['ids']:
                p = plan['points'][obj.name][index]
                t = (region['hi']-p[2])/(region['hi']-region['lo'])
                if region['role'] == 'hair':
                    values = _vertical_weights(region['names'][0], 'Head', t, smooth_transition=True)
                else:
                    angle = math.atan2(p[1]-region['center'][1], p[0]-region['center'][0]) % math.tau
                    sector = angle/math.tau*skirt_sectors
                    a = int(sector)
                    values = {}
                    for s, factor in ((a, 1-(sector-a)), ((a+1) % skirt_sectors, sector-a)):
                        for name, weight in _vertical_weights(region['names'][s], 'Hips', t).items():
                            values[name] = values.get(name, 0)+weight*factor
                assign(obj, [index], values)
        rig.data.pose_position = 'POSE'
        bpy.context.view_layer.update()
        metadata = _write_vrm(rig, meshes, plan, material_roles or {}, spring_center)
        if any(c['role']=='skirt' for c in plan['chains']):
            from avatar_contact_colliders import install_contact_colliders
            from avatar_skirt_binding import rebind_skirt_strips
            from avatar_directional_contacts import install_skirt_contact_rig
            metadata['contact_colliders']=install_contact_colliders(rig,meshes)
            metadata['strip_binding']=rebind_skirt_strips(rig,meshes)
            metadata['contact_rig']=install_skirt_contact_rig(rig,meshes)
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
                  deform_bones=len(plan['chains'])*segments,
                  terminal_bones=len(plan['chains']), rejected=plan['rejected'],
                  fixed_hair_roots=[c['names'][0] for c in plan['chains'] if c['role'] == 'hair'],
                  hair_attachments=[dict(root=c['names'][0], physics_start=c['names'][1],
                                         cap_position=list(c['line'][0]), fixed_end=list(c['line'][1]))
                                    for c in plan['chains'] if c['role'] == 'hair'],
                  leg_follow=[dict(bone=c['follow'], target=c['leg'], influence=c['follow_influence'])
                              for c in plan['chains'] if c['role'] == 'skirt'],
                  regions=[dict(object=r['obj'].name, role=r['role'], vertices=len(r['ids'])) for r in plan['regions']],
                  **metadata)
    rig['unimate_secondary_generator'] = 4
    from properties_hallway_rig import initialize
    from avatar_bone_collections import organize_bones
    initialize(rig)
    report['bone_collections'] = organize_bones(rig)
    return report


def _write_vrm(rig, meshes, plan, roles, spring_center=None):
    ext = rig.data.vrm_addon_extension
    ext.spec_version = '1.0'
    preserve_export_rest_frame(rig)
    for role, name in humanoid_roles().items():
        if name in rig.data.bones:
            key = re.sub(r'(?<!^)(?=[A-Z])', '_', role).lower()
            getattr(ext.vrm1.humanoid.human_bones, key).node.bone_name = name
    sb = ext.spring_bone1
    height = plan['height']
    for chain in plan['chains']:
        spring = sb.springs.add()
        spring.vrm_name = chain['names'][0].rsplit('_', 1)[0]
        # No implicit center: body/object translation should excite the springs.
        # Assign one only when the caller deliberately requests compensation.
        spring.center.bone_name = spring_center or ''
        hair = chain['role'] == 'hair'
        for i, name in enumerate(chain['names'][chain.get('physics_start', 0):]):
            joint = spring.joints.add()
            joint.node.bone_name = name
            joint.stiffness = (1.0 if hair else 1.6) * (1-.10*i)
            joint.drag_force = .4
            joint.gravity_power = .035 if hair else .025
            joint.gravity_dir = (0, 0, -1)
            # Bone-point collisions need a margin for the skinned cloth between
            # adjacent chains, which can lie inside their collision envelope.
            joint.hit_radius = height*(.004 if hair else .012)
    from avatar_colliders import rebuild_colliders
    metadata = rebuild_colliders(rig, meshes, roles)
    metadata.pop('height', None)
    return dict(**metadata, spring_names=[s.vrm_name for s in sb.springs if s.vrm_name.startswith(PREFIX)])
