"""Surface-fitted upper-skirt pivots and locally blended spring weights."""
import bpy
from mathutils import Vector

PREFIX = 'Secondary_SkirtSupport_'
GROUP = 'Skirt Support'
CONSTRAINT = 'Hallway upper skirt support'
from avatar_rig_defaults import FOLLOW
DEFAULT_INFLUENCE = FOLLOW['Skirt Support']


def support_name(root):
    return PREFIX + root.removeprefix('Secondary_Skirt_').rsplit('_', 1)[0]


def _surface(rig, meshes, family):
    from mathutils.bvhtree import BVHTree
    from avatar_apparel_weights import weights
    points, triangles = [], []
    for obj in meshes:
        owned = {v.index for v in obj.data.vertices if any(
            n.startswith(family + '_') or (n in rig.data.bones and
            str(rig.data.bones[n].get('hallway_support_chain', '')).startswith(family + '_'))
            for n in weights(obj, v.index))}
        slots = {p.material_index for p in obj.data.polygons if any(i in owned for i in p.vertices)}
        obj.data.calc_loop_triangles()
        matrix = rig.matrix_world.inverted() @ obj.matrix_world
        offset = len(points)
        points.extend(matrix @ v.co for v in obj.data.vertices)
        triangles.extend(tuple(offset + i for i in t.vertices) for t in obj.data.loop_triangles if t.material_index in slots)
    return BVHTree.FromPolygons(points, triangles, all_triangles=True) if triangles else None


def ensure_support(rig, meshes, families):
    """One pivot on the upper garment surface for every existing skirt chain."""
    from avatar_pelvis_binding import landmarks
    from avatar_skirt_follow import set_rest_frame, set_constraint, preserve_export_rest_frame
    from avatar_skirt_fit_io import rest_edit
    from avatar_bone_collections import organize_bones
    hips, _, _, thighs, *_ = landmarks(rig)
    lateral = (thighs[0].head_local - thighs[1].head_local).normalized()
    mid = (thighs[0].head_local + thighs[1].head_local) * .5
    targets = {}
    for family, chains in families.items():
        surface = _surface(rig, meshes, family)
        center = sum((points[0] for _, points in chains), Vector()) / len(chains)
        for names, points in chains:
            root = rig.data.bones[names[0]]
            leg = thighs[0 if (points[0]-mid).dot(lateral) > 0 else 1]
            top = float(root.get('hallway_garment_top', points[0].z))
            bottom = float(root.get('hallway_garment_bottom', points[-1].z))
            height = top - bottom
            # Keep the pivot just below the garment attachment. A lower pivot
            # leaves too little room for the upper spring to clear a raised thigh.
            z = min(top-height*.003, max(top-height*.005, leg.head_local.z+leg.length*.08))
            guide = points[0] + (points[0]-points[1]) * ((z-points[0].z)/(points[0].z-points[1].z))
            origin = Vector((center.x, center.y, z))
            direction = guide-origin
            direction.z = 0
            if direction.length < 1e-8:
                raise ValueError('Degenerate skirt support direction')
            direction.normalize()
            hits = []
            if surface:
                start = origin.copy()
                for _ in range(32):
                    hit, _, _, _ = surface.ray_cast(start, direction, height*4)
                    if hit is None:
                        break
                    hits.append(hit.copy()); start = hit + direction * max(height*1e-6, 1e-7)
            if hits:
                pivot = min(hits, key=lambda p: (p-guide).length_squared)
            elif surface:
                pivot, _, _, _ = surface.find_nearest(guide)
                if pivot is None or abs(pivot.z-z) > height*.04:
                    raise ValueError('No upper garment surface for ' + names[0])
            else:
                raise ValueError('No garment surface for ' + names[0])
            if pivot.z <= leg.head_local.z:
                raise ValueError('Skirt support must be above the leg socket')
            targets[support_name(names[0])] = dict(target=leg.name, pivot=pivot,
                root=names[0], physics=bool(rig.data.bones.get(support_name(names[0])) and rig.data.bones[support_name(names[0])].get('hallway_hip_physics')), length=min(leg.length*.2, (points[0]-points[1]).length*.3))
    settings = getattr(rig, 'hallway_rig', None)
    group = settings.follow_groups.get(GROUP) if settings else None
    influence = group.influence if group else DEFAULT_INFLUENCE
    selection = {p.name: (p.select if hasattr(p, 'select') else p.bone.select) for p in rig.pose.bones}
    active = rig.data.bones.active.name if rig.data.bones.active else None
    stale = {b.name for b in rig.data.bones if b.get('hallway_skirt_support') and b.name not in targets}
    for name in set(targets) | stale:
        bone = rig.data.bones.get(name)
        if bone and not bone.get('hallway_skirt_support'):
            raise ValueError('Unowned upper-skirt support bone: ' + name)
        if bone:
            constraints = rig.pose.bones[name].constraints
            if len(constraints) > 1 or any(c.name != CONSTRAINT or c.type != 'COPY_ROTATION' for c in constraints):
                raise ValueError('Unexpected constraint on ' + name)
    # Migration is restricted to our unused support helpers. Preserve a user's
    # added dependencies rather than silently breaking their rig.
    sb = rig.data.vrm_addon_extension.spring_bone1
    refs = {j.node.bone_name for s in sb.springs for j in s.joints} | {c.node.bone_name for c in sb.colliders}
    refs |= {s.center.bone_name for s in sb.springs}
    refs |= {c.subtarget for p in rig.pose.bones for c in p.constraints if getattr(c,'target',None)==rig and hasattr(c,'subtarget')}
    if stale & refs or any(rig.data.bones[n].children for n in stale):
        raise ValueError('Obsolete skirt support has external dependents')
    with rest_edit(rig):
        try:
            # Reassign obsolete owned groups before deletion, then the binder
            # reconstructs their garment domain from the existing spring rows.
            from avatar_apparel_weights import weights, assign
            for obj in meshes:
                for v in obj.data.vertices:
                    row = weights(obj, v.index)
                    amount = sum(row.pop(n, 0.) for n in stale)
                    if amount:
                        row[hips.name] = row.get(hips.name, 0.) + amount
                        assign(obj, [v.index], row)
                for name in stale:
                    vg = obj.vertex_groups.get(name)
                    if vg: obj.vertex_groups.remove(vg)
            rig.select_set(True)
            bpy.ops.object.mode_set(mode='EDIT')
            for name in stale:
                rig.data.edit_bones.remove(rig.data.edit_bones[name])
            for name, spec in targets.items():
                bone = rig.data.edit_bones.get(name) or rig.data.edit_bones.new(name)
                leg = rig.data.edit_bones[spec['target']]
                set_rest_frame(bone, leg)
                bone.head = spec['pivot']
                bone.tail = bone.head + (leg.tail-leg.head).normalized() * spec['length']
                if spec['physics']:
                    bone.parent=rig.data.edit_bones[hips.name]
                    bone.tail=rig.data.edit_bones[name+'_Tip'].head.copy()
                bone['hallway_skirt_support'] = 2
                bone['hallway_support_chain'] = spec['root']
                if not spec['physics']:
                    bone['hallway_constraint_group'] = GROUP
                    bone['hallway_follow_constraint'] = CONSTRAINT
            bpy.ops.object.mode_set(mode='OBJECT')
            for name, spec in targets.items():
                if spec['physics']:continue
                pb = rig.pose.bones[name]
                c = pb.constraints.get(CONSTRAINT) or pb.constraints.new('COPY_ROTATION')
                set_constraint(c, rig, spec['target'], influence); c.name = CONSTRAINT
            preserve_export_rest_frame(rig)
            organize_bones(rig)
        finally:
            if rig.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
            for pb in rig.pose.bones:
                if hasattr(pb, 'select'): pb.select = selection.get(pb.name, False)
                else: pb.bone.select = selection.get(pb.name, False)
            rig.data.bones.active = rig.data.bones.get(active) if active else None
    if settings and group is None and any(not spec['physics'] for spec in targets.values()):
        group = settings.follow_groups.add(); group.name = GROUP; group.influence = influence
    return dict(bones=list(targets), removed=sorted(stale), influence=influence,
                pivots={n:list(s['pivot']) for n,s in targets.items()})


def support_attachment(rig, body, spring_mass, bones, spring_weights, edges):
    """Broad support blend, locally limited to four weights with smooth masks.

    Neighboring supports start with the spring chain shares. Twenty passes
    smooth inside that original pair, then smooth removal masks enforce the
    four-weight limit without opening weights to any third chain.
    """
    import numpy as np
    from avatar_weight_smoothing import smooth_limited_weights
    helpers = {b.get('hallway_support_chain'):b.name for b in rig.data.bones if b.get('hallway_skirt_support') == 2}
    by_chain = {root.rsplit('_',1)[0]:name for root,name in helpers.items()}
    def ease(x):
        x=max(0.,min(1.,x)); return x*x*(3.-2.*x)
    candidates=[]
    for row, mass, spring in zip(body, spring_mass, spring_weights):
        values=dict(row)
        if 0 < mass < 1:
            strength=.9*ease(mass/.35)
            shares={}
            for n,w in zip(bones,spring):
                if w<=1e-10:continue
                helper=by_chain.get(n.rsplit('_',1)[0])
                if helper is None:raise ValueError('Missing surface support for '+n)
                shares[helper]=shares.get(helper,0.)+float(w)
            total=sum(shares.values())
            if len(shares)>2 or total<=0:raise ValueError('Invalid support chain patch')
            amount=sum(row.values())*strength
            values={n:v*(1.-strength) for n,v in row.items()}
            values.update({n:amount*w/total for n,w in shares.items()})
        values.update({n:float(w) for n,w in zip(bones,spring) if w>0})
        candidates.append(values)
    names=sorted({n for row in candidates for n in row})
    w=np.array([[row.get(n,0.) for n in names] for row in candidates])
    limited=smooth_limited_weights(w,edges,np.full(len(w),4),
        (spring_mass>0)&(spring_mass<1),iterations=20,factor=.2,mask_iterations=20,allowed=w>0)
    lookup={n:i for i,n in enumerate(names)}
    body_out=[{n:float(v) for n,v in zip(names,row) if n not in bones and v>1e-10} for row in limited]
    result=np.array([[row[lookup[n]] if n in lookup else 0. for n in bones] for row in limited])
    return body_out,result


def blend_hip_handoff(rig, obj, indices, body, bones, spring_weights, edges):
    """Spread the upper spring bend into its own first skirt segment.

    Crossfade along that segment's rest height, ending at its tail. Only the
    already assigned angular chain pair participates; fixed bodice, knee and
    lower skirt weights remain pinned. This operates on freshly generated
    weights, so regenerating does not repeatedly accumulate the crossfade.
    """
    import numpy as np
    from avatar_weight_smoothing import smooth_limited_weights
    roots = {b['hallway_support_chain']: b for b in rig.data.bones
             if b.get('hallway_skirt_support') and b.get('hallway_support_chain')}
    matrix = rig.matrix_world.inverted() @ obj.matrix_world
    candidates, changed = [], []
    for index, row, spring in zip(indices, body, spring_weights):
        values = dict(row)
        values.update({n: float(w) for n, w in zip(bones, spring) if w > 0})
        original = dict(values)
        z = (matrix @ obj.data.vertices[index].co).z
        for name, helper in roots.items():
            if name not in values:
                continue
            root = rig.data.bones[name]
            span = helper.head_local.z - root.tail_local.z
            if span <= 1e-8:
                continue
            t = max(0., min(1., (z-root.tail_local.z)/span))
            blend = .6*t*t*(3.-2.*t)
            if blend > 1e-6:
                values[helper.name] = values.get(helper.name, 0.) + values[name]*blend
                values[name] *= 1.-blend
        changed.append(values != original)
        candidates.append(values)
    active = [i for i, value in enumerate(changed) if value]
    if not active:
        return body, spring_weights
    lookup = {i:j for j,i in enumerate(active)}
    local_edges = [(lookup[a],lookup[b]) for a,b in edges if a in lookup and b in lookup]
    names = sorted({n for i in active for n in candidates[i]})
    weights = np.array([[candidates[i].get(n,0.) for n in names] for i in active])
    limited = smooth_limited_weights(weights, local_edges, np.full(len(active),4),
        np.ones(len(active),dtype=bool), iterations=20, factor=.2,
        mask_iterations=20, allowed=weights>0)
    for i,row in zip(active,limited):
        candidates[i] = {n:float(w) for n,w in zip(names,row) if w>1e-10}
    body_out=[{n:w for n,w in row.items() if n not in bones} for row in candidates]
    spring_out=np.array([[row.get(n,0.) for n in bones] for row in candidates])
    from avatar_hip_weight_interpolation import average_helper_handoff
    return average_helper_handoff(rig,body_out,bones,spring_out,edges)
