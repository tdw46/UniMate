"""Experimental neighboring-chain containment using native VRM colliders.

This is a bounded-separation approximation, not a cloth tension solver. BVT
remains the only simulator. Do not install on a working rig without validation.
"""
import math
import bpy
from mathutils import Vector
from avatar_colliders import segment_distance
from avatar_vrm_colliders import add_group, call_operator, organize_colliders


def add_segment_network(rig, meshes, slack=0.05, couple=True):
    """Isolated experiment: unique terminal nodes give per-segment groups.

    No joint is shared between SpringChains. Each segment has a helper tip,
    sibling to the next deform joint. Cross-chain evaluation order remains a
    portability concern, so this is not a default generation path.
    """
    if not bpy.app.background:
        raise RuntimeError('Experimental segment networks are restricted to isolated background tests')
    from avatar_colliders import plan_colliders
    from avatar_vrm_colliders import add_capsule
    sb=rig.data.vrm_addon_extension.spring_bone1
    if any(b.name.startswith('Hallway_ContactTip_') for b in rig.data.bones):
        raise ValueError('Segment contact network already exists')
    field_names=('hit_radius','stiffness','drag_force','gravity_power','gravity_dir')
    originals=[]
    for s in sb.springs:
        if not s.vrm_name.startswith('Secondary_Skirt_'):continue
        originals.append(dict(name=s.vrm_name,center=s.center.bone_name,joints=[dict(
            bone=j.node.bone_name,**{k:list(getattr(j,k)) if k=='gravity_dir' else getattr(j,k)
                                   for k in field_names}) for j in s.joints]))
    plan=plan_colliders(rig,meshes)
    body=[s for s in plan['collider_details'] if s['role']=='skirt']
    margin=plan['margin']
    bpy.context.view_layer.objects.active=rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    for chain in originals:
        for i,(head,tail) in enumerate(zip(chain['joints'],chain['joints'][1:])):
            tip=rig.data.edit_bones.new('Hallway_ContactTip_'+chain['name']+'_'+str(i))
            parent=rig.data.edit_bones[head['bone']]
            tip.head=rig.data.edit_bones[tail['bone']].head.copy()
            tip.tail=tip.head+(tip.head-parent.head).normalized()*.003
            tip.parent=parent;tip.use_connect=False;tip.use_deform=True
            head['tip']=tip.name
    bpy.ops.object.mode_set(mode='OBJECT')
    for i in reversed(range(len(sb.springs))):
        if sb.springs[i].vrm_name.startswith('Secondary_Skirt_'):sb.springs.remove(i)
    families={}
    for chain in originals:families.setdefault(chain['name'].rsplit('_',1)[0],[]).append(chain)
    neighbors={}
    for family in families.values():
        centers={c['name']:rig.data.bones[c['joints'][0]['bone']].head_local.copy() for c in family}
        center=sum(centers.values(),Vector())/len(centers)
        family.sort(key=lambda c:math.atan2(centers[c['name']].y-center.y,centers[c['name']].x-center.x))
        for i,chain in enumerate(family):neighbors[chain['name']]=(family[i-1],family[(i+1)%len(family)])
    for chain in originals:
        for index,head in enumerate(chain['joints'][:-1]):
            spring=sb.springs.add();spring.vrm_name=chain['name']+'_Contact_'+str(index)
            spring.center.bone_name=chain['center']
            for bone in (head['bone'],head['tip']):
                joint=spring.joints.add();joint.node.bone_name=bone
                for key in field_names:setattr(joint,key,head[key])
            endpoint=rig.data.bones[head['tip']].head_local
            group=add_group(rig,'Hallway_Segment_'+spring.vrm_name)
            spring.collider_groups.add().collider_group_uuid=group.uuid
            if couple:
                for neighbor in neighbors[chain['name']]:
                    other=neighbor['joints'][index]
                    if neighbor['name'].rsplit('_',1)[0]!=chain['name'].rsplit('_',1)[0]:
                        continue
                    other_tip=rig.data.bones[other['tip']].head_local
                    c=sb.add_collider(bpy.context,rig)
                    c.node.bone_name=other['bone'];c.ui_collider_type='uiColliderTypeSphereInside'
                    c.reset_bpy_object(bpy.context,rig);bpy.context.view_layer.update()
                    shape=c.extensions.vrmc_spring_bone_extended_collider.shape.sphere
                    shape.offset=rig.data.bones[other['bone']].matrix_local.inverted()@other_tip
                    bpy.context.view_layer.update()
                    shape.radius=(endpoint-other_tip).length*(1+slack)+head['hit_radius']+margin
                    bpy.context.view_layer.update()
                    c.bpy_object['unimate_generated_collider']=True
                    group.colliders.add().collider_uuid=c.uuid
            for spec in body:
                bone=rig.data.bones[spec['bone']]
                a,b=Vector(spec['offset']),Vector(spec['tail'])
                radius=min(spec['body_radius'],segment_distance(endpoint,bone.matrix_local@a,bone.matrix_local@b)-head['hit_radius']-margin)
                if radius<=0:continue
                c=add_capsule(rig,dict(bone=bone.name,offset=a,tail=b,radius=radius))
                group.colliders.add().collider_uuid=c.uuid
    organize_colliders(rig)
    return dict(segments=sum(len(c['joints'])-1 for c in originals),coupled=couple)


def add_neighbor_cages(rig, slack=0.05):
    """One inside capsule per directed neighboring pair, scoped to its chain.

    Plan in REST coordinates. Each capsule follows the neighboring chain's
    root segment and covers that chain's full rest length. Body capsules stay
    last in each collision list so they take precedence over coupling.
    """
    if not bpy.app.background:
        raise RuntimeError('Experimental neighbor cages are restricted to isolated background tests')
    if rig.data.vrm_addon_extension.spec_version != '1.0':
        raise ValueError('Neighbor cages require VRM 1 extended colliders')
    sb = rig.data.vrm_addon_extension.spring_bone1
    if any(g.vrm_name.startswith('Hallway_Tension_') for g in sb.collider_groups):
        raise ValueError('Neighbor cages already exist; refusing duplicates')
    families = {}
    for spring in sb.springs:
        if spring.vrm_name.startswith('Secondary_Skirt_') and len(spring.joints) > 1:
            names = [j.node.bone_name for j in spring.joints]
            families.setdefault(spring.vrm_name.rsplit('_', 1)[0], []).append({
                'name': spring.vrm_name, 'bones': names,
                'points': [rig.data.bones[n].head_local.copy() for n in names],
                'hits': [j.hit_radius for j in spring.joints[:-1]],
                'groups': [r.collider_group_uuid for r in spring.collider_groups],
            })
    plan = []
    for chains in families.values():
        if len(chains) < 3:
            continue
        center = sum((c['points'][0] for c in chains), Vector()) / len(chains)
        chains.sort(key=lambda c: math.atan2(c['points'][0].y-center.y,
                                            c['points'][0].x-center.x))
        for i, chain in enumerate(chains):
            for neighbor in (chains[i-1], chains[(i+1) % len(chains)]):
                a, b = neighbor['points'][0], neighbor['points'][-1]
                radius = max(segment_distance(p, a, b)+hit
                             for p, hit in zip(chain['points'][1:], chain['hits']))
                radius = radius * (1+slack) + (b-a).length*.005
                inverse = rig.data.bones[neighbor['bones'][0]].matrix_local.inverted()
                plan.append(dict(chain=chain['name'], bone=neighbor['bones'][0],
                                 offset=inverse@a, tail=inverse@b, radius=radius))
    for spec in plan:
        if callable(getattr(sb, 'add_collider', None)):
            c = sb.add_collider(bpy.context, rig)
        else:
            call_operator('add_spring_bone1_collider', rig)
            c = sb.colliders[-1]
        identifiers = {v.identifier for v in c.bl_rna.properties['ui_collider_type'].enum_items}
        if 'uiColliderTypeCapsuleInside' not in identifiers:
            raise RuntimeError('Installed VRM add-on lacks inside capsules')
        c.node.bone_name = spec['bone']
        c.ui_collider_type = 'uiColliderTypeCapsuleInside'
        c.reset_bpy_object(bpy.context, rig)
        bpy.context.view_layer.update()
        shape = c.extensions.vrmc_spring_bone_extended_collider.shape.capsule
        for key in ('offset', 'tail', 'radius'):
            setattr(shape, key, spec[key])
            bpy.context.view_layer.update()
        assert shape.inside
        c.bpy_object['unimate_generated_collider'] = True
        group = add_group(rig, 'Hallway_Tension_'+spec['chain'])
        group.colliders.add().collider_uuid = c.uuid
        spring = next(s for s in sb.springs if s.vrm_name == spec['chain'])
        spring.collider_groups.add().collider_group_uuid = group.uuid
        spring.collider_groups.move(len(spring.collider_groups)-1, 0)
    organize_colliders(rig)
    return {'directed_links': len(plan), 'slack': slack,
            'method': 'whole-chain inside-capsule containment',
            'equal_and_opposite_tension': False}
