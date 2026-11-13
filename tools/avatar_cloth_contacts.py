"""Portable rest-clear contacts for overlapping generated skirt ribbons.

A return guard prevents the lower hem folding through its upper section.
Sector guards preserve the order of neighboring cloth strips. No imported rig,
mesh mutation, custom simulator behavior, or extra runtime callbacks are used.
"""
import math
from mathutils import Vector
from avatar_colliders import segment_distance
from avatar_vrm_colliders import add_capsule,add_group,call_operator,organize_colliders,cleanup_orphan_displays

PREFIX='Secondary_ClothOverlayContact_'


def plan_overlay_contacts(rig, report):
    sb=rig.data.vrm_addon_extension.spring_bone1
    families={r['carrier'].rsplit('_',1)[0] for r in report.get('components',[]) if 'Secondary_Skirt_' in r['carrier']}
    if not families:return []
    height=report['height'];margin=max(height*.002,1e-6)
    hips=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones.hips.node.bone_name
    hip=rig.data.bones[hips]
    def family(name):return name.removesuffix('_Lower').rsplit('_',1)[0]
    springs=[s for s in sb.springs if s.joints and family(s.vrm_name) in families]
    upper=[s for s in springs if not rig.data.bones[s.joints[0].node.bone_name].get('hallway_dress_lower')]
    plans=[]
    for spring in springs:
        root=rig.data.bones[spring.joints[0].node.bone_name]
        samples=[(rig.data.bones[t.node.bone_name].head_local.copy(),j.hit_radius) for j,t in zip(spring.joints,spring.joints[1:])]
        if not samples:continue
        specs=[]
        def fit(bone,shape,radius):
            def clearance(bound):
                a,b=shape(bound)
                return min(segment_distance(p,a,b)-r-radius for p,r in samples)
            lo=-height;hi=0.
            if clearance(lo)<margin:return
            for _ in range(32):
                mid=(lo+hi)*.5
                if clearance(mid)>=margin:lo=mid
                else:hi=mid
            a,b=shape(lo);inverse=bone.matrix_local.inverted()
            specs.append(dict(bone=bone.name,offset=list(inverse@a),tail=list(inverse@b),radius=radius,
                              minimum_rest_clearance=clearance(lo)))
        if root.get('hallway_dress_lower'):
            parent=next(s for s in upper if s.vrm_name==root['hallway_dress_upper_spring'])
            bone=rig.data.bones[parent.joints[-2].node.bone_name]
            anchor=root.head_local.copy();normal=(bone.tail_local-bone.head_local).normalized()
            reach=sum(rig.data.bones[j.node.bone_name].length for j in spring.joints[:-1])+height*.04
            tangent=normal.cross(Vector((1,0,0)))
            if tangent.length<1e-6:tangent=normal.cross(Vector((0,1,0)))
            tangent.normalize();across=normal.cross(tangent).normalized()
            for shift in (-reach*.5,0.,reach*.5):
                def shape(bound):
                    center=anchor+normal*(bound-reach)+across*shift
                    return center-tangent*reach,center+tangent*reach
                fit(bone,shape,reach)
        base=spring.vrm_name.removesuffix('_Lower')
        owner=next(s for s in upper if s.vrm_name==base)
        radial=rig.data.bones[owner.joints[0].node.bone_name].head_local-hip.head_local;radial.z=0.
        if radial.length>height*1e-6:
            angle=math.atan2(radial.y,radial.x)
            peers=[s for s in upper if family(s.vrm_name)==family(base) and s!=owner]
            angles=[]
            for peer in peers:
                v=rig.data.bones[peer.joints[0].node.bone_name].head_local-hip.head_local
                angles.append(((math.atan2(v.y,v.x)-angle+math.pi)%(2*math.pi))-math.pi)
            neighbors=[max((a for a in angles if a<0),default=None),min((a for a in angles if a>0),default=None)]
            for neighbor in neighbors:
                if neighbor is None:continue
                theta=angle+neighbor*.5;normal=Vector((-math.sin(theta),math.cos(theta),0.))
                if normal.dot(radial)<0:normal.negate()
                radius=height*.6;start=hip.head_local.copy();start.z=0.;end=start+Vector((0,0,height*.7))
                def shape(bound):return start+normal*(bound-radius),end+normal*(bound-radius)
                fit(hip,shape,radius)
        if specs:plans.append(dict(name=PREFIX+spring.vrm_name,spring=spring.vrm_name,colliders=specs))
    return plans


def install_overlay_contacts(rig,report):
    plans=plan_overlay_contacts(rig,report)
    sb=rig.data.vrm_addon_extension.spring_bone1
    owned={g.uuid for g in sb.collider_groups if g.vrm_name.startswith(PREFIX)}
    targets={p['spring'] for p in plans}
    if any(r.collider_group_uuid in owned for s in sb.springs if s.vrm_name not in targets for r in s.collider_groups):
        raise ValueError('Generated overlay contacts have foreign spring owners')
    ids={r.collider_uuid for g in sb.collider_groups if g.uuid in owned for r in g.colliders}
    foreign={r.collider_uuid for g in sb.collider_groups if g.uuid not in owned for r in g.colliders}
    for i in reversed(range(len(sb.collider_groups))):
        if sb.collider_groups[i].uuid in owned:call_operator('remove_spring_bone1_collider_group',rig,collider_group_index=i)
    for i in reversed(range(len(sb.colliders))):
        if sb.colliders[i].uuid in ids-foreign:call_operator('remove_spring_bone1_collider',rig,collider_index=i)
    lookup={s.vrm_name:s for s in sb.springs};unique={}
    for plan in plans:
        group=add_group(rig,plan['name'])
        for spec in plan['colliders']:
            key=(spec['bone'],*spec['offset'],*spec['tail'],spec['radius'])
            if key not in unique:
                c=add_capsule(rig,spec,organize=False);c.bpy_object['hallway_cloth_overlay_contact']=True;unique[key]=c.uuid
            group.colliders.add().collider_uuid=unique[key]
        lookup[plan['spring']].collider_groups.add().collider_group_uuid=group.uuid
    if plans:organize_colliders(rig);cleanup_orphan_displays(rig)
    return dict(groups=len(plans),capsules=len(unique),portable=True,
                minimum_rest_clearance=min((s['minimum_rest_clearance'] for p in plans for s in p['colliders']),default=0.))
