"""Per-chain, rest-clear head/torso/arm contacts using portable VRM capsules.

Like skirt contacts, each chain gets independent coverage; a close fringe cannot
shrink the torso protection of a distant ponytail. Base geometry and newly bound
weights supply all measurements. No mesh, rest bone, weight or spring parameter
is changed. Artist groups remain owned by their authors.
"""
import numpy as np
from mathutils import Vector
from avatar_apparel_weights import weights
from avatar_contact_colliders import replace_contact_colliders

PREFIX = 'Secondary_Hair_'


def plan_hair_contacts(rig, meshes, material_roles=None):
    sb = rig.data.vrm_addon_extension.spring_bone1
    springs = [s for s in sb.springs if s.vrm_name.startswith(PREFIX) and len(s.joints)>1]
    if not springs:
        return dict(colliders=[], groups=[], minimum_rest_clearance=None)
    hum = rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    roles = ('head','neck','chest','upper_chest','spine','hips',
             'left_shoulder','right_shoulder','left_upper_arm','right_upper_arm',
             'left_lower_arm','right_lower_arm','left_hand','right_hand')
    names = list(dict.fromkeys(getattr(hum, r).node.bone_name for r in roles
                             if getattr(hum, r, None) and getattr(hum, r).node.bone_name in rig.data.bones))
    records = []
    body_records = []
    from avatar_material_roles import semantic_label
    from avatar_hair_regions import hair_images, extend_hair_region
    roles=material_roles or {}
    images=hair_images(meshes,roles)
    for obj in meshes:
        transform = rig.matrix_world.inverted_safe()@obj.matrix_world
        hair_ids={i for f in obj.data.polygons
                  if f.material_index<len(obj.data.materials) and obj.data.materials[f.material_index]
                  and 'hair' in roles.get(obj.data.materials[f.material_index].name,semantic_label(obj.data.materials[f.material_index].name))
                  for i in f.vertices}
        points=np.array([tuple(transform@v.co) for v in obj.data.vertices])
        hair_ids=extend_hair_region(obj,rig,points,hair_ids,roles,images)
        for v in obj.data.vertices:
            record=(transform@v.co,weights(obj,v.index))
            records.append(record)
            if v.index not in hair_ids:body_records.append(record)
    height = max(p.z for p,w in records)-min(p.z for p,w in records)
    margin = height*.0015
    body = []
    for name in names:
        bone = rig.data.bones[name]; inverse=bone.matrix_local.inverted_safe()
        pts=np.array([tuple(inverse@p) for p,w in body_records if w.get(name,0)>.25
                      and not any(n.startswith('Secondary_') and v>1e-6 for n,v in w.items())])
        if len(pts)<4:
            continue  # No anatomical surface evidence: do not invent a volume.
        # Longitudinal sections follow the actual weighted surface, including
        # clothing. Independent rounded sections fill head/torso/arm coverage.
        low,high=np.quantile(pts[:,1],(.05,.95))
        sections=3 if name in [getattr(hum,r).node.bone_name for r in ('head','chest','spine')] else 2
        for index in range(sections):
            lo=low+(high-low)*index/sections;hi=low+(high-low)*(index+1)/sections
            band=pts[(pts[:,1]>=lo)&(pts[:,1]<=hi)]
            if len(band)<4:continue
            center=np.median(band[:,(0,2)],axis=0)
            radius=float(np.quantile(np.linalg.norm(band[:,(0,2)]-center,axis=1),.85))
            radius=min(radius,max(bone.length*.65,height*.008))
            if radius<=height*.001:continue
            a=Vector((center[0],lo,center[1]));b=Vector((center[0],hi,center[1]))
            body.append(dict(bone=name,offset=list(a),tail=list(b),radius=radius))
    specs=[];lookup={};groups=[];minimum=float('inf');omitted=0
    for spring in springs:
        joint_names={j.node.bone_name for j in spring.joints}
        samples=[(rig.data.bones[t.node.bone_name].head_local.copy(),h.hit_radius)
                 for h,t in zip(spring.joints,spring.joints[1:])]
        # Surface clearance prevents a broad lock's edges from starting inside
        # a collider even when its center guide is clear.
        samples += [(p,0.) for p,w in records if any(n in joint_names and v>.05 for n,v in w.items())]
        points=np.array([tuple(p) for p,r in samples]);hits=np.array([r for p,r in samples])
        refs=[]
        for spec in body:
            bone=rig.data.bones[spec['bone']]
            a=np.array(bone.matrix_local@Vector(spec['offset']));b=np.array(bone.matrix_local@Vector(spec['tail']))
            delta=b-a;denom=float(delta@delta)
            t=np.clip((points-a)@delta/denom,0,1) if denom>1e-16 else np.zeros(len(points))
            distances=np.linalg.norm(points-a-t[:,None]*delta,axis=1)-hits
            radius=min(spec['radius'],float(distances.min())-margin)
            if radius<=height*.001:
                omitted+=1;continue
            minimum=min(minimum,float(distances.min())-radius)
            fitted=dict(spec,radius=radius,contact=True)
            key=(spec['bone'],*spec['offset'],*spec['tail'],radius)
            if key not in lookup:
                lookup[key]=len(specs);specs.append(fitted)
            refs.append(lookup[key])
        # A broad outward face per nearby body bone supplements the rounded
        # sections, as for skirt/leg guards. Its back stays inside the body;
        # fit the front against this chain only so it cannot push hair at rest.
        nearest={}
        for spec in body:
            bone=rig.data.bones[spec['bone']]
            a=np.array(bone.matrix_local@Vector(spec['offset']));b=np.array(bone.matrix_local@Vector(spec['tail']))
            axis=b-a;denom=float(axis@axis)
            t=np.clip((points-a)@axis/denom,0,1) if denom>1e-16 else np.zeros(len(points))
            residual=points-a-t[:,None]*axis;dist=np.linalg.norm(residual,axis=1)
            i=int(np.argmin(dist));distance=float(dist[i])
            if distance>spec['radius']*3 or distance<1e-8:continue
            if spec['bone'] not in nearest or distance<nearest[spec['bone']][0]:
                nearest[spec['bone']]=(distance,spec,a,b,residual[i]/distance)
        for distance,spec,a,b,normal in nearest.values():
            R=spec['radius']*2
            axis=b-a;denom=float(axis@axis)
            def clearance(bound):
                start=a+normal*(bound-R)
                t=np.clip((points-start)@axis/denom,0,1) if denom>1e-16 else np.zeros(len(points))
                return float((np.linalg.norm(points-start-t[:,None]*axis,axis=1)-hits-R).min())
            low=-R*4;high=spec['radius']
            if clearance(low)<margin:continue
            for _ in range(32):
                mid=(low+high)*.5
                if clearance(mid)>=margin:low=mid
                else:high=mid
            shift=normal*(low-R);inverse=rig.data.bones[spec['bone']].matrix_local.inverted_safe()
            start=list(inverse@Vector(a+shift));end=list(inverse@Vector(b+shift))
            key=(spec['bone'],*start,*end,R)
            if key not in lookup:
                lookup[key]=len(specs);specs.append(dict(bone=spec['bone'],offset=start,tail=end,radius=R,contact=True,directional=True))
            # Competing round sections can undo this outward projection after
            # the fixed-length constraint, just as with tight skirt contacts.
            # Head and neck keep rounded coverage around all sides; torso and
            # limb guards replace their competing local sections.
            if spec['bone'] not in (hum.head.node.bone_name,hum.neck.node.bone_name):
                refs=[i for i in refs if specs[i]['bone']!=spec['bone']]
            refs.append(lookup[key]);minimum=min(minimum,clearance(low))
        groups.append(dict(name='Secondary_HairContact_'+spring.vrm_name,
                           springs=[spring.vrm_name],colliders=refs))
    used=sorted({i for group in groups for i in group['colliders']})
    remap={old:new for new,old in enumerate(used)}
    specs=[specs[i] for i in used]
    for group in groups:group['colliders']=[remap[i] for i in group['colliders']]
    return dict(colliders=specs,groups=groups,minimum_rest_clearance=minimum,
                margin=margin,omitted=omitted,body_sections=len(body))


def install_hair_contacts(rig, meshes, material_roles=None):
    from avatar_skirt_fit_io import rest_edit
    with rest_edit(rig):
        plan=plan_hair_contacts(rig,meshes,material_roles)
        if not plan['groups']:
            return dict(colliders=0,groups=0)
        replace_contact_colliders(rig,plan,role='hair')
    return dict(colliders=len(plan['colliders']),groups=len(plan['groups']),
                references=sum(len(g['colliders']) for g in plan['groups']),
                minimum_rest_clearance=plan['minimum_rest_clearance'],
                body_sections=plan['body_sections'],omitted=plan['omitted'])


def fit_generated_hair_radii(rig, height):
    """Protect moving strand surfaces without changing their rest placement.

    Generated point springs need more clearance than their center guide alone.
    Grow their hit radius by at most 2% of character height, capped by the exact
    distance to every assigned rest-pose collider. Keep a 0.15% rest gap. This is
    generation-only: authored springs outside this module's prefix are untouched.
    The saved base makes repeated generation/validation calls idempotent.
    """
    from avatar_colliders import segment_distance
    from avatar_skirt_fit_io import rest_edit
    if not np.isfinite(height) or height <= 0:
        return dict(updated=0, minimum_rest_clearance=None)
    sb=rig.data.vrm_addon_extension.spring_bone1
    colliders={c.uuid:c for c in sb.colliders if c.bpy_object}
    groups={g.uuid:[colliders[x.collider_uuid] for x in g.colliders
                    if x.collider_uuid in colliders] for g in sb.collider_groups}
    updated=0;minimum=float('inf')
    with rest_edit(rig):
        for spring in sb.springs:
            if not spring.vrm_name.startswith(PREFIX):continue
            contacts=[c for ref in spring.collider_groups for c in groups.get(ref.collider_group_uuid,())]
            for head,tail in zip(spring.joints,spring.joints[1:]):
                bone=rig.data.bones.get(tail.node.bone_name)
                if bone is None or not contacts:continue
                clearances=[]
                for collider in contacts:
                    owner=rig.data.bones.get(collider.node.bone_name)
                    extended=getattr(getattr(collider,'extensions',None),'vrmc_spring_bone_extended_collider',None)
                    if owner is None or (extended and extended.enabled):
                        clearances=[];break
                    if collider.shape_type==collider.SHAPE_TYPE_CAPSULE:
                        shape=collider.shape.capsule
                        end=owner.matrix_local@Vector(shape.tail)
                    else:
                        shape=collider.shape.sphere
                        end=owner.matrix_local@Vector(shape.offset)
                    start=owner.matrix_local@Vector(shape.offset)
                    clearances.append(segment_distance(bone.head_local,start,end)-shape.radius)
                if not clearances:continue
                base=float(head.get('generated_hair_base_radius',head.hit_radius))
                head['generated_hair_base_radius']=base
                radius=max(base,min(base+height*.02,min(clearances)-height*.0015))
                if abs(head.hit_radius-radius)>1e-7:updated+=1
                head.hit_radius=radius
                minimum=min(minimum,min(clearances)-radius)
    return dict(updated=updated,minimum_rest_clearance=minimum if np.isfinite(minimum) else None)
