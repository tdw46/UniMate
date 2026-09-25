"""Directional skirt contact guards made from portable VRM capsules.

Finite, offset capsules approximate an outward support surface for one chain.
Their back sides stay inside the leg; broad fronts resist slipping around small
centerline colliders. Guards follow their chain's own leg or its pelvis support,
never the opposite moving thigh. Every guard is fitted against its spring endpoints
at rest. No mesh coordinates or skin weights are written here.
"""
import math
from mathutils import Vector,Quaternion
from avatar_colliders import plan_colliders,segment_distance
from avatar_contact_colliders import snapshot_contact_colliders,replace_contact_colliders


def skirt_owner_leg(rig,spring,thighs):
    """Use the chain's authored follow target; never switch ownership mid-chain."""
    bone=rig.data.bones[spring.joints[0].node.bone_name]
    root=bone
    while bone:
        for c in rig.pose.bones[bone.name].constraints:
            if c.type=='COPY_ROTATION' and c.target==rig and c.subtarget in thighs:
                return c.subtarget
        root=bone if bone.name.startswith('Secondary_Skirt_') else root
        bone=bone.parent
    point=root.head_local
    return min(thighs,key=lambda n:(rig.data.bones[n].head_local-point).length_squared)


def install_directional_contacts(rig,meshes,radius_factor=2.,fan_degrees=25.,both_legs=True,rest_envelope=False,side_scoped=False,smooth_fallback=False,opposite_fallback=False,pelvis_support=False,opposite_full_only=False,compact=True):
    # Optional flags support isolated comparisons; the production wrapper below
    # selects stable ownership, pelvis support and only true cross-leg fallback.
    plan=plan_colliders(rig,meshes);sb=rig.data.vrm_addon_extension.spring_bone1
    payload=snapshot_contact_colliders(rig)
    payload['groups']=[g for g in payload['groups'] if g['springs']]
    previous_guards={i for i,c in enumerate(payload['colliders']) if c.get('directional')}
    keep=[i for i in range(len(payload['colliders'])) if i not in previous_guards]
    mapping={old:i for i,old in enumerate(keep)}
    payload['colliders']=[payload['colliders'][i] for i in keep]
    for group in payload['groups']:group['colliders']=[mapping[i] for i in group['colliders'] if i in mapping]
    body={s['bone']:s for s in plan['collider_details'] if s['role']=='skirt'}
    hum=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    thighs=[getattr(hum,s+'_upper_leg').node.bone_name for s in ('left','right')]
    lower={getattr(hum,side+'_upper_leg').node.bone_name:getattr(hum,side+'_lower_leg').node.bone_name for side in ('left','right')}
    leg_names=set(thighs)|set(lower.values())
    lookup={s.vrm_name:s for s in sb.springs}
    guards=0;minimum=float('inf')
    for group in payload['groups']:
        if len(group['springs'])!=1:continue
        spring=lookup[group['springs'][0]]
        samples=[(rig.data.bones[t.node.bone_name].head_local.copy(),h.hit_radius) for h,t in zip(spring.joints,spring.joints[1:])]
        center=sum((p for p,r in samples),Vector())/len(samples)
        owner=skirt_owner_leg(rig,spring,thighs)
        if side_scoped and not opposite_fallback:
            allowed={owner,lower[owner]}
            group['colliders']=[i for i in group['colliders'] if payload['colliders'][i]['bone'] not in leg_names or payload['colliders'][i]['bone'] in allowed]
        if side_scoped and opposite_fallback and opposite_full_only:
            own={owner,lower[owner]}
            group['colliders']=[i for i in group['colliders'] if payload['colliders'][i]['bone'] not in leg_names or payload['colliders'][i]['bone'] in own or (Vector(payload['colliders'][i]['tail'])-Vector(payload['colliders'][i]['offset'])).length>=rig.data.bones[payload['colliders'][i]['bone']].length*.9]
        if smooth_fallback:
            group['colliders']=[i for i in group['colliders'] if payload['colliders'][i]['bone'] not in leg_names or (Vector(payload['colliders'][i]['tail'])-Vector(payload['colliders'][i]['offset'])).length>=rig.data.bones[payload['colliders'][i]['bone']].length*.9]
        chosen=[owner] if side_scoped else thighs if both_legs else [min(thighs,key=lambda n:segment_distance(center,rig.data.bones[n].head_local,rig.data.bones[n].tail_local))]
        if pelvis_support:
            chosen=[owner]+[n for n in thighs if n!=owner]
        for name in chosen:
            if name not in body:continue  # No safe body fit for this bone.
            bone=rig.data.bones[name];inverse=bone.matrix_local.inverted()
            local=[(inverse@p,r) for p,r in samples]
            middle=inverse@center;normal=Vector((middle.x,0,middle.z)).normalized()
            if normal.length<.5:continue
            R=bone.length*radius_factor
            if compact:
                # A leg-length radius creates a near-plane whose back side
                # spans other chains. Bound support by measured body thickness
                # rather than the longitudinal bone length. Keep a continuous
                # head-to-tail axis: short rounded supports introduce abrupt
                # changes in contact normal as the leg rotates.
                R=min(R,body[name]['body_radius'])
            for degrees in ((-fan_degrees,0,fan_degrees) if fan_degrees else (0,)):
                n=Quaternion((0,1,0),math.radians(degrees))@normal
                def shape(bound):
                    offset=n*(bound-R)
                    return offset,offset+Vector((0,bone.length,0))
                def clearance(bound):
                    a,b=shape(bound)
                    return min(segment_distance(p,a,b)-r-R for p,r in local)
                hi=max(p.dot(n)-r for p,r in local) if rest_envelope else body[name]['body_radius'];lo=-bone.length*4
                if clearance(lo)<plan['margin']:continue
                for _ in range(36):
                    mid=(lo+hi)/2
                    if clearance(mid)>=plan['margin']:lo=mid
                    else:hi=mid
                a,b=shape(lo)
                attachment=name
                if pelvis_support and name!=owner:
                    attachment=hum.hips.node.bone_name
                    transform=rig.data.bones[attachment].matrix_local.inverted()@bone.matrix_local
                    a,b=transform@a,transform@b
                index=len(payload['colliders']);payload['colliders'].append(dict(bone=attachment,offset=list(a),tail=list(b),radius=R,base=R,limit=R,contact=True,directional=True))
                group['colliders'].append(index);guards+=1;minimum=min(minimum,clearance(lo))
    used=sorted({i for g in payload['groups'] for i in g['colliders']})
    remap={old:i for i,old in enumerate(used)}
    payload['colliders']=[payload['colliders'][i] for i in used]
    for group in payload['groups']:group['colliders']=[remap[i] for i in group['colliders']]
    replace_contact_colliders(rig,payload)
    return dict(guards=guards,minimum_rest_clearance=minimum,radius_factor=radius_factor,fan_degrees=fan_degrees,compact=compact)


def split_contact_segments(rig):
    """Unique tip nodes give each simulated segment its own collider scope.

    Deform bones, rest frames and all mesh weights stay unchanged. The next
    deform segment and the preceding segment's tip are siblings: no joint is
    simulated twice, and parent-to-child dependencies are acyclic.
    """
    import bpy
    from avatar_vrm_colliders import add_group
    sb=rig.data.vrm_addon_extension.spring_bone1
    fields=('hit_radius','stiffness','drag_force','gravity_power','gravity_dir')
    originals=[]
    for spring in sb.springs:
        if not spring.vrm_name.startswith('Secondary_Skirt_'):continue
        if '_Contact_' in spring.vrm_name:raise ValueError('Already split into contact segments')
        originals.append(dict(name=spring.vrm_name,center=spring.center.bone_name,
            groups=[r.collider_group_uuid for r in spring.collider_groups],
            joints=[dict(bone=j.node.bone_name,**{f:list(getattr(j,f)) if f=='gravity_dir' else getattr(j,f) for f in fields}) for j in spring.joints]))
    bpy.context.view_layer.objects.active=rig;rig.select_set(True);bpy.ops.object.mode_set(mode='EDIT')
    for spring in originals:
        for i,(head,tail) in enumerate(zip(spring['joints'],spring['joints'][1:])):
            bone=rig.data.edit_bones[head['bone']];tip=rig.data.edit_bones.new('Secondary_SkirtTip_'+head['bone'])
            tip.head=rig.data.edit_bones[tail['bone']].head;tip.tail=tip.head+(tip.head-bone.head).normalized()*.003
            tip.parent=bone;tip.use_connect=False;head['tip']=tip.name
    bpy.ops.object.mode_set(mode='OBJECT')
    for i in reversed(range(len(sb.springs))):
        if sb.springs[i].vrm_name.startswith('Secondary_Skirt_'):sb.springs.remove(i)
    groups={g.uuid:[x.collider_uuid for x in g.colliders] for g in sb.collider_groups}
    for chain in originals:
        for i,head in enumerate(chain['joints'][:-1]):
            s=sb.springs.add();s.vrm_name=chain['name']+'_Contact_'+str(i);s.center.bone_name=chain['center']
            for name in (head['bone'],head['tip']):
                j=s.joints.add();j.node.bone_name=name
                for f in fields:setattr(j,f,head[f])
                j['hallway_stiffness_ratio']=head['stiffness']/max(chain['joints'][0]['stiffness'],1e-8)
            g=add_group(rig,'Secondary_SkirtContact_'+s.vrm_name)
            for uuid in dict.fromkeys(x for old in chain['groups'] for x in groups[old]):g.colliders.add().collider_uuid=uuid
            s.collider_groups.add().collider_group_uuid=g.uuid
    return dict(segments=sum(len(s['joints'])-1 for s in originals))


def make_surface_patches(rig,meshes):
    """Experimental independent spring flaps with contact at weighted surface centers."""
    import bpy
    if not bpy.app.background:raise RuntimeError('Surface patch experiments require isolated background Blender')
    from avatar_apparel_weights import weights
    sb=rig.data.vrm_addon_extension.spring_bone1
    chains=[];centers={};totals={}
    for s in sb.springs:
        if not s.vrm_name.startswith('Secondary_Skirt_'):continue
        names=[j.node.bone_name for j in s.joints]
        chains.append((names,rig.data.bones[names[0]].parent.name,rig.data.bones[names[0]].head_local.copy()))
        centers.update({n:Vector() for n in names[:-1]});totals.update({n:0. for n in names[:-1]})
    for o in meshes:
        tr=rig.matrix_world.inverted()@o.matrix_world
        for v in o.data.vertices:
            p=tr@v.co
            for n,w in weights(o,v.index).items():
                if n in centers:centers[n]+=p*w;totals[n]+=w
    for n in centers:
        if totals[n]:centers[n]/=totals[n]
        else:centers[n]=rig.data.bones[n].tail_local.copy()
    split_contact_segments(rig)
    bpy.context.view_layer.objects.active=rig;bpy.ops.object.mode_set(mode='EDIT')
    for names,parent,root in chains:
        for name in names[:-1]:
            bone=rig.data.edit_bones[name];bone.use_connect=False;bone.parent=rig.data.edit_bones[parent]
            tip=rig.data.edit_bones['Secondary_SkirtTip_'+name]
            bone.head=root;bone.tail=centers[name]
            if (bone.tail-bone.head).length<.005:bone.tail.z-=.005
            tip.head=bone.tail;tip.tail=bone.tail+(bone.tail-bone.head).normalized()*.003
    bpy.ops.object.mode_set(mode='OBJECT')
    for s in sb.springs:
        if s.vrm_name.startswith('Secondary_Skirt_'):
            for j in s.joints:j.hit_radius=.006
    return dict(patches=len(centers))


def install_skirt_contact_rig(rig,meshes):
    """Production physics-only upgrade; never writes a vertex or a skin weight."""
    from avatar_skirt_fit_io import rest_edit
    from avatar_mesh_invariant import snapshot,verify
    from avatar_apparel_weights import weights
    from avatar_bone_collections import organize_bones
    from properties_hallway_rig import initialize
    from avatar_continuous_contacts import merge_contact_segments
    from avatar_contact_colliders import install_contact_colliders
    before=snapshot(meshes)
    binding={o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}
    springs=[s for s in rig.data.vrm_addon_extension.spring_bone1.springs if s.vrm_name.startswith('Secondary_Skirt_')]
    if not springs:return dict(guards=0,segments=0)
    with rest_edit(rig):
        # VRM evaluates a continuous chain in order. Separate simulations for
        # dependent segments can feed stale parent poses into the next segment
        # and get stuck in a different collision basin after a fast reversal.
        # Merge only our generated segments; all deform bones/weights stay put.
        merge_contact_segments(rig)
        install_contact_colliders(rig,meshes)
        report=install_directional_contacts(rig,meshes,radius_factor=1.,fan_degrees=0.,
            rest_envelope=True,side_scoped=True,opposite_fallback=True,pelvis_support=True,opposite_full_only=True)
        initialize(rig);organize_bones(rig)
        verify(meshes,before)
        if binding!={o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}:
            raise RuntimeError('Contact generation unexpectedly changed skin weights')
        rig['hallway_skirt_contact_rig']=4
        chains=[s for s in rig.data.vrm_addon_extension.spring_bone1.springs if s.vrm_name.startswith('Secondary_Skirt_')]
        report['chains']=len(chains)
        report['segments']=sum(len(s.joints)-1 for s in chains)
    return report
