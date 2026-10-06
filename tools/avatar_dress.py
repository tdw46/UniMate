"""Knee-aware dress guides and portable upper/lower spring sections."""
import json
import math


def dress_profile(top, bottom, knee, ankle, segments):
    """Mid-calf and shorter remain thigh-followed; never add a tiny end joint."""
    top,bottom,knee,ankle=map(float,(top,bottom,knee,ankle))
    if not all(math.isfinite(x) for x in (top,bottom,knee,ankle)) or top<=bottom or knee<=ankle or segments<2:
        raise ValueError('Invalid dress or lower-leg height interval')
    mid_calf=(knee+ankle)*.5
    lower=bottom < mid_calf-(knee-ankle)*1e-5 and bottom<knee<top
    if bottom<knee<top:
        count=max(segments,4 if lower else 2)
        above=max(1,min(count-1,round(count*(top-knee)/(top-bottom))))
        if lower:above=max(2,min(count-2,above))
        below=count-above
        heights=[top+(knee-top)*i/above for i in range(above)]
        heights += [knee+(bottom-knee)*i/below for i in range(below+1)]
        knee_index=above
    else:
        heights=[top+(bottom-top)*i/segments for i in range(segments+1)]
        knee_index=None
    return dict(heights=heights,knee_index=knee_index,lower_follow=lower,
                knee_height=knee,mid_calf_height=mid_calf)


def full_chain_names(rig, spring):
    root=rig.data.bones[spring.joints[0].node.bone_name]
    if root.get('hallway_dress_lower'):return None
    names=root.get('hallway_dress_full_chain')
    return json.loads(names) if names else [j.node.bone_name for j in spring.joints]


def install_lower_follow(rig, chains):
    """Keep one continuous simulation per anatomical section, joined at knee.

    The upper spring's terminal carries an unweighted calf-frame helper. Its
    local rotation constraint adds the calf bend to inherited thigh/cloth
    motion. The lower spring starts there, so no simulated joint is constrained
    or shared between solvers and no cloth receives direct calf weights.
    """
    import bpy
    from avatar_skirt_follow import set_rest_frame,set_constraint
    rows=[c for c in chains if c['role']=='skirt' and c['dress']['lower_follow']]
    if not rows:return dict(lower_sections=0)
    sb=rig.data.vrm_addon_extension.spring_bone1
    fields=('hit_radius','stiffness','drag_force','gravity_power','gravity_dir')
    records=[]
    for chain in rows:
        spring=next(s for s in sb.springs if s.vrm_name==chain['names'][0].rsplit('_',1)[0])
        if rig.data.bones[chain['names'][0]].get('hallway_dress_full_chain'):
            raise ValueError('Dress chain already has lower-leg follow')
        records.append((chain,spring.vrm_name,[{f:list(getattr(j,f)) if f=='gravity_dir' else getattr(j,f) for f in fields} for j in spring.joints]))
    bpy.context.view_layer.objects.active=rig
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        for chain,name,values in records:
            index=chain['dress']['knee_index'];names=chain['names']
            root=rig.data.edit_bones[names[index]];previous=rig.data.edit_bones[names[index-1]]
            tip=rig.data.edit_bones.new('Secondary_SkirtTip_'+name)
            tip.head=root.head.copy();tip.tail=tip.head+(previous.tail-previous.head)*.1
            tip.parent=previous;tip.use_deform=False
            helper=rig.data.edit_bones.new(chain['follow']+'_Lower')
            calf=rig.data.edit_bones[chain['lower_leg']]
            set_rest_frame(helper,calf)
            helper.head=root.head.copy();helper.tail=helper.head+(calf.tail-calf.head)
            helper.parent=tip
            helper['hallway_dress_lower_follow']=True
            root.use_connect=False;root.parent=helper
            root['hallway_dress_lower']=True
            root['hallway_dress_upper_spring']=name
            upper=rig.data.edit_bones[names[0]]
            upper['hallway_dress_full_chain']=json.dumps(names)
            upper['hallway_dress_knee_index']=index
            chain['knee_tip']=tip.name;chain['lower_follow']=helper.name
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
    for chain,name,values in records:
        index=chain['dress']['knee_index'];names=chain['names']
        upper=next(s for s in sb.springs if s.vrm_name==name)
        center=upper.center.bone_name
        upper.joints.clear()
        lower=sb.springs.add();lower.vrm_name=name+'_Lower';lower.center.bone_name=center
        for spring,joint_names,settings in ((upper,names[:index]+[chain['knee_tip']],values[:index+1]),
                                            (lower,names[index:],values[index:])):
            for bone,record in zip(joint_names,settings):
                j=spring.joints.add();j.node.bone_name=bone
                for field,value in record.items():setattr(j,field,value)
                j['hallway_stiffness_ratio']=record['stiffness']/max(values[0]['stiffness'],1e-8)
        c=rig.pose.bones[chain['lower_follow']].constraints.new('COPY_ROTATION')
        set_constraint(c,rig,chain['lower_leg'],chain['follow_influence'])
    bpy.context.view_layer.update()
    return dict(lower_sections=len(records),follows=[dict(bone=c['lower_follow'],target=c['lower_leg'],
                knee_height=c['dress']['knee_height'],knee_joint=c['names'][c['dress']['knee_index']]) for c,_,_ in records])


def is_midline_dress_chain(rig, spring, lookup=None):
    """Classify both knee sections by the original waist attachment."""
    sb=rig.data.vrm_addon_extension.spring_bone1
    lookup=lookup or {s.vrm_name:s for s in sb.springs}
    root=rig.data.bones[spring.joints[0].node.bone_name]
    upper=lookup.get(root.get('hallway_dress_upper_spring',''))
    if upper:root=rig.data.bones[upper.joints[0].node.bone_name]
    if root.get('hallway_bilateral_chain') or not root.get('hallway_dress_full_chain'):return False
    hum=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    left=rig.data.bones[hum.left_upper_leg.node.bone_name]
    right=rig.data.bones[hum.right_upper_leg.node.bone_name]
    hips=rig.data.bones[hum.hips.node.bone_name]
    return abs(root.head_local.x-hips.head_local.x)<abs(left.head_local.x-right.head_local.x)*.25


def knee_profile_weights(front_cosine):
    """Smooth front/side/back weights, independent of chain count and names."""
    cosine=max(-1.,min(1.,float(front_cosine)))
    front=max(cosine,0.)**2
    back=max(-cosine,0.)**2
    return front,1.-front-back,back


def dress_knee_profile(rig,spring,lookup):
    from mathutils import Vector
    root=rig.data.bones[spring.joints[0].node.bone_name]
    upper=lookup[root['hallway_dress_upper_spring']]
    attachment=rig.data.bones[upper.joints[0].node.bone_name]
    hum=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    hips=rig.data.bones[hum.hips.node.bone_name]
    lateral=rig.data.bones[hum.left_upper_leg.node.bone_name].head_local-rig.data.bones[hum.right_upper_leg.node.bone_name].head_local
    lateral.z=0.
    forward=lateral.cross(Vector((0.,0.,1.))).normalized()
    radial=attachment.head_local-hips.head_local;radial.z=0.
    if forward.length<.5 or radial.length<1e-8:raise ValueError('Cannot resolve dress front from humanoid rest landmarks')
    return knee_profile_weights(radial.normalized().dot(forward))


def skirt_sector_angle(sector, count):
    """Mirror pairs straddle front/back; no chain lies on the sagittal plane."""
    if count < 6 or count % 2:
        raise ValueError('Skirt sectors must be even and at least six')
    step=math.tau/count
    return math.pi/2-(count//4)*step+(sector+.5)*step


def upgrade_knee_follow(rig):
    """One knee source and one portable constraint per unweighted helper."""
    from avatar_skirt_follow import set_constraint,CONSTRAINT_NAME
    from properties_hallway_rig import initialize,apply_follow
    sb=rig.data.vrm_addon_extension.spring_bone1
    lookup={s.vrm_name:s for s in sb.springs}
    records=[]
    for spring in sb.springs:
        root=rig.data.bones[spring.joints[0].node.bone_name] if spring.joints else None
        if not root or not root.get('hallway_dress_lower'):continue
        helper=root.parent
        if not helper or not helper.get('hallway_dress_lower_follow'):raise ValueError('Missing dress knee helper')
        if helper.parent.get('hallway_dress_knee_blend'):
            raise ValueError('Migrate the old center-chain layout with upgrade_bilateral_skirt first')
        constraints=list(rig.pose.bones[helper.name].constraints)
        if len(constraints)!=1 or constraints[0].name!=CONSTRAINT_NAME or constraints[0].type!='COPY_ROTATION':
            raise ValueError('Expected exactly one knee constraint on '+helper.name)
        records.append(dict(helper=helper.name,target=constraints[0].subtarget,
            family=root['hallway_dress_upper_spring'].rsplit('_',1)[0],profile=dress_knee_profile(rig,spring,lookup)))
    if not records:return dict(knee_sections=0,shared_sections=0)
    # The closest left/right front pair reaches the requested front maximum,
    # even though neither chain is on the centerline. Same for the rear pair.
    for row in records:
        peers=[r for r in records if r['family']==row['family']]
        front,_,back=row['profile']
        # Anchor the most lateral pair to the side control exactly; a paired
        # layout need not contain a chain at precisely 90 degrees from front.
        side=min(r['profile'][0]+r['profile'][2] for r in peers)
        front=max(0.,front-side)/max(max(r['profile'][0] for r in peers)-side,1e-8)
        back=max(0.,back-side)/max(max(r['profile'][2] for r in peers)-side,1e-8)
        row['normalized']=(front,max(0.,1-front-back),back)
    for row in records:
        pb=rig.pose.bones[row['helper']]
        pb.bone['hallway_follow_share']=1.
        pb.bone['hallway_knee_profile']=row['normalized']
        set_constraint(pb.constraints[0],rig,row['target'],1.)
    settings=initialize(rig)
    apply_follow(rig,settings.follow_groups['Skirt Knee'])
    rig['hallway_dress_knee_follow_revision']=3
    return dict(knee_sections=len(records),shared_sections=0,
        profile_controls={n:settings.follow_groups[n].influence for n in ('Skirt Knee','Skirt Knee Side','Skirt Knee Back')},
        chain_profiles=[dict(bone=r['helper'],target=r['target'],weights=r['normalized']) for r in records])


def upgrade_bilateral_skirt(rig, meshes, material_roles=None):
    """Refit owned skirt guides and their spring weights, never mesh positions.

    Explicit migration for the retired center-chain layout. Hair, humanoid
    bones, materials, mesh coordinates and shape keys are preserved. Contacts
    must then be regenerated for the new guides with install_skirt_contact_rig.
    """
    import bpy
    from avatar_springs import plan_secondary
    from avatar_skirt_follow import set_rest_frame,set_constraint
    from avatar_skirt_fit_io import rest_edit
    from avatar_mesh_invariant import snapshot,verify
    from avatar_skirt_binding import rebind_skirt_strips
    if rig.get('hallway_bilateral_skirt_layout')==1:return dict(changed=False)
    sb=rig.data.vrm_addon_extension.spring_bone1
    uppers=[s for s in sb.springs if s.vrm_name.startswith('Secondary_Skirt_') and
            not rig.data.bones[s.joints[0].node.bone_name].get('hallway_dress_lower')]
    if not uppers:return dict(changed=False)
    families={}
    for s in uppers:families.setdefault(s.vrm_name.rsplit('_',1)[0],[]).append(s)
    counts={len(v) for v in families.values()}
    resolutions={len(full_chain_names(rig,s))-1 for s in uppers}
    if len(counts)!=1 or len(resolutions)!=1:raise ValueError('Mixed skirt resolutions require explicit regeneration')
    count=next(iter(counts));resolution=next(iter(resolutions))
    plan=plan_secondary(rig,meshes,material_roles,skirt_sectors=count,skirt_segments=resolution,include_hair=False)
    chains=[c for c in plan['chains'] if c['role']=='skirt']
    expected={s.vrm_name:full_chain_names(rig,s) for s in uppers}
    if {c['names'][0].rsplit('_',1)[0]:c['names'] for c in chains}!=expected:
        raise ValueError('Skirt region ownership changed; refusing an ambiguous migration')
    for chain in chains:
        for name in (chain['follow'],chain['follow']+'_Lower'):
            pb=rig.pose.bones.get(name)
            if pb is not None and len(pb.constraints)!=1:
                raise ValueError('Expected exactly one follow constraint: '+name)
    before=snapshot(meshes)
    selected={p.name:bool(p.select if hasattr(p,'select') else p.bone.select) for p in rig.pose.bones}
    active=rig.data.bones.active;active_name=active.name if active else None
    with rest_edit(rig):
        bpy.context.view_layer.objects.active=rig
        bpy.ops.object.mode_set(mode='EDIT')
        try:
            eb=rig.data.edit_bones
            for chain in chains:
                names=chain['names'];line=chain['line'];upper=eb[names[0]]
                knee=upper.get('hallway_dress_knee_index')
                follow=eb[chain['follow']]
                set_rest_frame(follow,eb[chain['leg']])
                for name in names:eb[name].use_connect=False
                for i,(name,point) in enumerate(zip(names,line)):
                    bone=eb[name];bone.head=point
                    bone.tail=line[i+1] if i<len(line)-1 else point+(point-line[i-1])*.1
                    bone.use_connect=i>0 and i!=knee
                upper['hallway_bilateral_chain']=True
                if knee is not None:
                    helper=eb[chain['follow']+'_Lower'];parent=helper.parent
                    if parent.get('hallway_dress_knee_blend'):
                        bridge=parent;parent=bridge.parent
                        helper.parent=parent;eb.remove(bridge)
                    parent.head=line[knee];parent.tail=line[knee]+(line[knee]-line[knee-1])*.1
                    calf=eb[chain['lower_leg']]
                    set_rest_frame(helper,calf)
                    helper.parent=parent;helper.head=line[knee];helper.tail=helper.head+(calf.tail-calf.head)
                    helper['hallway_follow_share']=1.
        finally:
            bpy.ops.object.mode_set(mode='OBJECT')
            for pb in rig.pose.bones:
                if hasattr(pb,'select'):pb.select=selected.get(pb.name,False)
                else:pb.bone.select=selected.get(pb.name,False)
            rig.data.bones.active=rig.data.bones.get(active_name) if active_name else None
        for chain in chains:
            for name,target in ((chain['follow'],chain['leg']),(chain['follow']+'_Lower',chain['lower_leg'])):
                pb=rig.pose.bones.get(name)
                if pb is None:continue
                if len(pb.constraints)!=1:raise ValueError('Expected exactly one follow constraint: '+name)
                set_constraint(pb.constraints[0],rig,target,pb.constraints[0].influence)
        binding=rebind_skirt_strips(rig,meshes)
        verify(meshes,before)
        rig['hallway_bilateral_skirt_layout']=1
    return dict(changed=True,chains=len(chains),binding=binding)
