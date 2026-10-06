"""VRM-exportable skirt follow with the same rest frame as its source leg."""
import bpy

from avatar_rig_defaults import FOLLOW
INFLUENCE=FOLLOW['Skirt']
PREFIX='Secondary_SkirtFollow_'
CONSTRAINT_NAME='UniMate skirt leg follow'


def preserve_export_rest_frame(rig):
    # Automatic export posing can straighten a humanoid thigh without posing
    # its non-humanoid constraint helper, breaking their shared local frame.
    # Generated rigs already own an authored rest skeleton; export that frame.
    ext=getattr(rig.data,'vrm_addon_extension',None)
    humanoid=ext.vrm1.humanoid if ext else None
    if humanoid and getattr(humanoid,'pose',None)=='autoPose':
        humanoid.pose='restPositionPose'


def set_rest_frame(follow,leg):
    # A new edit bone has zero length. Assigning its matrix before setting its
    # length can discard the requested direction. Explicit endpoints and roll
    # preserve all three local axes, including mirrored and rolled thighs.
    follow.parent=leg.parent
    follow.use_connect=False
    follow.head=leg.head.copy()
    follow.tail=leg.tail.copy()
    follow.roll=leg.roll
    follow.use_deform=True  # Export retention only; no weights on this control.


def set_constraint(constraint,rig,leg_name,influence=INFLUENCE):
    constraint.name=CONSTRAINT_NAME
    constraint.target=rig;constraint.subtarget=leg_name
    constraint.owner_space=constraint.target_space='LOCAL'
    if hasattr(constraint,'mix_mode'):constraint.mix_mode='ADD'
    elif hasattr(constraint,'use_offset'):constraint.use_offset=True
    constraint.mute=False
    constraint.use_x=constraint.use_y=constraint.use_z=True
    constraint.invert_x=constraint.invert_y=constraint.invert_z=False
    constraint.influence=influence


def upgrade_skirt_follow(rig,influence=INFLUENCE):
    """Update our existing helpers without changing skinning or child rest bones.

    The caller suspends its physics runtime before changing the rest skeleton.
    Bone names and constraint targets identify ownership; no avatar identifiers
    or source mesh weights participate.
    """
    if not 0<=influence<=1:raise ValueError('Follow influence must be in [0,1]')
    targets={}
    for pb in rig.pose.bones:
        if not pb.name.startswith(PREFIX):continue
        constraints=[c for c in pb.constraints if c.type=='COPY_ROTATION' and c.name==CONSTRAINT_NAME]
        if len(constraints)!=1:raise ValueError('Expected one owned leg-follow constraint on '+pb.name)
        c=constraints[0]
        if c.target!=rig or c.subtarget not in rig.data.bones:raise ValueError('Invalid follow target')
        leg=rig.data.bones[c.subtarget]
        ancestor=leg
        while ancestor:
            if ancestor.name==pb.name:raise ValueError('Circular leg-follow dependency')
            ancestor=ancestor.parent
        targets[pb.name]=leg.name
    if not targets:return []
    active=bpy.context.view_layer.objects.active
    selected=list(bpy.context.selected_objects)
    mode=bpy.context.object.mode if bpy.context.object else 'OBJECT'
    rest={b.name:(b.head_local.copy(),b.tail_local.copy(),b.matrix_local.copy()) for b in rig.data.bones if b.name not in targets}
    try:
        if bpy.context.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT');rig.select_set(True);bpy.context.view_layer.objects.active=rig
        bpy.ops.object.mode_set(mode='EDIT')
        for name,source in targets.items():
            helper=rig.data.edit_bones[name];leg=rig.data.edit_bones[source]
            parent,pivot=helper.parent,helper.head.copy()
            dress=bool(helper.get('hallway_dress_lower_follow') or helper.get('hallway_dress_knee_blend'))
            set_rest_frame(helper,leg)
            if dress:
                helper.parent=parent;helper.head=pivot;helper.tail=pivot+(leg.tail-leg.head)
        bpy.ops.object.mode_set(mode='OBJECT')
        for name,source in targets.items():
            pb=rig.pose.bones[name];constraint=next(c for c in pb.constraints if c.name==CONSTRAINT_NAME)
            knee=pb.bone.get('hallway_dress_lower_follow') or pb.bone.get('hallway_dress_knee_blend')
            group=rig.hallway_rig.follow_groups.get('Skirt Knee') if hasattr(rig,'hallway_rig') else None
            value=(group.influence if group else FOLLOW['Skirt Knee']) if knee else influence
            if knee and group:
                from properties_hallway_rig import follow_influence
                value=follow_influence(rig,pb,group)
            else:
                from avatar_dress_fit import adapted_follow
                strength=rig.hallway_rig.dress_fit_strength if hasattr(rig,'hallway_rig') else 1.
                value=adapted_follow(value,pb.bone.get('hallway_dress_tightness',0.),strength)
                value*=float(pb.bone.get('hallway_follow_share',1.))
            set_constraint(constraint,rig,source,value)
            index=list(pb.constraints).index(constraint)
            if index:pb.constraints.move(index,0)
        bpy.context.view_layer.update()
        for name,(head,tail,matrix) in rest.items():
            b=rig.data.bones[name]
            assert (b.head_local-head).length<1e-6 and (b.tail_local-tail).length<1e-6
            error=max(abs(a-v) for row,old in zip(b.matrix_local,matrix) for a,v in zip(row,old))
            # Blender recomputes single-precision rest matrices on leaving Edit
            # Mode; retain positions exactly and allow only round-off in axes.
            assert error<1e-5,('Unrelated rest frame changed',name,error)
        for name,source in targets.items():
            a,b=rig.data.bones[name].matrix_local,rig.data.bones[source].matrix_local
            if rig.data.bones[name].get('hallway_dress_lower_follow') or rig.data.bones[name].get('hallway_dress_knee_blend'):a,b=a.to_3x3(),b.to_3x3()
            assert max(abs(x-y) for row,old in zip(a,b) for x,y in zip(row,old))<1e-6
        rig['unimate_skirt_follow_revision']=2
        preserve_export_rest_frame(rig)
        return [dict(bone=n,target=t,influence=next(c.influence for c in rig.pose.bones[n].constraints if c.name==CONSTRAINT_NAME)) for n,t in targets.items()]
    finally:
        if bpy.context.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        for obj in selected:obj.select_set(True)
        bpy.context.view_layer.objects.active=active
        if active and mode!='OBJECT':bpy.ops.object.mode_set(mode=mode)
