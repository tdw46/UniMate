"""Native upper-skirt spring segments, independent of lower-chain drivers."""
import bpy
from mathutils import Vector
PREFIX = 'Secondary_HipSkirt_'


def refresh_hip_contacts(rig):
    from avatar_hip_contacts import refresh
    return refresh(rig)


def fit_hip_origins(rig):
    """Refit existing upper pivots on the garment without rebinding any mesh."""
    from avatar_skirt_support import ensure_support
    from avatar_dress import full_chain_names
    families = {}
    for spring in rig.data.vrm_addon_extension.spring_bone1.springs:
        if not spring.vrm_name.startswith('Secondary_Skirt_'):
            continue
        names = full_chain_names(rig, spring)
        if names is not None:
            points = [rig.data.bones[n].head_local.copy() for n in names]
            families.setdefault(spring.vrm_name.rsplit('_', 1)[0], []).append((names, points))
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH' and o.find_armature() == rig]
    return ensure_support(rig, meshes, families) if families else None


def extend_hip_physics(rig):
    """Replace each surface support's rotation constraint with a VRM spring.

    Hips anchors the upper surface pivot. An unweighted tip reaches into the
    upper skirt so its capsule contacts can bend the support under motion.
    Existing skirt/knee spring hierarchies and their local drivers stay intact.
    """
    from avatar_skirt_fit_io import rest_edit
    from avatar_bone_collections import organize_bones
    from avatar_skirt_support import GROUP,CONSTRAINT
    sb=rig.data.vrm_addon_extension.spring_bone1
    hips=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones.hips.node.bone_name
    selected={p.name:(p.select if hasattr(p,'select') else p.bone.select) for p in rig.pose.bones}
    active=rig.data.bones.active.name if rig.data.bones.active else None
    rows=[]
    for bone in rig.data.bones:
        root=bone.get('hallway_support_chain')
        if not root or not bone.get('hallway_skirt_support'):continue
        spring=next((s for s in sb.springs if s.vrm_name==root.rsplit('_',1)[0]),None)
        if not spring:raise ValueError('Missing upper skirt spring for '+root)
        name=PREFIX+root.removeprefix('Secondary_Skirt_').rsplit('_',1)[0]
        if any(s.vrm_name==name for s in sb.springs):continue
        if any(c.name!=CONSTRAINT for c in rig.pose.bones[bone.name].constraints):raise ValueError('Foreign support constraint')
        source=rig.data.bones[root]
        # Modest extension into the already blended region, bounded by the
        # existing first segment rather than avatar-specific distances.
        end=source.head_local+(source.tail_local-source.head_local)*.2
        rows.append((bone.name,root,name,end,spring.joints[0]))
    with rest_edit(rig):
        if rows:
            try:
                rig.select_set(True);bpy.ops.object.mode_set(mode='EDIT')
                for name,root,_,end,_ in rows:
                    top=rig.data.edit_bones[name];top.parent=rig.data.edit_bones[hips];top.use_connect=False
                    top.tail=end;top['hallway_hip_physics']=1
                    for key in ('hallway_constraint_group','hallway_follow_constraint'):
                        if key in top:del top[key]
                    tip=rig.data.edit_bones.new(name+'_Tip');tip.head=end
                    tip.tail=end+(end-top.head).normalized()*.003;tip.parent=top;tip.use_deform=False
            finally:
                if rig.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
            for name,root,spring_name,_,source in rows:
                for c in list(rig.pose.bones[name].constraints):rig.pose.bones[name].constraints.remove(c)
                s=sb.springs.add();s.vrm_name=spring_name;s.center.bone_name=''
                for joint_name in (name,name+'_Tip'):
                    j=s.joints.add();j.node.bone_name=joint_name
                    for f in ('stiffness','drag_force','gravity_power','gravity_dir'):setattr(j,f,getattr(source,f))
                    j.hit_radius=min(source.hit_radius,rig.data.bones[name].length*.06)
                    j['hallway_stiffness_ratio']=1.
        settings=getattr(rig,'hallway_rig',None)
        if settings and GROUP in settings.follow_groups:settings.follow_groups.remove(settings.follow_groups.find(GROUP))
        origins=fit_hip_origins(rig)
        contacts=refresh_hip_contacts(rig)
        organize_bones(rig)
        for p in rig.pose.bones:
            if hasattr(p,'select'):p.select=selected.get(p.name,False)
            else:p.bone.select=selected.get(p.name,False)
        rig.data.bones.active=rig.data.bones.get(active) if active else None
    return dict(added_segments=len(rows),bones=[n for n,_,_,_,_ in rows],contacts=contacts, origins=origins)
