"""Regenerate owned secondary bones/weights while keeping final meshes immutable."""
import bpy
from avatar_mesh_invariant import snapshot,verify
from avatar_physics_preview import suspended
from avatar_vrm_colliders import call_operator,cleanup_orphan_displays
from avatar_springs import generate_secondary,plan_secondary
from avatar_apparel_weights import weights,assign


def regenerate(rig,meshes,follow=.55,segments=4,sectors=12,skirt_segments=None):
    if not rig.get('unimate_secondary_generator'):
        raise ValueError('This operation only replaces an existing generated secondary rig')
    plan=plan_secondary(rig,meshes,segments=segments,skirt_sectors=sectors,skirt_segments=skirt_segments)
    if not plan['chains']:raise ValueError('No supported secondary regions found')
    geometry=snapshot(meshes);sb=rig.data.vrm_addon_extension.spring_bone1
    names={b.name for b in rig.data.bones if b.name.startswith('Secondary_')}
    if any(b.name not in names and b.parent and b.parent.name in names for b in rig.data.bones):
        raise ValueError('Artist bones depend on the generated secondary rig')
    groups={g.uuid for g in sb.collider_groups if g.vrm_name.startswith('Secondary_')}
    if any(r.collider_group_uuid in groups for s in sb.springs if not s.vrm_name.startswith('Secondary_') for r in s.collider_groups):
        raise ValueError('Artist springs share generated collider groups')
    collider_ids={r.collider_uuid for g in sb.collider_groups if g.uuid in groups for r in g.colliders}
    foreign_ids={r.collider_uuid for g in sb.collider_groups if g.uuid not in groups for r in g.colliders}
    active=bpy.context.view_layer.objects.active;selected=list(bpy.context.selected_objects)
    mode=active.mode if active else 'OBJECT';pose_position=rig.data.pose_position
    with suspended():
        try:
            if active and active.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
            bpy.context.view_layer.objects.active=rig;rig.select_set(True)
            rig.data.pose_position='REST';bpy.context.view_layer.update()
            for i in reversed(range(len(sb.springs))):
                if sb.springs[i].vrm_name.startswith('Secondary_'):sb.springs.remove(i)
            for i in reversed(range(len(sb.collider_groups))):
                if sb.collider_groups[i].uuid in groups:call_operator('remove_spring_bone1_collider_group',rig,collider_group_index=i)
            for i in reversed(range(len(sb.colliders))):
                if sb.colliders[i].uuid in collider_ids-foreign_ids:call_operator('remove_spring_bone1_collider',rig,collider_index=i)
            for obj in meshes:
                for vertex in obj.data.vertices:
                    old=weights(obj,vertex.index)
                    affected={n:w for n,w in old.items() if n in names}
                    if not affected:continue
                    value={n:w for n,w in old.items() if n not in names}
                    for name,w in affected.items():
                        parent='Head' if name.startswith('Secondary_Hair_') else 'Hips'
                        value[parent]=value.get(parent,0.)+w
                    assign(obj,[vertex.index],value)
                for group in list(obj.vertex_groups):
                    if group.name in names:obj.vertex_groups.remove(group)
            bpy.ops.object.mode_set(mode='EDIT')
            for bone in list(rig.data.edit_bones):
                if bone.name in names:rig.data.edit_bones.remove(bone)
            bpy.ops.object.mode_set(mode='OBJECT')
            rig['hallway_contact_colliders']=0
            for key in ('hallway_pose_fit','hallway_pose_fit_bundle','hallway_pose_fit_path','hallway_pose_fit_report','hallway_pose_fit_settings_changed'):
                if key in rig:del rig[key]
            report=generate_secondary(rig,meshes,segments=segments,skirt_sectors=sectors,skirt_segments=skirt_segments)
            from properties_hallway_rig import initialize
            settings=initialize(rig);settings.follow_groups['Skirt'].influence=follow
            cleanup_orphan_displays(rig)
            report['mesh_signatures']=verify(meshes,geometry)
            rig['hallway_immutable_mesh']=True
            return report
        finally:
            if rig.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
            rig.data.pose_position=pose_position
            for obj in bpy.context.selected_objects:obj.select_set(False)
            for obj in selected:obj.select_set(True)
            bpy.context.view_layer.objects.active=active
            if active and mode!='OBJECT':bpy.ops.object.mode_set(mode=mode)
            bpy.context.view_layer.update()
