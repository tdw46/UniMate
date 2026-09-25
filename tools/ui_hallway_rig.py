"""Hallway sidebar for persistent generated-rig settings."""
import bpy
from properties_hallway_rig import active_rig, target_armature, initialize, apply_settings
from avatar_bone_collections import organize_bones


class HALLWAY_OT_ConfigureRig(bpy.types.Operator):
    bl_idname = 'hallway.configure_rig'
    bl_label = 'Organize and Configure Rig'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        rig = active_rig(context)
        return bool(rig and rig.mode != 'EDIT')

    def execute(self, context):
        rig = active_rig(context)
        initialize(rig)
        organize_bones(rig)
        return {'FINISHED'}


class HALLWAY_OT_ApplyRigSettings(bpy.types.Operator):
    bl_idname = 'hallway.apply_rig_settings'
    bl_label = 'Reapply Rig Settings'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        rig = active_rig(context)
        return bool(rig and rig.mode != 'EDIT' and rig.hallway_rig.initialized)

    def execute(self, context):
        count = apply_settings(active_rig(context))
        self.report({'INFO'}, f'Settings applied; {count} collider radii limited by rest clearance' if count else 'Rig settings applied')
        return {'FINISHED'}


class HALLWAY_OT_ShowColliders(bpy.types.Operator):
    bl_idname = 'hallway.show_colliders'
    bl_label = 'Show VRM Colliders'
    bl_options = {'REGISTER', 'UNDO'}
    visible: bpy.props.BoolProperty(default=True)

    @classmethod
    def poll(cls, context):
        rig = active_rig(context)
        return bool(rig and hasattr(rig.data, 'vrm_addon_extension'))

    def execute(self, context):
        from avatar_vrm_colliders import show_colliders
        show_colliders(active_rig(context), self.visible)
        return {'FINISHED'}


class HALLWAY_OT_RefitSkirtColliders(bpy.types.Operator):
    bl_idname = 'hallway.refit_skirt_colliders'
    bl_label = 'Refit Skirt Colliders'
    bl_description = 'Refit generated skirt capsules through the VRM API; preserve hair and artist collider groups'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        rig = active_rig(context)
        return bool(rig and rig.mode != 'EDIT' and hasattr(rig.data, 'vrm_addon_extension')
                    and rig.data.vrm_addon_extension.spec_version == '1.0')

    def execute(self, context):
        from avatar_colliders import rebuild_colliders
        rig = active_rig(context)
        meshes = [obj for obj in context.scene.objects if obj.type == 'MESH'
                  and any(m.type == 'ARMATURE' and m.object == rig for m in obj.modifiers)]
        if rig.get('hallway_skirt_contact_rig'):
            from avatar_directional_contacts import install_skirt_contact_rig
            report=install_skirt_contact_rig(rig,meshes)
            report['colliders']=report['guards']
        else:
            report = rebuild_colliders(rig, meshes, collider_roles=('skirt',))
        settings = initialize(rig)
        settings.skirt_thickness = 1.
        self.report({'INFO'}, f"Fitted {report['colliders']} VRM skirt capsules; thickness reset to 1x")
        return {'FINISHED'}


class HALLWAY_OT_ApplySkirtFit(bpy.types.Operator):
    bl_idname = 'hallway.apply_skirt_fit'
    bl_label = 'Apply Weight Fit'
    bl_description = 'Apply weights only; reject fits that modify vertices, topology or shape keys'
    bl_options = {'REGISTER', 'UNDO'}
    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(default='*.json', options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        rig=active_rig(context)
        return bool(rig and rig.mode!='EDIT')

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        from avatar_skirt_fit_io import apply_bundle
        try:
            report=apply_bundle(active_rig(context),bpy.path.abspath(self.filepath))
        except (ValueError, KeyError, OSError, RuntimeError) as error:
            self.report({'ERROR'},str(error))
            return {'CANCELLED'}
        self.report({'INFO'},'This fit is already applied' if report['already_applied'] else
                    f"Pose fit applied: {report['changed_vertices']} garment vertices, {report['skirt_colliders']} capsules")
        return {'FINISHED'}


class HALLWAY_OT_RegenerateSecondary(bpy.types.Operator):
    bl_idname = 'hallway.regenerate_secondary'
    bl_label = 'Regenerate Hair and Skirt Rig'
    bl_description = 'Rebuild generated springs and native colliders; preserve rest geometry, UVs and existing shape keys'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        rig=active_rig(context)
        return bool(rig and rig.get('unimate_secondary_generator') and rig.mode!='EDIT')

    def execute(self, context):
        from avatar_regenerate_secondary import regenerate
        rig=active_rig(context)
        meshes=[o for o in context.scene.objects if o.type=='MESH' and any(m.type=='ARMATURE' and m.object==rig for m in o.modifiers)]
        try:
            group=rig.hallway_rig.follow_groups.get('Skirt')
            result=regenerate(rig,meshes,follow=group.influence if group else .55)
        except (ValueError,RuntimeError) as error:
            self.report({'ERROR'},str(error));return {'CANCELLED'}
        self.report({'INFO'},f"Regenerated {result['chains']} spring chains; original mesh and shape keys preserved")
        return {'FINISHED'}


class HALLWAY_OT_SkirtContacts(bpy.types.Operator):
    bl_idname = 'hallway.skirt_contacts'
    bl_label = 'Upgrade Skirt Contacts'
    bl_description = 'Create segment-specific VRM contact guards while preserving mesh geometry and all skin weights'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        rig=active_rig(context)
        return bool(rig and rig.mode!='EDIT' and rig.get('unimate_secondary_generator'))

    def execute(self, context):
        from avatar_directional_contacts import install_skirt_contact_rig
        rig=active_rig(context)
        meshes=[o for o in context.scene.objects if o.type=='MESH' and any(m.type=='ARMATURE' and m.object==rig for m in o.modifiers)]
        try:report=install_skirt_contact_rig(rig,meshes)
        except (ValueError,RuntimeError) as error:
            self.report({'ERROR'},str(error));return {'CANCELLED'}
        self.report({'INFO'},f"{report['segments']} skirt contact segments; mesh and skin weights unchanged")
        return {'FINISHED'}


class HALLWAY_PT_Rig(bpy.types.Panel):
    bl_label = 'Rig Configuration'
    bl_idname = 'HALLWAY_PT_Rig'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Hallway'

    @classmethod
    def poll(cls, context):
        return target_armature(context) is not None

    def draw(self, context):
        rig = target_armature(context)
        layout = self.layout
        layout.label(text=rig.name, icon='ARMATURE_DATA')
        if active_rig(context) is None:
            layout.label(text='VRM target has no Hallway rig setup', icon='INFO')
            return
        settings = rig.hallway_rig
        if not settings.initialized:
            layout.operator('hallway.configure_rig')
            return
        from avatar_physics_preview import available, enabled as physics_enabled
        enabled = physics_enabled(context)
        row = layout.row(align=True)
        row.enabled = available(context)
        op = row.operator('hallway.set_physics', text='BVT Physics On' if enabled else 'BVT Physics Off', depress=enabled)
        op.enabled = not enabled
        row.operator('hallway.reset_physics', text='Reset')
        if not available(context):
            layout.label(text='Enable BVT for physics preview', icon='INFO')
        col = layout.column()
        col.enabled = rig.mode != 'EDIT'
        for group in settings.follow_groups:
            col.prop(group, 'influence', text=group.name + ' Follow', slider=True)
        if 'Skirt' in settings.spring_groups:
            col.prop(settings, 'skirt_thickness', text='Skirt Thickness')
            col.label(text='Limited by rest clearance', icon='INFO')
            col.operator('hallway.refit_skirt_colliders')
            if not rig.get('hallway_skirt_contact_rig'):col.operator('hallway.skirt_contacts')
        from avatar_vrm_colliders import colliders_visible
        visible = colliders_visible(rig) if hasattr(rig.data, 'vrm_addon_extension') else False
        op = col.operator('hallway.show_colliders', text='Hide VRM Colliders' if visible else 'Show VRM Colliders')
        op.visible = not visible
        for group in settings.spring_groups:
            box = col.box()
            box.label(text=group.name + ' Springs')
            box.prop(group, 'drag', slider=True)
            box.prop(group, 'stiffness')
            box.prop(group, 'non_root_stiffness', slider=True)
            box.prop(group, 'gravity')
        col.operator('hallway.regenerate_secondary')
        col.operator('hallway.apply_rig_settings')
        layout.label(text='Live updates · saved with the .blend')
        box = layout.box()
        box.label(text='Bone Collections')
        if hasattr(rig.data, 'collections'):
            for collection in rig.data.collections:
                if collection.get('hallway_role'):
                    row = box.row(align=True)
                    row.prop(collection, 'is_visible', text='', emboss=False,
                             icon='HIDE_OFF' if collection.is_visible else 'HIDE_ON')
                    row.label(text=collection.name)
        elif hasattr(rig.data, 'layers'):
            from avatar_bone_collections import NAMES
            for index, name in enumerate(NAMES):
                row = box.row(align=True)
                row.prop(rig.data, 'layers', index=index, text='', emboss=False,
                         icon='HIDE_OFF' if rig.data.layers[index] else 'HIDE_ON')
                row.label(text=name)
        box.operator('hallway.configure_rig', text='Refresh Bone Organization')


CLASSES = (HALLWAY_OT_ConfigureRig, HALLWAY_OT_ApplyRigSettings, HALLWAY_OT_ShowColliders, HALLWAY_OT_RefitSkirtColliders, HALLWAY_OT_ApplySkirtFit, HALLWAY_OT_RegenerateSecondary, HALLWAY_OT_SkirtContacts, HALLWAY_PT_Rig)


def register():
    from properties_hallway_rig import register as register_properties
    register_properties()
    import avatar_physics_preview
    avatar_physics_preview.register()
    for cls in CLASSES:
        if not getattr(cls, 'is_registered', False):
            bpy.utils.register_class(cls)
    import avatar_collider_overlay
    avatar_collider_overlay.register()


def unregister():
    import avatar_collider_overlay
    avatar_collider_overlay.unregister()
    for cls in reversed(CLASSES):
        if getattr(cls, 'is_registered', False):
            bpy.utils.unregister_class(cls)
    from properties_hallway_rig import unregister as unregister_properties
    unregister_properties()
    import avatar_physics_preview
    avatar_physics_preview.unregister()
