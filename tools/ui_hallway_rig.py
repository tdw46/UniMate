"""Hallway sidebar for persistent generated-rig settings."""
import bpy
from properties_hallway_rig import active_rig, initialize, apply_settings
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
    bl_label = 'Apply Rig Settings'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        rig = active_rig(context)
        return bool(rig and rig.mode != 'EDIT' and rig.hallway_rig.initialized)

    def execute(self, context):
        count = apply_settings(active_rig(context))
        self.report({'INFO'}, f'Settings applied; {count} collider radii limited by rest clearance' if count else 'Rig settings applied')
        return {'FINISHED'}


class HALLWAY_PT_Rig(bpy.types.Panel):
    bl_label = 'Rig Configuration'
    bl_idname = 'HALLWAY_PT_Rig'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Hallway'

    @classmethod
    def poll(cls, context):
        return active_rig(context) is not None

    def draw(self, context):
        rig = active_rig(context)
        layout = self.layout
        layout.label(text=rig.name, icon='ARMATURE_DATA')
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
        for group in settings.spring_groups:
            box = col.box()
            box.label(text=group.name + ' Springs')
            box.prop(group, 'drag', slider=True)
            box.prop(group, 'stiffness')
            box.prop(group, 'gravity')
        col.operator('hallway.apply_rig_settings')
        layout.label(text='Settings saved with the .blend')
        box = layout.box()
        box.label(text='Bone Collections')
        if hasattr(rig.data, 'collections'):
            for collection in rig.data.collections:
                if collection.get('hallway_role'):
                    box.prop(collection, 'is_visible', text=collection.name)
        box.operator('hallway.configure_rig', text='Refresh Bone Organization')


CLASSES = (HALLWAY_OT_ConfigureRig, HALLWAY_OT_ApplyRigSettings, HALLWAY_PT_Rig)


def register():
    from properties_hallway_rig import register as register_properties
    register_properties()
    import avatar_physics_preview
    avatar_physics_preview.register()
    for cls in CLASSES:
        if not getattr(cls, 'is_registered', False):
            bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        if getattr(cls, 'is_registered', False):
            bpy.utils.unregister_class(cls)
    from properties_hallway_rig import unregister as unregister_properties
    unregister_properties()
    import avatar_physics_preview
    avatar_physics_preview.unregister()
