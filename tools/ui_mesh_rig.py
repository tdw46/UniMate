"""Mesh-only automatic rigging UI; original rigs are never inference inputs."""
import json,re
from pathlib import Path
import bpy
from mesh_landmarks import jobs,blender_bridge


def runtime_root():
    return Path(bpy.utils.user_resource('DATAFILES'))/'hallway_rig'/'mesh_landmarks'


class HALLWAY_OT_PrepareMeshRig(jobs.WorkerOperator,bpy.types.Operator):
    bl_idname='hallway.prepare_mesh_rig'
    bl_label='Prepare Rigging Engine'
    action:bpy.props.EnumProperty(items=(('setup','Prepare Engine',''),('prepare','Download Model','')),default='setup')
    def execute(self,context):return self.start(context,runtime_root(),self.action)


class HALLWAY_OT_GenerateMeshRig(jobs.WorkerOperator,bpy.types.Operator):
    bl_idname='hallway.generate_mesh_rig'
    bl_label='Generate Rig in New Scene'
    bl_description='Predict humanoid joints from the selected meshes without using an existing skeleton or weights'
    bl_options={'REGISTER','UNDO'}
    @classmethod
    def poll(cls,context):return context.mode=='OBJECT' and context.window and any(o.type=='MESH' for o in context.selected_objects)
    def execute(self,context):
        from mesh_landmarks import skin_binding
        self._skin_options=dict(backend=context.scene.hallway_skinning_device,brand='Hallway Solver') if context.scene.hallway_skinning=='SOLVER' else None
        if self._skin_options and not skin_binding.available():
            self.report({'ERROR'},'Solver unavailable on this platform. Choose Blender Weighting.');return {'CANCELLED'}
        self._secondary=context.scene.hallway_rig_secondary
        self._breasts=context.scene.hallway_rig_breasts
        return self.start(context,runtime_root(),'infer',[o for o in context.selected_objects if o.type=='MESH'],context.scene.hallway_rig_device)
    def finish_generation(self,context,meshes,prediction):
        import rebuild_mmd_character as builder
        from avatar_springs import humanoid_roles,preserve_export_rest_frame
        from properties_hallway_rig import initialize
        from avatar_bone_collections import organize_bones
        def finish(rig,copies,report):
            ext=rig.data.vrm_addon_extension;ext.spec_version='1.0';preserve_export_rest_frame(rig)
            for role,name in dict(humanoid_roles(),upperChest='Chest',chest='UpperSpine').items():
                node=getattr(ext.vrm1.humanoid.human_bones,re.sub(r'(?<!^)(?=[A-Z])','_',role).lower(),None)
                if node and name in rig.data.bones:node.node.bone_name=name
            if self._breasts:
                from avatar_breasts import generate_breasts, install_physics
                report['breasts']=generate_breasts(context,rig,copies,prefix='Hallway_')
                install_physics(rig,report['breasts'],copies,bone_prefix='Hallway_')
            initialize(rig);organize_bones(rig)
            text=bpy.data.texts.new('Automatic Rigging Report');text.write(json.dumps(report,indent=2,default=str))
            rig['mesh_only_rig_report']=text.name
            bpy.ops.object.select_all(action='DESELECT');rig.select_set(True);context.view_layer.objects.active=rig
        blender_bridge.build(context,meshes,prediction,builder,prefix='Hallway',secondary=self._secondary,finalize=finish,binding=self._binding)



class HALLWAY_PT_MeshRig(bpy.types.Panel):
    bl_idname='HALLWAY_PT_MeshRig';bl_label='Automatic Rigging'
    bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='Hallway'
    def draw(self,context):
        layout=self.layout
        if not hasattr(bpy.types.Armature,'vrm_addon_extension'):
            layout.label(text='Enable VRM Add-on to create the rig',icon='INFO')
            layout.operator('wm.url_open',text='Download VRM Add-on').url='https://extensions.blender.org/add-ons/vrm/'
            return
        if jobs.ACTIVE:
            layout.label(text='Rigging task running. Esc to cancel.',icon='TIME');return
        status=jobs.state(runtime_root())
        if not status.get('ready'):
            layout.label(text='Set up local automatic rigging once.',icon='INFO')
            layout.operator('hallway.prepare_mesh_rig').action='setup';return
        if not status.get('models_ready'):
            layout.label(text='Download the model (about 907 MB).',icon='INFO')
            layout.operator('hallway.prepare_mesh_rig',text='Download Rigging Model').action='prepare';return
        layout.label(text='Select the complete character meshes.')
        layout.label(text='Use a neutral standing pose, facing -Y.')
        layout.label(text='Joints are predicted from geometry only.',icon='INFO')
        layout.label(text='Review placement and bending before export.')
        if 'mps' in status.get('devices',[]):layout.prop(context.scene,'hallway_rig_device')
        layout.prop(context.scene,'hallway_skinning',expand=True)
        if context.scene.hallway_skinning=='SOLVER':
            from mesh_landmarks import skin_binding
            if skin_binding.available():layout.prop(context.scene,'hallway_skinning_device',expand=True)
            else:layout.label(text='Solver unavailable. Choose Blender Weighting.',icon='INFO')
        layout.prop(context.scene,'hallway_rig_secondary')
        layout.prop(context.scene,'hallway_rig_breasts')
        row=layout.row();row.scale_y=1.8;row.operator('hallway.generate_mesh_rig',icon='ARMATURE_DATA')


CLASSES=(HALLWAY_OT_PrepareMeshRig,HALLWAY_OT_GenerateMeshRig,HALLWAY_PT_MeshRig)
def register():
    for cls in CLASSES:bpy.utils.register_class(cls)
    bpy.types.Scene.hallway_rig_device=bpy.props.EnumProperty(name='Processing',items=(('cpu','CPU','Portable local processing'),('mps','Metal','Apple Silicon GPU')),default='cpu')
    bpy.types.Scene.hallway_skinning=bpy.props.EnumProperty(name='Skin Weights',items=(('SOLVER','Hallway Solver','Use the bundled independent skinning solver'),('BLENDER','Blender Weighting','Use Blender heat weighting and cleanup')),default='SOLVER')
    bpy.types.Scene.hallway_skinning_device=bpy.props.EnumProperty(name='Skinning Device',items=(('auto','Automatic','Prefer Metal or Vulkan when available; otherwise use CPU',2),('cpu','CPU','Portable local processing',0),('gpu','GPU','Use Metal or the native Windows GPU backend',1)),default='auto')
    bpy.types.Scene.hallway_rig_secondary=bpy.props.BoolProperty(name='Generate Hair and Skirt Bones',default=True)
    bpy.types.Scene.hallway_rig_breasts=bpy.props.BoolProperty(name='Generate Breast Bones',default=False)
def unregister():
    jobs.cancel_all()
    for cls in reversed(CLASSES):
        if getattr(cls,'is_registered',False):bpy.utils.unregister_class(cls)
    for name in ('hallway_rig_device','hallway_rig_secondary','hallway_skinning','hallway_skinning_device','hallway_rig_breasts'):
        if hasattr(bpy.types.Scene,name):delattr(bpy.types.Scene,name)
