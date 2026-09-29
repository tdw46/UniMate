"""Isolated vertical-bounce comparison using the installed BVT solver.

Pass source.blend output.json baseline|capsules|plane and an optional outward slope.
The plane variant is diagnostic only; production uses portable capsules.
"""
import bpy,sys,json,math,types,importlib
from pathlib import Path
from mathutils import Matrix,Vector,Quaternion
import numpy as np
if not bpy.app.background:raise RuntimeError('Use isolated background Blender')
sys.path.insert(0,str(Path(__file__).resolve().parent))
arguments=sys.argv[sys.argv.index('--')+1:]
source,output,variant=arguments[:3]
slope=float(arguments[3]) if len(arguments)>3 else .7
for repo in getattr(getattr(bpy.context.preferences,'extensions',None),'repos',[]):
 if repo.module=='user_default':repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
from properties_hallway_rig import register,initialize
register()
bpy.ops.wm.open_mainfile(filepath=str(Path(source).resolve()))
name='bl_ext.user_default.beyond_vrm_extension_suite';package=types.ModuleType(name);package.__path__=[str(Path.home()/'Documents/Blender/extensions/user_default/beyond_vrm_extension_suite')];sys.modules[name]=package
solver=importlib.import_module(name+'.VRM_SpringSimulation')
for cls in (solver.BVT_OT_SpringSimulationEngine,solver.BVT_OT_SetSpringSimulation,solver.BVT_OT_SetSpringLoopPhysics):bpy.utils.register_class(cls)
solver.register_runtime()
from avatar_physics_preview import set_simulation,background_step
from avatar_skirt_fit_io import rest_edit
from avatar_skirt_ceiling import plan_ceiling,install_skirt_ceiling
from avatar_vrm_colliders import add_group,call_operator
from avatar_mesh_invariant import snapshot,verify
from avatar_apparel_weights import weights
from avatar_colliders import rest_contacts
set_simulation(False)
rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE' and o.get('unimate_secondary_generator'));bpy.context.view_layer.objects.active=rig
if bpy.context.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
rig.animation_data_clear()
for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
rig.location=(0,0,0);bpy.context.view_layer.update()
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.find_armature()==rig];geometry=snapshot(meshes);binding={o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}
plans=plan_ceiling(rig,slope);sb=rig.data.vrm_addon_extension.spring_bone1
setup={}
if variant.startswith('capsules'):
 with rest_edit(rig):setup=install_skirt_ceiling(rig,slope)
 assert not rest_contacts(rig)['contacts']
elif variant.startswith('plane'):
 with rest_edit(rig):
  for plan in plans:
   call_operator('add_spring_bone1_collider',rig);c=sb.colliders[-1]
   c.node.bone_name=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones.hips.node.bone_name
   c.ui_collider_type='uiColliderTypePlane';c.reset_bpy_object(bpy.context,rig);bpy.context.view_layer.update()
   shape=c.extensions.vrmc_spring_bone_extended_collider.shape.plane
   shape.offset=plan['offset'];bpy.context.view_layer.update();shape.normal=plan['normal'];bpy.context.view_layer.update()
   normal=rig.data.bones[c.node.bone_name].matrix_local.to_3x3()@Vector(shape.normal)
   expected=rig.data.bones[c.node.bone_name].matrix_local.to_3x3()@Vector(plan['normal'])
   assert normal.dot(expected)>.99999,tuple(normal)
   assert (Vector(shape.offset)-Vector(plan['offset'])).length<1e-5
   g=add_group(rig,plan['name']);g.colliders.add().collider_uuid=c.uuid
   for s in sb.springs:
    if s.vrm_name in plan['springs']:s.collider_groups.add().collider_group_uuid=g.uuid
   setup=dict(normal=list(normal),offset=list(shape.offset))
# Isolate skirt runtime costs; hair does not interact with skirt collision groups.
for i in reversed(range(len(sb.springs))):
 if not sb.springs[i].vrm_name.startswith('Secondary_Skirt_'):sb.springs.remove(i)
springs=list(sb.springs);tips=[j.node.bone_name for s in springs for j in s.joints[1:]]
ceiling=plans[0]['height'];height=max(b.head_local.z for b in rig.data.bones)-min(b.head_local.z for b in rig.data.bones)
def positions():return np.array([rig.pose.bones[n].head[:] for n in tips])
rows=[]
for amplitude,period,leg in ((.2,24,0),(.3,12,0),(.3,8,30),(.45,6,30)):
 set_simulation(False)
 rig.location=(0,0,0)
 for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
 if leg:
  for n,axis in ((rig.data.vrm_addon_extension.vrm1.humanoid.human_bones.left_upper_leg.node.bone_name,(1,0,0)),(rig.data.vrm_addon_extension.vrm1.humanoid.human_bones.right_upper_leg.node.bone_name,(0,0,1))):
   pb=rig.pose.bones[n];pb.rotation_mode='QUATERNION';pb.rotation_quaternion=Quaternion(axis,math.radians(leg))
 bpy.context.view_layer.update();set_simulation(True)
 for _ in range(90):background_step(1/60)
 start=positions();maxabove=0.;h=[]
 for frame in range(period*6+180):
  offset=height*amplitude*(1-math.cos(math.tau*(frame+1)/period))*.5 if frame<period*6 else 0.
  rig.location.z=offset;bpy.context.view_layer.update();background_step(1/60);bpy.context.view_layer.update()
  current=positions();assert np.isfinite(current).all();h.append(current)
  maxabove=max(maxabove,float(np.max(current[:,2]-ceiling)))
 h=np.array(h);error=float(np.linalg.norm(h[-1]-start,axis=1).max())
 row=dict(amplitude_m=height*amplitude,cycle_frames=period,leg_degrees=leg,peak_above_ceiling_mm=maxabove*1000,final_above_ceiling=int(np.sum(h[-1,:,2]>ceiling)),return_error_mm=error*1000,hold_step_rms_mm=float(np.sqrt(np.mean(np.diff(h[-60:],axis=0)**2))*1000))
 rows.append(row);print('CASE',variant,json.dumps(row),flush=True)
set_simulation(False);rig.location=(0,0,0)
for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
verify(meshes,geometry);assert binding=={o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}
report=dict(variant=variant,outward_slope=slope,setup=setup,cases=rows,geometry_and_weights_unchanged=True)
Path(output).write_text(json.dumps(report,indent=2));print('CEILING_TEST_OK',variant,flush=True)
