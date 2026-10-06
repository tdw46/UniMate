"""Isolated knee-bend and rapid dress motion using the installed BVT solver.

Pass source.blend output.json enabled|disabled. Reports surface intersections
and collider depth; does not infer zero mesh clipping from collider clearance.
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
target={p.name:p.matrix_basis.copy() for p in rig.pose.bones if not p.name.startswith('Secondary_')}
rig.animation_data_clear()
for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
rig.location=(0,0,0);bpy.context.view_layer.update()
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.find_armature()==rig];geometry=snapshot(meshes);binding={o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}

from avatar_colliders import segment_distance
from mathutils.bvhtree import BVHTree
sb=rig.data.vrm_addon_extension.spring_bone1
with rest_edit(rig):
 assert not rest_contacts(rig)['contacts']
 native={c.uuid:(c.node.bone_name,Vector(c.shape.capsule.offset),Vector(c.shape.capsule.tail),c.shape.capsule.radius) for c in sb.colliders}
for i in reversed(range(len(sb.springs))):
 if not sb.springs[i].vrm_name.startswith('Secondary_Skirt_'):sb.springs.remove(i)
springs=list(sb.springs);tips=[j.node.bone_name for s in springs for j in s.joints[1:]]
groups={g.uuid:[r.collider_uuid for r in g.colliders] for g in sb.collider_groups}
hum=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
legs={getattr(hum,s+'_'+part).node.bone_name for s in ('left','right') for part in ('upper_leg','lower_leg')}
parts=[]
for obj in meshes:
 obj.data.calc_loop_triangles();ws=binding[obj.name]
 skirt={i for i,w in enumerate(ws) if sum(v for n,v in w.items() if n.startswith('Secondary_Skirt_'))>.05}
 skin={i for i,w in enumerate(ws) if i not in skirt and sum(v for n,v in w.items() if n in legs)>.4}
 st=[tuple(t.vertices) for t in obj.data.loop_triangles if all(i in skirt for i in t.vertices)]
 lt=[tuple(t.vertices) for t in obj.data.loop_triangles if all(i in skin for i in t.vertices)]
 if st or lt:parts.append((obj,st,lt))
def surfaces():
 vertices=[];st=[];lt=[]
 for obj,garment,body in parts:
  ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh();n=len(vertices)
  vertices.extend(obj.matrix_world@v.co for v in mesh.vertices);ev.to_mesh_clear()
  st.extend(tuple(i+n for i in tri) for tri in garment);lt.extend(tuple(i+n for i in tri) for tri in body)
 return len(BVHTree.FromPolygons(vertices,st,all_triangles=True).overlap(BVHTree.FromPolygons(vertices,lt,all_triangles=True)))
def contacts():
 depths={'leg':[],'roof':[],'other':[]}
 for s in springs:
  for h,t in zip(s.joints,s.joints[1:]):
   p=rig.pose.bones[t.node.bone_name].head
   for ref in s.collider_groups:
    for uuid in groups[ref.collider_group_uuid]:
     bone,a,b,r=native[uuid];m=rig.pose.bones[bone].matrix
     depth=r+h.hit_radius-segment_distance(p,m@a,m@b)
     kind='leg' if bone in legs else 'roof' if any(g.vrm_name.startswith('Secondary_SkirtCeiling_') and g.uuid==ref.collider_group_uuid for g in sb.collider_groups) else 'other'
     if depth>1e-4:depths[kind].append(depth*1000)
 return {k:dict(count=len(v),max_mm=max(v,default=0.)) for k,v in depths.items()}
def positions():return np.array([rig.pose.bones[n].head[:] for n in tips])

followers=[p for p in rig.pose.bones if p.bone.get('hallway_dress_lower_follow')]
assert followers,'No lower dress followers to measure'
if variant=='disabled':
 for pb in followers:pb.constraints[0].influence=0.
rows=[]
for case in ('knee_hold','combined_hold','rapid_reversals'):
 set_simulation(False)
 for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
 bpy.context.view_layer.update();set_simulation(True)
 for _ in range(90):background_step(1/60)
 start=positions();history=[];samples=[]
 for frame in range(420):
  t=(1-math.cos(math.pi*frame/89))/2 if frame<90 else 1. if frame<240 else (1+math.cos(math.pi*(frame-240)/59))/2 if frame<300 else 0.
  bend=70*t;upper=0;side=0
  if case=='combined_hold':bend=90*t;upper=30*t;side=25*t
  if case=='rapid_reversals' and frame<120:
   bend=80*(.5-.5*math.cos(math.tau*frame/12));upper=30*math.sin(math.tau*frame/12);side=20*math.sin(math.tau*frame/24)
  elif case=='rapid_reversals':bend=upper=side=0.
  for n,rotation in ((hum.left_upper_leg.node.bone_name,Quaternion((1,0,0),math.radians(upper))@Quaternion((0,0,1),math.radians(side))),
                     (hum.left_lower_leg.node.bone_name,Quaternion((1,0,0),math.radians(bend)))):
   pb=rig.pose.bones[n];pb.rotation_mode='QUATERNION';pb.rotation_quaternion=rotation
  bpy.context.view_layer.update();background_step(1/60);bpy.context.view_layer.update()
  current=positions();assert np.isfinite(current).all();history.append(current)
  if frame in (44,89,179,239,299,419):samples.append(dict(frame=frame,triangle_intersections=surfaces(),collider_penetration=contacts()))
 h=np.array(history)
 row=dict(case=case,samples=samples,hold_step_rms_mm=float(np.sqrt(np.mean(np.diff(h[180:240],axis=0)**2))*1000),return_error_mm=float(np.linalg.norm(h[-1]-start,axis=1).max())*1000)
 rows.append(row);print('DRESS_CASE',variant,json.dumps(row),flush=True)
set_simulation(False)
verify(meshes,geometry);assert binding=={o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}
report=dict(variant=variant,geometry_and_weights_unchanged=True,cases=rows)
Path(output).write_text(json.dumps(report,indent=2));print('DRESS_MOTION_OK',variant,flush=True)
