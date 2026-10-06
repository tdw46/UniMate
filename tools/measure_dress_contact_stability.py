"""Isolated native-contact BVT benchmark; no renderer or solver modifications.

Usage: blender --background --factory-startup --python this.py -- source.blend
output_dir [--rebuild-contacts]. Reports 60 Hz discrete acceleration of joint
endpoints and sampled evaluated-mesh triangle intersections. Counts are not
penetration volume; this diagnostic does not prove all-pose collision freedom.
"""
import bpy,sys,math,json,importlib,types
from pathlib import Path
import numpy as np
from mathutils import Matrix,Quaternion,Vector
from mathutils.bvhtree import BVHTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
import argparse
if not bpy.app.background:raise RuntimeError('Use isolated background Blender')
parser=argparse.ArgumentParser()
parser.add_argument('source');parser.add_argument('output');parser.add_argument('--rebuild-contacts',action='store_true')
parser.add_argument('--ramp-frames',type=int,default=45)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
variant='rebuilt' if args.rebuild_contacts else 'baseline'
ramp=args.ramp_frames
if ramp<3:raise ValueError('At least three ramp frames required')
hold_end=ramp+90;return_end=hold_end+ramp;total=return_end+60
for repo in bpy.context.preferences.extensions.repos:
 folder=Path.home()/'Documents/Blender/extensions'/repo.module
 if folder.is_dir():repo.use_custom_directory=True;repo.custom_directory=str(folder)
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
import properties_hallway_rig as props
props.register()
bpy.ops.wm.open_mainfile(filepath=str(Path(args.source).resolve()))
bpy.context.scene.render.engine='BLENDER_WORKBENCH'
name='bl_ext.user_default.beyond_vrm_extension_suite';package=types.ModuleType(name);package.__path__=[str(Path.home()/'Documents/Blender/extensions/user_default/beyond_vrm_extension_suite')];sys.modules[name]=package
solver=importlib.import_module(name+'.VRM_SpringSimulation')
for cls in (solver.BVT_OT_SpringSimulationEngine,solver.BVT_OT_SetSpringSimulation,solver.BVT_OT_SetSpringLoopPhysics):bpy.utils.register_class(cls)
solver.register_runtime()
from avatar_physics_preview import set_simulation,background_step
from avatar_apparel_weights import weights
set_simulation(False)
rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE' and o.get('unimate_secondary_generator'));bpy.context.view_layer.objects.active=rig
if rig.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
rig.animation_data_clear()
for p in rig.pose.bones:p.matrix_basis=Matrix.Identity(4)
bpy.context.view_layer.update()
sb=rig.data.vrm_addon_extension.spring_bone1
# Independent hair simulations do not contribute to the dress benchmark.
for i in reversed(range(len(sb.springs))):
 if sb.springs[i].vrm_name.startswith('Secondary_Hair_'):sb.springs.remove(i)
if args.rebuild_contacts:
 from avatar_directional_contacts import install_skirt_contact_rig
 install_skirt_contact_rig(rig,[o for o in bpy.context.scene.objects if o.type=='MESH' and o.find_armature()==rig])

meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.find_armature()==rig]
parts=[]
hum=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
leg_names={getattr(hum,s+'_'+p).node.bone_name for s in ('left','right') for p in ('upper_leg','lower_leg')}
left_thigh=hum.left_upper_leg.node.bone_name;left_calf=hum.left_lower_leg.node.bone_name
for o in meshes:
 o.data.calc_loop_triangles();rows=[weights(o,v.index) for v in o.data.vertices]
 dress={i for i,w in enumerate(rows) if any(n.startswith(('Secondary_Skirt_','Secondary_SkirtSupport_')) and v>.05 for n,v in w.items())}
 skin={i for i,w in enumerate(rows) if i not in dress and sum(w.get(n,0) for n in leg_names)>.4}
 st=[tuple(t.vertices) for t in o.data.loop_triangles if all(i in dress for i in t.vertices)]
 lt=[tuple(t.vertices) for t in o.data.loop_triangles if all(i in skin for i in t.vertices)]
 if st or lt:parts.append((o,st,lt))
if not any(a for _,a,_ in parts) or not any(b for _,_,b in parts):
 raise ValueError('Fixture needs generated dress and leg-weighted body triangles')

def intersections(details=False):
 vs=[];st=[];lt=[]
 for o,a,b in parts:
  e=o.evaluated_get(bpy.context.evaluated_depsgraph_get());m=e.to_mesh();off=len(vs)
  vs.extend(o.matrix_world@v.co for v in m.vertices);e.to_mesh_clear()
  st.extend(tuple(i+off for i in t) for t in a);lt.extend(tuple(i+off for i in t) for t in b)
 pairs=BVHTree.FromPolygons(vs,st,all_triangles=True).overlap(BVHTree.FromPolygons(vs,lt,all_triangles=True))
 if details:
  rows=[dict(dress_center=list(sum((vs[v] for v in st[a]),Vector())/3),body_center=list(sum((vs[v] for v in lt[b]),Vector())/3)) for a,b in pairs]
  (out/(variant+'_surface_details.json')).write_text(json.dumps(rows))
 return len(pairs)
names=[j.node.bone_name for s in sb.springs for j in s.joints[1:]]
results=[]
for case,(x,z,knee) in enumerate(((65,0,-80),(0,35,0),(60,30,-65))):
 set_simulation(False)
 for p in rig.pose.bones:p.matrix_basis=Matrix.Identity(4)
 bpy.context.view_layer.update();set_simulation(True)
 for _ in range(35):background_step(1/60)
 history=[];hits=[]
 for frame in range(total):
  # Smooth rise, steady hold, smooth return, steady rest.
  t=.5-.5*math.cos(math.pi*frame/(ramp-1)) if frame<ramp else 1. if frame<hold_end else .5+.5*math.cos(math.pi*(frame-hold_end)/(ramp-1)) if frame<return_end else 0.
  p=rig.pose.bones[left_thigh];p.rotation_mode='QUATERNION';p.rotation_quaternion=Quaternion((1,0,0),math.radians(x)*t)@Quaternion((0,0,1),math.radians(z)*t)
  p=rig.pose.bones[left_calf];p.rotation_mode='QUATERNION';p.rotation_quaternion=Quaternion((1,0,0),math.radians(knee)*t)
  bpy.context.view_layer.update();background_step(1/60);bpy.context.view_layer.update()
  history.append([tuple(rig.pose.bones[n].head) for n in names])
  if frame in (ramp//2,ramp-1,hold_end-45,hold_end-1,hold_end+ramp//2,return_end-1,total-1):hits.append(intersections(details=case==0 and frame==hold_end-1))
 h=np.array(history);assert np.isfinite(h).all()
 quiet=h[hold_end-45:hold_end];jerks=np.linalg.norm(np.diff(h,n=2,axis=0),axis=2)*1000
 q=np.linalg.norm(np.diff(quiet,axis=0),axis=2)*1000
 worst=np.unravel_index(np.argmax(jerks),jerks.shape)
 row=dict(case=[x,z,knee],hits=hits,hold_step_max_mm=float(q.max()),hold_step_rms_mm=float(np.sqrt(np.mean(q*q))),acc_p99_mm=float(np.quantile(jerks,.99)),acc_max_mm=float(jerks.max()),worst=names[worst[1]])
 results.append(row);np.save(out/(variant+'_'+str(case)+'.npy'),h);print('BENCH',variant,json.dumps(row),flush=True)
set_simulation(False)
(out/(variant+'.json')).write_text(json.dumps(results,indent=2))
