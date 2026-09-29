"""Isolated BVT hair collider ablation: source.blend report.json variant.

Variants: baseline, none, head, nohead, refit. No live scene writes.
"""
import bpy,sys,json,types,importlib,math
from pathlib import Path
from mathutils import Matrix,Quaternion
import numpy as np
if not bpy.app.background:raise RuntimeError('Use isolated background Blender')
sys.path.insert(0,str(Path(__file__).resolve().parent))
source,output,variant=sys.argv[sys.argv.index('--')+1:]
output=Path(output)
for repo in getattr(getattr(bpy.context.preferences,'extensions',None),'repos',[]):
 if repo.module=='user_default':repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
bpy.ops.wm.open_mainfile(filepath=str(Path(source).resolve()))
name='bl_ext.user_default.beyond_vrm_extension_suite';package=types.ModuleType(name);package.__path__=[str(Path.home()/'Documents/Blender/extensions/user_default/beyond_vrm_extension_suite')];sys.modules[name]=package
solver=importlib.import_module(name+'.VRM_SpringSimulation')
for cls in (solver.BVT_OT_SpringSimulationEngine,solver.BVT_OT_SetSpringSimulation,solver.BVT_OT_SetSpringLoopPhysics):bpy.utils.register_class(cls)
solver.register_runtime()
from avatar_physics_preview import set_simulation,background_step
set_simulation(False)
r=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE' and o.get('unimate_secondary_generator'));bpy.context.view_layer.objects.active=r
if bpy.context.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
r.animation_data_clear();r.matrix_world=Matrix.Identity(4)
for pb in r.pose.bones:pb.matrix_basis=Matrix.Identity(4)
sb=r.data.vrm_addon_extension.spring_bone1
if variant=='refit':
 from avatar_colliders import rebuild_colliders
 meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.find_armature()==r]
 fit=rebuild_colliders(r,meshes,collider_roles=('hair',))
 print('HAIR_REFIT',json.dumps(fit),flush=True)
for i in reversed(range(len(sb.springs))):
 if not sb.springs[i].vrm_name.startswith('Secondary_Hair_'):sb.springs.remove(i)
cm={c.uuid:c for c in sb.colliders};gm={g.uuid:g for g in sb.collider_groups}
for s in sb.springs:
 for ref in s.collider_groups:
  g=gm[ref.collider_group_uuid]
  for i in reversed(range(len(g.colliders))):
   c=cm[g.colliders[i].collider_uuid]
   remove=variant=='none' or (variant=='head' and c.node.bone_name!='Head') or (variant=='nohead' and c.node.bone_name=='Head')
   if remove:g.colliders.remove(i)
tips=[j.node.bone_name for s in sb.springs for j in s.joints[1:]]
pairs=[];seen={}
for s in sb.springs:
 key=tuple(tuple(round(x,6) for x in r.data.bones[j.node.bone_name].head_local) for j in s.joints)
 if key in seen:pairs.append((seen[key].joints[-1].node.bone_name,s.joints[-1].node.bone_name))
 else:seen[key]=s
counts={};original=solver._CapsuleWorldCollider.calculate_collision
# Instrument contact occurrences without changing any collision response.
def collision(self,target,radius,*args):
 normal,distance=original(self,target,radius,*args)
 if distance<0:counts[self.radius]=counts.get(self.radius,0)+1
 return normal,distance
solver._CapsuleWorldCollider.calculate_collision=collision
rows=[]
for scenario in ('translate','headturn'):
 set_simulation(False);r.location=(0,0,0)
 for pb in r.pose.bones:pb.matrix_basis=Matrix.Identity(4)
 bpy.context.view_layer.update();set_simulation(True)
 for _ in range(60):background_step(1/60)
 h=[];pairdiff=[];counts.clear()
 for frame in range(300):
  t=frame/60;envelope=math.sin(math.pi*frame/179)**2 if frame<180 else 0.
  if scenario=='translate':r.location=(.10*math.sin(2*math.pi*t*1.1)*envelope,0.,.07*math.sin(2*math.pi*t*.7)*envelope)
  else:
   pb=r.pose.bones['Head'];pb.rotation_mode='QUATERNION';pb.rotation_quaternion=Quaternion((0,1,0),.45*math.sin(2*math.pi*t*.8)*envelope)
  bpy.context.view_layer.update();background_step(1/60);bpy.context.view_layer.update()
  h.append([tuple(r.pose.bones[n].head) for n in tips])
  pairdiff.append(max(((r.pose.bones[a].head-r.pose.bones[b].head).length for a,b in pairs),default=0))
 h=np.array(h);jerk=np.diff(h[:180],n=3,axis=0);score=np.sqrt(np.mean(jerk**2,axis=(0,2)))
 row=dict(scenario=scenario,motion_jerk_rms_mm=float(np.sqrt(np.mean(jerk**2))*1000),hold_step_rms_mm=float(np.sqrt(np.mean(np.diff(h[-60:],axis=0)**2))*1000),pair_divergence_mm=max(pairdiff)*1000,contacts_by_radius=dict(counts),worst=[dict(bone=tips[i],jerk_mm=float(score[i]*1000)) for i in np.argsort(score)[-8:][::-1]])
 rows.append(row);print('HAIR_CASE',variant,json.dumps(row),flush=True)
 np.save(output.with_name(output.stem+'_'+scenario+'.npy'),h)
set_simulation(False)
output.write_text(json.dumps(dict(variant=variant,cases=rows),indent=2));print('HAIR_ABLATION_DONE',variant,flush=True)
