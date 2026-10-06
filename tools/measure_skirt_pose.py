"""Isolated captured-pose sweep using the installed BVT solver.

Pass source.blend output.json pose.json [guard_width=1] [fan_degrees=0] [lower_thigh_support=0]
[broad_dress_roots=0] [smooth_fallback=0] [drag_override] [midline_contacts=0] [midline_transverse=0] [case=all].
Reports surface intersections
and collider depth; does not infer zero mesh clipping from collider clearance.
"""
import bpy,sys,json,math,types,importlib
from pathlib import Path
from mathutils import Matrix,Vector,Quaternion
import numpy as np
if not bpy.app.background:raise RuntimeError('Use isolated background Blender')
sys.path.insert(0,str(Path(__file__).resolve().parent))
arguments=sys.argv[sys.argv.index('--')+1:]
source,output,pose_path=arguments[:3]
width=float(arguments[3]) if len(arguments)>3 else 1.
fan=float(arguments[4]) if len(arguments)>4 else 0.
thigh=bool(int(arguments[5])) if len(arguments)>5 else False
broad=bool(int(arguments[6])) if len(arguments)>6 else False
smooth=bool(int(arguments[7])) if len(arguments)>7 else False
drag=float(arguments[8]) if len(arguments)>8 else None
midline=bool(int(arguments[9])) if len(arguments)>9 else False
transverse=bool(int(arguments[10])) if len(arguments)>10 else False
only_case=arguments[11] if len(arguments)>11 and arguments[11]!='all' else None
probe_config=json.loads(Path(arguments[12]).read_text()) if len(arguments)>12 and arguments[12]!='-' else []
surface_scope=arguments[13] if len(arguments)>13 else 'skirt'
if surface_scope not in ('skirt','waist'):raise ValueError('Unknown surface scope')
if only_case not in (None,'smooth','quick','cycle','direct'):raise ValueError('Unknown pose case')
variant=f'width_{width}_fan_{fan}_thigh_{thigh}_broad_{broad}_smooth_{smooth}_drag_{drag}_midline_{midline}_transverse_{transverse}'
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

from avatar_directional_contacts import install_directional_contacts
with rest_edit(rig):
 if width!=1. or fan or thigh or broad or smooth or midline or transverse:
  install_directional_contacts(rig,meshes,radius_factor=1.,fan_degrees=fan,rest_envelope=True,side_scoped=True,opposite_fallback=True,pelvis_support=True,opposite_full_only=True,guard_width=width,lower_thigh_support=thigh,broad_dress_roots=broad,smooth_fallback=smooth,midline_contacts=midline,midline_transverse=transverse)
from avatar_colliders import segment_distance
from mathutils.bvhtree import BVHTree
sb=rig.data.vrm_addon_extension.spring_bone1
with rest_edit(rig):
 assert not rest_contacts(rig)['contacts']
 native={c.uuid:(c.node.bone_name,Vector(c.shape.capsule.offset),Vector(c.shape.capsule.tail),c.shape.capsule.radius) for c in sb.colliders}
for i in reversed(range(len(sb.springs))):
 if not sb.springs[i].vrm_name.startswith(('Secondary_Skirt_','Secondary_HipSkirt_')):sb.springs.remove(i)
springs=list(sb.springs);tips=[j.node.bone_name for s in springs for j in s.joints[1:]]
if drag is not None:
 for s in springs:
  for j in s.joints:j.drag_force=drag
groups={g.uuid:[r.collider_uuid for r in g.colliders] for g in sb.collider_groups}
hum=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
legs={getattr(hum,s+'_'+part).node.bone_name for s in ('left','right') for part in ('upper_leg','lower_leg')}
parts=[]
for obj in meshes:
 obj.data.calc_loop_triangles();ws=binding[obj.name]
 skirt={i for i,w in enumerate(ws) if sum(v for n,v in w.items() if n.startswith('Secondary_Skirt_'))>.05}
 skin={i for i,w in enumerate(ws) if i not in skirt and sum(v for n,v in w.items() if n in legs)>.4}
 if surface_scope=='waist':
  from avatar_pelvis_binding import landmarks
  *_,waist_high=landmarks(rig)
  waist_low=min(b.head_local.z for b in rig.data.bones if 'hallway_garment_top' in b)-.05
  m=rig.matrix_world.inverted()@obj.matrix_world
  garment_slots={f.material_index for f in obj.data.polygons if any(i in skirt for i in f.vertices)}
  skirt={i for f in obj.data.polygons if f.material_index in garment_slots for i in f.vertices if waist_low<=(m@obj.data.vertices[i].co).z<=waist_high}
  skin=set()
  for f in obj.data.polygons:
   mat=obj.data.materials[f.material_index];label=mat.name.lower() if mat else ''
   if any(t in label for t in ('skin','body')) and not any(t in label for t in ('cloth','hair','shoe')):skin.update(f.vertices)
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

# Optional selected-vertex probes measure mesh behavior, not just spring tips.
probes=[]
for entry in probe_config:
 obj=bpy.data.objects[entry['object']];center=int(entry['index'])
 if obj not in meshes:raise ValueError('Probe must belong to the tested rig')
 neighbors={j for e in obj.data.edges if center in e.vertices for j in e.vertices}
 ids=[center]+sorted(neighbors-{center})
 triangles=[tuple(t.vertices) for t in obj.data.loop_triangles if center in t.vertices]
 probes.append((obj,ids,triangles))
def probe_state():
 points=[];creases=[]
 for obj,ids,triangles in probes:
  ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh()
  try:
   coords={i:obj.matrix_world@mesh.vertices[i].co for i in ids}
   points.extend(coords[i][:] for i in ids)
   normals=[(coords[b]-coords[a]).cross(coords[c]-coords[a]).normalized() for a,b,c in triangles]
   angles=[math.degrees(normals[i].angle(normals[j])) for i in range(len(triangles)) for j in range(i)
           if len(set(triangles[i])&set(triangles[j]))==2 and normals[i].length>.5 and normals[j].length>.5]
   creases.append(max(angles,default=0.))
  finally:ev.to_mesh_clear()
 return points,creases

pose=json.loads(Path(pose_path).read_text())
goals={n:Matrix(p['matrix']).to_quaternion() for n,p in pose.items() if n in rig.pose.bones and not n.startswith('Secondary_')}
rows=[]
for case in ((only_case,) if only_case else ('smooth','quick','cycle','direct')):
 set_simulation(False)
 for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
 if case=='direct':
  for n,q in goals.items():
   pb=rig.pose.bones[n];pb.rotation_mode='QUATERNION';pb.rotation_quaternion=q
 bpy.context.view_layer.update();set_simulation(True)
 for _ in range(90):background_step(1/60)
 start=positions();history=[];samples=[];mesh_history=[];held_creases=[]
 ramp=90 if case=='smooth' else 8
 for frame in range(360):
  t=(1-math.cos(math.pi*min(frame,ramp-1)/(ramp-1)))/2 if frame<ramp else 1. if frame<240 else (1+math.cos(math.pi*(frame-240)/59))/2 if frame<300 else 0.
  if case=='direct' and frame<240:t=1.
  if case=='cycle':t=(.5-.5*math.cos(math.tau*frame/90)) if frame<270 else 0.
  for n,q in goals.items():
   pb=rig.pose.bones[n];pb.rotation_mode='QUATERNION';pb.rotation_quaternion=Quaternion().slerp(q,t)
  bpy.context.view_layer.update();background_step(1/60);bpy.context.view_layer.update()
  current=positions();assert np.isfinite(current).all();history.append(current)
  if probes:
   mesh_points,creases=probe_state();mesh_history.append(mesh_points)
   if frame==239:held_creases=creases
  if frame in (7,44,89,179,239,299,359):samples.append(dict(frame=frame,triangle_intersections=surfaces(),collider_penetration=contacts()))
 h=np.array(history)
 jerk=np.linalg.norm(np.diff(h,n=3,axis=0),axis=2)*1000
 row=dict(case=case,samples=samples,held_joint_positions={name:list(map(float,h[239,i])) for i,name in enumerate(tips)},hold_step_rms_mm=float(np.sqrt(np.mean(np.diff(h[180:240],axis=0)**2))*1000) if case!='cycle' else None,
          rms_frame_jerk_mm=float(np.sqrt(np.mean(jerk**2))),peak_frame_jerk_mm=float(jerk.max()),
          peak_jerk_bone=tips[np.unravel_index(jerk.argmax(),jerk.shape)[1]],
          return_error_mm=float(np.linalg.norm(h[-1]-start,axis=1).max())*1000 if case!='direct' else None)
 if probes:
  mh=np.array(mesh_history);mj=np.linalg.norm(np.diff(mh,n=3,axis=0),axis=2)*1000
  row['mesh_probe']=dict(held_max_dihedral_degrees=held_creases,rms_frame_jerk_mm=float(np.sqrt(np.mean(mj**2))),peak_frame_jerk_mm=float(mj.max()),hold_step_rms_mm=float(np.sqrt(np.mean(np.diff(mh[180:240],axis=0)**2))*1000))
 rows.append(row);print('POSE_CASE',variant,json.dumps(row),flush=True)
set_simulation(False)
verify(meshes,geometry);assert binding=={o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}
report=dict(variant=variant,surface_scope=surface_scope,geometry_and_weights_unchanged=True,cases=rows)
Path(output).write_text(json.dumps(report,indent=2));print('POSE_MOTION_OK',variant,flush=True)
