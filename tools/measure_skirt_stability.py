"""Isolated BVT motion/hold/return diagnostic for skirt side ownership and chatter."""
import argparse,json,math,sys,time
from pathlib import Path
import bpy
if not bpy.app.background:raise RuntimeError("Run stability measurements in an isolated background Blender")
import numpy as np
from mathutils import Matrix,Quaternion,Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import intersect_line_line
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_physics_preview import set_simulation,background_step
from avatar_colliders import segment_distance,spring_samples,plan_colliders
from avatar_vrm_colliders import add_capsule,add_group,call_operator
from avatar_apparel_weights import weights
from properties_hallway_rig import initialize
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');p.add_argument('--variant',default='baseline');p.add_argument('--full',action='store_true');p.add_argument('--stress',action='store_true');p.add_argument('--loop-motion',action='store_true');p.add_argument('--yaw',action='store_true');p.add_argument('--bvt-runtime-only',action='store_true');p.add_argument('--follow',type=float,nargs='+');args=p.parse_args(sys.argv[sys.argv.index('--')+1:])
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
for repo in bpy.context.preferences.extensions.repos:
    if repo.module=='user_default':repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
bpy.ops.wm.open_mainfile(filepath=str(Path(args.source).resolve()))
if args.bvt_runtime_only:
    # Isolated tests can load the installed, unmodified solver independently
    # when unrelated BVT UI development prevents full addon registration.
    import importlib,types
    name='bl_ext.user_default.beyond_vrm_extension_suite'
    package=types.ModuleType(name)
    package.__path__=[str(Path.home()/'Documents/Blender/extensions/user_default/beyond_vrm_extension_suite')]
    sys.modules[name]=package
    solver=importlib.import_module(name+'.VRM_SpringSimulation')
    for cls in (solver.BVT_OT_SpringSimulationEngine,solver.BVT_OT_SetSpringSimulation,solver.BVT_OT_SetSpringLoopPhysics):bpy.utils.register_class(cls)
    solver.register_runtime()
else:
    bpy.ops.preferences.addon_enable(module='bl_ext.user_default.beyond_vrm_extension_suite')
set_simulation(False)
rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE' and o.get('unimate_secondary_generator'));bpy.context.view_layer.objects.active=rig
if rig.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
keep={rig,*rig.children_recursive}
for obj in list(bpy.data.objects):
    if obj not in keep:bpy.data.objects.remove(obj,do_unlink=True)
scene=bpy.context.scene
rig.animation_data_clear()
for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
bpy.context.view_layer.update()
settings=initialize(rig);sb=rig.data.vrm_addon_extension.spring_bone1
hum=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
legs=[getattr(hum,s+'_upper_leg').node.bone_name for s in ('left','right')]
leg_names={getattr(hum,s+'_'+n).node.bone_name for s in ('left','right') for n in ('upper_leg','lower_leg')}
meshes=[o for o in rig.children if o.type=='MESH']
height=max(b.head_local.z for b in rig.data.bones)-min(b.head_local.z for b in rig.data.bones)
margin=height*.0015
samples=spring_samples(rig,'skirt')
parts=[];garment=[]
for obj in meshes:
    obj.data.calc_loop_triangles();ws=[weights(obj,v.index) for v in obj.data.vertices]
    skirt={i for i,w in enumerate(ws) if sum(v for n,v in w.items() if n.startswith('Secondary_Skirt_'))>.05}
    skin={i for i,w in enumerate(ws) if i not in skirt and sum(v for n,v in w.items() if n in leg_names)>.4}
    st=[tuple(t.vertices) for t in obj.data.loop_triangles if all(i in skirt for i in t.vertices)]
    lt=[tuple(t.vertices) for t in obj.data.loop_triangles if all(i in skin for i in t.vertices)]
    if st or lt:parts.append((obj,st,lt))
    tr=rig.matrix_world.inverted()@obj.matrix_world
    garment.extend((tr@obj.data.vertices[i].co,0.,'surface') for i in skirt)
from avatar_directional_contacts import install_directional_contacts,skirt_owner_leg
from avatar_mesh_invariant import snapshot,verify
geometry=snapshot(meshes)
settings.follow_groups['Skirt'].influence=.55;settings.spring_groups['Skirt'].drag=.4
if args.variant.startswith('continuous'):
 from avatar_continuous_contacts import merge_contact_segments
 from avatar_contact_colliders import install_contact_colliders
 from avatar_skirt_fit_io import rest_edit
 with rest_edit(rig):
  merge_contact_segments(rig);install_contact_colliders(rig,meshes)
if args.variant!='baseline':
 from avatar_skirt_fit_io import rest_edit
 with rest_edit(rig):
  setup=install_directional_contacts(rig,meshes,1. if args.variant in ('pelvisflat','pelvisdetail','pelvishybrid','compactown') else .5,0.,rest_envelope=True,side_scoped=True,smooth_fallback=args.variant in ('smooth','soft','physical','physicalsoft','pelvis','pelvisflat'),opposite_fallback=args.variant.startswith('continuous') or args.variant.startswith('physical') or args.variant.startswith('pelvis') or args.variant=='compactown',pelvis_support=args.variant.startswith('pelvis') or args.variant=='continuous',opposite_full_only=args.variant in ('pelvishybrid','compactown','continuous','continuousown'))
 if args.variant in ('soft','physicalsoft'):settings.spring_groups['Skirt'].stiffness*=.5
skirt_springs=[s for s in sb.springs if s.vrm_name.startswith('Secondary_Skirt_')]
tips=[j.node.bone_name for s in skirt_springs for j in s.joints[1:]]
owners=[skirt_owner_leg(rig,s,legs) for s in skirt_springs for j in s.joints[1:]]
def positions():return np.array([rig.matrix_world@rig.pose.bones[n].head for n in tips])
def surfaces():
 vertices=[];st=[];lt=[]
 for obj,garment,body in parts:
  ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh();n=len(vertices)
  vertices.extend(obj.matrix_world@v.co for v in mesh.vertices);ev.to_mesh_clear()
  st.extend(tuple(i+n for i in tri) for tri in garment);lt.extend(tuple(i+n for i in tri) for tri in body)
 return len(BVHTree.FromPolygons(vertices,st,all_triangles=True).overlap(BVHTree.FromPolygons(vertices,lt,all_triangles=True)))
results=[]
for side in range(2):
 for x,z,y in ([(65,0,0),(0,60,0),(-60,-45,30)] if args.stress else [(30,0,0),(0,30,0),(-30,-30,-30)]):
  set_simulation(False)
  for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
  bpy.context.view_layer.update();set_simulation(True)
  for _ in range(90):background_step(1/60)
  start=positions();history=[];counts=[]
  ramp=8 if args.stress else 90; hold=180 if args.stress else 90; end=ramp+hold; returning=8 if args.stress else 60; total=end+returning+(90 if args.stress else 0)
  if args.loop_motion:total=360;ramp=120;end=270;returning=8
  for f in range(total):
   t=(1-math.cos(math.pi*f/(ramp-1)))/2 if f<ramp else 1. if f<end else (1+math.cos(math.pi*min(f-end,returning-1)/(returning-1)))/2
   ax,az,ay=x*t,z*t,y*t
   if args.loop_motion and f<120:
    phase=math.tau*f/24; ax=65*math.sin(phase);az=60*math.cos(phase);ay=30*math.sin(phase*.5)
   pb=rig.pose.bones[legs[side]];pb.rotation_mode='QUATERNION';pb.rotation_quaternion=Quaternion((1,0,0),math.radians(ax))@Quaternion((0,0,1),math.radians(az))@Quaternion((0,1,0),math.radians(ay))
   bpy.context.view_layer.update();background_step(1/60);bpy.context.view_layer.update();history.append(positions())
   if f==end-1:
    deepest=0.;penetrating=0
    cmap={c.uuid:c for c in sb.colliders};gmap={g.uuid:g for g in sb.collider_groups}
    for spring in skirt_springs:
     for h,tj in zip(spring.joints,spring.joints[1:]):
      point=rig.pose.bones[tj.node.bone_name].head
      for ref in spring.collider_groups:
       for cr in gmap[ref.collider_group_uuid].colliders:
        c=cmap[cr.collider_uuid];shape=c.shape.capsule;m=rig.pose.bones[c.node.bone_name].matrix
        depth=shape.radius+h.hit_radius-segment_distance(point,m@Vector(shape.offset),m@Vector(shape.tail))
        deepest=max(deepest,depth);penetrating+=depth>1e-4
   if f in (ramp//2,ramp-1,end-60,end-30,end-1,end+returning//2,total-1):counts.append(surfaces())
  h=np.asarray(history);np.save(out/f'positions_{side}_{x}_{z}_{y}.npy',h);quiet=h[end-60:end];acc=np.diff(quiet,n=2,axis=0);speed=np.diff(quiet,axis=0)
  opposite=np.array([n!=legs[side] for n in owners])
  row=dict(side=side,angles=[x,z,y],surfaces=counts,hold_jitter_rms_mm=float(np.sqrt(np.mean(acc*acc))*1000),hold_step_rms_mm=float(np.sqrt(np.mean(speed*speed))*1000),opposite_excursion_mm=float(np.max(np.linalg.norm(h[:,opposite]-start[None,opposite],axis=2)))*1000)
  row['motion_jerk_rms_mm']=float(np.sqrt(np.mean(np.diff(h[:ramp],n=3,axis=0)**2))*1000)
  row['return_jerk_rms_mm']=float(np.sqrt(np.mean(np.diff(h[end:end+returning],n=3,axis=0)**2))*1000)
  row['hold_collider_penetrations']=penetrating;row['hold_collider_depth_mm']=deepest*1000
  row['rest_return_error_mm']=float(np.max(np.linalg.norm(h[-1]-start,axis=1)))*1000
  results.append(row);print('STABILITY',args.variant,json.dumps(row),flush=True)
set_simulation(False)
for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
bpy.context.view_layer.update();verify(meshes,geometry)
report=dict(variant=args.variant,stiffness=settings.spring_groups['Skirt'].stiffness,drag=settings.spring_groups['Skirt'].drag,follow=settings.follow_groups['Skirt'].influence,colliders=len(sb.colliders),cases=results,geometry_unchanged=True)
(out/'results.json').write_text(json.dumps(report,indent=2));bpy.ops.wm.save_as_mainfile(filepath=str(out/'candidate.blend'));print('STABILITY_DONE',flush=True)
