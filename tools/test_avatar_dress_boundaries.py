"""Regression for lower-dress chain ridges on an isolated captured live file.

Arguments: source.blend probes.json output_dir. Probe indices are diagnostic
inputs only; production fitting uses rest geometry and generated chain metadata.
"""
import bpy,sys,json,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
if not bpy.app.background:raise RuntimeError('Use isolated Blender')
source,probe_path,destination=sys.argv[sys.argv.index('--')+1:];out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=True)
for repo in getattr(getattr(bpy.context.preferences,'extensions',None),'repos',[]):
 if repo.module=='user_default':repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
from properties_hallway_rig import register
register();bpy.ops.wm.open_mainfile(filepath=str(Path(source).resolve()))
if bpy.context.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
from avatar_mesh_invariant import snapshot,verify
from avatar_apparel_weights import weights
from avatar_skirt_binding import rebind_skirt_strips
from avatar_skirt_fit_io import rest_edit
from avatar_directional_contacts import skirt_owner_leg
from avatar_contact_colliders import snapshot_contact_colliders
r=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE' and o.get('hallway_generated_rig'))
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.find_armature()==r]
geometry=snapshot(meshes);poses={p.name:p.matrix_basis.copy() for p in r.pose.bones}
def binding():return {o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}
def restore_pose():
 for n,m in poses.items():r.pose.bones[n].matrix_basis=m
 bpy.context.view_layer.update()
probes=json.loads(Path(probe_path).read_text())
def angles():
 result=[]
 for entry in probes:
  o=bpy.data.objects[entry['object']];index=entry['index'];o.data.calc_loop_triangles()
  ev=o.evaluated_get(bpy.context.evaluated_depsgraph_get());me=ev.to_mesh()
  try:
   tris=[tuple(t.vertices) for t in o.data.loop_triangles if index in t.vertices]
   ns=[(me.vertices[b].co-me.vertices[a].co).cross(me.vertices[c].co-me.vertices[a].co).normalized() for a,b,c in tris]
   values=[math.degrees(ns[i].angle(ns[j])) for i,t in enumerate(tris) for j in range(i) if len(set(t)&set(tris[j]))==2 and ns[i].length>.5 and ns[j].length>.5]
   result.append(max(values,default=0.))
  finally:ev.to_mesh_clear()
 return result
before=binding();restore_pose();before_angles=angles();contacts=snapshot_contact_colliders(r)
hum=r.data.vrm_addon_extension.vrm1.humanoid.human_bones;thighs=[hum.left_upper_leg.node.bone_name,hum.right_upper_leg.node.bone_name]
chains={s.vrm_name:skirt_owner_leg(r,s,thighs) for s in r.data.vrm_addon_extension.spring_bone1.springs if s.vrm_name.startswith('Secondary_Skirt_') and not r.data.bones[s.joints[0].node.bone_name].get('hallway_dress_lower')}
with rest_edit(r):report=rebind_skirt_strips(r,meshes)
restore_pose();after=binding();after_angles=angles();verify(meshes,geometry)
assert contacts==snapshot_contact_colliders(r)
from avatar_weight_seams import coincident_groups
weld_sources={}
for obj in meshes:
 ids=[i for i,row in enumerate(before[obj.name]) if any(n.startswith(('Secondary_Skirt_','Secondary_SkirtSupport_')) for n in row)]
 matrix=r.matrix_world.inverted()@obj.matrix_world
 for group in coincident_groups([matrix@obj.data.vertices[i].co for i in ids]):
  members=[ids[k] for k in group]
  for index in members:weld_sources[obj.name,index]=members
def owners(row):
 result=set()
 for name in row:
  chain=name.rsplit('_',1)[0]
  if name.startswith('Secondary_Skirt_') and chain in chains:result.add(chains[chain])
  elif name in r.data.bones:
   root=r.data.bones[name].get('hallway_support_chain','')
   if root and root.rsplit('_',1)[0] in chains:result.add(chains[root.rsplit('_',1)[0]])
 return result
changed=[];bundle={}
for o in meshes:
 for i,(a,b) in enumerate(zip(before[o.name],after[o.name])):
  if max([abs(a.get(n,0)-b.get(n,0)) for n in a.keys()|b.keys()]+[0])<=1e-6:continue
  assert any(n.startswith(('Secondary_Skirt_','Secondary_SkirtSupport_')) for n in a),'Unrelated mesh weights changed'
  assert len(b)<=4
  assert abs(sum(b.values())-1)<1e-6
  new_chains={n.rsplit('_',1)[0] for n in b if n.startswith('Secondary_Skirt_')};old_chains={n.rsplit('_',1)[0] for n in a if n.startswith('Secondary_Skirt_')}
  assert len(new_chains)<=3
  # A duplicated center seam combines the two already present sides of
  # that same virtual vertex. Other rows may not introduce another owner.
  allowed=set().union(*(owners(before[o.name][j]) for j in weld_sources.get((o.name,i),[i])))
  assert owners(b)<=allowed,(o.name,i,owners(b),allowed)
  changed.append((o.name,i));bundle.setdefault(o.name,[]).append(dict(index=i,before=a,after=b))
assert changed
assert max(after_angles)<max(before_angles) and sum(after_angles)<sum(before_angles)
with rest_edit(r):rebind_skirt_strips(r,meshes)
repeat=binding()
assert all(max([abs(a.get(n,0)-b.get(n,0)) for n in a.keys()|b.keys()]+[0])<1e-6 for name,rows in after.items() for a,b in zip(rows,repeat[name]))
restore_pose();verify(meshes,geometry)
from avatar_weight_seams import coincident_groups
seam_groups=0;seam_error_before=0.
for obj in meshes:
 ids=[i for i,row in enumerate(after[obj.name]) if any(n.startswith(('Secondary_Skirt_','Secondary_SkirtSupport_')) for n in row)]
 matrix=r.matrix_world.inverted()@obj.matrix_world
 for group in coincident_groups([matrix@obj.data.vertices[i].co for i in ids]):
  if len(group)<2:continue
  group=[ids[k] for k in group];seam_groups+=1
  first=after[obj.name][group[0]]
  for index in group[1:]:
   other=after[obj.name][index]
   assert max([abs(first.get(n,0)-other.get(n,0)) for n in first.keys()|other.keys()]+[0])<1e-7
   a,b=before[obj.name][group[0]],before[obj.name][index]
   seam_error_before=max(seam_error_before,max([abs(a.get(n,0)-b.get(n,0)) for n in a.keys()|b.keys()]+[0]))
report.update(seam_groups=seam_groups,seam_error_before=seam_error_before,seam_error_after=0.)
report.update(before_angles=before_angles,after_angles=after_angles,changed_vertices=len(changed),geometry_preserved=True,unrelated_weights_preserved=True,contacts_unchanged=True,max_four_weights=True,ownership_scoped=True,repeatable=True)
(out/'validation.json').write_text(json.dumps(report,indent=2));(out/'bundle.json').write_text(json.dumps(dict(geometry=geometry,changes=bundle),indent=2))
bpy.ops.wm.save_as_mainfile(filepath=str(out/'fixed.blend'))
print('LOWER_BOUNDARIES_OK',json.dumps({k:report[k] for k in ('before_angles','after_angles','changed_vertices','repeatable')}))
