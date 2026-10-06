"""Isolated pelvis/garment migration, scope, invariants and repeatability."""
import bpy,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
if not bpy.app.background:raise RuntimeError('Use an isolated Blender')
source,dest=sys.argv[sys.argv.index('--')+1:];out=Path(dest).resolve();out.mkdir(exist_ok=True,parents=True)
for repo in bpy.context.preferences.extensions.repos:
 if repo.module=='user_default':repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
from properties_hallway_rig import register,follow_entries,follow_influence
register();bpy.ops.wm.open_mainfile(filepath=str(Path(source).resolve()))
from avatar_pelvis_binding import repair_pelvis_weights,transfer_garment_waist,landmarks,confine_pelvis_weights
from avatar_skirt_binding import rebind_skirt_strips
from avatar_skirt_fit_io import rest_edit
from avatar_mesh_invariant import snapshot,verify
from avatar_apparel_weights import weights
from avatar_leg_binding import _bone_distance
r=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE' and o.get('hallway_bilateral_skirt_layout'))
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.find_armature()==r];geometry=snapshot(meshes)
def binding():return {o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}
before=binding();pose={p.name:p.matrix_basis.copy() for p in r.pose.bones};counts=(len(bpy.data.objects),len(r.data.bones))
with rest_edit(r):
 a=repair_pelvis_weights(r,meshes);a['socket_falloff']=confine_pelvis_weights(r,meshes)
 b=rebind_skirt_strips(r,meshes);c=transfer_garment_waist(r,meshes)
 assert counts==(len(bpy.data.objects),len(r.data.bones))
 after=binding()
 assert not repair_pelvis_weights(r,meshes)['applied']
 assert transfer_garment_waist(r,meshes)['vertices']==0 and binding()==after
r.hallway_rig.follow_groups['Skirt Knee'].influence=.6
assert all(abs(con.influence-follow_influence(r,p,r.hallway_rig.follow_groups['Skirt Knee']))<1e-6 for p,con in follow_entries(r,'Skirt Knee'))
assert all(r.pose.bones[n].matrix_basis==v for n,v in pose.items())
verify(meshes,geometry)
changed=0
_,_,_,thighs,_,low,core_low,high=landmarks(r)
for obj in meshes:
 m=r.matrix_world.inverted()@obj.matrix_world
 for i,(old,new) in enumerate(zip(before[obj.name],after[obj.name])):
  assert abs(sum(new.values())-1)<1e-6
  if max([abs(old.get(n,0)-new.get(n,0)) for n in old.keys()|new.keys()]+[0])<1e-6:continue
  changed+=1;z=(m@obj.data.vertices[i].co).z
  if not any(n.startswith('Secondary_Skirt_') for n in old):
   assert a['low']-1e-6<=z<=a['high']+1e-6,(obj.name,i,z)
  assert not any(n.startswith('Secondary_Hair_') for n in old)
  p=m@obj.data.vertices[i].co
  if p.z<=core_low and not any(n.startswith('Secondary_') for n in old):
   own=min(thighs,key=lambda bone:_bone_distance(p,bone))
   other=next(bone for bone in thighs if bone!=own)
   assert new.get(other.name,0)<=old.get(other.name,0)+1e-6,(obj.name,i,'cross-leg contamination')
report=dict(pelvis=a,binding=b,attachment=c,changed_vertices=changed,geometry_unchanged=True,pose_preserved=True,idempotent=True,no_duplicate_objects=True)
(out/'regression.json').write_text(json.dumps(report,indent=2));bpy.ops.wm.save_as_mainfile(filepath=str(out/'fixed.blend'))
print('PELVIS_BINDING_OK',json.dumps(report),flush=True)
