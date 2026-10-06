"""Isolated socket falloff migration; geometry, locality and repeatability."""
import bpy,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
if not bpy.app.background:raise RuntimeError('Use an isolated Blender')
source,dest=sys.argv[sys.argv.index('--')+1:];out=Path(dest).resolve();out.mkdir(exist_ok=True,parents=True)
for repo in bpy.context.preferences.extensions.repos:
 if repo.module=='user_default':repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
from properties_hallway_rig import register
register();bpy.ops.wm.open_mainfile(filepath=str(Path(source).resolve()))
from avatar_pelvis_binding import confine_pelvis_weights,transfer_garment_waist,landmarks,thigh_envelope
from avatar_skirt_binding import rebind_skirt_strips
from avatar_skirt_fit_io import rest_edit
from avatar_mesh_invariant import snapshot,verify
from avatar_apparel_weights import weights
from avatar_leg_binding import _bone_distance
r=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE' and o.get('hallway_bilateral_skirt_layout'))
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.find_armature()==r];geometry=snapshot(meshes)
def binding():return {o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}
before=binding();pose={p.name:p.matrix_basis.copy() for p in r.pose.bones};counts=(len(bpy.data.objects),len(r.data.bones))
(out/'pose.json').write_text(json.dumps({n:dict(matrix=[list(row) for row in mat]) for n,mat in pose.items() if not n.startswith('Secondary_')}))
with rest_edit(r):
 a=confine_pelvis_weights(r,meshes);b=rebind_skirt_strips(r,meshes);c=transfer_garment_waist(r,meshes)
 after=binding();assert confine_pelvis_weights(r,meshes)['vertices']==0
 assert transfer_garment_waist(r,meshes)['vertices']==0 and binding()==after
assert counts==(len(bpy.data.objects),len(r.data.bones))
assert all(r.pose.bones[n].matrix_basis==v for n,v in pose.items());verify(meshes,geometry)
hips,spine,chest,thighs,arms,low,core_low,high=landmarks(r)
changed=0;above=0;maximum=0.
root=min(b.head_local.z for b in r.data.bones if 'hallway_garment_top' in b)
for obj in meshes:
 m=r.matrix_world.inverted()@obj.matrix_world
 skirt_ids={i for i,w in enumerate(before[obj.name]) if any(n.startswith("Secondary_Skirt_") for n in w)}
 slots={f.material_index for f in obj.data.polygons if any(i in skirt_ids for i in f.vertices)}
 slots|={i for i,mat in enumerate(obj.data.materials) if mat and any(t in mat.name.lower() for t in ("skin","body")) and not any(t in mat.name.lower() for t in ("cloth","hair","shoe"))}
 torso_ids={i for f in obj.data.polygons if f.material_index in slots for i in f.vertices}
 for i,(old,new) in enumerate(zip(before[obj.name],after[obj.name])):
  p=m@obj.data.vertices[i].co
  assert abs(sum(new.values())-1)<1e-6
  if i in torso_ids and thigh_envelope(p,hips,spine,thighs)<1e-10 and p.z<high and min(_bone_distance(p,b) for b in (hips,spine,chest))<=min(_bone_distance(p,b) for b in arms) and not any(n.startswith('Secondary_') for n in new):
   mass=sum(new.get(b.name,0) for b in thighs);maximum=max(maximum,mass);above+=1
   assert mass<1e-6,(obj.name,i,mass)
  if max([abs(old.get(n,0)-new.get(n,0)) for n in old.keys()|new.keys()]+[0])<1e-6:continue
  changed+=1
  assert not any(n.startswith('Secondary_Hair_') for n in old)
  if any(n.startswith('Secondary_Skirt_') for n in new):
   assert len(new)<=4
   assert sum(w for n,w in new.items() if n.startswith('Secondary_'))>0
  if not any(n.startswith('Secondary_Skirt_') for n in old):
   assert p.z>min(t.head_local.z-t.length*.30 for t in thighs),(obj.name,i,'distal leg changed')
report=dict(socket=a,binding=b,attachment=c,changed_vertices=changed,beyond_socket_vertices=above,maximum_thigh_weight_beyond_socket=maximum,geometry_unchanged=True,pose_preserved=True,idempotent=True)
(out/'regression.json').write_text(json.dumps(report,indent=2))
(out/'weights.json').write_text(json.dumps(dict(geometry=geometry,weights=after)))
bpy.ops.wm.save_as_mainfile(filepath=str(out/'fixed.blend'))
print('HIP_FALLOFF_OK',json.dumps(report),flush=True)
