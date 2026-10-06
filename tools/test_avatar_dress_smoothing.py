"""Verify local knee-weight smoothing, immutable meshes and repeatable binding.

Isolated Blender only: source.blend output_directory
"""
import bpy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
if not bpy.app.background:raise RuntimeError('Use isolated background Blender')
source,destination=sys.argv[sys.argv.index('--')+1:]
out=Path(destination).resolve();out.mkdir(exist_ok=True,parents=True)
for repo in getattr(getattr(bpy.context.preferences,'extensions',None),'repos',[]):
 if repo.module=='user_default':repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
from properties_hallway_rig import register
register()
bpy.ops.wm.open_mainfile(filepath=str(Path(source).resolve()))
from avatar_mesh_invariant import snapshot,verify
from avatar_skirt_binding import rebind_skirt_strips
from avatar_skirt_fit_io import rest_edit
from avatar_apparel_weights import weights
from avatar_dress import full_chain_names
r=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE' and o.get('hallway_bilateral_skirt_layout'))
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.find_armature()==r]
geom=snapshot(meshes)
def binding():return {o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}
before=binding();ranges=[];upper=set();lower=set()
for spring in r.data.vrm_addon_extension.spring_bone1.springs:
 if not spring.vrm_name.startswith('Secondary_Skirt_'):continue
 names=full_chain_names(r,spring)
 if not names:continue
 k=r.data.bones[names[0]].get('hallway_dress_knee_index')
 if k is None:continue
 points=[r.data.bones[n].head_local for n in names]
 ranges.append((points[k].z,.4*min(points[k-1].z-points[k].z,points[k].z-points[k+1].z)))
 upper.update(names[:k]);lower.update(names[k:])
def mixed(rows):
 return sum(sum(w.get(n,0) for n in upper)>1e-5 and sum(w.get(n,0) for n in lower)>1e-5 for vs in rows.values() for w in vs)
with rest_edit(r):report=rebind_skirt_strips(r,meshes,smooth_iterations=0)
after=binding();changes=0;max_delta=0
for o in meshes:
 m=r.matrix_world.inverted()@o.matrix_world
 for i,(a,b) in enumerate(zip(before[o.name],after[o.name])):
  delta=max([abs(a.get(n,0)-b.get(n,0)) for n in a.keys()|b.keys()]+[0.])
  if delta<1e-6:continue
  changes+=1;max_delta=max(max_delta,delta)
  z=(m@o.data.vertices[i].co).z
  assert any(abs(z-knee)<band+1e-6 for knee,band in ranges),(o.name,i,z)
  assert {n:w for n,w in a.items() if not n.startswith('Secondary_Skirt_')}=={n:w for n,w in b.items() if not n.startswith('Secondary_Skirt_')}
  assert abs(sum(b.values())-1)<1e-6
assert changes>0 and mixed(after)>mixed(before)
with rest_edit(r):rebind_skirt_strips(r,meshes,smooth_iterations=0)
assert binding()==after
verify(meshes,geom)
report.update(changed_vertices=changes,maximum_weight_delta=max_delta,mixed_vertices_before=mixed(before),mixed_vertices_after=mixed(after),only_knee_band_changed=True,humanoid_weights_unchanged=True,geometry_unchanged=True,idempotent=True)
(out/'binding.json').write_text(json.dumps(report,indent=2))
bpy.ops.wm.save_as_mainfile(filepath=str(out/'smooth.blend'))
print('SMOOTH_BINDING_OK',json.dumps(report),flush=True)
