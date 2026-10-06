"""Measure a localized garment crease in a frozen pose before/after rebinding.

Blender background args: source.blend output_directory mesh_name vertex_ids_csv
Vertex IDs are supplied diagnostic probes, never generation rules.
"""
import bpy,sys,json,math,importlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
if not bpy.app.background:raise RuntimeError('Use isolated Blender')
source,dest,mesh_name,probes=sys.argv[sys.argv.index('--')+1:]
out=Path(dest).resolve();out.mkdir(parents=True,exist_ok=True)
from mathutils import Vector
for repo in bpy.context.preferences.extensions.repos:
 if repo.module=='user_default':repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
from properties_hallway_rig import register
register();bpy.ops.wm.open_mainfile(filepath=str(Path(source).resolve()))
from avatar_skirt_binding import rebind_skirt_strips
from avatar_pelvis_binding import transfer_garment_waist
from avatar_skirt_fit_io import rest_edit
from avatar_mesh_invariant import snapshot,verify
from avatar_apparel_weights import weights
obj=bpy.data.objects[mesh_name];r=obj.find_armature();meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.find_armature()==r];geometry=snapshot(meshes)
ids=tuple(map(int,probes.split(',')))
obj.data.calc_loop_triangles()
def probe():
 bpy.context.view_layer.update();ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());me=ev.to_mesh();coords=[v.co.copy() for v in me.vertices];ev.to_mesh_clear()
 output={}
 for i in ids:
  ts=[tuple(t.vertices) for t in obj.data.loop_triangles if i in t.vertices]
  normals=[(coords[b]-coords[a]).cross(coords[c]-coords[a]).normalized() for a,b,c in ts]
  angles=[math.degrees(normals[j].angle(normals[k])) for j in range(len(ts)) for k in range(j) if len(set(ts[j])&set(ts[k]))==2 and normals[j].length>.5 and normals[k].length>.5]
  output[i]=max(angles,default=0)
 return output
before=probe()
with rest_edit(r):
 report=rebind_skirt_strips(r,meshes);transfer_garment_waist(r,meshes)
for o in meshes:o.data.update()
after=probe();verify(meshes,geometry)
assert max(after.values())<max(before.values()),(before,after)
assert all(after[i]<=before[i]+1e-3 for i in ids),(before,after)
(out/'probe.json').write_text(json.dumps(dict(before=before,after=after,binding=report),indent=2));bpy.ops.wm.save_as_mainfile(filepath=str(out/'measured.blend'));print('HITCH_PROBE',before,after)
