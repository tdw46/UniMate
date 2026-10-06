"""Native upper-skirt physics migration and VRM roundtrip regression."""
import bpy,sys,json,struct,importlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
if not bpy.app.background:raise RuntimeError('Use isolated Blender')
source,destination=sys.argv[sys.argv.index('--')+1:];out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=True)
for repo in bpy.context.preferences.extensions.repos:
 if repo.module=='user_default':repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
from properties_hallway_rig import register,apply_settings
register();bpy.ops.wm.open_mainfile(filepath=str(Path(source).resolve()))
from avatar_skirt_hip_physics import extend_hip_physics,PREFIX
from avatar_skirt_support import CONSTRAINT
from avatar_mesh_invariant import snapshot,verify
from avatar_skirt_fit_io import rig_signature,rest_edit
from avatar_apparel_weights import weights
from avatar_colliders import rest_contacts
from avatar_skirt_binding import rebind_skirt_strips
r=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE' and o.get('hallway_generated_rig'))
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.find_armature()==r]
with rest_edit(r):rebind_skirt_strips(r,meshes)
geometry=snapshot(meshes)
def binding():return {o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}
ws=binding();sb=r.data.vrm_addon_extension.spring_bone1
colliders=[c.uuid for c in sb.colliders];old_bones=len(r.data.bones)
rest={b.name:b.matrix_local.copy() for b in r.data.bones if not b.get('hallway_skirt_support')}
pose={p.name:p.matrix_basis.copy() for p in r.pose.bones if not p.name.startswith('Secondary_')}
report=extend_hip_physics(r);apply_settings(r)
assert report['added_segments']==12 and len(r.data.bones)==old_bones+12
assert binding()==ws;verify(meshes,geometry)
assert set(colliders)<={c.uuid for c in sb.colliders}
assert all(max(abs(a-b) for row,old in zip(r.data.bones[n].matrix_local,m) for a,b in zip(row,old))<1e-6 for n,m in rest.items())
assert all(r.pose.bones[n].matrix_basis==m for n,m in pose.items())
assert 'Skirt Support' not in r.hallway_rig.follow_groups
hips=[s for s in sb.springs if s.vrm_name.startswith(PREFIX)]
assert len(hips)==12 and all(len(s.joints)==2 and not s.center.bone_name for s in hips)
for s in hips:
 name=s.joints[0].node.bone_name
 assert not r.pose.bones[name].constraints
 assert any(c.get('hallway_role')=='Physics' for c in r.data.bones[name].collections)
 assert r.data.bones[s.joints[-1].node.bone_name].parent==r.data.bones[name]
 assert s.collider_groups
with rest_edit(r):assert not rest_contacts(r)['contacts']
signature=rig_signature(r);groups=[g.uuid for g in sb.collider_groups]
assert extend_hip_physics(r)['added_segments']==0
assert signature==rig_signature(r) and groups==[g.uuid for g in sb.collider_groups]
with rest_edit(r):rebind_skirt_strips(r,meshes)
assert signature==rig_signature(r),'Rebinding damaged physics setup'
assert all(max([abs(a.get(n,0)-b.get(n,0)) for n in a.keys()|b.keys()]+[0])<1e-6 for name,rows in ws.items() for a,b in zip(rows,binding()[name]))
assert 'Skirt Support' not in r.hallway_rig.follow_groups
bpy.ops.wm.save_as_mainfile(filepath=str(out/'fixed.blend'))
(out/'weights.json').write_text(json.dumps(dict(geometry=geometry,weights=ws)))
search=importlib.import_module('bl_ext.user_default.vrm.editor.search')
_,accepted,errors=search.export_constraints([r]+meshes,r);assert not errors
assert not any(n in accepted.rotation_constraints for n in report['bones'])
from mathutils import Matrix
bpy.context.view_layer.objects.active=r
if bpy.context.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
for p in r.pose.bones:p.matrix_basis=Matrix.Identity(4)
bpy.context.view_layer.update();bpy.ops.object.select_all(action='DESELECT')
for o in [r]+meshes:o.select_set(True)
meta=r.data.vrm_addon_extension.vrm1.meta
if not meta.authors:meta.authors.add().value='Local validation'
path=out/'hip_physics.vrm'
assert bpy.ops.export_scene.vrm(filepath=str(path),armature_object_name=r.name,export_only_selections=True,export_invisibles=True,export_gltf_animations=False,ignore_warning=True)=={'FINISHED'}
raw=path.read_bytes();size,_=struct.unpack_from('<II',raw,12);gltf=json.loads(raw[20:20+size])
exported=[s for s in gltf['extensions']['VRMC_springBone']['springs'] if s.get('name','').startswith(PREFIX)]
assert len(exported)==12 and all(len(s['joints'])==2 and 'center' not in s for s in exported)
for o in list(bpy.data.objects):bpy.data.objects.remove(o,do_unlink=True)
assert bpy.ops.import_scene.vrm(filepath=str(path))=={'FINISHED'}
r=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
assert len([s for s in r.data.vrm_addon_extension.spring_bone1.springs if s.vrm_name.startswith(PREFIX)])==12
report.update(geometry_and_weights_unchanged=True,lower_rest_frames_preserved=True,original_colliders_preserved=True,repeatable=True,rebind_preserves_physics=True,vrm_export_reimport=True)
(out/'validation.json').write_text(json.dumps(report,indent=2));print('HIP_PHYSICS_OK',json.dumps(report))
