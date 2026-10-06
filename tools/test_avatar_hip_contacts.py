"""Isolated upper-contact upgrade, ownership, refit and VRM roundtrip checks."""
import bpy,sys,json,struct,importlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
if not bpy.app.background:raise RuntimeError('Use isolated background Blender')
source,destination=sys.argv[sys.argv.index('--')+1:];out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=True)
for repo in bpy.context.preferences.extensions.repos:
 if repo.module=='user_default':repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
from properties_hallway_rig import register,apply_settings,apply_thickness
register();bpy.ops.wm.open_mainfile(filepath=str(Path(source).resolve()))
from avatar_skirt_hip_physics import extend_hip_physics,PREFIX
from avatar_hip_contacts import GROUP_PREFIX,LEGACY_GROUP_PREFIX
from avatar_mesh_invariant import snapshot,verify
from avatar_skirt_fit_io import rest_edit
from avatar_apparel_weights import weights
from avatar_colliders import rest_contacts
from avatar_skirt_binding import rebind_skirt_strips
r=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE' and o.get('hallway_generated_rig'))
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.find_armature()==r]
geometry=snapshot(meshes)
def binding():return {o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}
ws=binding();sb=r.data.vrm_addon_extension.spring_bone1
rest={b.name:b.matrix_local.copy() for b in r.data.bones if not b.name.startswith('Secondary_SkirtSupport_')}
pose={p.name:p.matrix_basis.copy() for p in r.pose.bones if not p.name.startswith('Secondary_')}
old_heads={b.name:b.head_local.copy() for b in r.data.bones if b.get('hallway_hip_physics')}
report=extend_hip_physics(r);apply_settings(r)
assert binding()==ws;verify(meshes,geometry)
changed={n:max(abs(a-b) for row,old in zip(r.data.bones[n].matrix_local,m) for a,b in zip(row,old)) for n,m in rest.items()}
assert not {n:d for n,d in changed.items() if d>1e-6},{n:d for n,d in changed.items() if d>1e-6}
assert all(r.pose.bones[n].matrix_basis==m for n,m in pose.items())
assert all(r.data.bones[n].head_local.z>p.z for n,p in old_heads.items())
assert all(len([w for n,w in row.items() if w>1e-6])<=4 for rows in ws.values() for row in rows if any(n.startswith(('Secondary_Skirt_', 'Secondary_SkirtSupport_')) for n in row))
hips=[s for s in sb.springs if s.vrm_name.startswith(PREFIX)]
def validate():
 groups={g.uuid:g for g in sb.collider_groups};colliders={c.uuid:c for c in sb.colliders}
 keys=[c.bpy_object.get('hallway_hip_guard') for c in sb.colliders if c.bpy_object and c.bpy_object.get('hallway_hip_guard')]
 assert len(keys)==len(set(keys))==len(hips)*6
 owned={g.uuid for g in sb.collider_groups if g.vrm_name.startswith(GROUP_PREFIX)}
 assert len(owned)==len(hips)
 assert not any(g.vrm_name.startswith(LEGACY_GROUP_PREFIX) for g in sb.collider_groups)
 for s in sb.springs:
  if s.vrm_name.startswith(PREFIX):
   assert len(s.collider_groups)==1 and s.collider_groups[0].collider_group_uuid in owned
   assert not s.center.bone_name and not r.pose.bones[s.joints[0].node.bone_name].constraints
  else:assert not any(ref.collider_group_uuid in owned for ref in s.collider_groups)
 for uuid in owned:
  patches=[colliders[ref.collider_uuid] for ref in groups[uuid].colliders if colliders[ref.collider_uuid].bpy_object.get('hallway_hip_guard')]
  assert len({c.node.bone_name for c in patches})==1
  assert all(c.shape_type=='Capsule' and c.bpy_object.parent_bone==c.node.bone_name and c.bpy_object.show_in_front and not c.bpy_object.hide_viewport and not c.bpy_object.hide_get() for c in patches)
 with rest_edit(r):assert not rest_contacts(r)['contacts']
validate();ids=[c.uuid for c in sb.colliders];groups=[g.uuid for g in sb.collider_groups]
assert extend_hip_physics(r)['added_segments']==0
assert ids==[c.uuid for c in sb.colliders] and groups==[g.uuid for g in sb.collider_groups]
# Panel thickness applies to the new groups and resets to their rest-safe fit.
original=r.hallway_rig.skirt_thickness
r.hallway_rig.skirt_thickness=.8;apply_thickness(r)
assert all(abs(c.shape.capsule.radius-c.bpy_object['hallway_base_radius']*.8)<1e-6 for c in sb.colliders if c.bpy_object.get('hallway_hip_guard'))
r.hallway_rig.skirt_thickness=original;apply_thickness(r)
with rest_edit(r):rebind_skirt_strips(r,meshes)
smoothed=binding()
roots={b['hallway_support_chain'] for b in r.data.bones if b.get('hallway_hip_physics')}
changed_rows=0
for name,rows in ws.items():
 for old,new in zip(rows,smoothed[name]):
  error=max([abs(old.get(n,0)-new.get(n,0)) for n in old.keys()|new.keys()]+[0])
  if error>1e-6:
   changed_rows+=1
   assert roots & old.keys(),'Smoothing escaped the first skirt segment'
   assert len(new)<=4
   assert not any(n in pose and new.get(n,0)>old.get(n,0)+1e-5 and 'Leg' in n for n in new)
assert changed_rows>0
with rest_edit(r):rebind_skirt_strips(r,meshes)
assert all(max([abs(a.get(n,0)-b.get(n,0)) for n in a.keys()|b.keys()]+[0])<1e-6 for name,rows in smoothed.items() for a,b in zip(rows,binding()[name]))

from avatar_directional_contacts import install_skirt_contact_rig
with rest_edit(r):report['refit']=install_skirt_contact_rig(r,meshes)
validate();counts=(len(sb.colliders),len(sb.collider_groups))
with rest_edit(r):install_skirt_contact_rig(r,meshes)
assert counts==(len(sb.colliders),len(sb.collider_groups)),'Refit duplicated colliders'
validate();verify(meshes,geometry)
bpy.ops.wm.save_as_mainfile(filepath=str(out/'fixed.blend'))
search=importlib.import_module('bl_ext.user_default.vrm.editor.search')
_,accepted,errors=search.export_constraints([r]+meshes,r);assert not errors
from mathutils import Matrix
bpy.context.view_layer.objects.active=r
if bpy.context.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
for p in r.pose.bones:p.matrix_basis=Matrix.Identity(4)
bpy.context.view_layer.update();bpy.ops.object.select_all(action='DESELECT')
for o in [r]+meshes:o.select_set(True)
meta=r.data.vrm_addon_extension.vrm1.meta
if not meta.authors:meta.authors.add().value='Local validation'
path=out/'hip_contacts.vrm'
assert bpy.ops.export_scene.vrm(filepath=str(path),armature_object_name=r.name,export_only_selections=True,export_invisibles=True,export_gltf_animations=False,ignore_warning=True)=={'FINISHED'}
raw=path.read_bytes();size,_=struct.unpack_from('<II',raw,12);gltf=json.loads(raw[20:20+size]);native=gltf['extensions']['VRMC_springBone']
exported=[s for s in native['springs'] if s.get('name','').startswith(PREFIX)]
assert len(exported)==len(hips) and all(len(s['joints'])==2 and 'center' not in s for s in exported)
for s in exported:
 assert len(s['colliderGroups'])==1
 group=native['colliderGroups'][s['colliderGroups'][0]]
 assert group['name'].startswith(GROUP_PREFIX)
 assert all('capsule' in native['colliders'][i]['shape'] for i in group['colliders'])
for o in list(bpy.data.objects):bpy.data.objects.remove(o,do_unlink=True)
assert bpy.ops.import_scene.vrm(filepath=str(path))=={'FINISHED'}
r=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE');sb=r.data.vrm_addon_extension.spring_bone1
assert len([s for s in sb.springs if s.vrm_name.startswith(PREFIX)])==len(exported)
report.update(geometry_unchanged=True,weights_changed_only_at_upper_handoff=changed_rows,lower_rest_frames_preserved=True,repeatable=True,refit_repeatable=True,live_thickness=True,vrm_export_reimport=True,colliders=counts[0])
(out/'validation.json').write_text(json.dumps(report,indent=2));print('HIP_CONTACTS_OK',counts)
