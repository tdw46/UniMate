"""Isolated support generation, locality, live controls and VRM export regression.

Blender --background --factory-startup --python this.py -- source.blend output_dir
"""
import bpy, importlib, json, struct, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
if not bpy.app.background:
    raise RuntimeError('Use isolated Blender')
source, destination = sys.argv[sys.argv.index('--') + 1:]
out = Path(destination).resolve(); out.mkdir(parents=True, exist_ok=True)
for repo in getattr(getattr(bpy.context.preferences, 'extensions', None), 'repos', []):
    if repo.module == 'user_default':
        repo.use_custom_directory = True
        repo.custom_directory = str(Path.home() / 'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
from properties_hallway_rig import register
register()
bpy.ops.wm.open_mainfile(filepath=str(Path(source).resolve()))
from avatar_skirt_support import PREFIX, GROUP, ensure_support
from avatar_skirt_binding import rebind_skirt_strips
from avatar_skirt_fit_io import rest_edit, rig_signature
from avatar_mesh_invariant import snapshot, verify
from avatar_apparel_weights import weights
from avatar_pelvis_binding import landmarks
r = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE' and o.get('hallway_bilateral_skirt_layout'))
meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH' and o.find_armature() == r]
geometry = snapshot(meshes)
rest = {b.name: b.matrix_local.copy() for b in r.data.bones}
pose = {p.name: p.matrix_basis.copy() for p in r.pose.bones}
def binding():
    return {o.name: [weights(o, v.index) for v in o.data.vertices] for o in meshes}
before = binding()
with rest_edit(r):
    report = rebind_skirt_strips(r, meshes)
after = binding(); signature = rig_signature(r)
with rest_edit(r):
    rebind_skirt_strips(r, meshes)
assert rig_signature(r) == signature
assert after == binding(), 'Repeated generation changed weights'
verify(meshes, geometry)
for name, matrix in rest.items():
    if name in report['support']['removed']:continue
    assert max(abs(a-b) for row, old in zip(r.data.bones[name].matrix_local, matrix) for a,b in zip(row,old)) < 1e-6
assert all(r.pose.bones[n].matrix_basis == m for n,m in pose.items() if n not in report['support']['removed'])
helpers = [b for b in r.data.bones if b.get('hallway_skirt_support') == 2]
from avatar_dress import full_chain_names
expected_roots={names[0] for s in r.data.vrm_addon_extension.spring_bone1.springs if s.vrm_name.startswith('Secondary_Skirt_') and (names:=full_chain_names(r,s))}
assert {b.get('hallway_support_chain') for b in helpers} == expected_roots
assert not any(b.name in (PREFIX+'L',PREFIX+'R') for b in r.data.bones)
hips, _, _, thighs, *_ = landmarks(r)
axis = (thighs[0].head_local-thighs[1].head_local).normalized()
center = (thighs[0].head_local+thighs[1].head_local)*.5
support_vertices = 0
for obj in meshes:
    m = r.matrix_world.inverted() @ obj.matrix_world
    for i, row in enumerate(after[obj.name]):
        support = {n:v for n,v in row.items() if n.startswith(PREFIX)}
        if support:
            support_vertices += 1
            assert len(row) <= 4 and len(support) <= 2
            assert abs(sum(row.values())-1) < 1e-6
            side = 'L' if ((m @ obj.data.vertices[i].co)-center).dot(axis) > 0 else 'R'
            x=((m @ obj.data.vertices[i].co)-center).dot(axis)
            if abs(x)>(thighs[0].head_local-thighs[1].head_local).length*.08+1e-6:
                assert all(r.pose.bones[n].constraints[0].subtarget == thighs[0 if side=='L' else 1].name for n in support)
        old = before[obj.name][i]
        if any(n.startswith('Secondary_Hair_') for n in old):
            assert row == old
assert support_vertices > 0
from avatar_skirt_support import _surface
surfaces={}
for b in helpers:
    root=b['hallway_support_chain'];family=root.rsplit('_',2)[0]
    if family not in surfaces:surfaces[family]=_surface(r,meshes,family)
    hit,_,_,distance=surfaces[family].find_nearest(b.head_local)
    assert hit is not None and distance<1e-5, (b.name,distance)
search = importlib.import_module('bl_ext.user_default.vrm.editor.search')
_, accepted, errors = search.export_constraints([r]+meshes, r)
assert not errors, errors
assert {b.name for b in helpers} <= set(accepted.rotation_constraints)
for value in (0., .08, .75):
    r.hallway_rig.follow_groups[GROUP].influence = value
    for b in helpers:
        p = r.pose.bones[b.name]
        assert len(p.constraints) == 1
        assert abs(p.constraints[0].influence-value) < 1e-6
        assert any(c.get('hallway_role') == 'Constraints' for c in b.collections)
        assert b.head_local.z > r.data.bones[p.constraints[0].subtarget].head_local.z
# Save the posed, validated file before the destructive export roundtrip.
bpy.ops.wm.save_as_mainfile(filepath=str(out/'fixed.blend'))
(out/'weights.json').write_text(json.dumps(dict(geometry=geometry, weights=after)))
from mathutils import Matrix
bpy.context.view_layer.objects.active = r
if bpy.context.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
for p in r.pose.bones:
    p.matrix_basis = Matrix.Identity(4)
bpy.context.view_layer.update()
meta = r.data.vrm_addon_extension.vrm1.meta
if not meta.authors:
    meta.authors.add().value = 'Local validation'
bpy.ops.object.select_all(action='DESELECT')
for o in [r]+meshes:
    o.select_set(True)
path = out/'support.vrm'
assert bpy.ops.export_scene.vrm(filepath=str(path), armature_object_name=r.name,
    export_only_selections=True, export_invisibles=True, export_gltf_animations=False,
    ignore_warning=True) == {'FINISHED'}
raw=path.read_bytes(); size,_=struct.unpack_from('<II',raw,12); gltf=json.loads(raw[20:20+size])
exported={n['name']:n['extensions']['VRMC_node_constraint']['constraint']['rotation']
          for n in gltf['nodes'] if 'VRMC_node_constraint' in n.get('extensions',{})}
helper_names=[b.name for b in helpers]
for name in helper_names:
    assert abs(exported[name].get('weight',1)-.75) < 1e-6
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o,do_unlink=True)
assert bpy.ops.import_scene.vrm(filepath=str(path)) == {'FINISHED'}
imported=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
for name in helper_names:
    p=imported.pose.bones[name]
    assert len(p.constraints)==1 and abs(p.constraints[0].influence-.75)<1e-6
result=dict(binding=report, support_vertices=support_vertices, geometry_preserved=True,
    existing_rest_frames_preserved=True, pose_preserved=True, idempotent=True,
    maximum_influences=4, bilateral_support=True, live_follow_control=True,
    vrm_export_reimport=True, exported_support_constraints=len(helper_names))
(out/'validation.json').write_text(json.dumps(result,indent=2))
print('SKIRT_SUPPORT_OK',json.dumps(result),flush=True)
