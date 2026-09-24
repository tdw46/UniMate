"""Isolated regressions for collection visibility and generated collider cleanup."""
import json, sys
from pathlib import Path
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parent))
from avatar_vrm_colliders import (organize_colliders, show_colliders, display_objects,
    cleanup_orphan_displays, archive_colliders, collider_collection, add_capsule, colliders_visible)
from avatar_colliders import rebuild_colliders
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/collider_cleanup_20260924'
for repo in bpy.context.preferences.extensions.repos:
    if repo.module == 'user_default':
        repo.use_custom_directory = True
        repo.custom_directory = str(Path.home() / 'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
bpy.ops.wm.open_mainfile(filepath=str(OUT/'before.blend'))
rig = bpy.data.objects['Belle_arm_UniMate']
bpy.context.view_layer.objects.active = rig
if rig.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
source = bpy.data.objects['Belle_arm - Original']
source_names = {o.name for o in display_objects(source)}
source_count = len(source.data.vrm_addon_extension.spring_bone1.colliders)
# Migrate both legacy hiding mechanisms on heads AND endpoints.
for obj in display_objects(rig):
    obj.hide_set(True)
    obj.hide_viewport = True
collection = organize_colliders(rig)
assert len(collection.objects) == 34
assert all(o.show_in_front and not o.hide_viewport and not o.hide_get()
           for o in collection.objects)
for visible in (False, True, False):
    show_colliders(rig, visible)
    bpy.context.view_layer.update()
    assert colliders_visible(rig) == visible
    assert not collection.hide_viewport
    assert all(not o.hide_viewport and not o.hide_get() for o in collection.objects)
    assert all(o.visible_get() == visible for o in collection.objects)
# Hidden capsule display must still evaluate bone and whole-rig movement.
from mathutils import Vector, Quaternion
sb = rig.data.vrm_addon_extension.spring_bone1
c = next(c for c in sb.colliders if c.node.bone_name == 'LeftUpLeg')
offset = Vector(c.shape.capsule.offset)
tail = Vector(c.shape.capsule.tail)
bone = rig.pose.bones['LeftUpLeg']
basis = bone.matrix_basis.copy()
world = rig.matrix_world.copy()
try:
    bone.rotation_mode = 'QUATERNION'
    bone.rotation_quaternion = Quaternion((0, 0, 1), -.6)
    rig.location += Vector((.23, -.17, .09))
    bpy.context.view_layer.update()
    matrix = rig.matrix_world @ bone.matrix
    assert (c.bpy_object.matrix_world.translation - matrix @ offset).length < 1e-5
    assert (c.bpy_object.children[0].matrix_world.translation - matrix @ tail).length < 1e-5
finally:
    bone.matrix_basis = basis
    rig.matrix_world = world
    bpy.context.view_layer.update()
# Interrupted generation: an orphan head/end and an ungrouped official record.
head = bpy.data.objects.new('Interrupted collider', None)
end = bpy.data.objects.new('Interrupted endpoint', None)
for obj in (head, end):
    collection.objects.link(obj)
    obj['unimate_generated_collider'] = True
    obj['hallway_collider_owner'] = rig
head.parent = rig
end.parent = head
assert cleanup_orphan_displays(rig) == ['Interrupted collider', 'Interrupted endpoint']
artist = bpy.data.objects.new('Artist empty', None)
collection.objects.link(artist)
artist.parent = rig
protected = bpy.data.objects.new('Protected generated reference', None)
collection.objects.link(protected)
protected['unimate_generated_collider'] = True
protected['hallway_collider_owner'] = rig
artist.constraints.new('COPY_LOCATION').target = protected
assert cleanup_orphan_displays(rig) == []
assert artist.constraints[0].target == protected
bpy.data.objects.remove(artist, do_unlink=True)
assert cleanup_orphan_displays(rig) == ['Protected generated reference']
sb = rig.data.vrm_addon_extension.spring_bone1
hair = {c.uuid for c in sb.colliders if c.node.bone_name in {'Head','Neck','Spine1','LeftArm','RightArm'}}
stray = add_capsule(rig, dict(bone='LeftUpLeg', offset=(0,.1,0), tail=(0,.2,0), radius=.01))
assert len(sb.colliders) == 18
meshes = [o for o in rig.children if o.type == 'MESH']
counts = []
for _ in range(3):
    rebuild_colliders(rig, meshes, collider_roles=('skirt',))
    counts.append((len(sb.colliders), len(collection.objects), len(bpy.data.objects)))
    assert not colliders_visible(rig)  # Refit preserves the collection toggle.
    assert all(o.show_in_front and not o.hide_viewport and not o.hide_get() for o in collection.objects)
    assert hair <= {c.uuid for c in sb.colliders}
assert counts[0] == counts[1] == counts[2], counts
assert counts[0][:2] == (9, 18), counts
assert len(source.data.vrm_addon_extension.spring_bone1.colliders) == source_count
assert all(n in bpy.data.objects for n in source_names)
archive = bpy.data.collections['Belle_arm - Original comparison']
legacy = [o for o in bpy.data.objects if o.rigid_body is not None and o.type == 'MESH']
archived = archive_colliders(source, archive, legacy)
assert archived == 447
child = next(c for c in archive.children if c.get('hallway_archived_colliders'))
assert len(child.objects) == 447
assert all(not o.hide_viewport for o in child.objects)
assert not any(o.name in bpy.context.scene.collection.objects for o in child.objects)
assert archive_colliders(source, archive, legacy) == 447
assert len(archive.children) == 1
assert not cleanup_orphan_displays(rig)
show_colliders(rig, True)
report = dict(repeated_refit_counts=counts, generated_display_objects=18,
              front_enabled=True, object_hide_flags_clear=True,
              collection_toggle_tested=True, hidden_colliders_follow_bones=True, orphan_recovery=True,
              artist_references_preserved=True, original_vrm_displays_preserved=len(source_names),
              archived_objects=archived)
(OUT/'validation.json').write_text(json.dumps(report, indent=2))
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'validated.blend'))
# Ensure ownership references and collection visibility survive save/reopen.
bpy.ops.wm.open_mainfile(filepath=str(OUT/'validated.blend'))
rig = bpy.data.objects['Belle_arm_UniMate']
collection = collider_collection(rig)
assert collection and len(collection.objects) == 18 and not collection.hide_viewport
assert all(o.show_in_front and not o.hide_viewport for o in collection.objects)
print('COLLIDER_LIFECYCLE_OK', json.dumps(report), flush=True)
