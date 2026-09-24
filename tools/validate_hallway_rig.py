"""Exercise Belle configuration, target isolation and blend persistence in isolation."""
import json
import sys
from pathlib import Path
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui_hallway_rig as ui
from avatar_bone_collections import organize_bones
from properties_hallway_rig import initialize, skirt_colliders, apply_settings, follow_constraints
from avatar_colliders import rest_contacts
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/hallway_rig_20260924'
for repo in bpy.context.preferences.extensions.repos:
    if repo.module == 'user_default':
        repo.use_custom_directory = True
        repo.custom_directory = str(Path.home() / 'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
bpy.ops.wm.open_mainfile(filepath=str(OUT/'belle_before.blend'))
ui.register()
rig = bpy.data.objects['Belle_arm_UniMate']
if rig.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
settings = initialize(rig)
original = bpy.data.objects['Belle_arm - Original']
original_groups = [(c.name,len(c.bones)) for c in original.data.collections]
counts = organize_bones(rig)
assert counts == {'Physics':528,'Constraints':12,'Deform':51,'Controls':1}, counts
assert organize_bones(rig) == counts
assert all(sum(bool(c.get('hallway_role')) for c in b.collections) == 1 for b in rig.data.bones)
# Artist collections survive organization.
custom = rig.data.collections.new('Artist Selection')
custom.assign(rig.data.bones['Head'])
organize_bones(rig)
assert rig.data.bones['Head'].name in custom.bones
skirt = skirt_colliders(rig)
sb = rig.data.vrm_addon_extension.spring_bone1
hair = {c.uuid:c.shape.capsule.radius for c in sb.colliders if c not in skirt}
bases = {c.uuid:c.bpy_object['hallway_base_radius'] for c in skirt}
flags = [(c.target.name,c.subtarget,c.mix_mode,c.target_space,c.owner_space,tuple((c.use_x,c.use_y,c.use_z,c.invert_x,c.invert_y,c.invert_z))) for c in follow_constraints(rig,'Skirt')]
settings.follow_groups['Skirt'].influence = .73
settings.skirt_thickness = .7
settings.spring_groups['Skirt'].drag = .63
settings.spring_groups['Hair'].drag = .42
bpy.context.view_layer.objects.active = rig
assert bpy.ops.hallway.apply_rig_settings() == {'FINISHED'}
assert all(abs(c.influence-.73)<1e-6 for c in follow_constraints(rig,'Skirt'))
assert all(abs(c.shape.capsule.radius-bases[c.uuid]*.7)<1e-6 for c in skirt)
apply_settings(rig)  # Must not compound the multiplier.
assert all(abs(c.shape.capsule.radius-bases[c.uuid]*.7)<1e-6 for c in skirt)
assert all(abs(c.shape.capsule.radius-hair[c.uuid])<1e-7 for c in sb.colliders if c.uuid in hair)
settings.skirt_thickness = 3.
limited = apply_settings(rig)
assert limited > 0
assert not rest_contacts(rig)['contacts'], rest_contacts(rig)
settings.skirt_thickness = .7
apply_settings(rig)
assert flags == [(c.target.name,c.subtarget,c.mix_mode,c.target_space,c.owner_space,tuple((c.use_x,c.use_y,c.use_z,c.invert_x,c.invert_y,c.invert_z))) for c in follow_constraints(rig,'Skirt')]
assert all(not s.center.bone_name for s in sb.springs)
assert original_groups == [(c.name,len(c.bones)) for c in original.data.collections]
for _ in range(2):
    ui.unregister()
    ui.register()
    assert rig.hallway_rig.initialized
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'configuration_roundtrip.blend'))
bpy.ops.wm.open_mainfile(filepath=str(OUT/'configuration_roundtrip.blend'))
rig = bpy.data.objects['Belle_arm_UniMate']
settings = rig.hallway_rig
assert settings.initialized and abs(settings.skirt_thickness-.7)<1e-6
assert abs(settings.follow_groups['Skirt'].influence-.73)<1e-6
assert abs(settings.spring_groups['Skirt'].drag-.63)<1e-6
assert abs(settings.spring_groups['Hair'].drag-.42)<1e-6
assert all(abs(c.influence-.73)<1e-6 for c in follow_constraints(rig,'Skirt'))
assert bpy.types.HALLWAY_PT_Rig.bl_category == 'Hallway'
result=dict(collections=counts,skirt_colliders=len(skirt_colliders(rig)),hair_colliders_unchanged=len(hair),thickness_clamped_at_rest=limited,save_reload=True,registration_cycles=2,centers_preserved=True,original_untouched=True)
(OUT/'validation.json').write_text(json.dumps(result,indent=2))
print('HALLWAY_CONFIG_OK',json.dumps(result),flush=True)
