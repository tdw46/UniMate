"""Verify direct RNA edits apply immediately without an Apply operator or solver reset."""
import json, sys
from pathlib import Path
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parent))
import properties_hallway_rig as props
import ui_hallway_rig as ui
from avatar_physics_preview import set_simulation, bvt_runtime, background_step
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/hallway_realtime_20260924'
for repo in bpy.context.preferences.extensions.repos:
    if repo.module == 'user_default':
        repo.use_custom_directory = True
        repo.custom_directory = str(Path.home() / 'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
bpy.ops.wm.open_mainfile(filepath=str(OUT/'before.blend'))
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.beyond_vrm_extension_suite')
ui.register()
rig = bpy.data.objects['Belle_arm_UniMate']
settings = props.initialize(rig)
original = bpy.data.objects['Belle_arm - Original']
original_springs = [(j.stiffness,j.drag_force,j.gravity_power) for s in original.data.vrm_addon_extension.spring_bone1.springs for j in s.joints]
set_simulation(True)
background_step(1/60)
runtime = bvt_runtime()
runtime_before = runtime._rig_runtimes.get(rig.as_pointer())
# Exercise a different active object: updates must resolve id_data ownership.
if bpy.context.object and bpy.context.object.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
other = bpy.data.objects.new('Unrelated active object', None)
bpy.context.scene.collection.objects.link(other)
bpy.context.view_layer.objects.active = other
selection = {o.name:o.select_get() for o in bpy.context.view_layer.objects}
pose = {p.name:tuple(p.matrix_basis) for p in rig.pose.bones if not p.name.startswith('Secondary_')}
constraints = list(props.follow_constraints(rig, 'Skirt'))
flags = [(c.mix_mode,c.target_space,c.owner_space,c.use_x,c.use_y,c.use_z,c.invert_x,c.invert_y,c.invert_z) for c in constraints]
for value in (.32, .9, 1.):
    settings.follow_groups['Skirt'].influence = value
    assert all(abs(c.influence-value)<1e-6 for c in constraints)
assert flags == [(c.mix_mode,c.target_space,c.owner_space,c.use_x,c.use_y,c.use_z,c.invert_x,c.invert_y,c.invert_z) for c in constraints]
hair_radii = {c.uuid:c.shape.capsule.radius for c in rig.data.vrm_addon_extension.spring_bone1.colliders if c not in props.skirt_colliders(rig)}
for value in (.55, .8, .8, 3., 1.):
    settings.skirt_thickness = value
    for c in props.skirt_colliders(rig):
        expected = min(c.bpy_object['hallway_base_radius']*value,c.bpy_object['hallway_radius_limit'])
        assert abs(c.shape.capsule.radius-expected)<1e-6
        assert abs(c.bpy_object.children[0].empty_display_size-expected)<1e-6
assert hair_radii == {c.uuid:c.shape.capsule.radius for c in rig.data.vrm_addon_extension.spring_bone1.colliders if c.uuid in hair_radii}
checked = {}
for name in ('Skirt','Hair'):
    group = settings.spring_groups[name]
    other_name = 'Hair' if name=='Skirt' else 'Skirt'
    others = [(j.stiffness,j.drag_force,j.gravity_power) for s in props.springs(rig,settings.spring_groups[other_name].prefix) for j in s.joints]
    joints = [j for s in props.springs(rig,group.prefix) for j in s.joints]
    ratios = [j.get('hallway_stiffness_ratio',j.stiffness/s.joints[0].stiffness) for s in props.springs(rig,group.prefix) for j in s.joints]
    group.drag = .72
    assert all(abs(j.drag_force-.72)<1e-6 for j in joints)
    group.gravity = .047
    assert all(abs(j.gravity_power-.047)<1e-6 for j in joints)
    for value in (2., 0., 1.3):
        group.stiffness = value
        assert all(abs(j.stiffness-value*ratio)<1e-6 for j,ratio in zip(joints,ratios))
    assert others == [(j.stiffness,j.drag_force,j.gravity_power) for s in props.springs(rig,settings.spring_groups[other_name].prefix) for j in s.joints]
    checked[name] = len(joints)
assert bpy.context.scene.bvt_spring_simulation_enabled
assert runtime._rig_runtimes.get(rig.as_pointer()) is runtime_before
assert bpy.context.view_layer.objects.active == other
assert selection == {o.name:o.select_get() for o in bpy.context.view_layer.objects}
assert pose == {p.name:tuple(p.matrix_basis) for p in rig.pose.bones if not p.name.startswith('Secondary_')}
assert original_springs == [(j.stiffness,j.drag_force,j.gravity_power) for s in original.data.vrm_addon_extension.spring_bone1.springs for j in s.joints]
# Initialization of partial groups must not apply their intermediate defaults.
before = [(j.stiffness,j.drag_force,j.gravity_power) for s in props.springs(rig,'Secondary_Hair_') for j in s.joints]
settings.spring_groups.remove(settings.spring_groups.find('Hair'))
props.initialize(rig)
assert before == [(j.stiffness,j.drag_force,j.gravity_power) for s in props.springs(rig,'Secondary_Hair_') for j in s.joints]
# BVT consumes the modified RNA without reconstructing the runtime.
background_step(1/60)
assert runtime._rig_runtimes.get(rig.as_pointer()) is runtime_before
# Verify eye-icon UI calls without taking over the user's viewport.
class Layout:
    def __init__(self): self.calls=[]
    def row(self, **kwargs): return self
    def column(self, **kwargs): return self
    def box(self): return self
    def label(self, **kwargs): pass
    def prop(self, owner, field, **kwargs): self.calls.append((field,kwargs))
    def operator(self, *args, **kwargs): return type('Op',(),{})()
bpy.context.view_layer.objects.active = rig
layout = Layout()
ui.HALLWAY_PT_Rig.draw(type('Panel',(),{'layout':layout})(), bpy.context)
eyes = [kw for field,kw in layout.calls if field=='is_visible']
assert len(eyes)==4 and all(kw['icon'] in ('HIDE_ON','HIDE_OFF') and kw['text']=='' and kw['emboss']==False for kw in eyes)
set_simulation(False)
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'validated.blend'))
bpy.ops.wm.open_mainfile(filepath=str(OUT/'validated.blend'))
rig=bpy.data.objects['Belle_arm_UniMate']
rig.hallway_rig.follow_groups['Skirt'].influence=.67
assert all(abs(c.influence-.67)<1e-6 for c in props.follow_constraints(rig,'Skirt'))
report=dict(follow_constraints=len(constraints),capsules=len(props.skirt_colliders(rig)),spring_joints=checked,
            callback_updates=True,selection_and_pose_preserved=True,other_rig_preserved=True,
            taper_survives_zero=True,solver_not_restarted=True,eye_icons=len(eyes),reload_persistence=True)
(OUT/'validation.json').write_text(json.dumps(report,indent=2))
print('HALLWAY_REALTIME_OK',json.dumps(report),flush=True)
