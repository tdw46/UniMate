"""Exercise every panel setting and actual reset operator in isolated Blender."""
import bpy, sys, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
if not bpy.app.background:raise RuntimeError('Use isolated Blender')
source,output=sys.argv[sys.argv.index('--')+1:]
for repo in getattr(getattr(bpy.context.preferences,'extensions',None),'repos',[]):
    if repo.module=='user_default':
        repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
import ui_hallway_rig
ui_hallway_rig.register()
bpy.ops.wm.open_mainfile(filepath=str(Path(source).resolve()))
from avatar_rig_defaults import FOLLOW, SPRINGS
from properties_hallway_rig import active_rig, initialize, follow_entries, follow_influence, skirt_colliders, non_root_joint_roles, springs
from avatar_vrm_colliders import show_colliders, colliders_visible
from avatar_mesh_invariant import snapshot,verify
r=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE' and o.get('hallway_generated_rig'))
bpy.context.view_layer.objects.active=r
if bpy.context.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.find_armature()==r]
geometry=snapshot(meshes);pose={p.name:p.matrix_basis.copy() for p in r.pose.bones if not p.name.startswith('Secondary_')}
initialize(r)
for group in r.hallway_rig.follow_groups:group.influence=.13
for group in r.hallway_rig.spring_groups:
    group.drag=.11;group.stiffness=.31;group.non_root_stiffness=.23;group.gravity=.009
r.hallway_rig.skirt_thickness=.35
for c in r.data.collections:
    if c.get('hallway_role'):c.is_visible=False
show_colliders(r,True)
# Resolve target through VRM even while a bound mesh is active.
bpy.ops.object.select_all(action='DESELECT');meshes[0].select_set(True);bpy.context.view_layer.objects.active=meshes[0]
assert active_rig(bpy.context)==r
assert bpy.ops.hallway.reset_rig_settings()=={'FINISHED'}
for name,value in FOLLOW.items():
    if name in r.hallway_rig.follow_groups:assert abs(r.hallway_rig.follow_groups[name].influence-value)<1e-6
for group in r.hallway_rig.follow_groups:
    for pb,c in follow_entries(r,group.name):assert abs(c.influence-follow_influence(r,pb,group))<1e-6
for group in r.hallway_rig.spring_groups:
    for field,value in SPRINGS[group.name].items():assert abs(getattr(group,field)-value)<1e-6
    chains=springs(r,group.prefix)
    for spring,roles in zip(chains,non_root_joint_roles(r,chains)):
        for joint,nonroot in zip(spring.joints,roles):
            assert abs(joint.drag_force-group.drag)<1e-6 and abs(joint.gravity_power-group.gravity)<1e-6
            expected=group.stiffness*joint['hallway_stiffness_ratio']*(group.non_root_stiffness if nonroot else 1.)
            assert abs(joint.stiffness-expected)<1e-6
assert r.hallway_rig.skirt_thickness==1.
for c in skirt_colliders(r):
    o=c.bpy_object;assert abs(c.shape.capsule.radius-min(o['hallway_base_radius'],o['hallway_radius_limit']))<1e-6
assert not colliders_visible(r)
assert all(c.is_visible for c in r.data.collections if c.get('hallway_role'))
assert bpy.context.view_layer.objects.active==meshes[0]
assert bpy.ops.hallway.reset_rig_settings()=={'FINISHED'}
verify(meshes,geometry);assert all(r.pose.bones[n].matrix_basis==m for n,m in pose.items())
# New controls must use the same canonical defaults; initializing existing
# settings must not overwrite a saved custom value.
r.hallway_rig.follow_groups['Skirt'].influence=.33
initialize(r);assert abs(r.hallway_rig.follow_groups['Skirt'].influence-.33)<1e-6
r.hallway_rig.follow_groups.clear()
initialize(r)
assert all(abs(r.hallway_rig.follow_groups[n].influence-v)<1e-6 for n,v in FOLLOW.items() if n in r.hallway_rig.follow_groups)
report=dict(follow=FOLLOW,springs=SPRINGS,thickness=1.,bone_collections_visible=True,colliders_hidden=True,
    native_constraints_and_springs_and_radii=True,mesh_active_target_resolution=True,
    reset_without_bvt=True,idempotent=True,geometry_and_humanoid_pose_preserved=True,
    new_control_defaults=True,saved_settings_preserved_on_initialize=True)
Path(output).write_text(json.dumps(report,indent=2));print('HALLWAY_DEFAULTS_OK',json.dumps(report))
