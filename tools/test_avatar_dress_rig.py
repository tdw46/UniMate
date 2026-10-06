"""Verify dress junctions, live follow updates, calf groups and immutable binding."""
import bpy,json,sys,importlib
from pathlib import Path
from mathutils import Matrix,Quaternion
sys.path.insert(0,str(Path(__file__).resolve().parent))
if not bpy.app.background:raise RuntimeError('Use isolated background Blender')
source,output=sys.argv[sys.argv.index('--')+1:]
for repo in getattr(getattr(bpy.context.preferences,'extensions',None),'repos',[]):
 if repo.module=='user_default':repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
from properties_hallway_rig import register,initialize,follow_constraints,follow_entries,follow_influence
register()
bpy.ops.wm.open_mainfile(filepath=str(Path(source).resolve()))
from avatar_mesh_invariant import snapshot,verify
from avatar_apparel_weights import weights
from avatar_skirt_binding import rebind_skirt_strips
from avatar_skirt_fit_io import rest_edit
rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE' and o.get('unimate_secondary_generator'))
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.find_armature()==rig]
geom=snapshot(meshes);binding={o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}
sb=rig.data.vrm_addon_extension.spring_bone1;settings=initialize(rig)
helpers=[p for p in rig.pose.bones if p.bone.get('hallway_dress_lower_follow')]
assert helpers
constraints=list(follow_constraints(rig,'Skirt'));old=settings.follow_groups['Skirt'].influence
for value in (0.,.3,1.,old):
 settings.follow_groups['Skirt'].influence=value
 assert all(abs(c.influence-value)<1e-6 for c in constraints)
knee=settings.follow_groups['Skirt Knee']
for name in ('Skirt Knee','Skirt Knee Side','Skirt Knee Back'):
 control=settings.follow_groups.get(name)
 if control is None:continue
 previous=control.influence
 for value in (0.,.15,.4,previous):
  control.influence=value
  assert all(abs(c.influence-follow_influence(rig,pb,knee))<1e-6 for pb,c in follow_entries(rig,'Skirt Knee'))
  assert all(abs(c.influence-old)<1e-6 for c in constraints)
groups={g.uuid:g for g in sb.collider_groups};colliders={c.uuid:c for c in sb.colliders}
for helper in helpers:
 c=helper.constraints[0];calf=rig.data.bones[c.subtarget]
 lower=next(s for s in sb.springs if rig.data.bones[s.joints[0].node.bone_name].parent==helper.bone)
 root=rig.data.bones[lower.joints[0].node.bone_name]
 upper=next(s for s in sb.springs if s.vrm_name==root['hallway_dress_upper_spring'])
 assert abs(root.head_local.z-calf.head_local.z)<1e-6
 junction=helper.parent.parent if helper.parent.bone.get('hallway_dress_knee_blend') else helper.parent
 assert junction.name==upper.joints[-1].node.bone_name
 assert not set(j.node.bone_name for j in upper.joints)&set(j.node.bone_name for j in lower.joints)
 assert any(colliders[ref.collider_uuid].node.bone_name==calf.name for cg in lower.collider_groups for ref in groups[cg.collider_group_uuid].colliders)
 assert any(colliders[ref.collider_uuid].bpy_object.get('hallway_directional_guard') and colliders[ref.collider_uuid].node.bone_name==calf.name for cg in lower.collider_groups for ref in groups[cg.collider_group_uuid].colliders)
 assert not any(helper.name in w for rows in binding.values() for w in rows)
 for axis in ((1,0,0),(0,1,0),(0,0,1)):
  pb=rig.pose.bones[calf.name];pb.rotation_mode='QUATERNION';pb.rotation_quaternion=Quaternion(axis,.8);bpy.context.view_layer.update()
  assert (rig.pose.bones[root.name].head-helper.parent.head).length<1e-6
  pb.matrix_basis=Matrix.Identity(4)
# No sagittal center chain or blending helpers; each side follows only its knee.
shared=[p for p in helpers if p.parent.bone.get('hallway_dress_knee_blend')]
assert not shared
assert all(len(p.constraints)==1 for p in rig.pose.bones if p.name.startswith('Secondary_SkirtFollow_'))
hum=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
calves=[getattr(hum,side+'_lower_leg').node.bone_name for side in ('left','right')]
front=[p for p in helpers if p.bone['hallway_knee_profile'][0]>.999]
assert len(front)==2 and {p.constraints[0].subtarget for p in front}==set(calves)
for helper in front:
 for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
 bpy.context.view_layer.update();baseline=helper.matrix.to_quaternion()
 for source in calves:
  pb=rig.pose.bones[source];pb.rotation_mode='QUATERNION';pb.rotation_quaternion=Quaternion((1,0,0),1.57079632679)
  bpy.context.view_layer.update()
  angle=baseline.rotation_difference(helper.matrix.to_quaternion()).angle
  expected=1.57079632679*helper.constraints[0].influence if source==helper.constraints[0].subtarget else 0.
  assert abs(angle-expected)<1e-5,(helper.name,source,angle,expected)
  pb.matrix_basis=Matrix.Identity(4);bpy.context.view_layer.update()
with rest_edit(rig):rebind_skirt_strips(rig,meshes)
maximum=max(abs(w.get(n,0)-weights(o,i).get(n,0)) for o in meshes for i,w in enumerate(binding[o.name]) for n in set(w)|set(weights(o,i)))
assert maximum<1e-6,maximum
verify(meshes,geom)
report=dict(lower_sections=len(helpers),shared_knee_sections=len(shared),upper_constraints=len(constraints),knee_constraints=len(list(follow_constraints(rig,'Skirt Knee'))),knee_junctions_continuous=True,calf_contacts_scoped=True,live_follow_all_constraints=True,geometry_unchanged=True,rebind_max_weight_error=maximum)
Path(output).write_text(json.dumps(report,indent=2));print('DRESS_RIG_OK',json.dumps(report),flush=True)
