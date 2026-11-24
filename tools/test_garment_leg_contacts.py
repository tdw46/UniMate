"""Unlabelled leg surfaces get continuous, rest-clear garment contacts."""
import bpy,sys,json,os,math
from pathlib import Path
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_garment_leg_contacts import plan_leg_contacts
bpy.ops.wm.read_factory_settings(use_empty=True)
for repo in bpy.context.preferences.extensions.repos:
 if repo.module=='user_default':repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
rig=bpy.data.objects.new('Rig',bpy.data.armatures.new('Rig'));bpy.context.collection.objects.link(rig);bpy.context.view_layer.objects.active=rig;rig.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
for side,x in [('L',.12),('R',-.12)]:
 for part,z in [('Thigh',.9),('Shin',.5)]:
  b=rig.data.edit_bones.new(part+'.'+side);b.head=(x,0,z);b.tail=(x,0,z-.4)
names=['Secondary_Accessory_SurfacePanel_00_01_'+str(i).zfill(2) for i in range(5)]
hips = rig.data.edit_bones.new('Hips')
hips.head, hips.tail = (0, 0, .9), (0, 0, 1.)
for side in ('L', 'R'):
 rig.data.edit_bones['Thigh.' + side].parent = hips
 rig.data.edit_bones['Shin.' + side].parent = rig.data.edit_bones['Thigh.' + side]
heights = (.85, .70, .50, .35, .19, .17)
for i,n in enumerate(names):
 b=rig.data.edit_bones.new(n)
 b.head, b.tail = (.12, .13, heights[i]), (.12, .13, heights[i+1])
 b.parent = rig.data.edit_bones[names[i-1]] if i else hips
short_names = [f'Secondary_Accessory_Jacket_short_{i:02d}' for i in range(2)]
for i,n in enumerate(short_names):
 b = rig.data.edit_bones.new(n)
 b.head, b.tail = (.12, .13, .93 - i*.05), (.12, .13, .89 - i*.05)
 b.parent = rig.data.edit_bones[short_names[i-1]] if i else hips
bpy.ops.object.mode_set(mode='OBJECT')
ext=rig.data.vrm_addon_extension;ext.spec_version='1.0'
for side,suffix in [('left','L'),('right','R')]:
 for part,bone in [('upper_leg','Thigh'),('lower_leg','Shin')]:getattr(ext.vrm1.humanoid.human_bones,side+'_'+part).node.bone_name=bone+'.'+suffix
spring=ext.spring_bone1.springs.add();spring.vrm_name=names[0].rsplit('_',1)[0]
for n in names:
 j=spring.joints.add();j.node.bone_name=n;j.hit_radius=.004
verts=[];groups={}
for side in ('L','R'):
 for part in ('Thigh','Shin'):
  n=part+'.'+side;b=rig.data.bones[n];ids=[]
  for y in (.1,.2,.3):
   for a in range(32):
    ids.append(len(verts));verts.append(b.matrix_local@Vector((.08*math.cos(a*math.tau/32),y,.08*math.sin(a*math.tau/32))))
  groups[n]=ids
mesh=bpy.data.meshes.new('Anonymous legs');mesh.from_pydata(verts,[],[]);obj=bpy.data.objects.new('Anonymous legs',mesh);bpy.context.collection.objects.link(obj)
for n,ids in groups.items():obj.vertex_groups.new(name=n).add(ids,1.,'REPLACE')
chain=dict(names=names,garment=True)
result,fits=plan_leg_contacts(rig,[obj],[chain],1.,'Secondary_')
assert all(abs(f['radius']-.08)<1e-5 for f in fits.values()),fits
specs=result[spring.vrm_name];rounds=[s for s in specs if s['kind']=='round'];guards=[s for s in specs if s['kind']=='directional']
assert len(rounds)==4 and len(guards)==2,(rounds,guards)
assert all(s['bone'].endswith('.L') for s in guards),'Opposite side directional support leaked'
assert all(abs(Vector(s['tail']).y-Vector(s['offset']).y-.4)<1e-5 for s in specs)
assert all(s['minimum_rest_clearance']>=.0015-1e-7 for s in specs)
assert min(s['radius'] for s in rounds)>.07,'Legs fell back to undersized material-based fit'
assert plan_leg_contacts(rig,[obj],[dict(names=names,garment=False)],1.,'Secondary_')[0]=={},'Non-garment spring captured'
# A below-mid-calf panel splits at its knee while a short hem keeps its attachment.
short = ext.spring_bone1.springs.add()
short.vrm_name = short_names[0].rsplit('_',1)[0]
for name in short_names:
 short.joints.add().node.bone_name = name
from avatar_garment_follow import install_garment_follow
sections, follows = install_garment_follow(rig, [chain, dict(names=short_names, garment=True)])
assert [f['target'] for f in follows] == ['Thigh.L', 'Shin.L']
assert len(sections) == 3 and len(short.joints) == 2
upper, lower, _ = sections
assert rig.data.bones[lower['names'][0]].head_local.z == .5
assert all(len(rig.pose.bones[f['bone']].constraints) == 1 for f in follows)
all_joints = [j.node.bone_name for s in ext.spring_bone1.springs for j in s.joints]
assert len(set(all_joints)) == len(all_joints)
assert not install_garment_follow(rig, sections)[1]
contacts, _ = plan_leg_contacts(rig, [obj], sections, 1., 'Secondary_')
assert any(s['bone'] == 'Shin.L' for s in contacts[lower['spring_name']])
bpy.ops.object.mode_set(mode='EDIT')
front_names = [f'Secondary_Accessory_Jacket_front_{i:02d}' for i in range(5)]
for i, name in enumerate(front_names):
 bone = rig.data.edit_bones.new(name)
 bone.head = (.10, -.24, heights[i])
 bone.tail = (.10, -.24, heights[i+1])
 bone.parent = rig.data.edit_bones[front_names[i-1]] if i else rig.data.edit_bones['Hips']
bpy.ops.object.mode_set(mode='OBJECT')
front = ext.spring_bone1.springs.add()
front.vrm_name = front_names[0].rsplit('_', 1)[0]
for name in front_names:
 front.joints.add().node.bone_name = name
contacts, _ = plan_leg_contacts(rig, [obj], [dict(names=front_names, garment=True, coat_hem=.46)], 1., 'Secondary_')
front_faces = [spec for spec in contacts[front.vrm_name] if spec['kind'] == 'directional']
assert front_faces and all(spec['radius'] < 1.54*.08 for spec in front_faces)
assert max(spec['radius'] for spec in guards) > max(spec['radius'] for spec in front_faces)
bpy.ops.object.mode_set(mode='EDIT')
edit = rig.data.edit_bones
spine = edit.new('Spine')
spine.head, spine.tail = (0.,0.,1.), (0.,0.,1.15)
spine.parent = edit['Hips']
upper_spine = edit.new('UpperSpine')
upper_spine.head, upper_spine.tail = (0.,0.,1.15), (0.,0.,1.3)
upper_spine.parent = spine
upper_names = [f'Secondary_Accessory_SurfacePanel_unit_Torso_U{i:02d}' for i in range(3)]
for i, name in enumerate(upper_names):
 bone = edit.new(name)
 bone.head, bone.tail = (.10,-.14,1.15-i*.08), (.10,-.14,1.07-i*.08)
 bone.parent = edit[upper_names[i-1]] if i else edit['Hips']
bpy.ops.object.mode_set(mode='OBJECT')
upper = ext.spring_bone1.springs.add()
upper.vrm_name = upper_names[0].rsplit('_', 1)[0]
for name in upper_names[1:]:
 upper.joints.add().node.bone_name = name
upper_chain = dict(names=upper_names, garment=True, coat_torso=True, coat_spine='UpperSpine')
contacts_plan, torso_follow = install_garment_follow(rig, [upper_chain])
assert len(torso_follow) == 1 and torso_follow[0]['target'] == 'UpperSpine'
helper = rig.pose.bones[torso_follow[0]['bone']]
assert len(helper.constraints) == 1 and helper.parent.parent.name == 'Spine'
assert upper.joints[-1].node.bone_name == upper_names[-1]
assert plan_leg_contacts(rig, [obj], contacts_plan, 1., 'Secondary_')[0] == {}
Path(os.environ['BEYOND_UI_TEST_RESULT']).write_text(json.dumps(dict(passed=True,
    round_capsules=len(rounds), directional_capsules=len(guards), rest_clear=True,
    coat_follows=len(follows), knee_split=True)))
