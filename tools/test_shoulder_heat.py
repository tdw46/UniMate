"""Local shoulder heat continuity and protected-domain regression checks."""
import bpy,bmesh,os,sys,json
import numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_apparel_weights import weights,assign
from avatar_shoulder_transition import refine_shoulder_heat
import avatar_arp_binding
from mathutils import Matrix,Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
rig=bpy.data.objects.new('Test',bpy.data.armatures.new('Test'));bpy.context.collection.objects.link(rig);bpy.context.view_layer.objects.active=rig;rig.select_set(True);bpy.ops.object.mode_set(mode='EDIT')
for side,sign in [('L',1),('R',-1)]:
 for name,a,b,parent in [('Clavicle',(.1*sign,0,.9),(.3*sign,0,.9),None),('UpperArm',(.3*sign,0,.9),(.3*sign,0,.55),'Clavicle'),('Forearm',(.3*sign,0,.55),(.3*sign,0,.15),'UpperArm'),('Hand',(.3*sign,0,.15),(.3*sign,0,.08),'Forearm'),('Index1',(.3*sign,0,.08),(.3*sign,0,.03),'Hand')]:
  bone=rig.data.edit_bones.new(name+'.'+side);bone.head=a;bone.tail=b
  if parent:bone.parent=rig.data.edit_bones[parent+'.'+side]
bone=rig.data.edit_bones.new('Hips');bone.head=(0,0,.5);bone.tail=(0,0,.6)
bpy.ops.object.mode_set(mode='OBJECT')
# Two cut-bank vertices at the same shoulder position must get one continuous
# source-surface heat solution. The distal arm and nearby fingers stay isolated.
points=[(.3,0,.865),(.3,0,.865),(.3,0,.62),(.3,0,.88),(.3,0,.89)]
data=bpy.data.meshes.new('Shoulder');data.from_pydata(points,[],[(0,2,3),(1,3,4)])
obj=bpy.data.objects.new('Shoulder',data);bpy.context.collection.objects.link(obj)
data.attributes.new(name='hallway_arm_bank',type='INT',domain='POINT')
for i,row in enumerate([{'UpperArm.L':.9999,'Hips':.0001},{'Clavicle.L':1},{'UpperArm.L':1},{'Index1.L':1},{'Hips':1}]):assign(obj,[i],row)
source_data=data.copy();source=bpy.data.objects.new('Uncut reference',source_data);bpy.context.collection.objects.link(source)
source.location=(4,2,3)
data.normals_split_custom_set([(0.,.6,.8)]*len(data.loops));original_normals=np.array([n.vector[:] for n in data.corner_normals])
original=np.array([v.co[:] for v in data.vertices]);bones={b.name:tuple(b.head_local) for b in rig.data.bones};objects=set(bpy.data.objects);meshes=set(bpy.data.meshes);armatures=set(bpy.data.armatures)
called=[]
def fake_heat(target,mesh,scale=1.):
    np.testing.assert_allclose(np.array([v.co[:] for v in mesh.data.vertices]).min(0),original.min(0),atol=1e-6)
    called.append(True)
    return [{'Clavicle.L':.5,'UpperArm.L':.5} for v in mesh.data.vertices],''
real=avatar_arp_binding._solve;avatar_arp_binding._solve=fake_heat
try:
    result=refine_shoulder_heat(rig,[obj],[source],world_to_rig=Matrix.Translation(-source.location))
finally:avatar_arp_binding._solve=real
assert called and result[0]['vertices']==2,result
assert weights(obj,0)==weights(obj,1)=={'Clavicle.L':.5,'UpperArm.L':.5}
assert weights(obj,2)=={'UpperArm.L':1} and weights(obj,3)=={'Index1.L':1} and weights(obj,4)=={'Hips':1}
assert bones=={b.name:tuple(b.head_local) for b in rig.data.bones}
np.testing.assert_array_equal(original_normals,np.array([n.vector[:] for n in data.corner_normals]))
np.testing.assert_array_equal(original,np.array([v.co[:] for v in data.vertices]))
assert objects==set(bpy.data.objects) and meshes==set(bpy.data.meshes) and armatures==set(bpy.data.armatures)
try:refine_shoulder_heat(rig,[obj],[])
except ValueError as error:assert 'one-to-one' in str(error)
else:raise AssertionError('Misaligned reference mesh mapping was accepted')
def failing_heat(*args,**kwargs):raise RuntimeError('injected solve failure')
avatar_arp_binding._solve=failing_heat
try:
    try:refine_shoulder_heat(rig,[obj],[source],world_to_rig=Matrix.Translation(-source.location))
    except RuntimeError as error:assert 'injected solve failure' in str(error)
    else:raise AssertionError('Expected the solver failure to propagate')
finally:avatar_arp_binding._solve=real
assert objects==set(bpy.data.objects) and meshes==set(bpy.data.meshes) and armatures==set(bpy.data.armatures)
data.attributes.remove(data.attributes['hallway_arm_bank']);assert refine_shoulder_heat(rig,[obj],[source])==[]
from mesh_landmarks.arm_separation import limit_generated_fill_weights
flag=data.attributes.new(name='hallway_generated_fill',type='INT',domain='POINT');flag.data[2].value=1
row={'UpperArm.L':.7,'Clavicle.L':.2,'Forearm.L':.09,'Hand.L':.009,'Index1.L':.001}
assign(obj,[2,3],row);assert len(weights(obj,2))==5
limited=limit_generated_fill_weights([obj]);assert limited['vertices']==1
assert len(weights(obj,2))==4 and abs(sum(weights(obj,2).values())-1)<1e-6
assert len(weights(obj,3))==5, 'Unmarked source row was modified by fill cleanup'
# The underside retains raw donor support mass until after mixing. A 60%
# supported donor at half strength must not become an amplified 100% donor.
data.attributes.new(name='hallway_arm_bank',type='INT',domain='POINT')
obj.data.vertices[2].co=(.3,0,.795);assign(obj,[2],{'UpperArm.L':1.})
def weak_heat(target,mesh,scale=1.):return [{'Clavicle.L':.3,'UpperArm.L':.3} for v in mesh.data.vertices],''
avatar_arp_binding._solve=weak_heat
try:refine_shoulder_heat(rig,[obj],[source],world_to_rig=Matrix.Translation(-source.location),include_root=False)
finally:avatar_arp_binding._solve=real
assert abs(weights(obj,2)['Clavicle.L']-.1875)<1e-6,weights(obj,2)
Path(os.environ['BEYOND_UI_TEST_RESULT']).write_text(json.dumps(dict(passed=True,heat=result,coincident_attachment_continuous=True,protected_domains_unchanged=True,temporary_data_cleaned=True)))
