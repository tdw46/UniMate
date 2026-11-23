"""Hand skin stays owned; distant garment vertices cannot follow fingertips."""
import bpy,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_hand_binding import confine_hand_weights
from avatar_apparel_weights import assign,weights
bpy.ops.wm.read_factory_settings(use_empty=True)
rig=bpy.data.objects.new('Rig',bpy.data.armatures.new('Rig'));bpy.context.scene.collection.objects.link(rig);bpy.context.view_layer.objects.active=rig;rig.select_set(True);bpy.ops.object.mode_set(mode='EDIT')
for n,a,b in [('Hips',(0,0,0),(0,0,.2)),('Thigh.L',(.1,0,0),(.1,0,-.3)),('Hand.L',(.2,0,.1),(.2,0,.04)),('Index1.L',(.2,0,.04),(.2,0,.01)),('Index2.L',(.2,0,.01),(.2,0,-.02)),('Index3.L',(.2,0,-.02),(.2,0,-.04))]:
 bone=rig.data.edit_bones.new(n);bone.head=a;bone.tail=b
bpy.ops.object.mode_set(mode='OBJECT')
mesh=bpy.data.meshes.new('Surface');mesh.from_pydata([(.205,0,-.025),(.15,0,-.18),(.13,0,-.16)],[],[(0,1,2)]);obj=bpy.data.objects.new('Surface',mesh);bpy.context.scene.collection.objects.link(obj)
assign(obj,[0],{'Index3.L':1});assign(obj,[1],{'Index3.L':1});assign(obj,[2],{'Index2.L':.7,'Thigh.L':.3})
report=confine_hand_weights(rig,[obj])
assert weights(obj,0)=={'Index3.L':1.},'Real fingertip changed'
for i in (1,2):
 row=weights(obj,i);assert not any(n.startswith('Index') for n in row),row;assert abs(sum(row.values())-1)<1e-6
assert weights(obj,2)=={'Thigh.L':1.}
from mesh_landmarks.arm_separation import enforce_banks
bank=mesh.attributes.new(name='hallway_arm_bank',type='INT',domain='POINT');bank.data.foreach_set('value',[1,0,0])
strength=mesh.attributes.new(name='hallway_arm_bank_strength',type='FLOAT',domain='POINT');strength.data.foreach_set('value',[1.,1.,1.])
assign(obj,[1],{'Index3.L':1.});assign(obj,[2],{'Hand.L':1.})
assert enforce_banks(rig,[obj])[0]['body_bank_vertices_corrected']==2
assert weights(obj,0)=={'Index3.L':1.}
assert all(not n.startswith(('Hand','Index')) for i in (1,2) for n in weights(obj,i))
from mesh_landmarks.arm_separation import close_shoulder_weights
mesh2=bpy.data.meshes.new('Shoulder contact rims')
mesh2.from_pydata([(.1,0,.2),(.1,0,.2),(.2,0,.05),(.2,0,.05)],[],[])
obj2=bpy.data.objects.new('Shoulder contact rims',mesh2);bpy.context.scene.collection.objects.link(obj2)
for name,kind,values in [('hallway_arm_bank','INT',[0,1,0,1]),('hallway_arm_join','FLOAT',[1.,1.,0.,0.])]:
 attr=mesh2.attributes.new(name=name,type=kind,domain='POINT');attr.data.foreach_set('value',values)
assign(obj2,[0,2],{'Hips':1.});assign(obj2,[1,3],{'Hand.L':1.})
assert close_shoulder_weights(rig,[obj2])['shoulder_join_vertices']==2
assert weights(obj2,0)==weights(obj2,1), 'Coincident shoulder rims split under pose'
assert weights(obj2,2)=={'Hips':1.} and weights(obj2,3)=={'Hand.L':1.},'Wrist contact was reattached'
Path(os.environ['BEYOND_UI_TEST_RESULT']).write_text(json.dumps(dict(passed=True,report=report)))
print('HAND_DOMAIN_PASSED')
