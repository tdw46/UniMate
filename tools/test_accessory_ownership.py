"""Mixed hip/arm accessory spill must not pull the accessory or claim fingers."""
import bpy,sys,os,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_accessory_ownership import confine_pelvis_accessories
from avatar_apparel_weights import assign,weights
bpy.ops.wm.read_factory_settings(use_empty=True)
rig=bpy.data.objects.new('Rig',bpy.data.armatures.new('Rig'));bpy.context.collection.objects.link(rig);bpy.context.view_layer.objects.active=rig;rig.select_set(True);bpy.ops.object.mode_set(mode='EDIT')
for n in ('Hips','Thigh.L','Forearm.L','Hand.L','Index1.L','Secondary_Panel'):
 b=rig.data.edit_bones.new(n);b.head=(0,0,1);b.tail=(0,0,.8)
bpy.ops.object.mode_set(mode='OBJECT')
m=bpy.data.meshes.new('Test');m.from_pydata([(0,0,.9)]*4+[(0,0,1.3)],[],[]);o=bpy.data.objects.new('Test',m);bpy.context.collection.objects.link(o)
rows=[{'Hips':.5,'Thigh.L':.15,'Forearm.L':.35},{'Index1.L':.8,'Hand.L':.2},{'Secondary_Panel':.2,'Hips':.6,'Forearm.L':.2},{'Forearm.L':.9,'Hips':.1},{'Hips':.7,'Forearm.L':.3}]
for i,row in enumerate(rows):assign(o,[i],row)
original=[weights(o,i) for i in range(5)];coords=[v.co.copy() for v in m.vertices]
result=confine_pelvis_accessories(rig,[o]);assert result['vertices']==1
assert 'Forearm.L' not in weights(o,0),'Pelvis accessory still follows the forearm'
assert abs(weights(o,0)['Hips']/weights(o,0)['Thigh.L']-.5/.15)<1e-5
assert [weights(o,i) for i in range(1,5)]==original[1:],'Legitimate finger, arm, cloth or torso weights changed'
assert all(v.co==c for v,c in zip(m.vertices,coords))
Path(os.environ['BEYOND_UI_TEST_RESULT']).write_text(json.dumps(dict(passed=True,**result)))
