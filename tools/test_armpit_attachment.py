"""Local shoulder heat continuity and protected-domain regression checks."""
import bpy,bmesh,os,sys,json
import numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_apparel_weights import weights,assign
from avatar_armpit_attachment import close_attachment
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
points=[];faces=[];flags=[];banks=[];rows=29;columns=9
for side in (0,1):
 start=len(points)
 for z in np.linspace(.6,.89,rows):
  for y in np.linspace(-.04,.04,columns):points.append((.3,y,z));flags.append(int(y>0));banks.append(side)
 for r in range(rows-1):
  for c in range(columns-1):
   a=start+r*columns+c;faces.append((a,a+1,a+1+columns,a+columns))
data=bpy.data.meshes.new('Two cut banks');data.from_pydata(points,[],faces);data.update();obj=bpy.data.objects.new('Two cut banks',data);bpy.context.collection.objects.link(obj)
data.attributes.new(name='hallway_generated_fill',type='INT',domain='POINT').data.foreach_set('value',flags)
data.attributes.new(name='hallway_arm_bank',type='INT',domain='POINT').data.foreach_set('value',banks)
assign(obj,list(range(rows*columns)),{'Clavicle.L':1.});assign(obj,list(range(rows*columns,2*rows*columns)),{'UpperArm.L':1.})
finger=rows*columns+18*columns+4;assign(obj,[finger],{'Index1.L':1.})
original=np.array([v.co[:] for v in data.vertices]);before=[weights(obj,v.index) for v in data.vertices]
report=close_attachment(rig,obj);assert report['pairs']>0,report
# Fully shared core: body and arm at z~.80, below the preserved cap.
a=20*columns+4;b=a+rows*columns
assert before[a]!=before[b] and weights(obj,a)==weights(obj,b), (weights(obj,a),weights(obj,b))
assert weights(obj,finger)=={'Index1.L':1.}
assert all(weights(obj,v.index).get('Index1.L',0)==0 for v in data.vertices if v.index!=finger), 'Finger weights leaked into attachment smoothing'
for v in data.vertices:
 if v.co.z>.86 or v.co.z<.68:assert weights(obj,v.index)==before[v.index], 'Cap or distal arm changed'
 assert len(weights(obj,v.index))<=4 and abs(sum(weights(obj,v.index).values())-1)<1e-6
np.testing.assert_array_equal(original,np.array([v.co[:] for v in data.vertices]))
assert close_attachment(rig,obj)=={'already_applied':True}
data.attributes.remove(data.attributes['hallway_generated_fill']);del obj['hallway_armpit_attachment_revision'];assert close_attachment(rig,obj)=={'vertices':0}
Path(os.environ['BEYOND_UI_TEST_RESULT']).write_text(json.dumps(dict(passed=True,attachment=report,matched_cut_rim_weights=True,protected_fingers_and_cap_unchanged=True,geometry_exact=True)))
