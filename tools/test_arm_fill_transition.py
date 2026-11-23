"""Legacy generated-fill experiment invariants; not part of rig generation."""
import bpy,bmesh,os,sys,json
import numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_apparel_weights import weights,assign
from avatar_arm_fill import round_generated_arm_fills,FILL_ATTRIBUTE
bpy.ops.wm.read_factory_settings(use_empty=True)
rig=bpy.data.objects.new('Test',bpy.data.armatures.new('Test'));bpy.context.collection.objects.link(rig);bpy.context.view_layer.objects.active=rig;rig.select_set(True);bpy.ops.object.mode_set(mode='EDIT')
for side,sign in [('L',1),('R',-1)]:
 for name,a,b,parent in [('Clavicle',(.1*sign,0,.9),(.3*sign,0,.9),None),('UpperArm',(.3*sign,0,.9),(.3*sign,0,.55),'Clavicle'),('Forearm',(.3*sign,0,.55),(.3*sign,0,.15),'UpperArm'),('Hand',(.3*sign,0,.15),(.3*sign,0,.08),'Forearm'),('Index1',(.3*sign,0,.08),(.3*sign,0,.03),'Hand')]:
  bone=rig.data.edit_bones.new(name+'.'+side);bone.head=a;bone.tail=b
  if parent:bone.parent=rig.data.edit_bones[parent+'.'+side]
bone=rig.data.edit_bones.new('Hips');bone.head=(0,0,.5);bone.tail=(0,0,.6)
bpy.ops.object.mode_set(mode='OBJECT')
n=96;levels=np.linspace(.18,.85,41);points=[];generated=[]
for z in levels:
 for angle in np.arange(n)*2*np.pi/n:
  fill=np.cos(angle)<-1e-6
  points.append((.3+(.02*abs(np.cos(angle)) if fill else .06*np.cos(angle)),.06*np.sin(angle),z));generated.append(int(fill))
faces=[(r*n+j,r*n+(j+1)%n,(r+1)*n+(j+1)%n,(r+1)*n+j) for r in range(len(levels)-1) for j in range(n)]
data=bpy.data.meshes.new('Fused repair');data.from_pydata(points,[],faces);data.update();obj=bpy.data.objects.new('Fused repair',data);bpy.context.collection.objects.link(obj)
fill=data.attributes.new(name=FILL_ATTRIBUTE,type='INT',domain='POINT');fill.data.foreach_set('value',generated)
bank=data.attributes.new(name='hallway_arm_bank',type='INT',domain='POINT');bank.data.foreach_set('value',np.ones(len(points),dtype=np.int32));assign(obj,list(range(len(points))),{'UpperArm.L':1.})
data.attributes.new(name='_hallway_source_normal_corner',type='INT',domain='CORNER')
source_normal=np.array([.2,.3,.93]);source_normal/=np.linalg.norm(source_normal)
data.normals_split_custom_set([tuple(source_normal)]*len(data.loops))
source={tuple(v.co) for v,g in zip(data.vertices,generated) if not g};center_index=20*n+n//2;center_before=data.vertices[center_index].co.x
bm=bmesh.new();bm.from_mesh(data);original_defects=sum(not e.is_manifold for e in bm.edges);bm.free()
report=round_generated_arm_fills(rig,[obj]);assert report and report[0]['generated_vertices_changed']>0,report
assert source <= {tuple(v.co) for v in data.vertices},'Round only generated cap interiors; original surface moved'
for face in data.polygons:
 if all(not data.attributes[FILL_ATTRIBUTE].data[j].value for j in face.vertices):
  assert all(np.dot(data.corner_normals[j].vector,source_normal)>.9999 for j in face.loop_indices),'Original custom normals were lost during fill refinement'
assert data.vertices[center_index].co.x<center_before-.01,'Concave generated center did not move toward the arm envelope'
bm=bmesh.new();bm.from_mesh(data);assert sum(not e.is_manifold for e in bm.edges)==original_defects,'Fill subdivision created non-manifold diagonals';bm.free()
assert data.attributes.get('_hallway_source_normal_corner'), 'Original corner attribute was removed'
assert not data.attributes.get('_hallway_source_normal_corner_'), 'Temporary normal correspondence leaked'
assert max(len(weights(obj,v.index)) for v in data.vertices)<=4
before=np.array([v.co[:] for v in data.vertices]);assert not round_generated_arm_fills(rig,[obj]);np.testing.assert_array_equal(before,np.array([v.co[:] for v in data.vertices]))
Path(os.environ['BEYOND_UI_TEST_RESULT']).write_text(json.dumps(dict(passed=True,curvature=report)))
