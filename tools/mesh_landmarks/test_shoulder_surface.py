"""Recover buried fused-arm sockets, preserving good sockets and distal joints."""
import bpy,copy,json,os,sys,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_landmarks.blender_bridge import geometry,geometry_digest
from mesh_landmarks.shoulder_surface import fit_fused_shoulders
from mesh_landmarks.arm_separation import plan


def fixture(fused,low=True):
 bpy.ops.wm.read_factory_settings(use_empty=True);objects=[];x=.19 if fused else .33
 for at,scale in [((0,0,.45),(.15,.10,.38)),((x,0,.42),(.09,.075,.37)),((-x,0,.42),(.09,.075,.37))]:
  bpy.ops.mesh.primitive_uv_sphere_add(segments=48,ring_count=32,location=at);o=bpy.context.object;o.scale=scale;bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);objects.append(o)
 o=objects[0];bpy.context.view_layer.objects.active=o
 for other in objects[1:]:
  mod=o.modifiers.new('Fixture union','BOOLEAN');mod.operation='UNION';mod.solver='EXACT';mod.object=other;bpy.ops.object.modifier_apply(modifier=mod.name);bpy.data.objects.remove(other,do_unlink=True)
 joints={'Hips':dict(head=[0,0,.4],tail=[0,0,.5]),'Neck':dict(head=[0,0,.80],tail=[0,0,.9])}
 for side,sign in [('Left',1),('Right',-1)]:
  head=.64 if low else .71
  for name,a,b in [('Shoulder',[sign*.05,0,.74],[sign*x,0,head]),('Arm',[sign*x,0,head],[sign*x,0,.35]),('ForeArm',[sign*x,0,.35],[sign*x,0,.12]),('Hand',[sign*x,0,.12],[sign*x,0,.06])]:joints[side+name]=dict(head=a,tail=b)
 return o,dict(joints=joints)

results=[]
for fused,low in ((True,True),(True,False),(False,True)):
 obj,pred=fixture(fused,low);v,f=geometry([obj]);digest=geometry_digest(v,f);original=copy.deepcopy(pred)
 report=fit_fused_shoulders(v,f,pred)
 assert bool(report['adjusted'])==(fused and low),(fused,low,report)
 for side in ('Left','Right'):
  assert pred['joints'][side+'ForeArm']==original['joints'][side+'ForeArm']
  assert pred['joints'][side+'Hand']==original['joints'][side+'Hand']
  assert pred['joints'][side+'Arm']['tail']==original['joints'][side+'Arm']['tail']
 if report['adjusted']:
  for row in report['adjusted']:
   assert row['after'][2]>row['before'][2]+.03
   assert pred['joints'][row['side']+'Shoulder']['tail']==row['after']
  for cut in plan(v,f,pred):
   a=np.array(pred['joints'][cut['side']+'Arm']['head']);b=np.array(pred['joints'][cut['side']+'ForeArm']['head'])
   assert cut['curve'][0][1]>=a[2]-.15*np.linalg.norm(a-b),'Underarm release stops too far below socket'
  moved=copy.deepcopy(original);offset=np.array([2.,-3.,4.]);scale=2.3
  for joint in moved['joints'].values():
   for key in ('head','tail'):joint[key]=(np.array(joint[key])*scale+offset).tolist()
  transformed=fit_fused_shoulders(v*scale+offset,f,moved)
  for a,b in zip(report['adjusted'],transformed['adjusted']):np.testing.assert_allclose(b['after'],np.array(a['after'])*scale+offset,atol=2e-6)
  # Prepared caps must not trigger a second anatomical fit.
  pred['shoulder_socket_refinement']=report;before=copy.deepcopy(pred)
  assert fit_fused_shoulders(v,f,pred)==report and pred==before
 assert geometry_digest(*geometry([obj]))==digest
 results.append(dict(fused=fused,low=low,adjustments=report['adjusted']))
Path(os.environ['BEYOND_UI_TEST_RESULT']).write_text(json.dumps(dict(passed=True,cases=results)))
print('SHOULDER_SURFACE_PASSED',results)
