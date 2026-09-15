"""Concave attachment cuts preserve limb heat and reject torso arm leakage."""
import json,math,sys
from pathlib import Path
import bpy
from mathutils import Matrix
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_apparel_weights import assign,weights
from avatar_arm_boundary import separate_arm_weights
report=[]
for segments,scale,offset in [(12,.5,(0,0,0)),(24,1.,(3,-2,4)),(32,2.3,(-1,4,2))]:
 bpy.ops.wm.read_factory_settings(use_empty=True)
 rig=bpy.data.objects.new('Anonymous',bpy.data.armatures.new('Anonymous'));bpy.context.scene.collection.objects.link(rig);bpy.context.view_layer.objects.active=rig;rig.select_set(True);bpy.ops.object.mode_set(mode='EDIT')
 specs=[('Root',(0,0,0),(0,0,.2)),('Spine',(0,0,.2),(0,0,.5)),('Chest',(0,0,.5),(0,0,.9)),('Neck',(0,0,.9),(0,0,1.1))]
 for side,sign in [('L',1),('R',-1)]:specs += [('UpperArm.'+side,(sign*.5,0,.8),(sign*1.,0,.8)),('Forearm.'+side,(sign*1.,0,.8),(sign*1.5,0,.8))]
 for name,a,b in specs:
  bone=rig.data.edit_bones.new(name);bone.head=a;bone.tail=b
 bpy.ops.object.mode_set(mode='OBJECT')
 profile=[(-1.5,.13),(-1.2,.15),(-.9,.16),(-.7,.16),(-.55,.12),(-.5,.085),(-.45,.14),(-.3,.35),(0,.4),(.3,.35),(.45,.14),(.5,.085),(.55,.12),(.7,.16),(.9,.16),(1.2,.15),(1.5,.13)]
 points=[(x,r*math.cos(math.tau*j/segments),.8+r*math.sin(math.tau*j/segments)) for x,r in profile for j in range(segments)]
 faces=[(k*segments+j,k*segments+(j+1)%segments,(k+1)*segments+(j+1)%segments,(k+1)*segments+j) for k in range(len(profile)-1) for j in range(segments)]
 faces += [tuple(reversed(range(segments))),tuple((len(profile)-1)*segments+j for j in range(segments))]
 data=bpy.data.meshes.new('Surface');data.from_pydata(points,[],faces);obj=bpy.data.objects.new('Anonymous surface',data);bpy.context.scene.collection.objects.link(obj);obj['binding_role']='body';obj['binding_surface']='skin'
 for i,p in enumerate(points):assign(obj,[i],{'Chest':.1,'UpperArm.'+('L' if p[0]>=0 else 'R'):.9})
 before=[weights(obj,i) for i in range(len(points))]
 transform=Matrix.Translation(offset)@Matrix.Scale(scale,4);data.transform(transform);rig.data.transform(transform)
 audit=separate_arm_weights([obj],rig)
 for i,p in enumerate(points):
  v=weights(obj,i);arm=sum(w for n,w in v.items() if n.endswith(('.L','.R')))
  assert abs(sum(v.values())-1)<1e-6 and len(v)<=4
  if abs(p[0])<.3:assert arm==0 and v.get('Chest')==1.
  if abs(p[0])>1.1:assert v==before[i],(segments,'distal heat changed',p,v)
 assert audit['maximum_torso_interior_arm_weight']==0
 assert audit['maximum_crease_torso_arm_weight']<=.100001
 assert all(s['concave_cut_edges']>0 for s in audit['sides'])
 report.append({'segments':segments,'scale':scale,'offset':offset,'crease_cut':audit,'passed':True})
p=Path(__file__).resolve().parents[1]/'outputs/complex_avatar_grid/arm_boundary_generalization.json';p.write_text(json.dumps(report,indent=2));print('CREASE_BOUNDARY_PASSED',json.dumps(report),flush=True)
