"""Read saved spread/aligned rigs and independently verify the skin rest bake."""
import json,sys,os
from pathlib import Path
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from avatar_apparel_weights import weights
from avatar_source_import import material_role
from avatar_fingers import DIGITS
from validate_seam_locality import validate as validate_seam_patch
FULLBODY='--full-body' in sys.argv
out=Path(os.environ.get('AVATAR_EVAL_ROOT',ROOT/('outputs/fullbody_avatar_grid' if FULLBODY else 'outputs/finger_avatar_grid')));entries=json.loads((out/'sources/manifest.json').read_text());report=[]

def read(path):
 bpy.ops.wm.open_mainfile(filepath=str(path));r=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
 result={}
 for o in bpy.context.scene.objects:
  if o.type!='MESH':continue
  result[o.name]={'co':np.array([v.co[:] for v in o.data.vertices]),'weights':[weights(o,v.index) for v in o.data.vertices],
  'faces':[tuple(f.vertices) for f in o.data.polygons],'uv':[[tuple(x.uv) for x in layer.data] for layer in o.data.uv_layers]}
 return result,{b.name:np.array(b.matrix_local) for b in r.data.bones},{b.name:b.length for b in r.data.bones}

for entry in entries:
 folder=out/'avatars'/entry['id']
 seam=validate_seam_patch(folder/'02_heat_baseline.blend',folder/'03_spread_bound.blend')
 spread,old,old_lengths=read(folder/'03_spread_bound.blend');aligned,new,new_lengths=read(folder/'04_aligned_rest.blend')
 assert spread.keys()==aligned.keys() and old.keys()==new.keys() and len(new)==(52 if FULLBODY else 43)
 length_error=max(abs(new_lengths[n]-old_lengths[n]) for n in old)
 thumb_error=max(float(np.abs(np.linalg.inv(new['Hand.'+s])@new[f'Thumb{i}.{s}']-np.linalg.inv(old['Hand.'+s])@old[f'Thumb{i}.{s}']).max()) for s in ('L','R') for i in (1,2,3))
 alignment=min(new[f'{d}{i}.{s}'][0,1]*sign for s,sign in [('L',1),('R',-1)] for d in DIGITS[1:] for i in (1,2,3))
 assert length_error<1e-5 and thumb_error<1e-5 and alignment>.99999
 error=change=0.
 for name,before in spread.items():
  after=aligned[name]
  for field in ('weights','faces','uv'):assert before[field]==after[field],(entry['id'],name,field)
  points=np.column_stack((before['co'],np.ones(len(before['co']))));expected=np.zeros((len(points),3))
  for bone in old:
   influence=np.array([w.get(bone,0) for w in before['weights']])
   expected+=((new[bone]@np.linalg.inv(old[bone])@points.T).T[:,:3])*influence[:,None]
  error=max(error,float(np.linalg.norm(expected-after['co'],axis=1).max()))
  change=max(change,float(np.linalg.norm(before['co']-after['co'],axis=1).max()))
 assert error<1e-5 and change>.01,(entry['id'],error,change)
 rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
 assert all(np.max(np.abs(np.array(pb.matrix_basis)-np.eye(4)))<1e-6 for pb in rig.pose.bones)
 bpy.ops.wm.open_mainfile(filepath=str(folder/'02_fresh_rig.blend'));rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
 bpy.context.scene.frame_set(0)
 skin=[o for o in bpy.context.scene.objects if o.type=='MESH' and any(material_role(m)=='body' for m in o.data.materials)]
 refs=[(o,v.index) for o in skin for v in o.data.vertices]
 coordinates=np.array([o.data.vertices[i].co[:] for o,i in refs]);tip_checks=[]
 for side in ('L','R'):
  for digit in DIGITS:
   tip=np.array(rig.data.bones[f'{digit}3.{side}'].tail_local)
   nearest=np.argsort(np.linalg.norm(coordinates-tip,axis=1))[:5]
   own=[];other=[]
   for idx in nearest:
    o,i=refs[idx];w=weights(o,i)
    own.append(sum(w.get(f'{digit}{j}.{side}',0) for j in (1,2,3)))
    other.append(sum(w.get(f'{d}{j}.{side}',0) for d in DIGITS if d!=digit for j in (1,2,3)))
   # Independent surface-motion check on those fingertip samples.
   frame=555 if FULLBODY else 120+12+24*DIGITS.index(digit);bpy.context.scene.frame_set(frame);deps=bpy.context.evaluated_depsgraph_get();cloud={}
   for o in skin:
    ev=o.evaluated_get(deps);mesh=ev.to_mesh();cloud[o.name]=np.array([v.co[:] for v in mesh.vertices]);ev.to_mesh_clear()
   if FULLBODY:
    # Remove wrist/arm motion so this proves actual finger deformation.
    hand=rig.pose.bones['Hand.'+side]
    unpose=np.array(rig.data.bones[hand.name].matrix_local@hand.matrix.inverted())
    for name,points in cloud.items():cloud[name]=(unpose@np.column_stack((points,np.ones(len(points)))).T).T[:,:3]
   motion=float(np.mean([np.linalg.norm(cloud[refs[k][0].name][refs[k][1]]-coordinates[k]) for k in nearest]))
   length=sum(rig.data.bones[f'{digit}{j}.{side}'].length for j in (1,2,3))
   check={'side':side,'digit':digit,'tip_own_chain_mean_weight':float(np.mean(own)),'tip_other_digits_mean_weight':float(np.mean(other)),'tip_motion_in_finger_lengths':motion/length}
   assert np.mean(own)>.85 and np.mean(other)<.15 and motion/length>.20,check
   tip_checks.append(check)
 case={'id':entry['id'],'bones':len(new),'mesh_uv_topology_and_weights_preserved_during_rest_bake':True,
       'maximum_independent_skinning_rest_error':error,'maximum_rest_vertex_change':change,'zero_pose_after_bake':True,'maximum_bone_length_change':length_error,'maximum_thumb_relative_hand_rest_error':thumb_error,'minimum_finger_alignment_dot':float(alignment),'local_seam_validation':seam,'tip_checks':tip_checks,'passed':True}
 report.append(case);print('HAND_VALIDATED',json.dumps(case),flush=True)
 (out/'hand_validation.json').write_text(json.dumps({'cases':report,'passed':len(report)==9},indent=2))
