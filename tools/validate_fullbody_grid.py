"""Verify full-body saved rigs, lower-limb skinning and planted ankle motion."""
import json,sys
from pathlib import Path
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from avatar_apparel_weights import weights
out=ROOT/'outputs/fullbody_avatar_grid';cases=[]
for entry in json.loads((out/'sources/manifest.json').read_text()):
 folder=out/'avatars'/entry['id'];audit=json.loads((folder/'preparation.json').read_text())
 assert audit['full_body'] and not audit['booleans'] and audit['cutoff_z_before_normalization'] is None
 bpy.ops.wm.open_mainfile(filepath=str(folder/'02_fresh_rig.blend'));scene=bpy.context.scene;scene.frame_set(0)
 rig=next(o for o in scene.objects if o.type=='ARMATURE');assert len(rig.data.bones)==52 and not rig.data.bones['Root'].use_deform
 meshes=[o for o in scene.objects if o.type=='MESH'];refs=[(o,v.index) for o in meshes for v in o.data.vertices]
 co=np.array([o.data.vertices[i].co[:] for o,i in refs]);values=[weights(o,i) for o,i in refs]
 norm=max(abs(sum(w.values())-1) for w in values);root=max(w.get('Root',0) for w in values)
 assert norm<1e-5 and root==0,(entry['id'],norm,root)
 feet=[];ankles={s:rig.pose.bones['Foot.'+s].head.copy() for s in ('L','R')};maximum_drift=0.
 for side in ('L','R'):
  tip=np.array(rig.data.bones['Toe.'+side].tail_local);ids=np.argsort(np.linalg.norm(co-tip,axis=1))[:10]
  own=float(np.mean([sum(values[k].get(b+'.'+side,0) for b in ('Thigh','Shin','Foot','Toe')) for k in ids]))
  assert own>.8,(entry['id'],side,own)
  feet.append({'side':side,'toe_sample_own_leg_mean_weight':own})
 hip_z=[]
 for frame in range(360,480):
  scene.frame_set(frame);hip_z.append(rig.pose.bones['Hips'].head.z)
  for s in ('L','R'):maximum_drift=max(maximum_drift,(rig.pose.bones['Foot.'+s].head-ankles[s]).length)
 assert maximum_drift<1e-5 and max(hip_z)-min(hip_z)>.15,(entry['id'],maximum_drift,hip_z)
 lifts={s:0. for s in ('L','R')}
 for frame in range(240,360):
  scene.frame_set(frame)
  for s in lifts:lifts[s]=max(lifts[s],rig.pose.bones['Foot.'+s].head.z-ankles[s].z)
 assert min(lifts.values())>.05,lifts
 # Check the evaluated surface as well as the pose bones.
 scene.frame_set(300);deps=bpy.context.evaluated_depsgraph_get();motion=0.
 for o in meshes:
  ev=o.evaluated_get(deps);mesh=ev.to_mesh();low=[v.index for v in o.data.vertices if v.co.z<rig.data.bones['Hips'].head_local.z*.75]
  if low:motion=max(motion,max((mesh.vertices[i].co-o.data.vertices[i].co).length for i in low))
  ev.to_mesh_clear()
 assert motion>.02
 case={'id':entry['id'],'bones':52,'source_full_body_preserved_without_boolean_cut':True,'nondeforming_root_has_no_skin_weights':True,'maximum_weight_sum_error':norm,'maximum_squat_ankle_drift':maximum_drift,'squat_hips_vertical_range':max(hip_z)-min(hip_z),'march_ankle_lifts':lifts,'lower_body_surface_motion':motion,'foot_samples':feet,'passed':True};cases.append(case)
 print('FULLBODY_VALIDATED',json.dumps(case),flush=True)
 (out/'fullbody_validation.json').write_text(json.dumps({'cases':cases,'passed':len(cases)==9},indent=2))
