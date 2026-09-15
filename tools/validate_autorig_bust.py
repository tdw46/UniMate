"""Read-back geometry, skin weights, crease ownership, and evaluated motion."""
import json,sys
from pathlib import Path
import bpy,numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_apparel_weights import weights
from avatar_arm_boundary import crease_field
p=Path(__file__).resolve().parents[1]/'outputs/stitched_autorig'
bpy.ops.wm.open_mainfile(filepath=str(p/'00_imported.blend'))
o=next(o for o in bpy.context.scene.objects if o.type=='MESH')
geometry=([tuple(v.co) for v in o.data.vertices],[tuple(f.vertices) for f in o.data.polygons],[[tuple(v.uv) for v in uv.data] for uv in o.data.uv_layers])
bpy.ops.wm.open_mainfile(filepath=str(p/'rigged.blend'))
o=next(o for o in bpy.context.scene.objects if o.type=='MESH');r=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
assert geometry==([tuple(v.co) for v in o.data.vertices],[tuple(f.vertices) for f in o.data.polygons],[[tuple(v.uv) for v in uv.data] for uv in o.data.uv_layers])
points,_,signed,_=crease_field([o],r);tree=cKDTree(points);_,match=tree.query(np.array(geometry[0]))
body=[];crease=[];max_arm_body=0.;max_arm_crease=0.;min_torso_crease=1.
for v,i in zip(o.data.vertices,match):
 w=weights(o,v.index);assert len(w)<=4 and abs(sum(w.values())-1)<1e-5
 arm=sum(x for n,x in w.items() if n.endswith(('.L','.R')))
 if all(signed[side][i]<-r.data.bones['UpperArm.'+side].length*.22*.35 for side in ('L','R')):
  body.append(v.index);max_arm_body=max(max_arm_body,arm)
 elif all(signed[side][i]<=0 for side in ('L','R')):
  crease.append(v.index);max_arm_crease=max(max_arm_crease,arm);min_torso_crease=min(min_torso_crease,sum(w.get(n,0) for n in ('Root','Spine','Chest')))
assert max_arm_body<1e-7 and max_arm_crease<=.100001
assert min_torso_crease>=.89999
clouds={}
for frame in (0,26,106,200):
 bpy.context.scene.frame_set(frame);ev=o.evaluated_get(bpy.context.evaluated_depsgraph_get());m=ev.to_mesh();clouds[frame]=np.array([v.co[:] for v in m.vertices]);ev.to_mesh_clear()
# The arm phase must leave the torso surface still while the arms actually rise.
delta=np.linalg.norm(clouds[200]-clouds[0],axis=1)
assert delta[body].max()<1e-6 and delta.max()>.1
# Coincident imported atlas vertices must remain coincident throughout motion.
_,canonical=np.unique(np.round(np.array(geometry[0]),7),axis=0,return_inverse=True)
order=np.argsort(canonical);pairs=np.column_stack((order[:-1],order[1:]));pairs=pairs[canonical[pairs[:,0]]==canonical[pairs[:,1]]]
gap=max(np.linalg.norm(c[pairs[:,0]]-c[pairs[:,1]],axis=1).max() for c in clouds.values())
assert gap<1e-6
report={'blender':bpy.app.version_string,'vertices':len(geometry[0]),'triangles':len(geometry[1]),'bones':len(r.data.bones),'geometry_and_uvs_unchanged':True,'weights_normalized_max_four':True,'torso_vertices_checked':len(body),'crease_vertices_checked':len(crease),'maximum_torso_arm_weight':max_arm_body,'maximum_crease_arm_weight':max_arm_crease,'minimum_crease_torso_weight':min_torso_crease,'maximum_torso_motion_during_arm_raise':float(delta[body].max()),'maximum_arm_motion':float(delta.max()),'maximum_coincident_uv_vertex_gap':float(gap),'passed':True}
(p/'saved_rig_validation.json').write_text(json.dumps(report,indent=2));print('SAVED_RIG_PASSED',json.dumps(report),flush=True)
