"""Read back geometry, heat preservation, atlas placement, and evaluated motion."""
import json,sys
from pathlib import Path
import bpy,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_apparel_weights import weights
p=Path(__file__).resolve().parents[1]/'outputs/stitched_autorig'
bpy.ops.wm.open_mainfile(filepath=str(p/'00_imported.blend'))
o=next(o for o in bpy.context.scene.objects if o.type=='MESH')
geometry=([tuple(v.co) for v in o.data.vertices],[tuple(f.vertices) for f in o.data.polygons],[[tuple(v.uv) for v in uv.data] for uv in o.data.uv_layers])
bpy.ops.wm.open_mainfile(filepath=str(p/'rigged.blend'))
o=next(o for o in bpy.context.scene.objects if o.type=='MESH');r=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
assert geometry==([tuple(v.co) for v in o.data.vertices],[tuple(f.vertices) for f in o.data.polygons],[[tuple(v.uv) for v in uv.data] for uv in o.data.uv_layers])
final_weights=[weights(o,i) for i in range(len(o.data.vertices))]
for w in final_weights:assert len(w)<=4 and abs(sum(w.values())-1)<1e-5
atlas=json.loads(bpy.context.scene['autorig_audit'])['atlas_correction']
assert atlas['applied']
assert np.allclose(r.data.bones['Head'].head_local,r.data.bones['Neck'].tail_local)
assert atlas['neck_center']<r.data.bones['Head'].head_local.y<atlas['neck_back']
clouds={}
for frame in (0,26,106,200):
 bpy.context.scene.frame_set(frame);ev=o.evaluated_get(bpy.context.evaluated_depsgraph_get());m=ev.to_mesh();clouds[frame]=np.array([v.co[:] for v in m.vertices]);ev.to_mesh_clear()
# Confirm the diagnostic really raises the arms.
delta=np.linalg.norm(clouds[200]-clouds[0],axis=1)
assert delta.max()>.1
# Coincident imported atlas vertices must remain coincident throughout motion.
_,canonical=np.unique(np.round(np.array(geometry[0]),7),axis=0,return_inverse=True)
order=np.argsort(canonical);pairs=np.column_stack((order[:-1],order[1:]));pairs=pairs[canonical[pairs[:,0]]==canonical[pairs[:,1]]]
gap=max(np.linalg.norm(c[pairs[:,0]]-c[pairs[:,1]],axis=1).max() for c in clouds.values())
assert gap<1e-6
# The seam pass must not reintroduce the rejected shoulder/arm constraint.
bpy.ops.wm.open_mainfile(filepath=str(p/'ordinary_heat.blend'))
o=next(o for o in bpy.context.scene.objects if o.type=='MESH')
heat=[weights(o,i) for i in range(len(o.data.vertices))]
# This fixture's atlas UV splits already share heat weights: exact preservation.
assert final_weights==heat
report={'blender':bpy.app.version_string,'vertices':len(geometry[0]),'triangles':len(geometry[1]),'bones':11,'geometry_and_uvs_unchanged':True,'weights_normalized_max_four':True,'all_weights_match_ordinary_heat':True,'atlas_correction':atlas,'maximum_arm_motion':float(delta.max()),'maximum_coincident_uv_vertex_gap':float(gap),'passed':True}
(p/'saved_rig_validation.json').write_text(json.dumps(report,indent=2));print('SAVED_RIG_PASSED',json.dumps(report),flush=True)
