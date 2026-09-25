"""Check experimental rest edits, shape-key deltas and native VRM export."""
import json
from pathlib import Path
import struct
import sys
import bpy
import numpy as np

import argparse
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('candidate');p.add_argument('output');p.add_argument('--rest-repair-report')
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);source,candidate,output=a.source,a.candidate,a.output
removed=json.loads(Path(a.rest_repair_report).read_text()).get('removed_polygons',{}) if a.rest_repair_report else {}
out=Path(output);out.mkdir(parents=True,exist_ok=True)
for repo in bpy.context.preferences.extensions.repos:
    if repo.module=='user_default':
        repo.use_custom_directory=True
        repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
bpy.ops.wm.open_mainfile(filepath=str(Path(source).resolve()))
rig=next(o for o in bpy.data.objects if o.type=='ARMATURE' and o.get('unimate_secondary_generator'))

def arrays(obj):
    vertices=np.array([tuple(v.co) for v in obj.data.vertices])
    keys={k.name:np.array([tuple(v.co) for v in k.data]) for k in obj.data.shape_keys.key_blocks} if obj.data.shape_keys else {}
    topology=[(tuple(p.vertices),p.material_index) for p in obj.data.polygons]
    uv=[(u.name,[[tuple(u.data[i].uv) for i in p.loop_indices] for p in obj.data.polygons]) for u in obj.data.uv_layers]
    return vertices,keys,topology,uv

before={o.name:arrays(o) for o in rig.children if o.type=='MESH'}
def binding(obj):
    groups={g.index:g.name for g in obj.vertex_groups}
    return [[(groups[g.group],g.weight) for g in v.groups] for v in obj.data.vertices]
bindings={o.name:binding(o) for o in rig.children if o.type=='MESH'}
bpy.ops.wm.open_mainfile(filepath=str(Path(candidate).resolve()))
rig=next(o for o in bpy.data.objects if o.type=='ARMATURE' and o.get('unimate_secondary_generator'))
changed=0;maximum=0.;key_error=0.
meshes=[o for o in rig.children if o.type=='MESH']
for obj in meshes:
    original,old_keys,old_topology,old_uv=before[obj.name]
    vertices,keys,topology,uv=arrays(obj)
    dropped=set(removed.get(obj.name,()))
    assert [p for i,p in enumerate(old_topology) if i not in dropped]==topology
    assert [(name,[p for i,p in enumerate(data) if i not in dropped]) for name,data in old_uv]==uv
    assert old_keys.keys()==keys.keys()
    if a.rest_repair_report:assert bindings[obj.name]==binding(obj),'Weights changed during rest repair'
    delta=vertices-original
    changed+=int(np.count_nonzero(np.linalg.norm(delta,axis=1)>1e-7))
    maximum=max(maximum,float(np.max(np.linalg.norm(delta,axis=1))))
    for name,positions in keys.items():
        key_error=max(key_error,float(np.max(np.abs((positions-old_keys[name])-delta))))
assert key_error<1e-6,key_error
sb=rig.data.vrm_addon_extension.spring_bone1
joint_names=[j.node.bone_name for s in sb.springs for j in s.joints]
assert len(joint_names)==len(set(joint_names)), 'Duplicated spring joints'
for obj in bpy.context.view_layer.objects:obj.select_set(False)
for obj in [rig]+meshes:obj.select_set(True)
bpy.context.view_layer.objects.active=rig
meta=rig.data.vrm_addon_extension.vrm1.meta
meta.vrm_name='Skirt Experiment'
if not meta.authors:meta.authors.add().value='Local validation'
target=out/'candidate.vrm'
assert bpy.ops.export_scene.vrm(filepath=str(target),armature_object_name=rig.name,
    export_only_selections=True,export_invisibles=True,export_gltf_animations=False,
    ignore_warning=True)=={'FINISHED'}
raw=target.read_bytes();size,_=struct.unpack_from('<II',raw,12)
gltf=json.loads(raw[20:20+size]);spring_data=gltf['extensions']['VRMC_springBone']
assert len(spring_data['colliders'])==len(sb.colliders)
report=dict(changed_vertices=changed,maximum_rest_displacement=maximum,
            maximum_shape_key_delta_error=key_error,uv_material_slots_preserved=True,topology_preserved=not bool(removed),removed_hidden_cap_faces=sum(map(len,removed.values())),weights_preserved=True if a.rest_repair_report else None,
            exported_colliders=len(spring_data['colliders']),exported_springs=len(spring_data['springs']),
            unique_spring_joints=True,
            extended_colliders=sum('VRMC_springBone_extended_collider' in c.get('extensions',{})
                                   for c in spring_data['colliders']))
(out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('SKIRT_EXPORT_VALIDATED',json.dumps(report),flush=True)
