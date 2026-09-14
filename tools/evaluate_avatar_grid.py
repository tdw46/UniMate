"""Validate nine prepared avatars through the UniMate reconstruction path."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bpy
import numpy as np
from scipy.spatial import cKDTree
from data_process.motion_export.export_general import export_asset
from data_process.mesh_animation.animate_npz import animate_character

OUT = Path(os.environ.get('AVATAR_EVAL_ROOT', ROOT / 'outputs/avatar_grid')).resolve()
SAMPLE_FRAMES = (0,30,59,90,119,150,179,210,239,270,299,330,359)


def snapshot(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 30
    bpy.ops.import_scene.gltf(filepath=str(path))
    rig = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
    names = sorted(rig.pose.bones.keys())
    rest = np.array([rig.matrix_world @ rig.data.bones[n].matrix_local for n in names],dtype=np.float64)
    transforms, vertices = [], []
    meshes = [o for o in bpy.context.scene.objects if o.type=='MESH' and any(m.type=='ARMATURE' for m in o.modifiers)]
    for frame in range(360):
        bpy.context.scene.frame_set(frame)
        transforms.append(np.array([rig.matrix_world @ rig.pose.bones[n].matrix for n in names],dtype=np.float64))
        if frame in SAMPLE_FRAMES:
            deps = bpy.context.evaluated_depsgraph_get()
            cloud = []
            for obj in meshes:
                evaluated = obj.evaluated_get(deps)
                mesh = evaluated.to_mesh()
                cloud.extend(tuple(evaluated.matrix_world @ v.co) for v in mesh.vertices)
                evaluated.to_mesh_clear()
            vertices.append(np.array(cloud))
    return names, rest, np.array(transforms), vertices


report = {'blender':bpy.app.version_string, 'motion_source':'Authored diagnostics reconstructed through UniMate; no pretrained model inference', 'avatars':[]}
for entry in json.loads((OUT/'sources/manifest.json').read_text()):
    name = entry['id']
    folder = OUT/'avatars'/name
    # Prove the saved intermediate, not only an in-memory counter.
    bpy.ops.wm.open_mainfile(filepath=str(folder/'01_cut_unrigged.blend'))
    meshes = [o for o in bpy.context.scene.objects if o.type=='MESH']
    assert len(bpy.data.armatures) == 0
    assert not any(o.vertex_groups for o in meshes)
    assert not any(v.groups for o in meshes for v in o.data.vertices)
    assert not any(m.type in ('BOOLEAN','ARMATURE') for o in meshes for m in o.modifiers)
    source = folder/'02_fresh_rig.glb'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    export_dir = folder/'export'/digest[:12]
    export_asset(str(source),str(export_dir),save_name=name,prune=False,remove_tpose=False,save_vis=False)
    motion = export_dir/'motions'/f'{name}-evaluation.npz'
    assert motion.exists(), motion
    input_names,input_rest,original,input_vertices = snapshot(source)
    animate_character(str(source),str(motion),str(folder/'reconstructed'),extra_bones_strategy='keep')
    result_path = folder/'reconstructed'/f'{name}-evaluation.glb'
    names,rest,result,result_vertices = snapshot(result_path)
    assert names == input_names
    d0 = original[...,:3,:3] @ np.linalg.inv(input_rest[...,:3,:3])
    d1 = result[...,:3,:3] @ np.linalg.inv(rest[...,:3,:3])
    u,_,vt = np.linalg.svd(d1 @ np.linalg.inv(d0))
    delta = u @ vt
    angle = np.degrees(np.arccos(np.clip((np.trace(delta,axis1=-2,axis2=-1)-1)/2,-1,1)))
    positions = np.linalg.norm(result[...,:3,3]-original[...,:3,3],axis=-1)
    surface = max(max(cKDTree(a).query(b)[0].max(),cKDTree(b).query(a)[0].max()) for a,b in zip(input_vertices,result_vertices))
    # Check that every requested phase genuinely changes the intended joints.
    motion_amplitudes = {}
    for phase,bone in enumerate(('Head','Neck','UpperArm.L')):
        matrices = result[phase*120:(phase+1)*120,names.index(bone),:3,:3]
        u,_,vt = np.linalg.svd(matrices @ np.linalg.inv(matrices[0]))
        variation = np.degrees(np.arccos(np.clip((np.trace(u@vt,axis1=-2,axis2=-1)-1)/2,-1,1)))
        motion_amplitudes[bone] = float(variation.max())
    case = {'id':name,'frames':360,'bones':len(names),'saved_unrigged_checkpoint_verified':True,
            'source_glb_sha256':digest,'max_rotation_error_degrees':float(angle.max()),
            'max_joint_position_error':float(positions.max()),'max_surface_error':float(surface),
            'surface_sample_frames':SAMPLE_FRAMES,'evaluated_motion_degrees':motion_amplitudes,
            'per_bone_rotation_error_degrees':dict(zip(names,map(float,angle.max(axis=0))))}
    case['passed'] = bool(angle.max()<.1 and positions.max()<1e-4 and surface<1e-4 and min(motion_amplitudes.values())>5)
    report['avatars'].append(case)
    (OUT/'pipeline_validation.json').write_text(json.dumps(report,indent=2))
    print('AVATAR_VALIDATED',json.dumps(case),flush=True)
    assert case['passed'],case
report['passed'] = len(report['avatars'])==9 and all(c['passed'] for c in report['avatars'])
(OUT/'pipeline_validation.json').write_text(json.dumps(report,indent=2))
print('NINE_AVATAR_PIPELINE_PASSED',flush=True)
