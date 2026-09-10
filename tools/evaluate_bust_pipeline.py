"""Exercise actual UniMate GLB -> NPZ -> GLB, checking every bone/frame."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bpy
import numpy as np
from data_process.motion_export.export_general import export_asset
from data_process.mesh_animation.animate_npz import animate_character
from data_process.utils.blender_export import bind_action, clear_animation_state

OUT = ROOT / 'outputs/bust_evaluation'


def snapshot(path, action_name=None):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 30
    bpy.ops.import_scene.gltf(filepath=str(path))
    rig = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
    if action_name:
        clear_animation_state(rig)
        bind_action(rig, bpy.data.actions[action_name])
    names = sorted(rig.pose.bones.keys())
    # Compare neutral-relative world transforms, avoiding bone display-axis
    # assumptions in independently imported glTF armatures.
    neutral = np.array([rig.matrix_world @ rig.data.bones[n].matrix_local for n in names], dtype=np.float64)
    frames = []
    vertices = []
    meshes = sorted((o for o in bpy.context.scene.objects if o.type == 'MESH'), key=lambda o:o.name)
    for frame in range(120):
        bpy.context.scene.frame_set(frame)
        frames.append(np.array([rig.matrix_world @ rig.pose.bones[n].matrix for n in names], dtype=np.float64))
        # Every evaluated vertex, sampled on motion extrema and endpoints.
        if frame in (0, 30, 60, 90, 119):
            deps = bpy.context.evaluated_depsgraph_get()
            cloud = []
            for obj in meshes:
                ev = obj.evaluated_get(deps)
                mesh = ev.to_mesh()
                cloud.extend(tuple(ev.matrix_world @ v.co) for v in mesh.vertices)
                ev.to_mesh_clear()
            vertices.append(cloud)
    return names, neutral, np.array(frames), np.array(vertices)


report = {'blender': bpy.app.version_string, 'motion_source': 'authored diagnostic fixtures; no model inference', 'cases': []}
for name in ('slender', 'broad'):
    source = OUT / f'{name}.glb'
    exports = OUT / 'export' / name
    export_asset(str(source), str(exports), prune=False, remove_tpose=False, save_vis=False)
    for clip in ('head', 'neck', 'arms'):
        npz = exports / 'motions' / f'{name}-{clip}.npz'
        assert npz.is_file(), npz
        original_names, original_rest, original, original_vertices = snapshot(source, clip)
        animate_character(str(source), str(npz), str(OUT / 'reconstructed' / name), extra_bones_strategy='keep')
        target = OUT / 'reconstructed' / name / f'{name}-{clip}.glb'
        names, rest, result, result_vertices = snapshot(target)
        assert names == original_names
        original_delta = original[..., :3, :3] @ np.linalg.inv(original_rest[..., :3, :3])
        result_delta = result[..., :3, :3] @ np.linalg.inv(rest[..., :3, :3])
        relative = result_delta @ np.linalg.inv(original_delta)
        # Orthogonalize the float32 Blender matrices before angle extraction.
        u, _, vt = np.linalg.svd(relative)
        relative = u @ vt
        angle = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1)))
        position_error = np.linalg.norm(result[..., :3, 3] - original[..., :3, 3], axis=-1)
        from scipy.spatial import cKDTree
        surface_error = max(max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max()) for a,b in zip(original_vertices, result_vertices))
        entry = {'bust':name, 'clip':clip, 'frames':len(result), 'bones':len(names),
                 'max_rotation_error_degrees':float(angle.max()),
                 'max_joint_position_error':float(position_error.max()),
                 'max_surface_distance_at_5_frames':float(surface_error),
                 'per_bone_rotation_error_degrees': dict(zip(names, map(float, angle.max(axis=0))))}
        entry['passed'] = bool(angle.max() < .1 and position_error.max() < 1e-4 and surface_error < 1e-4)
        report['cases'].append(entry)
        (OUT / 'validation.json').write_text(json.dumps(report, indent=2))
        print('CASE_RESULT', json.dumps(entry), flush=True)
        assert entry['passed'], entry
report['passed'] = all(c['passed'] for c in report['cases']) and len(report['cases']) == 6
(OUT / 'validation.json').write_text(json.dumps(report, indent=2))
print('PIPELINE_VALIDATED', report['passed'], flush=True)
