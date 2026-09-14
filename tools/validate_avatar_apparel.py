"""Check garment semantics with isolated bone motions, separate from round trips."""
import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from avatar_apparel_weights import components, weights

OUT = ROOT/'outputs/avatar_grid'
BEFORE = '--before' in sys.argv


def capture(obj):
    deps = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(deps)
    mesh = evaluated.to_mesh()
    result = [evaluated.matrix_world @ v.co for v in mesh.vertices]
    evaluated.to_mesh_clear()
    return result


def motion(rig, obj, bone):
    for pb in rig.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    if bone:
        rig.pose.bones[bone].rotation_mode = 'QUATERNION'
        rig.pose.bones[bone].rotation_quaternion = Quaternion((0,0,1), .5)
    bpy.context.view_layer.update()
    return capture(obj)


report = {'baseline': BEFORE, 'avatars': [], 'checks': []}
for entry in json.loads((OUT/'sources/manifest.json').read_text()):
    name = entry['id']
    folder = OUT/('before_apparel_fix' if BEFORE else 'avatars')/name
    bpy.ops.wm.open_mainfile(filepath=str(folder/'02_fresh_rig.blend'))
    rig = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
    rig.animation_data_clear()
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    valid = all(abs(sum(g.weight for g in v.groups)-1) < 1e-5 and
                sum(g.weight > 1e-8 for g in v.groups) <= 4
                for o in meshes for v in o.data.vertices)
    rigid = json.loads(bpy.context.scene['preparation_audit'])['fresh_rig']['rigid_parts']
    neck = rig.data.bones['Neck']
    lower_movement = 0.
    count = 0
    for obj in meshes:
        if obj.name in rigid:
            continue
        indices = [v.index for v in obj.data.vertices
                   if v.co.z < neck.head_local.z-neck.length*.35]
        if not indices:
            continue
        count += len(indices)
        rest = motion(rig, obj, None)
        for bone in ('Head', 'Neck'):
            posed = motion(rig, obj, bone)
            lower_movement = max(lower_movement, max((posed[i]-rest[i]).length for i in indices))
    report['avatars'].append({'id': name, 'normalized_at_most_four_weights': valid,
                              'body_vertices_below_collar': count,
                              'lower_body_head_or_neck_only_displacement': lower_movement,
                              'passed': valid and count > 0 and lower_movement < 1e-5})
    if name in ('monk', 'rogue'):
        obj = bpy.data.objects['Monk.001' if name == 'monk' else 'Face']
        size = 146 if name == 'monk' else 29
        indices = [i for part in components(obj.data) if len(part) == size for i in part]
        assert len(indices) == (1460 if name == 'monk' else 29)
        rest = motion(rig, obj, None)
        head = motion(rig, obj, 'Head')
        neck = motion(rig, obj, 'Neck')
        transform = rig.matrix_world @ rig.pose.bones['Neck'].matrix @ rig.data.bones['Neck'].matrix_local.inverted() @ rig.matrix_world.inverted()
        head_move = max((head[i]-rest[i]).length for i in indices)
        neck_error = max((neck[i]-transform @ rest[i]).length for i in indices)
        fully_neck = all(abs(weights(obj, i).get('Neck', 0)-1) < 1e-6 for i in indices)
        if name == 'monk':
            hair = [i for part in components(obj.data) if len(part) != size for i in part]
            assert all(abs(weights(obj, i).get('Head', 0)-1) < 1e-6 for i in hair), 'Neckwear classification captured facial hair'
        report['checks'].append({'id': name, 'region': 'neckwear', 'vertices': len(indices),
                                 'fully_neck': fully_neck, 'head_only_displacement': head_move,
                                 'rigid_neck_transform_error': neck_error,
                                 'passed': fully_neck and head_move < 1e-5 and neck_error < 1e-5})
    if name in ('scifi', 'adventurer'):
        obj = bpy.data.objects['SciFi_Body' if name == 'scifi' else 'Adventurer_Body']
        parts = components(obj.data)
        body = max(parts, key=len)
        indices = [i for part in parts if part is not body for i in part]
        neck = rig.data.bones['Neck']
        lower = [i for i in indices if obj.data.vertices[i].co.z < neck.head_local.z-neck.length*.35]
        assert indices and lower
        rest = motion(rig, obj, None)
        head = motion(rig, obj, 'Head')
        neck_pose = motion(rig, obj, 'Neck')
        head_move = max((head[i]-rest[i]).length for i in indices)
        lower_neck_move = max((neck_pose[i]-rest[i]).length for i in lower)
        response = {}
        for bone in ('Spine', 'Chest', 'Neck', 'Clavicle.L', 'UpperArm.L', 'UpperArm.R'):
            posed = motion(rig, obj, bone)
            response[bone] = max((posed[i]-rest[i]).length for i in indices)
        report['checks'].append({'id': name, 'region': 'torso_clothing', 'vertices': len(indices),
                                 'head_only_displacement': head_move,
                                 'lower_torso_neck_only_displacement': lower_neck_move,
                                 'isolated_bone_displacements': response,
                                 'passed': head_move < 1e-5 and lower_neck_move < 1e-5 and min(response.values()) > 1e-4})
    if name == 'adventurer':
        obj = bpy.data.objects['Adventurer_Legs']
        rest = motion(rig, obj, None)
        moved = max((posed-rest[i]).length for bone in ('UpperArm.L', 'UpperArm.R', 'Forearm.L', 'Forearm.R')
                    for i, posed in enumerate(motion(rig, obj, bone)))
        report['checks'].append({'id': name, 'region': 'cut_base_fittings',
                                 'arm_only_displacement': moved, 'passed': moved < 1e-5})
report['passed'] = all(a['passed'] for a in report['avatars']) and all(c['passed'] for c in report['checks'])
(OUT/('apparel_validation_before.json' if BEFORE else 'apparel_validation.json')).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
if not BEFORE:
    assert report['passed'], report
