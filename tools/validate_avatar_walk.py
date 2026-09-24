"""Validate constrained skirt motion and actual VRM 1 export/readback."""
import argparse
import importlib
import json
import math
from pathlib import Path
import re
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from demo_avatar_springs import enable
from avatar_source_import import read_gltf
from avatar_springs import PREFIX
from avatar_apparel_weights import head_cap_vertices, weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT/'outputs/spring_walk_demo/final'))
    parser.add_argument('--source', default=str(ROOT/'outputs/spring_walk_demo/sources/AvatarSample_O.vrm'))
    parser.add_argument('--name', default='sample_o_walk')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    out = Path(args.output)
    for repo in bpy.context.preferences.extensions.repos:
        if repo.module == 'user_default':
            repo.use_custom_directory = True
            repo.custom_directory = str(Path.home()/'Documents/Blender/extensions/user_default')
    enable('vrm')
    bpy.ops.wm.open_mainfile(filepath=str(out/(args.name+'.blend')))
    scene = bpy.context.scene
    rig = next(o for o in scene.objects if o.type == 'ARMATURE')
    meshes = [o for o in scene.objects if o.type == 'MESH']
    cap = head_cap_vertices(meshes, rig)
    assert cap
    for obj in meshes:
        for i in cap.get(obj.name, []):
            assert weights(obj, i) == {'Head': 1.}, (obj.name, i, weights(obj, i))
    followers = [p for p in rig.pose.bones if p.name.startswith(PREFIX+'SkirtFollow_')]
    assert followers
    search = importlib.import_module('bl_ext.user_default.vrm.editor.search')
    bpy.context.view_layer.update()
    _, accepted, errors = search.export_constraints([rig]+meshes, rig)
    assert not errors, errors
    assert set(accepted.rotation_constraints) == {p.name for p in followers}
    for pb in followers:
        c = pb.constraints[0]
        assert c.type == 'COPY_ROTATION' and c.mix_mode == 'ADD'
        assert not c.mute and c.is_valid and c.target == rig
        assert c.owner_space == c.target_space == 'LOCAL'
        assert c.use_x and c.use_y and c.use_z
        assert not (c.invert_x or c.invert_y or c.invert_z)
        assert not getattr(c, 'vertex_group', '')
        assert c.subtarget in ('Thigh.L', 'Thigh.R')
        from avatar_skirt_follow import INFLUENCE
        assert abs(c.influence-INFLUENCE) < 1e-6
    # Real frame changes must drive the controls, with no doubled hip rotation.
    max_follow_error = 0.
    for frame in range(scene.frame_start, scene.frame_end+1):
        scene.frame_set(frame)
        for pb in followers:
            c = pb.constraints[0]
            rest = pb.bone.parent.matrix_local.inverted() @ pb.bone.matrix_local
            local = (pb.parent.matrix @ rest).inverted() @ pb.matrix
            actual = local.to_quaternion()
            target = rig.pose.bones[c.subtarget].rotation_quaternion
            expected = Quaternion().slerp(target, c.influence)
            error = min(max(abs(a-b) for a,b in zip(actual, expected)),
                        max(abs(a+b) for a,b in zip(actual, expected)))
            max_follow_error = max(max_follow_error, error)
    assert max_follow_error < 1e-5, max_follow_error
    sb = rig.data.vrm_addon_extension.spring_bone1
    fixed_roots = []
    for spring in sb.springs:
        if 'Hair' in spring.vrm_name:
            first = rig.data.bones[spring.joints[0].node.bone_name]
            assert first.name.endswith('_01') and first.parent.get('unimate_fixed_hair_root')
            assert first.parent.parent.name == 'Head'
            fixed_roots.append(first.parent.name)
        elif 'Skirt' in spring.vrm_name:
            assert all(abs(j.drag_force-.6) < 1e-6 for j in spring.joints)
    thighs = [c for c in sb.colliders if c.node.bone_name.startswith('Thigh.')]
    assert len(thighs) == 6
    for side in ('L', 'R'):
        cs = [c for c in thighs if c.node.bone_name == 'Thigh.'+side]
        assert len(cs) == 3
        assert min(c.shape.capsule.offset[1] for c in cs) < 1e-5
        assert abs(max(c.shape.capsule.tail[1] for c in cs)-rig.data.bones['Thigh.'+side].length) < 1e-5
        assert all(c.shape.capsule.radius > 0 for c in cs)
    groups = {g.uuid:{r.collider_uuid for r in g.colliders} for g in sb.collider_groups}
    thigh_ids = {c.uuid for c in thighs}
    for spring in sb.springs:
        if 'Skirt' in spring.vrm_name:
            assert thigh_ids <= set().union(*(groups[g.collider_group_uuid] for g in spring.collider_groups))
    # Keep source attribution and permission fields on the test export.
    source_meta = read_gltf(Path(args.source))['extensions']['VRMC_vrm']['meta']
    meta = rig.data.vrm_addon_extension.vrm1.meta
    meta.vrm_name = source_meta['name']+' - UniMate rig test'
    meta.authors.clear()
    for name in source_meta['authors']:
        meta.authors.add().value = name
    for key, value in source_meta.items():
        prop = re.sub(r'(?<!^)(?=[A-Z])', '_', key).lower()
        if key not in ('name', 'authors', 'thumbnailImage', 'licenseUrl') and hasattr(meta, prop):
            setattr(meta, prop, value)
    rig.animation_data_clear()
    for pb in rig.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action='DESELECT')
    for obj in [rig]+meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = rig
    export_path = out/(args.name+'.vrm')
    result = bpy.ops.export_scene.vrm(filepath=str(export_path),
        armature_object_name=rig.name, export_only_selections=True,
        export_invisibles=True, export_gltf_animations=False, ignore_warning=True)
    assert result == {'FINISHED'}, result
    gltf = read_gltf(export_path)
    nodes = gltf['nodes']
    constraints = {n['name']:n['extensions']['VRMC_node_constraint']['constraint']['rotation']
                   for n in nodes if 'VRMC_node_constraint' in n.get('extensions', {})}
    assert set(constraints) == {p.name for p in followers}, constraints
    for pb in followers:
        exported = constraints[pb.name]
        assert nodes[exported['source']]['name'] == pb.constraints[0].subtarget
        assert abs(exported.get('weight', 1)-pb.constraints[0].influence) < 1e-6
    assert len(gltf['extensions']['VRMC_springBone']['colliders']) == len(sb.colliders)
    exported_springs = gltf['extensions']['VRMC_springBone']['springs']
    exported_joint_names = {nodes[j['node']]['name'] for s in exported_springs for j in s['joints']}
    assert not set(fixed_roots) & exported_joint_names
    assert set(fixed_roots) <= {n['name'] for n in nodes}
    # Reimport through the official VRM importer, not a custom JSON reconstruction.
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    assert bpy.ops.import_scene.vrm(filepath=str(export_path)) == {'FINISHED'}
    imported = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
    for name, value in constraints.items():
        c = imported.pose.bones[name].constraints[0]
        assert c.type == 'COPY_ROTATION' and c.mix_mode == 'ADD'
        assert c.owner_space == c.target_space == 'LOCAL'
        assert c.subtarget == nodes[value['source']]['name']
    # Recorded spring poses must also survive saving with the constraints active.
    bpy.ops.wm.open_mainfile(filepath=str(out/(args.name+'_baked.blend')))
    rig = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
    samples = json.loads((out/'motion_samples.json').read_text())
    max_rotation, max_head = 0., 0.
    for sample in samples:
        bpy.context.scene.frame_set(sample['frame'])
        if 'object_location' in sample:
            assert (rig.location-Vector(sample['object_location'])).length < 1e-6
        for name, q in sample['bones'].items():
            actual = rig.pose.bones[name].rotation_quaternion
            max_rotation = max(max_rotation, min(max(abs(a-b) for a,b in zip(actual,q)), max(abs(a+b) for a,b in zip(actual,q))))
            max_head = max(max_head, (rig.pose.bones[name].head-Vector(sample['heads'][name])).length)
    assert max_rotation < 1e-6 and max_head < 2e-5, (max_rotation, max_head)
    report = dict(exported_rotation_constraints=len(constraints), imported_rotation_constraints=len(constraints),
                  fixed_hair_roots_excluded_from_physics=len(fixed_roots),
                  head_only_cap_vertices=sum(map(len, cap.values())),
                  exporter_accepted=True, circular_dependencies=errors, upper_leg_colliders=len(thighs),
                  max_leg_follow_quaternion_error=max_follow_error,
                  saved_bake_rotation_error=max_rotation, saved_bake_head_error=max_head,
                  recorded_frames=len(samples), export=str(export_path))
    (out/'export_validation.json').write_text(json.dumps(report, indent=2))
    print('WALK_EXPORT_VALIDATED', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
