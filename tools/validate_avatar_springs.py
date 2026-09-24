"""Read back spring scenes and verify generation across the nine prepared rigs."""
import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Quaternion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from avatar_springs import generate_secondary, plan_secondary, PREFIX
from avatar_apparel_weights import weights, head_cap_vertices, enforce_head_cap
from demo_avatar_springs import fingerprint, enable


def evaluated_points(obj):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        return [v.co.copy() for v in mesh.vertices]
    finally:
        evaluated.to_mesh_clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT/'outputs/spring_avatar_demo'))
    parser.add_argument('--generation-only', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    out = Path(args.output)
    for repo in bpy.context.preferences.extensions.repos:
        if repo.module == 'user_default':
            repo.use_custom_directory = True
            repo.custom_directory = str(Path.home()/'Documents/Blender/extensions/user_default')
    enable('vrm')
    results = []
    for path in sorted((ROOT/'outputs/fullbody_avatar_grid/avatars').glob('*/02_fresh_rig.blend')):
        bpy.ops.wm.open_mainfile(filepath=str(path))
        bpy.context.scene.frame_set(0)
        rig = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
        meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
        geometry = {o.name: fingerprint(o) for o in meshes}
        before = {o.name: [weights(o, i) for i in range(len(o.data.vertices))] for o in meshes}
        plan = plan_secondary(rig, meshes)
        touched = {o.name: set() for o in meshes}
        for region in plan['regions']:
            touched[region['obj'].name].update(region['ids'])
        result = generate_secondary(rig, meshes) if plan['chains'] else {'chains': 0, 'rejected': plan['rejected']}
        assert all(not s.center.bone_name for s in rig.data.vrm_addon_extension.spring_bone1.springs if s.vrm_name.startswith(PREFIX))
        if not plan['chains']:
            enforce_head_cap(meshes, rig)
        for chain in plan['chains']:
            if chain['role'] != 'hair':
                continue
            root_bone = rig.data.bones[chain['names'][0]]
            assert root_bone.parent.name == 'Head' and root_bone.get('unimate_fixed_hair_root')
            assert (root_bone.head_local-chain['line'][0]).length < 1e-6
            spring = next(s for s in rig.data.vrm_addon_extension.spring_bone1.springs
                          if s.joints[0].node.bone_name == chain['names'][1])
            assert root_bone.name not in {j.node.bone_name for j in spring.joints}
        result['avatar'] = path.parent.name
        assert all(geometry[o.name] == fingerprint(o) for o in meshes)
        assert all(before[o.name][i] == weights(o, i) for o in meshes for i in range(len(o.data.vertices)) if i not in touched[o.name])
        assert all(abs(sum(weights(o, i).values())-1) < 1e-6 for o in meshes for i in touched[o.name])
        for obj in meshes:
            for i in touched[obj.name]:
                assert len(weights(obj, i)) <= 4
        cap = head_cap_vertices(meshes, rig)
        for obj in meshes:
            for i in cap.get(obj.name, []):
                assert weights(obj, i) == {'Head': 1.}, (obj.name, i, weights(obj, i))
        # Rest skinning must not change the evaluated surface when chains are added.
        rig.animation_data_clear()
        for pb in rig.pose.bones:
            pb.matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.update()
        rest_error = max((a-b.co).length for o in meshes for a, b in zip(evaluated_points(o), o.data.vertices))
        assert rest_error < 1e-5, (path, rest_error)
        if cap:
            # Move neck/head independently of the arms and free hair. A cap
            # vertex must still equal the rigid Head transform exactly.
            for name, axis, angle in [('Neck',(1,0,0),.3), ('Head',(0,0,1),.5),
                                      ('UpperArm.L',(0,1,0),1.2), ('UpperArm.R',(0,1,0),-1.2)]:
                rig.pose.bones[name].rotation_mode = 'QUATERNION'
                rig.pose.bones[name].rotation_quaternion = Quaternion(axis, angle)
            for pb in rig.pose.bones:
                if pb.name.startswith(PREFIX+'Hair_'):
                    pb.rotation_mode = 'QUATERNION'
                    pb.rotation_quaternion = Quaternion((1,0,0), .7)
            bpy.context.view_layer.update()
            deformation = rig.pose.bones['Head'].matrix @ rig.data.bones['Head'].matrix_local.inverted()
            cap_error = 0.
            for obj in meshes:
                tr = rig.matrix_world.inverted() @ obj.matrix_world
                evaluated = evaluated_points(obj)
                for i in cap.get(obj.name, []):
                    cap_error = max(cap_error, (tr@evaluated[i]-deformation@tr@obj.data.vertices[i].co).length)
            assert cap_error < 1e-5, (path, cap_error)
            result['posed_head_cap_rigid_error'] = cap_error
            result['head_only_cap_vertices'] = sum(map(len, cap.values()))
        from avatar_colliders import rest_contacts
        for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
        bpy.context.view_layer.update()
        clearance=rest_contacts(rig)
        assert not clearance['contacts'], clearance
        result['collider_rest_clearance']=clearance
        from validate_skirt_follow import check_rotations
        result['skirt_follow_axis_checks']=check_rotations(rig)
        result['rest_surface_error'] = rest_error
        result['geometry_and_unrelated_weights_preserved'] = True
        results.append(result)
        print('GENERALIZATION_OK', path.parent.name, result['chains'], rest_error, flush=True)
    (out/'generalization.json').write_text(json.dumps(results, indent=2))
    if args.generation_only:
        return
    from avatar_physics_preview import set_simulation, enabled
    bpy.ops.preferences.addon_enable(module='bl_ext.user_default.beyond_vrm_extension_suite')
    bpy.ops.wm.open_mainfile(filepath=str(out/'sample_b_springs.blend'))
    assert enabled()
    bpy.ops.wm.open_mainfile(filepath=str(out/'sample_b_springs_baked.blend'))
    assert not enabled()
    set_simulation(False)
    rig = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    samples = json.loads((out/'motion_samples.json').read_text())
    maximum_rotation_error = 0
    surface_motion = {'hair': 0., 'skirt': 0.}
    from avatar_springs import _material_region
    regions = {role: [(o, _material_region(o, role, {})) for o in meshes] for role in surface_motion}
    for sample in samples:
        bpy.context.scene.frame_set(sample['frame'])
        for name, values in sample['bones'].items():
            a = rig.pose.bones[name].rotation_quaternion
            b = Quaternion(values)
            error = min(max(abs(x-y) for x, y in zip(a, b)), max(abs(x+y) for x, y in zip(a, b)))
            maximum_rotation_error = max(maximum_rotation_error, error)
        if sample['frame'] % 15:
            continue
        posed = {o.name: evaluated_points(o) for o in meshes}
        for pb in rig.pose.bones:
            if pb.name.startswith(PREFIX):
                pb.matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.update()
        neutral = {o.name: evaluated_points(o) for o in meshes}
        for role, objects in regions.items():
            for obj, ids in objects:
                surface_motion[role] = max([surface_motion[role]] + [(posed[obj.name][i]-neutral[obj.name][i]).length for i in ids])
    assert maximum_rotation_error < 1e-6, maximum_rotation_error
    assert min(surface_motion.values()) > .005, surface_motion
    report = dict(frames=len(samples), baked_rotation_component_error=maximum_rotation_error,
                  maximum_surface_motion_from_springs=surface_motion,
                  simulation_disabled_during_bake_validation=True, editable_simulation_enabled=True, baked_simulation_disabled=True)
    (out/'saved_scene_validation.json').write_text(json.dumps(report, indent=2))
    print('SAVED_SPRING_SCENES_OK', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
