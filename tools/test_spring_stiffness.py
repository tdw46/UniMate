"""Blender background regression against a generated skirt/hair fixture.

blender --background --factory-startup --python tools/test_spring_stiffness.py -- source.blend output.json
"""
import json
from pathlib import Path
import sys
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import properties_hallway_rig as props


def run(source, output):
    if not bpy.app.background:
        raise RuntimeError('Use an isolated background session')
    if hasattr(bpy.context.preferences, 'extensions'):
        for repo in bpy.context.preferences.extensions.repos:
            if repo.module == 'user_default':
                repo.use_custom_directory = True
                repo.custom_directory = str(Path.home() / 'Documents/Blender/extensions/user_default')
    try:
        bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
    except (RuntimeError, ModuleNotFoundError):
        bpy.ops.preferences.addon_enable(module='VRM_Addon_for_Blender')
    props.register()
    bpy.ops.wm.open_mainfile(filepath=str(Path(source).resolve()))
    rig = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE' and o.get('unimate_secondary_generator'))
    props.initialize(rig)
    results = {}
    for group in rig.hallway_rig.spring_groups:
        chains = props.springs(rig, group.prefix)
        roles = props.non_root_joint_roles(rig, chains)
        baseline = [[j.stiffness for j in s.joints] for s in chains]
        others = [(j, j.stiffness) for s in rig.data.vrm_addon_extension.spring_bone1.springs
                  if not s.vrm_name.startswith(group.prefix) for j in s.joints]
        # Independent check: root segments have no simulated ancestor, even
        # though every split skirt segment starts at VRM joint index zero.
        heads = {j.node.bone_name for s in chains for j in s.joints[:-1]}
        root_count = distal_count = 0
        for spring, flags in zip(chains, roles):
            for joint, distal in zip(spring.joints[:-1], flags[:-1]):
                parent = rig.data.bones[joint.node.bone_name].parent
                expected = False
                while parent:
                    expected |= parent.name in heads
                    parent = parent.parent
                assert distal == expected
                root_count += not distal
                distal_count += distal
        assert root_count and distal_count
        assert group.non_root_stiffness == 1.
        try:
            for factor in (.25, 0., 2., 1.):
                group.non_root_stiffness = factor
                for spring, flags, values in zip(chains, roles, baseline):
                    for joint, distal, value in zip(spring.joints, flags, values):
                        assert abs(joint.stiffness - value * (factor if distal else 1.)) < 1e-5
                assert all(j.stiffness == value for j, value in others)
            # Root edits still preserve the original taper and distal multiplier.
            original_root = group.stiffness
            group.non_root_stiffness = .25
            try:
                group.stiffness = original_root * 1.1
                for spring, flags, values in zip(chains, roles, baseline):
                    for joint, distal, value in zip(spring.joints, flags, values):
                        assert abs(joint.stiffness - value * 1.1 * (.25 if distal else 1.)) < 1e-5
            finally:
                group.stiffness = original_root
        finally:
            group.non_root_stiffness = 1.
        results[group.name] = dict(roots=root_count,non_root_segments=distal_count)
    group = rig.hallway_rig.spring_groups['Skirt']
    group.non_root_stiffness = .25
    rig_name = rig.name
    expected = [[j.stiffness for j in s.joints] for s in props.springs(rig, group.prefix)]
    saved = Path(output).with_suffix('.blend').resolve()
    bpy.ops.wm.save_as_mainfile(filepath=str(saved))
    bpy.ops.wm.open_mainfile(filepath=str(saved))
    rig = bpy.data.objects[rig_name]
    group = rig.hallway_rig.spring_groups['Skirt']
    assert group.non_root_stiffness == .25
    assert expected == [[j.stiffness for j in s.joints] for s in props.springs(rig, group.prefix)]
    group.non_root_stiffness = 0.
    assert all(j.stiffness == 0. for s, flags in zip(props.springs(rig, group.prefix), props.non_root_joint_roles(rig, props.springs(rig, group.prefix))) for j, distal in zip(s.joints, flags) if distal)
    Path(output).write_text(json.dumps(dict(groups=results,live_callback=True,zero_round_trip=True,owner_isolation=True,save_reload=True),indent=2))
    print('SPRING_STIFFNESS_REGRESSION_OK', results, flush=True)


if __name__ == '__main__':
    run(*sys.argv[sys.argv.index('--') + 1:])
