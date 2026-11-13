"""Native VRM reference checks for new and legacy generated breast names."""
import json
import os
import sys
from pathlib import Path
import bpy
import addon_utils

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.environ.get('VRM_ADDON_PATH', str(Path.home() / 'Documents/Blender/extensions/user_default')))
assert addon_utils.enable('vrm', default_set=True)
from avatar_breasts import install_physics

results = []
for prefix in ('UniMate_', 'Hallway_'):
    data = bpy.data.armatures.new('Brand test')
    rig = bpy.data.objects.new('Brand test', data)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    chest = data.edit_bones.new('Chest')
    chest.head, chest.tail = (0, 0, 0), (0, 0, 1)
    for side, x in (('L', .2), ('R', -.2)):
        bone = data.edit_bones.new(prefix + 'Breast.' + side)
        bone.head, bone.tail, bone.parent = (x, 0, .6), (x, -.2, .6), chest
        end = data.edit_bones.new(prefix + 'BreastEnd.' + side)
        end.head, end.tail, end.parent = bone.tail, (x, -.25, .6), bone
    bpy.ops.object.mode_set(mode='OBJECT')
    data.vrm_addon_extension.spec_version = '1.0'
    data.vrm_addon_extension.vrm1.humanoid.human_bones.chest.node.bone_name = 'Chest'
    mesh = bpy.data.meshes.new('Bounds')
    mesh.from_pydata([(-.4, -.3, 0), (.4, .3, 1.5)], [], [])
    obj = bpy.data.objects.new('Bounds', mesh)
    bpy.context.scene.collection.objects.link(obj)
    # The default stays compatible with callers using legacy generated names.
    kwargs = {} if prefix == 'UniMate_' else {'bone_prefix': prefix}
    install_physics(rig, ['L', 'R'], [obj], **kwargs)
    springs = data.vrm_addon_extension.spring_bone1.springs
    assert len(springs) == 2
    assert all(j.node.bone_name in data.bones and j.node.bone_name.startswith(prefix)
               for spring in springs for j in spring.joints)
    results.append({'prefix': prefix, 'springs': len(springs)})
    rig.select_set(False)
Path(os.environ['BEYOND_UI_TEST_RESULT']).write_text(json.dumps({
    'passed': True, 'version': bpy.app.version_string, 'cases': results,
}))
