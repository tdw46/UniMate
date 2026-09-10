"""Focused action regressions; run with Blender --python-exit-code 1."""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bpy
from data_process.utils.blender_actions import bind_action_slot, iter_fcurves, new_fcurves


# Legacy compatibility without depending on an installed 4.0 binary.
legacy_curves = [SimpleNamespace(data_path='pose.bones["Head"].rotation_quaternion')]
legacy = SimpleNamespace(fcurves=legacy_curves)
assert list(iter_fcurves(legacy)) == legacy_curves
assert new_fcurves(legacy, None) is legacy_curves
bind_action_slot(SimpleNamespace(animation_data=SimpleNamespace()), legacy)
assert list(iter_fcurves(None)) == []

owner = bpy.data.objects.new('Action regression rig', bpy.data.armatures.new('Rig'))
bpy.context.collection.objects.link(owner)
owner.animation_data_create()
action = bpy.data.actions.new('Regression action')
owner.animation_data.action = action
curves = new_fcurves(action, owner)
curve = curves.new(data_path='location', index=0)
curve.keyframe_points.insert(0, 0)
curve.keyframe_points.insert(10, 2)
curve.update()
assert len(list(iter_fcurves(action))) == 1
bind_action_slot(owner, action)
bpy.context.scene.frame_set(10)
assert abs(owner.location.x - 2) < 1e-6

if hasattr(action, 'slots'):
    action.slots.new(id_type='OBJECT', name='Another owner')
    # Named owner remains deterministic even with another compatible slot.
    bind_action_slot(owner, action)
    assert owner.animation_data.action_slot.name_display == owner.name
    stranger = bpy.data.objects.new('Unknown owner', None)
    stranger.animation_data_create()
    stranger.animation_data.action = action
    try:
        bind_action_slot(stranger, action)
    except ValueError:
        pass
    else:
        raise AssertionError('Ambiguous action slots must be rejected')
print('ACTION_REGRESSIONS_PASSED', bpy.app.version_string)
