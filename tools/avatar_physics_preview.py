"""Delegate Hallway's live preview to BVT; no solver or continuous callbacks."""
import sys
from contextlib import contextmanager
import bpy


def available(context=None):
    context = context or bpy.context
    if not hasattr(context.scene, 'bvt_spring_simulation_enabled'):
        return False
    try:
        return 'enabled' in bpy.ops.bvt.set_spring_simulation.get_rna_type().properties
    except (AttributeError, RuntimeError):
        return False


def enabled(context=None):
    return bool(getattr((context or bpy.context).scene, 'bvt_spring_simulation_enabled', False))


def set_simulation(value, context=None):
    context = context or bpy.context
    if not available(context):
        raise RuntimeError('Enable Beyond VTuber Tools to use physics preview')
    if enabled(context) != bool(value):
        with bpy.context.temp_override(scene=context.scene):
            result = bpy.ops.bvt.set_spring_simulation(enabled=bool(value))
        if result != {'FINISHED'}:
            raise RuntimeError('BVT could not change spring simulation')
    return enabled(context)


@contextmanager
def suspended(context=None):
    """Release BVT's pose/cache ownership while changing spring metadata."""
    was_enabled = enabled(context)
    if was_enabled:
        set_simulation(False, context)
    try:
        yield
    finally:
        if was_enabled:
            set_simulation(True, context)


def reset(context=None):
    with suspended(context):
        pass


def bvt_runtime(*required):
    """Resolve the enabled BVT runtime without copying any of its solver code."""
    if not available():
        raise RuntimeError('Enable Beyond VTuber Tools before running physics')
    modules = [module for name, module in sys.modules.items()
               if name.endswith('.VRM_SpringSimulation') and 'beyond_vrm_extension_suite' in name]
    if len(modules) != 1:
        raise RuntimeError('Could not resolve a single enabled BVT spring runtime')
    module = modules[0]
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError('Installed BVT lacks the required simulation API: ' + ', '.join(missing))
    return module


def background_step(seconds, scene=None):
    """Headless numerical checks only; all integration and collision math is BVT's."""
    if not bpy.app.background:
        raise RuntimeError('Explicit stepping is only allowed in background Blender')
    scene = scene or bpy.context.scene
    if not enabled():
        raise RuntimeError('Enable BVT simulation before stepping')
    module = bvt_runtime('_simulate_scene_duration')
    return module._simulate_scene_duration(scene, seconds, update_dependencies=True)


@contextmanager
def cached_animation(scene):
    """Use BVT's exact playback cache for deterministic headless recording."""
    if not bpy.app.background:
        raise RuntimeError('Bake recording must run in background Blender')
    module = bvt_runtime('prepare_true_preview_animation_render',
                         'apply_true_preview_render_spring_frame',
                         'finish_true_preview_animation_render')
    try:
        if not module.prepare_true_preview_animation_render(scene):
            raise RuntimeError('BVT could not prepare the spring animation cache')
        def apply(frame):
            scene.frame_set(frame)
            module.apply_true_preview_render_spring_frame(scene)
            bpy.context.view_layer.update()
        yield apply
    finally:
        module.finish_true_preview_animation_render(scene)


class HALLWAY_OT_SetPhysics(bpy.types.Operator):
    bl_idname = 'hallway.set_physics'
    bl_label = 'Toggle BVT Physics'
    bl_description = 'Enable or disable Beyond VTuber Tools spring simulation'
    enabled: bpy.props.BoolProperty(default=True)

    @classmethod
    def poll(cls, context):
        return available(context)

    def execute(self, context):
        set_simulation(self.enabled, context)
        return {'FINISHED'}


class HALLWAY_OT_ResetPhysics(bpy.types.Operator):
    bl_idname = 'hallway.reset_physics'
    bl_label = 'Reset BVT Physics'

    @classmethod
    def poll(cls, context):
        return available(context) and enabled(context)

    def execute(self, context):
        with suspended(context):
            pass
        return {'FINISHED'}


CLASSES = (HALLWAY_OT_SetPhysics, HALLWAY_OT_ResetPhysics)


def register():
    for cls in CLASSES:
        if not getattr(cls, 'is_registered', False):
            bpy.utils.register_class(cls)


def unregister():
    # BVT owns its runtime; disabling Hallway must not disable BVT.
    for cls in reversed(CLASSES):
        if getattr(cls, 'is_registered', False):
            bpy.utils.unregister_class(cls)
