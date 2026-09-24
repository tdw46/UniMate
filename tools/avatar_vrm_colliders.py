"""Create and remove standard VRM 1 colliders through the official add-on API.

Shape edits use VRM RNA setters exclusively. The add-on owns its empty display
objects, bone parenting and export metadata; no collider meshes are authored.
"""
import bpy
from mathutils import Vector


def call_operator(name, rig, **kwargs):
    operator = getattr(bpy.ops.vrm, name)
    result = operator(armature_object_name=rig.name, **kwargs)
    if result != {'FINISHED'}:
        raise RuntimeError(f'VRM {name} failed: {result}')


def add_group(rig, name):
    data = rig.data.vrm_addon_extension.spring_bone1
    if callable(getattr(data, 'add_collider_group', None)):
        group = data.add_collider_group()
    else:
        call_operator('add_spring_bone1_collider_group', rig)
        group = data.collider_groups[-1]
    group.vrm_name = name
    return group


def add_capsule(rig, spec):
    if rig.data.vrm_addon_extension.spec_version != '1.0':
        raise ValueError('Capsules require VRM 1; VRM 0 supports sphere colliders')
    if spec['bone'] not in rig.data.bones:
        raise ValueError('Unknown collider attachment bone: ' + spec['bone'])
    data = rig.data.vrm_addon_extension.spring_bone1
    if callable(getattr(data, 'add_collider', None)):
        collider = data.add_collider(bpy.context, rig)
    else:
        call_operator('add_spring_bone1_collider', rig)
        collider = data.colliders[-1]
    collider.node.bone_name = spec['bone']
    if hasattr(collider, 'ui_collider_type'):
        collider.ui_collider_type = 'uiColliderTypeCapsule'
    else:
        collider.shape_type = 'Capsule'
    collider.reset_bpy_object(bpy.context, rig)
    bpy.context.view_layer.update()
    for field in ('offset', 'tail', 'radius'):
        setattr(collider.shape.capsule, field, spec[field])
        bpy.context.view_layer.update()
    assert collider.bpy_object.type == 'EMPTY'
    assert collider.bpy_object.parent == rig
    assert collider.bpy_object.parent_type == 'BONE'
    assert collider.bpy_object.parent_bone == spec['bone']
    assert (Vector(collider.shape.capsule.offset)-Vector(spec['offset'])).length < 1e-5
    assert (Vector(collider.shape.capsule.tail)-Vector(spec['tail'])).length < 1e-5
    assert abs(collider.shape.capsule.radius-spec['radius']) < 1e-6
    collider.bpy_object.name = spec['bone'] + ' Collider'
    collider.bpy_object.children[0].name = collider.bpy_object.name + ' End'
    collider.bpy_object['unimate_generated_collider'] = True
    collider.bpy_object.hide_render = True
    return collider


def display_objects(rig):
    """Read display references from the active VRM schema, including VRM 0."""
    ext = rig.data.vrm_addon_extension
    colliders = (ext.spring_bone1.colliders if ext.spec_version == '1.0' else
                 [c for g in ext.vrm0.secondary_animation.collider_groups for c in g.colliders])
    return [obj for collider in colliders if collider.bpy_object
            for obj in (collider.bpy_object, *collider.bpy_object.children)]


def show_colliders(rig, visible=True):
    for obj in display_objects(rig):
        obj.hide_viewport = not visible
        obj.hide_set(not visible)
        obj.hide_render = True
