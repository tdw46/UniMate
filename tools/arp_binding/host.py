# SPDX-License-Identifier: GPL-3.0-or-later
"""Small transactional utilities for the upstream binding excerpt."""
import bpy


def get_object(name):
    return bpy.data.objects.get(name)


def set_active_object(name):
    obj = get_object(name)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def duplicate_object(*, method, obj):
    if method != 'data':
        raise ValueError('Unsupported duplicate method')
    copy = obj.copy()
    copy.data = obj.data.copy()
    bpy.context.scene.collection.objects.link(copy)
    return copy


def is_object_id(obj, value, suffix_only=False):
    identity = str(obj.get('arp_id', ''))
    return identity.endswith(value) if suffix_only else identity == value


def get_object_id(value):
    return next((obj for obj in bpy.context.scene.objects if is_object_id(obj, value)), None)


def delete_object(obj):
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if isinstance(data, bpy.types.Mesh) and data.users == 0:
        bpy.data.meshes.remove(data)


def disable_mask_modifiers(objects):
    saved = [(mod, mod.show_viewport) for obj in objects for mod in obj.modifiers if mod.type == 'MASK']
    for mod, _ in saved:
        mod.show_viewport = False
    return saved


def restore_mask_modifiers(saved):
    for mod, visible in saved:
        mod.show_viewport = visible
