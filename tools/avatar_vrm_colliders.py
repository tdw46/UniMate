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
    organize_colliders(rig)
    return collider


def display_objects(rig):
    """Read display references from the active VRM schema, including VRM 0."""
    ext = rig.data.vrm_addon_extension
    colliders = (ext.spring_bone1.colliders if ext.spec_version == '1.0' else
                 [c for g in ext.vrm0.secondary_animation.collider_groups for c in g.colliders])
    return [obj for collider in colliders if collider.bpy_object
            for obj in (collider.bpy_object, *collider.bpy_object.children)]


def collider_collection(rig, create=False):
    """A rig-owned collection, identified by an ID reference rather than a name."""
    collection = next((c for c in bpy.data.collections
                       if c.get('hallway_collider_rig') == rig), None)
    if collection is None and create:
        collection = bpy.data.collections.new(rig.name + ' - VRM Colliders')
        collection['hallway_collider_rig'] = rig
        for scene in rig.users_scene:
            scene.collection.children.link(collection)
        collection.hide_render = True
    return collection


def clear_object_hiding(obj):
    """Migrate old per-object flags; subsequent visibility is collection-only."""
    obj.hide_viewport = False
    for scene in obj.users_scene:
        for layer in scene.view_layers:
            if obj.name in layer.objects and obj.hide_get(view_layer=layer):
                obj.hide_set(False, view_layer=layer)


def move_to_collection(obj, collection, keep_collections=()):
    if obj.name not in collection.objects:
        collection.objects.link(obj)
    clear_object_hiding(obj)
    for old in tuple(obj.users_collection):
        if old != collection and old not in keep_collections:
            old.objects.unlink(obj)


def organize_colliders(rig):
    """Group generated VRM display objects without changing their shape or pose."""
    collection = collider_collection(rig, create=True)
    ext = rig.data.vrm_addon_extension
    colliders = (ext.spring_bone1.colliders if ext.spec_version == '1.0' else
                 [c for g in ext.vrm0.secondary_animation.collider_groups for c in g.colliders])
    for collider in colliders:
        head = collider.bpy_object
        if not head or not head.get('unimate_generated_collider'):
            continue
        # The VRM add-on uses the first child as the capsule endpoint. Other
        # artist children must not acquire our ownership tag.
        objects = [head]
        if getattr(collider, 'shape_type', None) == 'Capsule' and head.children:
            objects.append(head.children[0])
        for obj in objects:
            obj['unimate_generated_collider'] = True
            obj['hallway_collider_owner'] = rig
            obj.show_in_front = True
            obj.hide_render = True
            move_to_collection(obj, collection)
    return collection


def colliders_visible(rig):
    collection = collider_collection(rig)
    if collection:
        return any(not layer.hide_viewport and not layer.exclude
                   for layer in layer_collections(bpy.context.view_layer.layer_collection, collection))
    return any(o.visible_get() for o in display_objects(rig)
               if o.get('unimate_generated_collider'))


def layer_collections(root, collection):
    if root.collection == collection:
        yield root
    for child in root.children:
        yield from layer_collections(child, collection)


def show_colliders(rig, visible=True):
    collection = organize_colliders(rig)
    # The monitor/disable flag drops empties from dependency-graph evaluation.
    # Use only the collection eye: hidden colliders must still follow bones.
    collection.hide_viewport = False
    for scene in rig.users_scene:
        for view_layer in scene.view_layers:
            for layer in layer_collections(view_layer.layer_collection, collection):
                layer.hide_viewport = not visible


def all_referenced_displays():
    """Protect references in BOTH schemas, including inactive/archived rigs."""
    result = set()
    for data in bpy.data.armatures:
        ext = getattr(data, 'vrm_addon_extension', None)
        if ext is None:
            continue
        colliders = list(ext.spring_bone1.colliders)
        colliders.extend(c for g in ext.vrm0.secondary_animation.collider_groups for c in g.colliders)
        for collider in colliders:
            if collider.bpy_object:
                result.add(collider.bpy_object)
                result.update(collider.bpy_object.children)
    return result


def cleanup_orphan_displays(rig):
    """Remove only unreferenced generated empties owned by this rig.

    No name matching or global orphan purge: artist objects and references from
    other rigs remain intact. Tag endpoints as well as heads so interrupted
    regeneration can be cleaned up on the next run.
    """
    referenced = all_referenced_displays()
    candidates = {o for o in bpy.data.objects if o.type == 'EMPTY'
                  and o.get('unimate_generated_collider')
                  and (o.get('hallway_collider_owner') == rig or o.parent == rig)
                  and o not in referenced}
    # Preserve custom children and externally referenced objects, including their
    # ancestors. Removing a collider must not clear an unrelated constraint.
    users = bpy.data.user_map(subset=candidates) if candidates else {}
    protected = {o for o in candidates if any(ch not in candidates for ch in o.children)
                 or any(not isinstance(u, (bpy.types.Collection, bpy.types.Scene))
                        and u not in candidates for u in users.get(o, ()))}
    for obj in tuple(protected):
        parent = obj.parent
        while parent in candidates:
            protected.add(parent)
            parent = parent.parent
    removed = sorted(o.name for o in candidates - protected)
    for name in removed:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    return removed


def archive_colliders(source, archive, legacy_objects=()):
    """Retain comparison physics behind its collection, never object hide flags.

    The caller supplies verified source-owned MMD bodies. RigidBodyWorld's
    unlinked technical collection remains intact for the original comparison.
    """
    objects = set(display_objects(source)) | set(legacy_objects)
    archive.hide_viewport = True
    archive.hide_render = True
    child = next((c for c in archive.children if c.get('hallway_archived_colliders')), None)
    if child is None:
        child = bpy.data.collections.new(source.name + ' - Archived Colliders')
        child['hallway_archived_colliders'] = True
        archive.children.link(child)
    technical = {s.rigidbody_world.collection for s in bpy.data.scenes
                 if s.rigidbody_world and s.rigidbody_world.collection}
    old_collections = {c for obj in objects for c in obj.users_collection
                       if c not in technical and c not in (archive, child)
                       and not c.children and set(c.objects) <= objects
                       and c not in {s.collection for s in bpy.data.scenes}}
    for obj in objects:
        move_to_collection(obj, child, keep_collections=technical)
    for collection in old_collections:
        users = bpy.data.user_map(subset={collection}).get(collection, ())
        if not collection.objects and all(isinstance(u, (bpy.types.Scene, bpy.types.Collection)) for u in users):
            bpy.data.collections.remove(collection)
    return len(objects)
