"""Install a validated replacement on existing mesh objects, preserving edits.

A hidden comparison collection retains the source rig, meshes and weights.
Existing mesh IDs, geometry, shape keys, materials and unrelated scene objects
stay in place. The current blend file is never saved by this operation.
"""
import json
from pathlib import Path
import bpy
from rebuild_mmd_character import fingerprint
from avatar_physics_preview import set_simulation, enabled


def install(directory):
    directory=Path(directory)
    inventory=json.loads((directory/'source_inventory.json').read_text())
    generation=json.loads((directory/'generation_report.json').read_text())
    validation=json.loads((directory/'validation.json').read_text())
    new_weights=json.loads((directory/'validated_weights.json').read_text())
    assert bpy.context.mode=='OBJECT'
    assert bpy.data.filepath==inventory['file'],'The active file changed'
    source=bpy.data.objects[inventory['active']]
    meshes=[bpy.data.objects[n] for n in inventory['affected_meshes']]
    assert all(fingerprint(o)==generation['mesh_hashes'][o.name] for o in meshes), 'Source mesh or shape keys changed since snapshot'
    archive_name=source.name+' - Original comparison'
    assert archive_name not in bpy.data.collections,'Comparison already installed'
    assert validation['geometry_and_all_shape_keys_unchanged'] and validation['changed_frames']>0
    for obj in meshes:
        assert len(new_weights[obj.name])==len(obj.data.vertices)
        assert all(abs(sum(w.values())-1)<1e-5 for w in new_weights[obj.name])
    file_before=bpy.data.filepath
    pointers={o.name:o.as_pointer() for o in meshes}
    unrelated={o.name:o.as_pointer() for o in bpy.context.scene.objects if o not in meshes and o!=source}
    set_simulation(False)
    existing_ids={o.as_pointer() for o in bpy.data.objects}
    with bpy.data.libraries.load(str(directory/'validated.blend'),link=False) as (available,loaded):
        loaded.objects=[name for name in available.objects if name not in inventory['affected_meshes']]
    imported=[o for o in loaded.objects if o]
    rig=next(o for o in imported if o.type=='ARMATURE')
    assert len([o for o in imported if o.type=='ARMATURE'])==1
    assert all(n in rig.data.bones for values in new_weights.values() for weights in values for n in weights)
    # RNA references can pull mesh dependencies into a rig-only library load.
    # Redirect those references to the preserved live objects, then discard
    # only the newly imported, verified-identical mesh copies.
    originals={generation['mesh_hashes'][o.name]:o for o in meshes}
    for obj in list(bpy.data.objects):
        if obj.as_pointer() in existing_ids or obj.type!='MESH':continue
        original=originals.get(fingerprint(obj))
        assert original is not None and not obj.users_collection,'Unexpected imported mesh dependency'
        obj.user_remap(original)
        mesh_data=obj.data
        bpy.data.objects.remove(obj,do_unlink=True)
        if mesh_data.users==0:bpy.data.meshes.remove(mesh_data)
    generated=bpy.data.collections.new('UniMate generated rig and colliders')
    bpy.context.scene.collection.children.link(generated)
    for obj in imported:generated.objects.link(obj)
    archive=bpy.data.collections.new(archive_name)
    bpy.context.scene.collection.children.link(archive)
    archived=[]
    for obj in meshes:
        copy=obj.copy();copy.data=obj.data.copy();copy.name=obj.name+' - Original'
        archive.objects.link(copy);archived.append(copy.name)
    original_name=source.name
    source.name=original_name+' - Original'
    archive.objects.link(source)
    for collection in list(source.users_collection):
        if collection!=archive:collection.objects.unlink(source)
    for obj in meshes:
        world=obj.matrix_world.copy()
        obj.vertex_groups.clear()
        names=sorted({n for values in new_weights[obj.name] for n in values})
        groups={n:obj.vertex_groups.new(name=n) for n in names}
        for i,values in enumerate(new_weights[obj.name]):
            for n,w in values.items():groups[n].add([i],w,'REPLACE')
        for mod in list(obj.modifiers):
            if mod.type=='ARMATURE' and mod.object==source:obj.modifiers.remove(mod)
        mod=obj.modifiers.new('UniMate deform','ARMATURE');mod.object=rig
        obj.parent=rig;obj.matrix_world=world
    archive.hide_render=True
    def exclude(layer):
        if layer.collection==archive:layer.exclude=True
        for child in layer.children:exclude(child)
    for layer in bpy.context.scene.view_layers:exclude(layer.layer_collection)
    # Archive source physics without accumulating object-level hide flags.
    # Ambiguous bodies in scenes with multiple source rigs are left alone.
    from avatar_vrm_colliders import archive_colliders, show_colliders
    legacy=[]
    legacy_objects=[]
    other_sources=[o for o in bpy.context.scene.objects if o.type=='ARMATURE'
                   and o!=source and o!=rig and not o.get('unimate_secondary_generator')]
    for obj in bpy.context.scene.objects:
        if getattr(obj,'mmd_type',None)=='RIGID_BODY':
            bone=getattr(getattr(obj,'mmd_rigid',None),'bone','')
            if bone not in source.data.bones or any(bone in o.data.bones for o in other_sources):
                continue
            legacy.append(dict(name=obj.name,hide_render=obj.hide_render,hide_viewport=obj.hide_viewport,hide_set=obj.hide_get()))
            legacy_objects.append(obj)
    archive_colliders(source,archive,legacy_objects)
    if bpy.context.scene.rigidbody_world:bpy.context.scene.rigidbody_world.enabled=False
    show_colliders(rig,False)
    bpy.ops.object.select_all(action='DESELECT')
    rig.hide_set(False);rig.select_set(True);bpy.context.view_layer.objects.active=rig
    bpy.context.view_layer.update()
    assert all(fingerprint(o)==generation['mesh_hashes'][o.name] and o.as_pointer()==pointers[o.name] for o in meshes)
    assert all(bpy.data.objects[n].as_pointer()==p for n,p in unrelated.items())
    assert bpy.data.filepath==file_before
    from properties_hallway_rig import initialize
    from avatar_bone_collections import organize_bones
    initialize(rig)
    organize_bones(rig)
    set_simulation(True)
    result=dict(rig=rig.name,bones=len(rig.data.bones),meshes=len(meshes),comparison_collection=archive.name,
                original_rig=source.name,original_meshes=archived,legacy_colliders=legacy,
                springs=len(rig.data.vrm_addon_extension.spring_bone1.springs),
                colliders=len(rig.data.vrm_addon_extension.spring_bone1.colliders),
                simulation_enabled=enabled(),
                geometry_shape_keys_and_mesh_ids_preserved=True,unrelated_objects_preserved=len(unrelated),
                current_file_unchanged=True,saved_current_file=False)
    (directory/'live_installation.json').write_text(json.dumps(result,indent=2))
    return {k:v for k,v in result.items() if k not in ('legacy_colliders','original_meshes')}
