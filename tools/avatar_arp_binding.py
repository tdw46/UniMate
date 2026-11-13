# SPDX-License-Identifier: GPL-3.0-or-later
"""Auto-Rig Pro Heat Maps core on disposable surfaces and generated bones.

Original vertex order, topology, shape keys and material data never enter an
operator that can change them. Cranial surface ownership is applied before
secondary hair chains; it is not inferred from an imported rig's weights.
"""
from io import StringIO
from types import SimpleNamespace
import bpy
from mathutils import Matrix
from arp_binding.core import bind_parts
from avatar_apparel_weights import assign, weights
from avatar_influence_domains import head_domains


def _solve(rig, obj, scale=1.):
    objects_before={o.as_pointer() for o in bpy.data.objects}
    meshes_before={o.as_pointer() for o in bpy.data.meshes}
    armatures_before={o.as_pointer() for o in bpy.data.armatures}
    try:
        # Reconstruct Basis-only data; copies containing Key datablocks can make
        # split/transfer operators evaluate the artist's active expression.
        data=bpy.data.meshes.new('Temporary binding surface')
        data.from_pydata([v.co*scale for v in obj.data.vertices],
                        [tuple(e.vertices) for e in obj.data.edges],
                        [tuple(p.vertices) for p in obj.data.polygons]);data.update()
        temp=bpy.data.objects.new('Temporary binding surface',data)
        bpy.context.scene.collection.objects.link(temp)
        target=rig
        if scale!=1.:
            target=bpy.data.objects.new('Temporary binding skeleton',rig.data.copy())
            target.data.transform(Matrix.Scale(scale,4))
            bpy.context.scene.collection.objects.link(target)
        owner=SimpleNamespace(rig=target,bind_rig=True,bind_rig_add=False,skin_prints=StringIO())
        settings=SimpleNamespace(arp_bind_split=True,arp_bind_engine='HEAT_MAP',arp_bind_preserve=False)
        temp.select_set(True)
        bpy.context.view_layer.objects.active=temp
        bind_parts(owner,settings,temp)
        return [weights(temp,v.index) for v in temp.data.vertices],owner.skin_prints.getvalue()
    finally:
        if bpy.context.object and bpy.context.object.mode!='OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        # All objects created by this synchronous solve belong to this call,
        # including separate()'s unnamed parts on an exception path.
        for o in list(bpy.data.objects):
            if o.as_pointer() not in objects_before:bpy.data.objects.remove(o,do_unlink=True)
        for collection,before in ((bpy.data.meshes,meshes_before),(bpy.data.armatures,armatures_before)):
            for data in list(collection):
                if data.as_pointer() not in before and data.users==0:collection.remove(data)


def bind_generated(rig, meshes):
    context=bpy.context
    selected=list(context.selected_objects);active=context.view_layer.objects.active
    pose=rig.data.pose_position
    reports=[];heads=head_domains(rig,meshes)
    try:
        if context.object and context.object.mode!='OBJECT':
            raise ValueError('Binding requires Object Mode')
        rig.data.pose_position='REST'
        for obj in meshes:
            rows,log=_solve(rig,obj)
            missing=[i for i,row in enumerate(rows) if not row and i not in heads[obj.name]]
            retried=len(missing)
            if missing:
                # Upstream's optional Scale Fix uses 20x. Preserve every good
                # primary result and retry only the failed rows at that scale.
                retry,retry_log=_solve(rig,obj,20.)
                for i in missing:rows[i]=retry[i]
                log+=retry_log
            unresolved=[i for i,row in enumerate(rows) if not row and i not in heads[obj.name]]
            if unresolved:
                raise ValueError(f'Automatic binding left {len(unresolved)} vertices without weights on {obj.name}')
            for i,row in enumerate(rows):
                if i in heads[obj.name]:row={'Head':1.}
                total=sum(row.values())
                if total<=1e-8:raise ValueError('Automatic binding produced an empty influence row')
                assign(obj,[i],{name:value/total for name,value in row.items()})
            mod=obj.modifiers.new('BVT deform','ARMATURE');mod.object=rig
            if hasattr(mod,'use_deform_preserve_volume'):mod.use_deform_preserve_volume=False
            mod.show_in_editmode=True;mod.show_on_cage=True
            obj.parent=rig
            reports.append(dict(object=obj.name,vertices=len(rows),cranial_vertices=len(heads[obj.name]),
                                scale_retry_vertices=retried,warnings=log.strip()))
    finally:
        rig.data.pose_position=pose
        for obj in list(context.selected_objects):obj.select_set(False)
        for obj in selected:
            if obj.name in context.view_layer.objects:obj.select_set(True)
        if active and active.name in context.view_layer.objects:context.view_layer.objects.active=active
    return dict(engine='Auto-Rig Pro 3.78.34 Heat Maps core',split_parts=True,
                preserve_volume=False,source_weights_used=False,objects=reports)
