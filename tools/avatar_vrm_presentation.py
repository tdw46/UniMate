"""Preserve VRM 1 presentation metadata when replacing an imported skeleton.

Copy through RNA so collection entries and property setters remain valid.
Humanoid mappings, constraints and spring definitions belong to the new rig.
"""
import bpy


def copy_presentation(source, target, mesh_map):
    if source.data.vrm_addon_extension.spec_version != '1.0':
        raise ValueError('Presentation copy currently requires VRM 1')
    id_map = {old.as_pointer():new for old,new in mesh_map.items()}
    names = {old.name:new.name for old,new in mesh_map.items()}
    for old,new in mesh_map.items():
        for a,b in zip(old.data.materials,new.data.materials):
            if a and b:id_map[a.as_pointer()]=b

    def copy_group(src, dst):
        for prop in src.bl_rna.properties:
            key=prop.identifier
            if key=='rna_type':continue
            value=getattr(src,key)
            if prop.type=='COLLECTION':
                collection=getattr(dst,key)
                collection.clear()
                for item in value:copy_group(item,collection.add())
            elif prop.type=='POINTER' and isinstance(value,bpy.types.PropertyGroup):
                copy_group(value,getattr(dst,key))
            elif not prop.is_readonly:
                if isinstance(value,bpy.types.ID):value=id_map.get(value.as_pointer(),value)
                elif key=='mesh_object_name':value=names.get(value,value)
                elif getattr(prop,'is_array',False):value=list(value)
                setattr(dst,key,value)

    src=source.data.vrm_addon_extension.vrm1
    dst=target.data.vrm_addon_extension.vrm1
    for key in ('meta','expressions','first_person'):
        copy_group(getattr(src,key),getattr(dst,key))
    binds=0
    for expression in dst.expressions.all_name_to_expression_dict().values():
        for bind in expression.morph_target_binds:
            mesh=bpy.data.objects.get(bind.node.mesh_object_name)
            if mesh not in mesh_map.values() or not mesh.data.shape_keys or bind.index not in mesh.data.shape_keys.key_blocks:
                raise ValueError('Copied expression does not resolve to a replacement shape key')
            binds+=1
    return dict(expression_binds=binds,meta_name=dst.meta.vrm_name)
