"""Recover unnamed hair surfaces from shared color texture and head attachment.

A shared atlas alone is insufficient: a complete connected surface must attach
near the generated head and have hanging-hair geometry. Explicit roles and other
semantic materials always win. No original rig or weight data is read.
"""
import numpy as np
from avatar_material_roles import semantic_label


def color_image(material):
    try:
        mtoon=material.vrm_addon_extension.mtoon1
        if mtoon.enabled or getattr(mtoon,'is_outline_material',False):
            image=mtoon.pbr_metallic_roughness.base_color_texture.index.source
            if image is not None:return image
    except AttributeError:
        pass
    tree=material.node_tree if material.use_nodes else None
    if tree is None:return None
    base=tree.nodes.get('mmd_base_tex')
    if base is not None and base.type=='TEX_IMAGE' and base.image is not None:
        return base.image
    for node in tree.nodes:
        if node.type not in ('BSDF_PRINCIPLED','BSDF_DIFFUSE','EMISSION'):continue
        socket=node.inputs.get('Base Color') if node.type=='BSDF_PRINCIPLED' else node.inputs.get('Color')
        if socket is None or not socket.is_linked:continue
        source=socket.links[0].from_node;seen=set()
        while source.type=='REROUTE' and source.inputs[0].is_linked and source.as_pointer() not in seen:
            seen.add(source.as_pointer());source=source.inputs[0].links[0].from_node
        if source.type=='TEX_IMAGE':return source.image
    return None


def hair_images(meshes, roles=None):
    roles=roles or {};result=set()
    for obj in meshes:
        used={face.material_index for face in obj.data.polygons}
        for index,mat in enumerate(obj.data.materials):
            if index not in used or mat is None:continue
            if roles.get(mat.name)=='hair' or (mat.name not in roles and 'hair' in semantic_label(mat.name)):
                image=color_image(mat)
                if image is not None:result.add(image.as_pointer())
    return result


def extend_hair_region(obj, rig, points, named_ids, roles, images):
    from avatar_apparel_weights import components, hair_free_top
    if not images:return set(named_ids)
    excluded=('skin','face','eye','mouth','teeth','skirt','shoe','sock','glove','body','cloth','accessory')
    slots=set()
    for i,mat in enumerate(obj.data.materials):
        if mat is None or mat.name in roles:continue
        label=semantic_label(mat.name)
        if any(token in label for token in excluded):continue
        image=color_image(mat)
        if image is not None and image.as_pointer() in images:slots.add(i)
    candidates={v for f in obj.data.polygons if f.material_index in slots for v in f.vertices}
    result=set(named_ids);allowed=result|candidates
    head=rig.data.bones.get('Head');neck=rig.data.bones.get('Neck')
    if head is None or neck is None:return result
    a=np.array(head.head_local);b=np.array(head.tail_local);axis=b-a
    length=float(np.linalg.norm(axis));height=float(np.ptp(points[:,2]))
    if length<1e-8:return result
    from avatar_virtual_topology import connected_graph, protected_extremities
    allowed -= set(np.flatnonzero(protected_extremities(rig, points, height)))
    _, regions, _ = connected_graph(obj.data, points, allowed, height*.0001)
    for component in regions:
        ids=set(component)
        if len(ids)<8 or not ids<=allowed or ids<=result:continue
        p=points[sorted(ids)];top=float(p[:,2].max())
        if top<a[2]+length*.4 or top>b[2]+length*.5:continue
        attachment=p[p[:,2]>=top-min(length*.2,float(np.ptp(p[:,2]))*.1)]
        center=np.median(attachment,axis=0)
        t=np.clip(float((center-a)@axis)/(length*length),0,1)
        if np.linalg.norm(center-a-axis*t)>length*.9:continue
        if hair_free_top(p,height,neck.head_local.z,head.tail_local.z) is None:continue
        result.update(ids)
    return result
