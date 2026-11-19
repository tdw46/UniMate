"""Preserve VRM's named shader groups on local material copies and appends."""
import re

import bpy
from bpy.app.handlers import persistent

_ALIAS = re.compile(r"(VRM Add-on MToon 1\.0 (?:Output|Normal|UV|UV Animation|MatCap UV|Outline Geometry) Revision [0-9]+)\.\d{3,}\Z")


def _value(v):
    if isinstance(v, bpy.types.ID) or v is None or isinstance(v, (int, float, str)):
        return v
    return tuple(v)


def _properties(item, skip=()):
    result = []
    for prop in item.bl_rna.properties:
        key = prop.identifier
        if key in skip or key == "rna_type":
            continue
        if prop.is_readonly and prop.type not in {"POINTER", "COLLECTION"}:
            continue
        v = getattr(item, key)
        if isinstance(item, bpy.types.ColorRamp) and key == "elements":
            result.append((key, tuple(_properties(element) for element in v)))
            continue
        if isinstance(v, (bpy.types.ImageUser, bpy.types.TexMapping, bpy.types.ColorMapping, bpy.types.ColorRamp)):
            result.append((key, _properties(v)))
            continue
        if prop.type == "COLLECTION" or (prop.type == "POINTER" and v and not isinstance(v, bpy.types.ID)):
            # An unfamiliar nested setting (for example a custom color ramp)
            # cannot establish equivalence by pointer or socket layout alone.
            raise ValueError(key)
        result.append((key, _value(v)))
    return tuple(result)


def _signature(tree, material=None):
    def driver_target(target):
        return tuple((key, ("self",) if material is not None and value == material else value)
                     for key, value in _properties(target))

    anim = tree.animation_data
    drivers = []
    if anim:
        if anim.action or anim.nla_tracks:
            raise ValueError("authored group animation")
        for f in anim.drivers:
            d = f.driver
            if f.modifiers or f.sampled_points or any(v.type not in {'CONTEXT_PROP', 'SINGLE_PROP'} for v in d.variables):
                raise ValueError("authored driver")
            drivers.append((f.data_path, f.array_index, f.mute, f.extrapolation, d.type, d.expression, d.use_self,
                            tuple((v.name, v.type, tuple(driver_target(t) for t in v.targets)) for v in d.variables),
                            tuple(_properties(k) for k in f.keyframe_points)))
    def sockets(values):
        return tuple((s.identifier, s.bl_idname, _value(getattr(s, "default_value", None)))
                     for s in values)
    interface = tuple((i.item_type, i.name, getattr(i.parent, 'identifier', None),
                       getattr(i, "identifier", None), getattr(i, "in_out", None),
                       getattr(i, "socket_type", None), _properties(i, {"parent", "interface_items"}))
                      for i in tree.interface.items_tree)
    core = set(bpy.types.Node.bl_rna.properties.keys())
    nodes = tuple(sorted((n.name, n.bl_idname, n.mute, sockets(n.inputs), sockets(n.outputs),
                          _properties(n, core)) for n in tree.nodes))
    links = tuple((l.from_node.name, l.from_socket.identifier, l.to_node.name, l.to_socket.identifier,
                   l.is_muted, getattr(l, 'multi_input_sort_id', 0)) for l in tree.links)
    return tree.bl_idname, interface, nodes, links, drivers


def restore_groups(materials, objects=()):
    """Touch only references on supplied locally authored MToon owners."""
    matches = {}
    def restore(owner, attr):
        group = getattr(owner, attr, None)
        if group is None or group.library:
            return
        if group not in matches:
            name = _ALIAS.fullmatch(group.name)
            canonical = bpy.data.node_groups.get((name[1], None)) if name else None
            matches[group] = None
            if canonical is not None and canonical != group:
                try:
                    if _signature(group) == _signature(canonical):
                        matches[group] = canonical
                except (ValueError, TypeError):
                    pass
        if matches[group] is not None:
            setattr(owner, attr, matches[group])

    for mat in materials:
        if mat is None or mat.library or mat.override_library or not mat.node_tree:
            continue
        root = getattr(getattr(mat, "vrm_addon_extension", None), "mtoon1", None)
        if root is None or not (root.get("enabled") or root.is_outline_material):
            continue
        for node in mat.node_tree.nodes:
            if node.type == 'GROUP':
                restore(node, "node_tree")
    for obj in objects:
        if obj.library or obj.override_library:
            continue
        for modifier in obj.modifiers:
            if modifier.type == 'NODES':
                group = modifier.node_group
                if group and group.name.startswith("VRM Add-on MToon 1.0 Outline Geometry Revision "):
                    restore(modifier, "node_group")


def already_mtoon(material):
    """An authored output alias is still an MToon graph, even if it differs."""
    root = material.vrm_addon_extension.mtoon1
    if root.is_outline_material or root.enabled:
        return True
    if not root.get("enabled") or not material.node_tree:
        return False
    node = material.node_tree.nodes.get("Mtoon1Material.Mtoon1Output")
    group = getattr(node, "node_tree", None)
    return bool(group and _ALIAS.fullmatch(group.name)
                and group.name.startswith("VRM Add-on MToon 1.0 Output Revision "))


def clean_name(name):
    # Blender suffix and the suffix of the Eevee source used in a BVT conversion.
    return re.sub(r"(?:\.\d{3,})+(?=$| · MToon$)", "", re.sub(r"(?:\.\d{3,})+$", "", name))


def material_name(stem, qualifier):
    # Purposeful variants need distinct names, including at Blender's byte limit.
    def trim(text, limit):
        return text.encode("utf8")[:limit].decode("utf8", errors="ignore")
    qualifier = trim(qualifier, 28)
    index = 1
    while True:
        suffix = " · " + qualifier + (f" {index}" if index > 1 else "")
        name = trim(clean_name(stem), 63 - len(suffix.encode("utf8"))) + suffix
        if bpy.data.materials.get(name) is None:
            return name
        index += 1


def material_copy(material, qualifier=None):
    """Make an independent editable variant, including its VRM outline."""
    result = material.copy()
    outline_copy = None
    try:
        if qualifier:
            result.name = material_name(material.name, qualifier)
        root = getattr(getattr(result, "vrm_addon_extension", None), "mtoon1", None)
        if root and root.outline_material:
            outline_copy = root.outline_material.copy()
            outline_copy.name = material_name(result.name, "Outline")
            root.outline_material = outline_copy
        restore_groups((result, outline_copy))
        return result
    except Exception:
        bpy.data.materials.remove(result)
        if outline_copy and not outline_copy.users:
            bpy.data.materials.remove(outline_copy)
        raise


def needs_group_repair(material):
    root = getattr(getattr(material, "vrm_addon_extension", None), "mtoon1", None)
    if not root or not root.get("enabled"):
        return False
    for owner in (material, root.outline_material):
        if owner and owner.node_tree and any(
            n.type == 'GROUP' and n.node_tree and _ALIAS.fullmatch(n.node_tree.name)
            for n in owner.node_tree.nodes
        ):
            return True
    return False


def _stored(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "to_list"):
        return value.to_list()
    return value


def _same_material(a, b):
    if not b or b.library or a.animation_data or b.animation_data or a.is_grease_pencil or b.is_grease_pencil:
        return False
    def settings(mat):
        stored = {k: _stored(v) for k, v in mat.items()}
        properties = []
        for prop in mat.bl_rna.properties:
            key = prop.identifier
            if key in bpy.types.ID.bl_rna.properties or prop.is_readonly and not prop.is_runtime:
                continue
            if prop.is_runtime and not mat.is_property_set(key):
                continue
            value = getattr(mat, key)
            if isinstance(value, bpy.types.PropertyGroup):
                value = {k: _stored(v) for k, v in value.items()}
            else:
                value = _value(value)
            properties.append((key, value))
        return stored, properties, _properties(mat.lineart)
    try:
        if settings(a) != settings(b):
            return False
        if not a.node_tree or not b.node_tree:
            return a.node_tree == b.node_tree
        if any(len(node.keys()) for tree in (a.node_tree, b.node_tree) for node in tree.nodes):
            return False
        return _signature(a.node_tree, a) == _signature(b.node_tree, b)
    except (TypeError, ValueError, AttributeError):
        return False


def rig_material(material, rig_label):
    """Rig generation reads materials. Normally the original ID is sufficient."""
    name = clean_name(material.name)
    if name == material.name and not needs_group_repair(material):
        return material
    original = bpy.data.materials.get((name, None))
    if original != material and _same_material(material, original):
        return original
    # Numerical suffixes alone cannot prove two independently edited materials
    # are equivalent. Keep the source intact when it already has a distinct look.
    result = material_copy(material, rig_label)
    if bpy.data.materials.get(name) is None:
        result.name = name
    return result


@persistent
def after_append(context):
    """Only a newly appended generated rig owns an automatic repair."""
    if 'LINK' in context.options:
        return
    objects, materials, scenes = set(), set(), set()
    for item in context.import_items:
        if item.append_action not in {'MAKE_LOCAL', 'COPY_LOCAL'} or not item.id:
            continue
        if item.id_type == 'OBJECT':
            objects.add(item.id)
        elif item.id_type == 'MATERIAL':
            materials.add(item.id)
        elif item.id_type == 'SCENE':
            scenes.add(item.id)
    rigs = [o for o in objects if o.type == 'ARMATURE' and
            (o.get('bvt_generated_rig') or o.get('mesh_only_rig_report'))]
    if len(rigs) != 1:
        return
    def owner(obj):
        arm = obj.find_armature()
        if arm:
            return arm
        parent = obj.parent
        while parent and parent.type != 'ARMATURE':
            parent = parent.parent
        return parent
    meshes = {o for o in objects if o.type == 'MESH' and owner(o) == rigs[0]}
    targets = {s.material for o in meshes for s in o.material_slots if s.material in materials}
    def with_outlines(mats):
        result = set(mats)
        for mat in mats:
            root = getattr(getattr(mat, "vrm_addon_extension", None), "mtoon1", None)
            if root and root.outline_material in materials:
                result.add(root.outline_material)
        return result
    other = {s.material for o in objects - meshes if o.type == 'MESH'
             for s in o.material_slots if s.material in materials}
    allowed = with_outlines(targets) - with_outlines(other)
    if not allowed:
        return
    owners = bpy.data.user_map(subset=allowed)
    owned = meshes | {obj.data for obj in meshes} | {rigs[0], rigs[0].data} | allowed
    owned.update(sc for sc in scenes if all(o == rigs[0] for o in sc.objects if o.type == 'ARMATURE'))
    # Follow references between materials too. A shared outline can otherwise
    # expose a private repair through a surface on a different character.
    while True:
        private = {m for m in allowed if owners[m] <= owned}
        if private == allowed:
            break
        owned -= allowed - private
        allowed = private
    restore_groups(allowed, meshes)
    # Scene-level references may belong to other characters. Only consolidate
    # an imported ID when every direct user belongs to this rig.
    for mat in targets & allowed:
        name = clean_name(mat.name)
        if name == mat.name or owners[mat] - owned:
            continue
        original = bpy.data.materials.get((name, None))
        if _same_material(mat, original):
            mat.user_remap(original)
        elif not original:
            mat.name = name
        else:
            mat.name = material_name(name, clean_name(rigs[0].name))
        outline = getattr(getattr(mat, "vrm_addon_extension", None), "mtoon1", None)
        outline = outline.outline_material if outline else None
        if outline in allowed and not owners[outline] - owned:
            outline.name = material_name(mat.name, "Outline")



def install():
    handlers = getattr(bpy.app.handlers, "blend_import_post", None)
    if handlers is not None and after_append not in handlers:
        handlers.append(after_append)


def uninstall():
    handlers = getattr(bpy.app.handlers, "blend_import_post", None)
    if handlers is not None and after_append in handlers:
        handlers.remove(after_append)
