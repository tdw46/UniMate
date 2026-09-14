"""VRM metadata adapter for the shared destructive avatar preparation flow."""
import json
import re
import struct

import bpy
import bmesh
from mathutils import Matrix


def read_gltf(path):
    data = path.read_bytes()
    if data[:4] == b'glTF':
        length = struct.unpack_from('<I', data, 12)[0]
        return json.loads(data[20:20+length])
    return json.loads(data)


def adapt_humanoid(path, rig):
    gltf = read_gltf(path)
    extensions = gltf.get('extensions', {})
    if 'VRM' in extensions:
        bones = {b['bone']: b['node'] for b in extensions['VRM']['humanoid']['humanBones']}
    elif 'VRMC_vrm' in extensions:
        bones = {name: b['node'] for name,b in extensions['VRMC_vrm']['humanoid']['humanBones'].items()}
    else:
        return None
    aliases = {'hips':'Hips', 'spine':'Abdomen', 'chest':'Torso', 'neck':'Neck', 'head':'Head'}
    for side, suffix in [('left','L'),('right','R')]:
        aliases.update({side+'Shoulder':'Shoulder.'+suffix,
                        side+'UpperArm':'UpperArm.'+suffix,
                        side+'LowerArm':'LowerArm.'+suffix,
                        side+'Hand':'Wrist.'+suffix})
    renamed = {}
    for role, alias in aliases.items():
        if role not in bones:
            raise ValueError('Required humanoid landmark missing: '+role)
        original = gltf['nodes'][bones[role]]['name']
        bone = rig.data.bones.get(original)
        if bone is None:
            raise ValueError('Imported humanoid bone not found: '+original)
        bone.name = alias
        assert bone.name == alias, 'Humanoid alias collides with another source bone'
        renamed[role] = {'source_bone':original, 'landmark':alias}
    # Canonical frame from semantic landmarks: anatomical left +X, up +Z.
    # Rotate all imported roots together, preserving skinning and child transforms.
    point = lambda name: rig.matrix_world @ rig.data.bones[name].head_local
    up = point('Neck')-point('Hips')
    if up.length < 1e-8:
        raise ValueError('Coincident hips and neck cannot define humanoid up')
    up.normalize()
    left = point('UpperArm.L')-point('UpperArm.R')
    left -= up*left.dot(up)
    if left.length < 1e-8:
        raise ValueError('Degenerate shoulder landmarks cannot define humanoid left')
    left.normalize()
    back = up.cross(left).normalized()
    rotation = Matrix((left, back, up)).to_4x4()
    for obj in bpy.context.scene.objects:
        if obj.parent is None:
            obj.matrix_world = rotation @ obj.matrix_world
    bpy.context.view_layer.update()
    return {'format':'VRM', 'humanoid_landmarks':renamed,
            'canonical_rotation': [list(row) for row in rotation],
            'material_roles':'VRoid semantic material tokens; no avatar identities or weights'}


def material_role(material):
    if material is None:
        return None
    # Authoring metadata, not an avatar allowlist. Preserve explicit overrides.
    if material.get('binding_role'):
        return material['binding_role']
    words = set(re.sub(r'([a-z])([A-Z])', r'\1_\2', material.name).lower().split('_'))
    if 'accessoryneck' in words or ('accessory' in words and 'neck' in words):
        # Cloth at the neckline (ties, bows, ribbon tails) drapes onto the
        # chest. Use the shared body-surface transfer and anatomical Neck cap;
        # the accessory's location alone does not imply a rigid Neck binding.
        return 'torso' if 'cloth' in words else 'neckwear'
    if words & {'hair','hairback','face','eye'}:
        return 'head'
    if 'body' in words and 'skin' in words:
        return 'body'
    if 'cloth' in words:
        return 'torso'
    return None


def split_material_regions(meshes, rigid_parts):
    """Split mixed meshes by semantic material role after baking and stripping."""
    result, audit = [], []
    for obj in meshes:
        for material in obj.data.materials:
            if material and not material.get('binding_surface') and 'skin' in material.name.lower().split('_'):
                material['binding_surface'] = 'skin'
        roles = {i: material_role(m) for i,m in enumerate(obj.data.materials)}
        used = {roles.get(p.material_index) for p in obj.data.polygons}
        if used == {None}:
            result.append(obj)
            continue
        old_name = obj.name
        for role in sorted(used, key=str):
            data = obj.data.copy()
            bm = bmesh.new()
            bm.from_mesh(data)
            bmesh.ops.delete(bm, geom=[f for f in bm.faces if roles.get(f.material_index) != role], context='FACES')
            bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
            bm.to_mesh(data)
            bm.free()
            if not data.polygons:
                bpy.data.meshes.remove(data)
                continue
            part = bpy.data.objects.new(old_name+'__'+str(role or 'unclassified'), data)
            bpy.context.scene.collection.objects.link(part)
            part.matrix_world = obj.matrix_world.copy()
            if role:
                part['binding_role'] = role
            if role == 'head':
                rigid_parts[part.name] = 'Head'
            result.append(part)
            audit.append({'source_mesh':old_name, 'mesh':part.name, 'role':role,
                          'vertices':len(data.vertices)})
        rigid_parts.pop(old_name, None)
        bpy.data.objects.remove(obj, do_unlink=True)
    return result, audit
