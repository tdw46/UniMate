"""The final rig must preserve the supplied mesh and every existing shape key."""
import hashlib
import numpy as np


def signature(obj):
    h=hashlib.sha256();mesh=obj.data
    def floats(items):h.update(np.asarray(items,dtype=np.float32).tobytes())
    floats([v.co[:] for v in mesh.vertices])
    for edge in mesh.edges:h.update(repr(tuple(edge.vertices)).encode())
    for poly in mesh.polygons:h.update(repr((tuple(poly.vertices),poly.material_index)).encode())
    for layer in mesh.uv_layers:
        h.update(layer.name.encode());floats([v.uv[:] for v in layer.data])
    if mesh.shape_keys:
        for key in mesh.shape_keys.key_blocks:
            h.update(repr((key.name,key.relative_key.name,key.vertex_group)).encode())
            floats([v.co[:] for v in key.data])
    return h.hexdigest()


def snapshot(meshes):return {o.name:signature(o) for o in meshes}


def verify(meshes,before):
    after=snapshot(meshes)
    if after!=before:raise RuntimeError('Rig operation changed final mesh geometry, topology, UVs or existing shape keys')
    return after
