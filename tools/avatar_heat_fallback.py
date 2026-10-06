"""Retry complete combined heat failures on independent material surfaces.

Only geometry and the newly generated skeleton participate; source weights are
never read. Shared vertices average their incident domain solutions, so material
boundaries cannot depend on iteration order. Normal combined heat is unchanged.
"""
import bpy
from avatar_apparel_weights import weights, assign


def seed_material_heat(proxy, rig, domains):
    from autorig_bust import repaired_heat
    accumulated = {}
    reports = []
    for label, faces in domains:
        ids = sorted({i for face in faces for i in face})
        if not ids:
            continue
        lookup = {i: j for j, i in enumerate(ids)}
        data = bpy.data.meshes.new('Temporary isolated heat surface')
        data.from_pydata([proxy.data.vertices[i].co for i in ids], [],
                         [[lookup[i] for i in face] for face in faces])
        data.update()
        obj = bpy.data.objects.new('Temporary isolated heat surface', data)
        bpy.context.scene.collection.objects.link(obj)
        try:
            report = repaired_heat([obj], rig, allow_unweighted=True)
            seeded = 0
            for local, global_id in enumerate(ids):
                row = weights(obj, local)
                if not row:
                    continue
                seeded += 1
                target = accumulated.setdefault(global_id, {})
                for name, value in row.items():
                    target[name] = target.get(name, 0.0) + value
            reports.append(dict(domain=label, vertices=len(ids), seeded=seeded,
                                heat=report))
        finally:
            bpy.data.objects.remove(obj, do_unlink=True)
            if not data.users:
                bpy.data.meshes.remove(data)
    for index, row in accumulated.items():
        total = sum(row.values())
        assign(proxy, [index], {name: value / total for name, value in row.items()})
    return dict(method='material-isolated fresh heat after complete combined failure',
                seeded_vertices=len(accumulated), domains=reports,
                source_weights_used=False)
