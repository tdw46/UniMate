"""Constrain fresh humanoid heat to local anatomy before secondary rigging.

Known facial/hair surfaces have an explicit head owner even where a strand was
not accepted for physics. Unknown cranial attachments can qualify geometrically.
Body surfaces retain the upstream heat solver results.
No source armature, source weights, pose or character identifier participates.
"""
from avatar_apparel_weights import components
from avatar_material_roles import semantic_label
from avatar_leg_binding import _bone_distance


def head_domains(rig, meshes):
    head=rig.data.bones['Head'];neck=rig.data.bones['Neck']
    result={}
    for obj in meshes:
        transform=rig.matrix_world.inverted()@obj.matrix_world
        points=[transform@v.co for v in obj.data.vertices]
        known=set()
        for face in obj.data.polygons:
            mat=obj.data.materials[face.material_index] if face.material_index<len(obj.data.materials) else None
            label=semantic_label(mat.name) if mat else ''
            if any(token in label for token in ('hair','face','eye','iris','mouth','teeth','headaccessory','double')):
                known.update(face.vertices)
        # Whole detached pieces whose upper attachment is cranial. This covers
        # unnamed head ornaments without mistaking sleeves near a cheek for hair.
        for ids in components(obj.data):
            if not ids or all(i in known for i in ids):continue
            pts=[points[i] for i in ids];low=min(p.z for p in pts);high=max(p.z for p in pts)
            if high<head.head_local.z+head.length*.5 or low<neck.head_local.z-head.length*.6:continue
            top=[p for p in pts if p.z>=high-(high-low)*.2]
            if not top:continue
            torso=[rig.data.bones[n] for n in ('Chest','UpperArm.L','UpperArm.R') if n in rig.data.bones]
            if all(_bone_distance(p,head)<min(_bone_distance(p,b) for b in torso) for p in top):
                # Limit detached ornaments to the cranial neighborhood.
                if max((p-head.head_local).length for p in top)<head.length*1.7:
                    known.update(ids)
        result[obj.name]=known
    return result

