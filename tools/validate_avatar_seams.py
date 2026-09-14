"""Measure skin-boundary separation under evaluated motion, independently of transfer."""
import hashlib
import json
import os
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Quaternion
from mathutils.kdtree import KDTree

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('AVATAR_EVAL_ROOT',ROOT/'outputs/complex_avatar_grid')).resolve()
BEFORE='--before' in sys.argv
FIRST='--first-only' in sys.argv


def fingerprint(obj):
    # Geometry, face/material assignments, UVs, and texture references must stay
    # identical to the pre-correction scene. Canonicalize polygon order because
    # rerunning Exact Boolean can reorder cap faces; keep each corner's UVs.
    faces=[]
    for p in obj.data.polygons:
        corners=[(obj.data.loops[i].vertex_index,
                  tuple(tuple(layer.data[i].uv) for layer in obj.data.uv_layers)) for i in p.loop_indices]
        start=min(range(len(corners)),key=lambda i:corners[i])
        faces.append((p.material_index,tuple(corners[start:]+corners[:start])))
    values={'vertices':[tuple(v.co) for v in obj.data.vertices],
            'polygons_and_corner_uvs':sorted(faces),
            'materials':[m.name if m else None for m in obj.data.materials]}
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()


def capture(objects):
    deps=bpy.context.evaluated_depsgraph_get()
    result={}
    for obj in objects:
        ev=obj.evaluated_get(deps)
        mesh=ev.to_mesh()
        result[obj.name]=[ev.matrix_world @ v.co for v in mesh.vertices]
        ev.to_mesh_clear()
    return result


report={'baseline':BEFORE,'avatars':[]}
entries=json.loads((OUT/'sources/manifest.json').read_text())
for entry in entries[:1] if FIRST else entries:
    folder=OUT/('before_voxel_seam_fix' if BEFORE else 'avatars')/entry['id']
    previous={}
    if not BEFORE:
        baseline=OUT/'before_voxel_seam_fix'/entry['id']/'02_fresh_rig.blend'
        if not baseline.exists():baseline=folder/'01_cut_unrigged.blend'
        bpy.ops.wm.open_mainfile(filepath=str(baseline))
        previous={o.name:fingerprint(o) for o in bpy.context.scene.objects if o.type=='MESH'}
    bpy.ops.wm.open_mainfile(filepath=str(folder/'02_fresh_rig.blend'))
    scene=bpy.context.scene
    rig=next(o for o in scene.objects if o.type=='ARMATURE')
    meshes=[o for o in scene.objects if o.type=='MESH']
    intact=BEFORE or previous=={o.name:fingerprint(o) for o in meshes}
    assert intact, 'Original geometry, UVs or material assignments changed'
    # Independent ground truth: nearest skin vertices on different objects.
    # Use creator SKIN material semantics so this also works on old scenes.
    parts=[]
    for obj in meshes:
        if obj.get('binding_role') not in ('body','head'):continue
        ids={i for p in obj.data.polygons if 'skin' in obj.data.materials[p.material_index].name.lower().split('_') for i in p.vertices}
        if ids:parts.append((obj,ids))
    pairs=[]
    tolerance=rig.data.bones['Neck'].length*.00625
    for k,(obj,ids) in enumerate(parts):
        tree=KDTree(len(ids))
        for i in ids:tree.insert(obj.data.vertices[i].co,i)
        tree.balance()
        for other,other_ids in parts[k+1:]:
            for j in other_ids:
                _,i,distance=tree.find(other.data.vertices[j].co)
                if distance<=tolerance:pairs.append((obj.name,i,other.name,j,distance))
    assert pairs
    fully_head=True
    for a,i,b,j,_ in pairs:
        for name,index in ((a,i),(b,j)):
            obj=bpy.data.objects[name]
            head=obj.vertex_groups.get('Head')
            try:weight=head.weight(index) if head else 0.
            except RuntimeError:weight=0.
            fully_head=fully_head and abs(weight-1)<1e-6
    max_gap=max_growth=0.
    for frame in range(360):
        scene.frame_set(frame)
        points=capture(meshes)
        for a,i,b,j,rest_distance in pairs:
            distance=(points[a][i]-points[b][j]).length
            max_gap=max(max_gap,distance)
            max_growth=max(max_growth,distance-rest_distance)
    # Stress each requested joint about all three world axes, beyond a single
    # authored animation trajectory. Compare the same original seam pairs.
    rig.animation_data_clear()
    stress_growth=0.
    for bone in ('Head','Neck'):
        for axis in ((1,0,0),(0,1,0),(0,0,1)):
            for angle in (-.8,.8):
                for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
                basis=rig.data.bones[bone].matrix_local.to_quaternion()
                rig.pose.bones[bone].rotation_quaternion=basis.inverted() @ Quaternion(axis,angle) @ basis
                bpy.context.view_layer.update()
                points=capture(meshes)
                stress_growth=max(stress_growth,max((points[a][i]-points[b][j]).length-d for a,i,b,j,d in pairs))
    case={'id':entry['id'],'paired_skin_vertices':len(pairs),
          'max_rest_gap':max(p[4] for p in pairs),'max_animated_gap':max_gap,
          'max_gap_growth_360_frames':max_growth,'max_gap_growth_12_stress_poses':stress_growth,
          'geometry_uv_material_fingerprint_preserved':intact,
          'head_neck_boundary_fully_head':fully_head,
          'passed':fully_head and max(max_growth,stress_growth)<1e-5}
    report['avatars'].append(case)
    print('SKIN_SEAM',json.dumps(case),flush=True)
report['passed']=all(a['passed'] for a in report['avatars'])
(OUT/('seam_validation_before.json' if BEFORE else 'seam_validation.json')).write_text(json.dumps(report,indent=2))
if not BEFORE:assert report['passed'],report
