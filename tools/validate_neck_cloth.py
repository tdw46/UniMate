"""Check draped neckline cloth under isolated motion, with optional old evidence."""
import json
import os
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Quaternion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from avatar_apparel_weights import weights
OUT = Path(os.environ.get('AVATAR_EVAL_ROOT', ROOT/'outputs/complex_avatar_grid')).resolve()
BEFORE = '--before' in sys.argv


def vertices(obj, rig, bone=None):
    for pb in rig.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    if bone:
        basis = rig.data.bones[bone].matrix_local.to_quaternion()
        rig.pose.bones[bone].rotation_quaternion = basis.inverted() @ Quaternion((0,1,0), .5) @ basis
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    points = [evaluated.matrix_world @ v.co for v in mesh.vertices]
    evaluated.to_mesh_clear()
    return points


report = {'baseline':BEFORE,'checks':[]}
for entry in json.loads((OUT/'sources/manifest.json').read_text()):
    folder = OUT/('before_neck_cloth_fix' if BEFORE else 'avatars')/entry['id']
    if not (folder/'02_fresh_rig.blend').exists():
        continue
    bpy.ops.wm.open_mainfile(filepath=str(folder/'02_fresh_rig.blend'))
    rig = next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
    rig.animation_data_clear()
    neck = rig.data.bones['Neck']
    for obj in [o for o in bpy.context.scene.objects if o.type=='MESH']:
        # Material regions provide ground truth even after merging with a shirt.
        slots = {i for i,m in enumerate(obj.data.materials)
                 if m and 'accessoryneck' in m.name.lower() and 'cloth' in m.name.lower()}
        indices = sorted({i for p in obj.data.polygons if p.material_index in slots for i in p.vertices})
        if not indices:
            continue
        values = [weights(obj,i) for i in indices]
        neck_values = [w.get('Neck',0) for w in values]
        torso_values = [sum(w.get(n,0) for n in ('Spine','Chest','Clavicle.L','Clavicle.R')) for w in values]
        rest = vertices(obj,rig)
        motion = {n:vertices(obj,rig,n) for n in ('Head','Neck','Chest')}
        lower = [i for i in indices if obj.data.vertices[i].co.z < neck.head_local.z-neck.length*.35]
        distance = lambda bone,ids: max(((motion[bone][i]-rest[i]).length for i in ids),default=0.)
        case = {'id':entry['id'],'vertices':len(indices),
                'neck_weight_min':min(neck_values),'neck_weight_max':max(neck_values),
                'neck_weight_mean':sum(neck_values)/len(indices),
                'torso_weight_mean':sum(torso_values)/len(indices),
                'head_only_displacement':distance('Head',indices),
                'neck_only_displacement':distance('Neck',indices),
                'chest_only_displacement':distance('Chest',indices),
                'below_collar_vertices':len(lower),
                'below_collar_neck_displacement':distance('Neck',lower)}
        # A bow entirely below the neck base can correctly follow almost only
        # the shirt. Require appreciable Neck response only for cloth that
        # actually extends onto the neck, rather than forcing an arbitrary mix.
        on_neck = [i for i in indices if obj.data.vertices[i].co.z > neck.head_local.z]
        case['vertices_above_neck_base'] = len(on_neck)
        neck_attachment = not on_neck or (max(neck_values)>.01 and case['neck_only_displacement']>.001)
        case['passed'] = (min(neck_values)<.05 and max(neck_values)<.999
                          and case['torso_weight_mean']>.5
                          and case['head_only_displacement']<1e-5
                          and neck_attachment
                          and case['chest_only_displacement']>.01
                          and bool(lower) and case['below_collar_neck_displacement']<1e-5)
        report['checks'].append(case)
        print('NECK_CLOTH',json.dumps(case),flush=True)
report['passed'] = bool(report['checks']) and all(c['passed'] for c in report['checks'])
(OUT/('neck_cloth_validation_before.json' if BEFORE else 'neck_cloth_validation.json')).write_text(json.dumps(report,indent=2))
if not BEFORE:
    assert report['passed'],report
