"""Validate semantic regions under isolated motion, without avatar-specific fixtures."""
import json
import os
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Quaternion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from avatar_apparel_weights import weights
OUT = Path(os.environ.get('AVATAR_EVAL_ROOT', ROOT/'outputs/complex_avatar_grid')).resolve()


def capture(meshes):
    deps = bpy.context.evaluated_depsgraph_get()
    result = {}
    for obj in meshes:
        evaluated = obj.evaluated_get(deps)
        mesh = evaluated.to_mesh()
        result[obj.name] = [evaluated.matrix_world @ v.co for v in mesh.vertices]
        evaluated.to_mesh_clear()
    return result


def pose(rig, meshes, bone=None):
    for pb in rig.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    if bone:
        pb = rig.pose.bones[bone]
        pb.rotation_mode = 'QUATERNION'
        basis = (rig.matrix_world @ rig.data.bones[bone].matrix_local).to_quaternion()
        pb.rotation_quaternion = basis.inverted() @ Quaternion((0,1,0), .5) @ basis
    bpy.context.view_layer.update()
    return capture(meshes)


report = {'blender':bpy.app.version_string, 'avatars':[]}
for entry in json.loads((OUT/'sources/manifest.json').read_text()):
    bpy.ops.wm.open_mainfile(filepath=str(OUT/'avatars'/entry['id']/'02_fresh_rig.blend'))
    rig = next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
    rig.animation_data_clear()
    meshes = [o for o in bpy.context.scene.objects if o.type=='MESH']
    assert len(rig.data.bones)==13
    assert all(abs(sum(g.weight for g in v.groups)-1)<1e-5 and
               sum(g.weight>1e-8 for g in v.groups)<=4
               for obj in meshes for v in obj.data.vertices)
    rest = pose(rig,meshes)
    motions = {n:pose(rig,meshes,n) for n in
               ('Head','Neck','Spine','Chest','Clavicle.L','Clavicle.R','UpperArm.L','UpperArm.R')}
    neck = rig.data.bones['Neck']
    case = {'id':entry['id'], 'vertices':sum(len(o.data.vertices) for o in meshes),
            'triangles':sum(sum(len(p.vertices)-2 for p in o.data.polygons) for o in meshes),
            'normalized_at_most_four_weights':True, 'regions':[]}
    for obj in meshes:
        role = obj.get('binding_role')
        assert role in ('body','head','neckwear','torso'), (obj.name,role)
        indices = list(range(len(obj.data.vertices)))
        displacement = lambda bone, ids: max(((motions[bone][obj.name][i]-rest[obj.name][i]).length for i in ids),default=0.)
        result = {'mesh':obj.name,'role':role,'vertices':len(indices),
                  'max_displacement':{n:displacement(n,indices) for n in motions}}
        if role in ('head','neckwear'):
            target = 'Head' if role=='head' else 'Neck'
            assert all(abs(weights(obj,i).get(target,0)-1)<1e-6 for i in indices)
            posed = pose(rig,meshes,target)[obj.name]
            transform = rig.matrix_world @ rig.pose.bones[target].matrix @ rig.data.bones[target].matrix_local.inverted() @ rig.matrix_world.inverted()
            result['rigid_transform_error'] = max((posed[i]-transform @ rest[obj.name][i]).length for i in indices)
            assert result['rigid_transform_error'] < 1e-5
            if role=='neckwear':
                assert displacement('Head',indices)<1e-5
        if role in ('body','torso'):
            lower = [i for i in indices if obj.data.vertices[i].co.z < neck.head_local.z-neck.length*.35]
            result['below_collar_vertices'] = len(lower)
            result['below_collar_neck_displacement'] = displacement('Neck',lower)
            assert displacement('Neck',lower)<1e-5
            if role=='torso':
                assert all(weights(obj,i).get('Head',0)<1e-8 for i in indices)
                assert displacement('Head',indices)<1e-5
                assert displacement('Spine',indices)>.01
                assert displacement('Chest',indices)>.01
                for suffix,sign in [('L',1),('R',-1)]:
                    shoulder = rig.data.bones['UpperArm.'+suffix].head_local
                    lateral = [i for i in indices if sign*(obj.data.vertices[i].co.x-shoulder.x)>neck.length]
                    if lateral:
                        result['sleeve_'+suffix+'_vertices'] = len(lateral)
                        assert displacement('UpperArm.'+suffix,lateral)>.01
        case['regions'].append(result)
    case['passed'] = True
    report['avatars'].append(case)
    print('REGIONS_PASSED',entry['id'],flush=True)
report['passed'] = len(report['avatars'])==9
(OUT/'region_validation.json').write_text(json.dumps(report,indent=2))
