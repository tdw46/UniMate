"""Exercise VRM metadata adaptation independently of downloaded characters."""
import json
from pathlib import Path
import sys
import tempfile

import bpy
from mathutils import Quaternion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from avatar_source_import import adapt_humanoid, split_material_regions

landmarks = {'hips':(0,0,0), 'spine':(0,0,.4), 'chest':(0,0,.8),
             'neck':(0,0,1), 'head':(0,0,1.3)}
for side,sign in [('left',1),('right',-1)]:
    for role,x in [('Shoulder',.2),('UpperArm',.4),('LowerArm',.8),('Hand',1.1)]:
        landmarks[side+role]=(sign*x,0,.9)

with tempfile.TemporaryDirectory() as folder:
    for version in (0,1):
        for angle in (0.,1.2,3.141592653589793):
            bpy.ops.wm.read_factory_settings(use_empty=True)
            rig=bpy.data.objects.new('Unrecognized source',bpy.data.armatures.new('Source'))
            bpy.context.scene.collection.objects.link(rig)
            bpy.context.view_layer.objects.active=rig
            rig.select_set(True)
            bpy.ops.object.mode_set(mode='EDIT')
            nodes=[]
            for i,(role,position) in enumerate(landmarks.items()):
                bone=rig.data.edit_bones.new('arbitrary_joint_'+str(i))
                bone.head=position
                bone.tail=(position[0],position[1],position[2]+.1)
                nodes.append({'name':bone.name})
            bpy.ops.object.mode_set(mode='OBJECT')
            root=bpy.data.objects.new('Imported root',None)
            bpy.context.scene.collection.objects.link(root)
            rig.parent=root
            root.matrix_world=Quaternion((.3,.2,.9),angle).to_matrix().to_4x4()
            bpy.context.view_layer.update()
            human={role:{'node':i} for i,role in enumerate(landmarks)}
            if version==0:
                ext={'VRM':{'humanoid':{'humanBones':[{'bone':role,**value} for role,value in human.items()]}}}
            else:
                ext={'VRMC_vrm':{'humanoid':{'humanBones':human}}}
            path=Path(folder)/'metadata.gltf'
            path.write_text('{}')
            original_matrix=rig.matrix_world.copy()
            assert adapt_humanoid(path,rig) is None
            assert rig.matrix_world==original_matrix
            path.write_text(json.dumps({'nodes':nodes,'extensions':ext}))
            result=adapt_humanoid(path,rig)
            for role,position in landmarks.items():
                alias=result['humanoid_landmarks'][role]['landmark']
                actual=rig.matrix_world@rig.data.bones[alias].head_local
                assert max(abs(actual[i]-position[i]) for i in range(3))<1e-6,(version,angle,role,actual)
            print('HUMANOID_FRAME_PASSED',version,angle)

bpy.ops.wm.read_factory_settings(use_empty=True)
mesh=bpy.data.meshes.new('Mixed geometry')
mesh.from_pydata([(x*2+dx,dy,0) for x in range(4) for dx,dy in [(0,0),(1,0),(0,1)]],
                 [],[(3*i,3*i+1,3*i+2) for i in range(4)])
obj=bpy.data.objects.new('Anonymous mixed mesh',mesh)
bpy.context.scene.collection.objects.link(obj)
for name in ('F00_Body_SKIN','F00_Tops_CLOTH','F00_AccessoryNeck_CLOTH','F00_AccessoryNeck_METAL'):
    mesh.materials.append(bpy.data.materials.new(name))
uv=mesh.uv_layers.new()
for p in mesh.polygons:
    p.material_index=p.index
    for loop in p.loop_indices:
        uv.data[loop].uv=(loop/10,loop/20)
expected=sorted(tuple(v.uv) for v in uv.data)
parts,audit=split_material_regions([obj],{})
assert sorted(p['binding_role'] for p in parts)==['body','neckwear','torso']
assert sorted(tuple(v.uv) for p in parts for v in p.data.uv_layers.active.data)==expected
assert sum(len(p.data.polygons) for p in parts)==4
torso=next(p for p in parts if p['binding_role']=='torso')
assert {torso.data.materials[p.material_index].name for p in torso.data.polygons}=={'F00_Tops_CLOTH','F00_AccessoryNeck_CLOTH'}
assert not any(p.vertex_groups for p in parts)
print('SEMANTIC_SPLIT_UV_PASSED')
print('SOURCE_ADAPTER_TESTS_PASSED')
