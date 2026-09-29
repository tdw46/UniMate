"""Isolated Blender test for bounded body-weight projection and fallback."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import bpy
from mathutils import Matrix, Vector
sys.path.insert(0, str(Path(__file__).resolve().parent))
from avatar_waist_attachment import WaistAttachment
from avatar_mesh_invariant import snapshot, verify


def run():
    if not bpy.app.background:
        raise RuntimeError('Use isolated background Blender for attachment fixtures')
    fields = {'hips':'Hips', 'spine':'Spine', 'chest':'', 'upper_chest':'',
              'left_upper_leg':'LeftUpLeg', 'right_upper_leg':'RightUpLeg'}
    human = NS(**{k:NS(node=NS(bone_name=v)) for k,v in fields.items()})
    bones = {n:NS(length=1., head_local=Vector((0,0,.5))) for n in ('Hips','Spine')}
    rig = NS(matrix_world=Matrix.Identity(4), data=NS(bones=bones,
        vrm_addon_extension=NS(vrm1=NS(humanoid=NS(human_bones=human)))))
    mesh = bpy.data.meshes.new('Waist test mesh')
    obj = bpy.data.objects.new('Waist test body', mesh)
    mat = bpy.data.materials.new('Waist test skin')
    try:
        mesh.from_pydata([(0,-.1,0),(0,.1,0),(0,.1,1),(0,-.1,1)], [], [(0,1,2,3)])
        mesh.materials.append(mat)
        obj.vertex_groups.new(name='Hips').add([0,1],1.,'REPLACE')
        obj.vertex_groups.new(name='Spine').add([2,3],1.,'REPLACE')
        before = snapshot([obj])
        sample = WaistAttachment(rig, [obj])
        value = sample.sample(Vector((.05,0,.5)))
        assert abs(value['Hips']-.5)<1e-6 and abs(value['Spine']-.5)<1e-6
        assert sample.sample(Vector((2,0,.5))) == {'Hips':1.}
        assert WaistAttachment(rig, []).sample(Vector((.05,0,.5))) == {'Hips':1.}
        verify([obj], before)
        print('WAIST_ATTACHMENT_REGRESSION_OK: barycentric body weights; bounded projection; pelvis fallback; geometry unchanged')
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)
        bpy.data.materials.remove(mat)


if __name__ == '__main__':
    run()
