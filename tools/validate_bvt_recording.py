"""Read back BVT demo output and compare saved bake keys to captured poses."""
import argparse
import json
import sys
from pathlib import Path
import bpy
from mathutils import Quaternion
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_physics_preview import enabled
parser=argparse.ArgumentParser()
parser.add_argument('directory',type=Path)
parser.add_argument('--name',default='sample_b_springs')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
for repo in bpy.context.preferences.extensions.repos:
    if repo.module=='user_default':
        repo.use_custom_directory=True
        repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
for module in ('vrm','beyond_vrm_extension_suite'):
    bpy.ops.preferences.addon_enable(module='bl_ext.user_default.'+module)
bpy.ops.wm.open_mainfile(filepath=str((args.directory/(args.name+'.blend')).resolve()))
assert enabled(), 'Editable take must use BVT'
bpy.ops.wm.open_mainfile(filepath=str((args.directory/(args.name+'_baked.blend')).resolve()))
assert not enabled(), 'Baked take must not simulate again'
rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
samples=json.loads((args.directory/'motion_samples.json').read_text())
error=0.
for sample in samples:
    bpy.context.scene.frame_set(sample['frame'])
    for name,values in sample['bones'].items():
        a=rig.pose.bones[name].rotation_quaternion
        b=Quaternion(values)
        error=max(error,min(max(abs(x-y) for x,y in zip(a,b)),max(abs(x+y) for x,y in zip(a,b))))
assert error<1e-5,error
assert 'avatar_spring_runtime' not in sys.modules
result={'frames':len(samples),'maximum_saved_quaternion_error':error,'editable_bvt_enabled':True,'baked_bvt_disabled':True,'solver':'BVT'}
(args.directory/'bake_readback.json').write_text(json.dumps(result,indent=2))
print('BVT_BAKE_READBACK_OK',json.dumps(result),flush=True)
