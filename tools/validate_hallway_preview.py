"""The installed Hallway addon must not load or schedule its old solver."""
import sys
from pathlib import Path
import bpy
import addon_utils
sys.path.insert(0,str(Path.home()/'Documents/Blender/addons'))
addon_utils.enable('hallway_rig',default_set=True)
assert addon_utils.check('hallway_rig')[1]
assert not (Path(__file__).resolve().parent/'avatar_spring_runtime.py').exists()
assert 'avatar_spring_runtime' not in sys.modules
import avatar_physics_preview as preview
assert not preview.available()
assert not bpy.ops.hallway.set_physics.poll()
assert bpy.types.HALLWAY_PT_Rig.bl_category=='Hallway'
assert not any(getattr(fn,'__module__','').startswith('avatar_') for fn in bpy.app.handlers.frame_change_post)
for _ in range(2):
    addon_utils.disable('hallway_rig',default_set=True)
    assert not hasattr(bpy.types,'HALLWAY_PT_Rig')
    addon_utils.enable('hallway_rig',default_set=True)
assert 'avatar_spring_runtime' not in sys.modules
print('HALLWAY_BVT_DELEGATION_OK: no internal solver, missing-BVT controls disabled, reload passed',flush=True)
