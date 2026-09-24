"""Install the project loader in Blender's configured shared addon directory.

Run inside Blender. Does not copy implementation modules or save the scene.
"""
import json
import shutil
from pathlib import Path
import addon_utils
import bpy


def install():
    root = Path(__file__).resolve().parents[1]
    scripts = Path.home() / 'Documents/Blender'
    target = scripts / 'addons/hallway_rig'
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / 'blender_addon/__init__.py', target / '__init__.py')
    (target / 'source.json').write_text(json.dumps({'tools_directory': str(root / 'tools')}, indent=2))
    paths = bpy.context.preferences.filepaths
    if hasattr(paths, 'script_directories'):
        entries = paths.script_directories
        if not any(Path(entry.directory).expanduser() == scripts for entry in entries if entry.directory):
            entry = entries.new()
            entry.name = 'Shared Blender Scripts'
            entry.directory = str(scripts)
    elif hasattr(paths, 'script_directory'):
        paths.script_directory = str(scripts)
    bpy.utils.refresh_script_paths()
    addon_utils.modules_refresh()
    addon_utils.enable('hallway_rig', default_set=True, persistent=True)
    if not addon_utils.check('hallway_rig')[1]:
        raise RuntimeError('Hallway addon did not enable')
    bpy.ops.wm.save_userpref()
    return str(target)


if __name__ == '__main__':
    print(install())
