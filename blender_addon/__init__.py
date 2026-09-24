"""Small development loader; the implementation stays in the project source."""
import json
import sys
from pathlib import Path

bl_info = {
    'name': 'Hallway Rig Tools',
    'author': 'Hallway',
    'version': (0, 1, 0),
    'blender': (3, 6, 0),
    'location': 'View3D > Sidebar > Hallway',
    'description': 'Generated avatar rig configuration and spring preview',
    'category': 'Rigging',
}


def register():
    root = Path(__file__).resolve().parent
    config = root / 'source.json'
    source = Path(json.loads(config.read_text())['tools_directory']) if config.exists() else root.parent / 'tools'
    if not (source / 'ui_hallway_rig.py').is_file():
        raise RuntimeError('Hallway rig source is missing; reinstall the development loader from the project')
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import ui_hallway_rig
    ui_hallway_rig.register()


def unregister():
    module = sys.modules.get('ui_hallway_rig')
    if module is not None:
        module.unregister()
