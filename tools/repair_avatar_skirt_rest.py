"""Repair an existing pose fit in an isolated Blender worker.

Arguments: current.blend original-reference.blend output-directory --rig NAME.
Use the original pre-optimization mesh as the shape reference, preserving the
current weights, bones, colliders, simulation parameters and constraints.
"""
import argparse,json,sys
from pathlib import Path
import bpy
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_skirt_fit_io import source_record,write_bundle
from avatar_rest_layers import repair_rig_layers


def run(current,reference,output,rig_name):
    raise RuntimeError("Geometry-editing rest repair is disabled; use immutable rig fitting")
    if not bpy.app.background:raise RuntimeError('Run in an isolated background Blender process')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    for repo in getattr(bpy.context.preferences,'extensions',()).repos if hasattr(bpy.context.preferences,'extensions') else ():
        if repo.module=='user_default':repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
    bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm' if hasattr(bpy.context.preferences,'extensions') else 'VRM_Addon_for_Blender')
    bpy.ops.wm.open_mainfile(filepath=str(Path(reference).resolve()))
    rig=bpy.data.objects[rig_name];reference=source_record(rig,[o for o in rig.children if o.type=='MESH'])
    bpy.ops.wm.open_mainfile(filepath=str(Path(current).resolve()))
    rig=bpy.data.objects[rig_name]
    bpy.context.window.scene=next(s for s in bpy.data.scenes if rig.name in s.objects)
    meshes=[o for o in rig.children if o.type=='MESH'];source=source_record(rig,meshes)
    for name,record in source['meshes'].items():
        if record['polygons']!=reference['meshes'][name]['polygons']:
            raise ValueError('Rest reference topology does not match the current fit')
    report=repair_rig_layers(rig,reference)
    (out/'source.json').write_text(json.dumps(source))
    (out/'report.json').write_text(json.dumps(report,indent=2))
    # Separate rest bundle; motion validation is supplied by the independent
    # BVT sweep before installation, never copied from an earlier fit report.
    write_bundle(rig,source,dict(rest_layers=report,repair_only=True),out/'rest-fit.json')
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'repaired.blend'))
    print('REST_LAYERS_REPAIRED',json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('current');p.add_argument('reference');p.add_argument('output');p.add_argument('--rig',required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);run(a.current,a.reference,a.output,a.rig)
