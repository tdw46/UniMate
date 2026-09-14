"""Read back the packed final gallery scenes without touching the live session."""
import json
import os
from pathlib import Path
import bpy

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('AVATAR_EVAL_ROOT', ROOT/'outputs/avatar_grid')).resolve()
report=[]
for view in ('front','oblique'):
    path=OUT/f'avatar_grid_{view}.blend'
    bpy.ops.wm.open_mainfile(filepath=str(path))
    scene=bpy.context.scene
    rigs=[o for o in scene.objects if o.type=='ARMATURE']
    images=[i for i in bpy.data.images if i.source=='FILE' and i.type=='IMAGE']
    case={'view':view,'file':path.name,'rigs':len(rigs),
          'bones':sum(len(r.data.bones) for r in rigs),
          'packed_images':sum(bool(i.packed_file) for i in images),
          'file_images':len(images),
          'unapplied_booleans':sum(m.type=='BOOLEAN' for o in scene.objects for m in o.modifiers),
          'frames':[scene.frame_start,scene.frame_end],'fps':scene.render.fps,
          'resolution':[scene.render.resolution_x,scene.render.resolution_y],
          'engine':scene.render.engine}
    assert case['rigs']==9 and case['bones']==117
    assert case['unapplied_booleans']==0
    assert case['packed_images']==case['file_images'] and images
    assert case['frames']==[0,359] and case['fps']==30
    assert case['resolution']==[1920,1920]
    assert all(r.animation_data and r.animation_data.action for r in rigs)
    case['passed']=True
    report.append(case)
(OUT/'scene_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
