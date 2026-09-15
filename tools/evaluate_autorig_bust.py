"""Validate the auto-rigged bust through the UniMate reconstruction path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bpy
import numpy as np
from scipy.spatial import cKDTree
from data_process.motion_export.export_general import export_asset
from data_process.mesh_animation.animate_npz import animate_character

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT/'outputs/stitched_autorig')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []);OUT=args.root.resolve()
SAMPLE_FRAMES = (0,26,53,106,133,180,200,220,239)


def snapshot(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 30
    bpy.ops.import_scene.gltf(filepath=str(path))
    rig = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
    names = sorted(rig.pose.bones.keys())
    rest = np.array([rig.matrix_world @ rig.data.bones[n].matrix_local for n in names],dtype=np.float64)
    transforms, vertices = [], []
    meshes = [o for o in bpy.context.scene.objects if o.type=='MESH' and any(m.type=='ARMATURE' for m in o.modifiers)]
    for frame in range(240):
        bpy.context.scene.frame_set(frame)
        transforms.append(np.array([rig.matrix_world @ rig.pose.bones[n].matrix for n in names],dtype=np.float64))
        if frame in SAMPLE_FRAMES:
            deps = bpy.context.evaluated_depsgraph_get()
            cloud = []
            for obj in meshes:
                evaluated = obj.evaluated_get(deps)
                mesh = evaluated.to_mesh()
                cloud.extend(tuple(evaluated.matrix_world @ v.co) for v in mesh.vertices)
                evaluated.to_mesh_clear()
            vertices.append(np.array(cloud))
    return names, rest, np.array(transforms), vertices


source=OUT/'rigged.glb'
export_dir=OUT/'export'/hashlib.sha256(source.read_bytes()).hexdigest()[:12]/OUT.name
export_asset(str(source),str(export_dir),save_name=OUT.name,prune=False,remove_tpose=False,save_vis=False)
motions=list((export_dir/'motions').glob('*.npz'));assert len(motions)==1
n0,r0,a,va=snapshot(source)
animate_character(str(source),str(motions[0]),str(OUT/'reconstructed'),extra_bones_strategy='keep')
files=list((OUT/'reconstructed').glob(motions[0].stem+'*.glb'));assert len(files)==1,files
n1,r1,b,vb=snapshot(files[0]);assert n0==n1
u,_,vt=np.linalg.svd((b[...,:3,:3]@np.linalg.inv(r1[...,:3,:3]))@np.linalg.inv(a[...,:3,:3]@np.linalg.inv(r0[...,:3,:3])))
angle=np.degrees(np.arccos(np.clip((np.trace(u@vt,axis1=-2,axis2=-1)-1)/2,-1,1)))
position=np.linalg.norm(a[...,:3,3]-b[...,:3,3],axis=-1)
surface=max(max(cKDTree(x).query(y)[0].max(),cKDTree(y).query(x)[0].max()) for x,y in zip(va,vb))
report={'frames':240,'bones':len(n0),'max_rotation_error_degrees':float(angle.max()),'max_joint_position_error':float(position.max()),'max_surface_error':float(surface),'sample_frames':SAMPLE_FRAMES,'motion_source':'Authored diagnostic motion; no pretrained inference','passed':bool(angle.max()<.1 and position.max()<1e-4 and surface<1e-4)}
(OUT/'pipeline_validation.json').write_text(json.dumps(report,indent=2));print('BUST_PIPELINE',json.dumps(report),flush=True);assert report['passed']
