"""Two-stage MIA joint prediction, using ONLY triangle positions and indices.

No bpy, object names, materials, bones, vertex groups or humanoid metadata.
The first pass estimates orientation; the second attends to predicted hands.
Normalization and hand-region clipping follow the official demo pipeline.
"""
import gc
import time
import numpy as np
import torch
import trimesh
from .model import PCAE
from .skeleton import NAMES, tree


def validate_geometry(vertices, faces):
    v=np.asarray(vertices,dtype=np.float64);f=np.asarray(faces)
    if v.ndim!=2 or v.shape[1]!=3 or len(v)<100 or not np.isfinite(v).all():
        raise ValueError('Select a complete character mesh with finite coordinates')
    if f.ndim!=2 or f.shape[1]!=3 or not np.issubdtype(f.dtype,np.integer) or not len(f):
        raise ValueError('Character geometry must contain triangles')
    if f.min()<0 or f.max()>=len(v) or np.ptp(v,axis=0).max()<1e-8:
        raise ValueError('Invalid character geometry')
    if len(v)>5000000 or len(f)>10000000:
        raise ValueError('Character exceeds the automatic rigging input limit')
    return v,f.astype(np.int64,copy=False)


def predict(vertices, faces, weights, device='cpu', progress=print):
    from pathlib import Path
    started=time.monotonic();torch.set_num_threads(4);torch.manual_seed(42)
    if device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Metal is not available; select CPU')
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available; select CPU')
    v,f=validate_geometry(vertices,faces)
    # Canonicalize before sampling to avoid scale-dependent triangle-area and
    # floating-point cancellation errors. This frame comes from geometry only.
    input_center=(v.min(0)+v.max(0))*.5
    input_scale=float(np.ptp(v,axis=0).max())
    v=(v-input_center)/input_scale
    # Fixed Blender convention, not any source bone's orientation.
    v=v[:,[0,2,1]].copy();v[:,2]*=-1
    mesh=trimesh.Trimesh(v,f,process=False)
    pts=trimesh.sample.sample_surface(mesh,32768,seed=42)[0].astype(np.float32)
    center=(pts.max(0)+pts.min(0))/2;scale=float(2/np.ptp(pts,axis=0).max())
    points=(pts-center)*scale
    def forward(path,points,refine=False):
        model=PCAE(N=32768,predict_bw=False,predict_joints=True,predict_joints_tail=True,
            output_dim=52,kinematic_tree=tree() if refine else None,
            hierarchical_ratio=.5 if refine else 0.,joints_attn_causal=refine).eval()
        model.load(str(Path(weights)/path));model=model.to(device)
        with torch.inference_mode():
            result=model(torch.from_numpy(points.astype(np.float32))[None].to(device)).joints.cpu().numpy()[0].reshape(52,2,3)
        del model;gc.collect()
        if device=='mps':torch.mps.empty_cache()
        if device=='cuda':torch.cuda.empty_cache()
        return result
    progress('Locating the body and hands')
    joints=forward('joints_coarse.pth',points)
    hip=joints[0,0];left=joints[NAMES.index('mixamorig:LeftUpLeg'),0];right=joints[NAMES.index('mixamorig:RightUpLeg'),0]
    x=left-right;z=np.cross(right-hip,left-hip)
    if min(np.linalg.norm(x),np.linalg.norm(z))<1e-7:
        raise ValueError('Could not identify the body orientation; use a neutral standing pose')
    x/=np.linalg.norm(x);z/=np.linalg.norm(z);y=np.cross(z,x);rotation=np.stack([x,y,z])
    aligned=np.einsum('ni,ji->nj',(v-center)*scale-hip,rotation)
    aligned_mesh=trimesh.Trimesh(aligned,f,process=False)
    parts=[trimesh.sample.sample_surface(aligned_mesh,16384,seed=43)[0]]
    radius=.15*(aligned.max()-aligned.min())
    for name in ('LeftHand','RightHand'):
        c=np.einsum('i,ji->j',joints[NAMES.index('mixamorig:'+name),1]-hip,rotation)
        box=trimesh.creation.box(extents=(radius,)*3);box.vertices+=c
        cut=aligned_mesh.slice_plane(box.facets_origin,-box.facets_normal)
        if len(cut.faces):parts.append(trimesh.sample.sample_surface(cut,8192,seed=44)[0])
    points=np.concatenate(parts);remaining=32768-len(points)
    if remaining:points=np.concatenate([points,trimesh.sample.sample_surface(aligned_mesh,remaining,seed=45)[0]])
    scale2=1/np.abs(points).max()
    progress('Refining the humanoid joints')
    joints=forward('joints.pth',points*scale2,True)
    joints=(np.einsum('nki,ij->nkj',joints/scale2,rotation)+hip)/scale+center
    joints=joints[:,:,[0,2,1]];joints[:,:,1]*=-1
    joints=joints*input_scale+input_center
    if not np.isfinite(joints).all():raise ValueError('Joint prediction failed')
    return dict(schema=1,device=device,seconds=time.monotonic()-started,
        source='mesh_geometry_only',joints={n.split(':')[-1]:dict(head=a.tolist(),tail=b.tolist()) for n,(a,b) in zip(NAMES,joints)})
