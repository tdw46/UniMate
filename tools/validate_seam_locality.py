"""Compare saved pre/post seam weights, mesh/UV data, and independent graph bounds."""
import argparse, json, sys
from collections import deque
from pathlib import Path
import bpy
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_apparel_weights import weights
from avatar_voxel_seams import skin_faces, seam_clusters


def snapshot(path):
 bpy.ops.wm.open_mainfile(filepath=str(path))
 return {o.name:{'weights':[weights(o,i) for i in range(len(o.data.vertices))],
                 'positions':[tuple(v.co) for v in o.data.vertices],
                 'faces':[tuple(p.vertices) for p in o.data.polygons],
                 'uvs':[[tuple(v.uv) for v in uv.data] for uv in o.data.uv_layers]}
         for o in bpy.context.scene.objects if o.type=='MESH'}


def validate(before,after):
 baseline=snapshot(before);final=snapshot(after);assert baseline.keys()==final.keys()
 rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
 parts=[(o,skin_faces(o)) for o in bpy.context.scene.objects if o.type=='MESH'];parts=[(o,f) for o,f in parts if f]
 explicit={o.get('binding_role') for o,_ in parts}>={'body','head'}
 neck=rig.data.bones['Neck'];tol=neck.length/160 if explicit else neck.length*1e-5
 clusters=seam_clusters(parts,tol)
 # Independent breadth-first expansion: weld-equivalent vertices become a
 # single node, then take precisely two mesh-edge steps from the seam.
 canonical={}
 for g in clusters:
  keys=[(o.name,i) for o,i,_ in g]
  for key in keys:canonical[key]=keys[0]
 node=lambda k:canonical.get(k,k)
 adjacency={}
 for o,faces in parts:
  for f in faces:
   for a,b in f.edge_keys:
    a,b=node((o.name,a)),node((o.name,b))
    adjacency.setdefault(a,set()).add(b);adjacency.setdefault(b,set()).add(a)
 seeds=set()
 radius=abs(rig.data.bones['UpperArm.L'].head_local.x-neck.head_local.x)*.85
 for g in clusters:
  roles={o.get('binding_role') for o,_,_ in g};center=sum((p for _,_,p in g),g[0][2]*0)/len(g)
  values=[baseline[o.name]['weights'][i] for o,i,_ in g];names=set().union(*(v.keys() for v in values))
  mismatch=any(max(v.get(n,0) for v in values)-min(v.get(n,0) for v in values)>1e-5 for n in names)
  is_seed=roles>={'head','body'} if explicit else (mismatch and neck.head_local.z<=center.z<=neck.tail_local.z+neck.length*.25 and abs(center.x-neck.head_local.x)<radius)
  if is_seed:seeds.add(node((g[0][0].name,g[0][1])))
 allowed=set(seeds);front=set(seeds)
 for _ in range(2):
  front={b for a in front for b in adjacency.get(a,())}-allowed;allowed|=front
 changed=[]
 for name,a in baseline.items():
  b=final[name]
  for field in ('positions','faces','uvs'):assert a[field]==b[field],(name,field)
  for i,(x,y) in enumerate(zip(a['weights'],b['weights'])):
   assert len(y)<=4 and abs(sum(y.values())-1)<1e-5
   if x!=y:
    assert node((name,i)) in allowed,(name,i,'outside two loops')
    changed.append((name,i))
 return {'id':after.parent.name,'vertices':sum(len(o['weights']) for o in final.values()),'changed_vertices':len(changed),
         'outside_two_loops_changed':0,'geometry_and_uvs_unchanged':True,'normalized_max_four_influences':True,'passed':True}


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--bust',action='store_true')
 a=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);root=Path(a.root).resolve()
 pairs=[(root/'heat_baseline.blend',root/'rigged.blend')] if a.bust else [(p/'02_heat_baseline.blend',p/'02_fresh_rig.blend') for p in sorted((root/'avatars').iterdir()) if p.is_dir()]
 report={'cases':[validate(x,y) for x,y in pairs],'passed':True}
 (root/'locality_validation.json').write_text(json.dumps(report,indent=2));print('LOCALITY_PASSED',json.dumps(report),flush=True)
