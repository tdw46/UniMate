"""Bounded face-part planning and mixed mouth/skull/body final writeback.

The legacy route admits explicit teeth selection; optional region/full-context
planners admit their separately qualified subsets. Region writes preserve per-group accumulation.
All input geometry participates in diffusion, including rigid teeth meshes.
"""
import numpy as np
from .teeth_plan import prepare,rigid_skin,skin_explicit_part
from .teeth_geometry import classify
from .teeth_split import arch_mask
from .mouth_bind import bind as mouth_bind

def components(n,edges,*,bounded=True):
 p=list(range(n))
 def root(i):
  while i!=p[i]:p[i]=p[p[i]];i=p[i]
  return i
 for a,b in np.asarray(edges,'i8'):
  if not (0<=a<n and 0<=b<n):
   if bounded:raise ValueError('invalid part edge')
   continue
  a=root(int(a));b=root(int(b));p[max(a,b)]=min(a,b)
 labels={};result=[]
 for i in range(n):
  r=root(i)
  if r not in labels:labels[r]=len(labels)
  result.append(labels[r])
 return np.array(result,'i4')

def skin_arch(plan,v,c,*,bounded=True):
 if plan['upper'] is None or plan['lower'] is None or not plan['head'] or not plan['jaw']:raise ValueError('combined teeth owners missing')
 mask=arch_mask(v,np.array(plan['upper'],'f4').reshape(-1,3),np.array(plan['lower'],'f4').reshape(-1,3),c,plan['split_hint'],bounded=bounded)
 total=len(mask);upper=int(mask.sum())
 if upper==total:return rigid_skin(total,plan['head'],bounded=bounded)
 return ([plan['head'],plan['jaw']],np.ones((total,1),'f4'),(1-mask.astype('i4')).reshape(-1,1),f"rigid, upper arch ({upper} verts) on {plan['head']}, lower arch ({total-upper}) on {plan['jaw']}")

def plan_parts(runtime,meshes,n,h,t,p,keep):
 plan=prepare(runtime,n,p,np.ones(len(n),bool),h,t,keep)
 if any(s['label'] not in ('teeth','teeth_top','teeth_bot') for s in plan['shapes']):raise ValueError('other part classifiers remain unqualified')
 parts=[]
 for idx,(_,v,f,e) in enumerate(meshes):
  if len(v)==0 or len(v)>4096:raise ValueError('part mesh size outside research bounds')
  c=components(len(v),e);hit=classify(v,plan['shapes'])
  if hit is None:
   if len(set(c.tolist()))>1 and any(classify(v[c==label],plan['shapes']) for label in set(c.tolist())):raise ValueError('connected region routing remains unqualified')
   continue
  label,score=hit
  skin=skin_arch(plan,v,c) if label=='teeth' else skin_explicit_part(plan,label,len(v))
  parts.append((idx,label,float(score),skin))
 return dict(parts=parts,regions=[],part_bones=plan['part_bones'],prepared=plan)

def bind(runtime,meshes,n,h,t,p,keep,*,max_depth=0,region_planner=False,full_face_planner=False,finger_isolate=False,finger_isolate_thresh=.15,finger_isolate_smooth=2,face_smooth=0,face_split=False,face_weight_smooth=0.,face_weight_smooth_iters=0,weight_hash_seed=None,weight_random_states=None):
 planner=plan_parts
 if region_planner:
  from .teeth_regions import plan_parts as planner
 if full_face_planner:
  from .full_face_parts import plan_parts as planner
 plan=planner(runtime,meshes,n,h,t,p,keep)
 ids=[i for i,name in enumerate(n) if name not in plan['part_bones']];lookup={old:new for new,old in enumerate(ids)}
 if any(p[i]>=0 and p[i] not in lookup for i in ids):raise ValueError('retained children of reserved part bones remain unqualified')
 pp=np.array([-1 if p[i]<0 else lookup[p[i]] for i in ids],'i8')
 vv=[];ff=[];ee=[];offsets=[0]
 for _,v,f,e in meshes:vv.append(v);ff.append(f+offsets[-1]);ee.append(e+offsets[-1]);offsets.append(offsets[-1]+len(v))
 body=mouth_bind(runtime,np.concatenate(vv),np.concatenate(ff),np.concatenate(ee),[n[i] for i in ids],h[ids],t[ids],pp,max_depth=max_depth,full_rig_context=(n,h,t,p) if full_face_planner else None,finger_isolate=finger_isolate,finger_isolate_thresh=finger_isolate_thresh,finger_isolate_smooth=finger_isolate_smooth,face_smooth=face_smooth,face_split=face_split,face_weight_smooth=face_weight_smooth,face_weight_smooth_iters=face_weight_smooth_iters,weight_hash_seed=weight_hash_seed,weight_random_states=weight_random_states)
 bones=body['bones'].copy();valid=bones>=0;bones[valid]=np.array(ids,'i4')[bones[valid]];weights=body['weights'].copy()
 for meshidx,label,score,skin in plan['parts']:
  sn,sw,si,_=skin;sl=slice(offsets[meshidx],offsets[meshidx+1]);weights[sl]=0;bones[sl]=-1
  weights[sl,:sw.shape[1]]=sw;bones[sl,:si.shape[1]]=np.array([n.index(name) for name in sn],'i4')[si]
 from .region_writeback import apply_regions
 weights,bones=apply_regions(weights,bones,n,offsets,plan['regions'])
 return dict(weights=weights,bones=bones,plan=plan,body=body,complete_native_bind=False)
