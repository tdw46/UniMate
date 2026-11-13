"""Recovered bounded disconnected-component teeth region planning and writeback."""
import numpy as np
from .teeth_bind import components,skin_arch
from .teeth_geometry import classify
from .teeth_plan import prepare,skin_explicit_part

def plan_parts(runtime,meshes,n,h,t,p,keep):
 plan=prepare(runtime,n,p,np.ones(len(n),bool),h,t,keep)
 if any(s['label'] not in ('teeth','teeth_top','teeth_bot') for s in plan['shapes']):raise ValueError('non-teeth region routing remains unqualified')
 parts=[];regions=[]
 for meshidx,(_,v,f,e) in enumerate(meshes):
  if len(v)==0 or len(v)>4096:raise ValueError('part mesh size outside research bounds')
  c=components(len(v),e);hit=classify(v,plan['shapes'])
  if hit:
   label,score=hit;skin=skin_arch(plan,v,c) if label=='teeth' else skin_explicit_part(plan,label,len(v));parts.append((meshidx,label,float(score),skin));continue
  groups=[np.flatnonzero(c==i).astype('i8') for i in range(int(c.max())+1)]
  # A later equally large component wins, as in the native scan.
  largest=max(range(len(groups)),key=lambda i:(len(groups[i]),i));counts=[];selected={}
  for i,ids in enumerate(groups):
   if i==largest or len(ids)<12:continue
   hit=classify(v[ids],plan['shapes'])
   if hit is None:continue
   label,score=hit;counts.append((label,len(ids),float(score)));selected.setdefault(label,[]).extend(ids.tolist())
  if not selected:continue
  skins=[]
  for label,indices in selected.items():
   ids=np.array(indices,'i8')
   if label=='teeth':
    skin=skin_arch(plan,v[ids],c[ids]);sn,sw,si,description=skin
    total=len(ids);upper=int(np.count_nonzero(si==0))
    if len(sn)==1:description=f'{total} rigid on {sn[0]}'
    else:description=f'{upper} rigid on {sn[0]}, {total-upper} rigid on {sn[1]}'
    skin=(sn,sw,si,description)
   else:skin=skin_explicit_part(plan,label,len(ids))
   skins.append((label,ids,skin,True))
  touched=np.array(sorted({i for values in selected.values() for i in values}),'i8');regions.append((meshidx,touched,counts,skins))
 return dict(parts=parts,regions=regions,part_bones=plan['part_bones'],prepared=plan)
