"""Bounded ARP full-context eye/tongue/teeth planner and mixed-bind adapter.

Auxiliary eye owners and component regions; arbitrary surface cuts remain open.
"""
import numpy as np
from .teeth_plan import prepare,skin_explicit_part
from .teeth_bind import components,skin_arch
from .face_geometry import classify
from .face_part_skin import skin
from .tongue_selection import select,surface_guard

def plan_parts(runtime,meshes,n,h,t,p,keep,*,skin_vertices=None,whole_meshes=None,deform=None,rig_context=None):
 bounded=rig_context is None
 d=np.ones(len(n),bool) if deform is None else np.asarray(deform,bool)
 plan=prepare(runtime,n,p,d,h,t,keep,full_context=True,rig_context=rig_context)
 if not plan['shapes'] and not plan['part_bones'] and not any(plan['auxiliary'].values()):
  return dict(parts=[],regions=[],part_bones=[],prepared=plan,tongue_selection=None)
 tongue=select(runtime,meshes if whole_meshes is None else whole_meshes,plan,bounded=bounded) if meshes else None;parts=[];regions=[]
 body_parts={i for i,label in rig_context['parts'] if label=='body'} if rig_context is not None else set()
 def write(label,v,c):
  if not bounded and not len(v):return None
  if label=='teeth':return skin_arch(plan,v,c,bounded=bounded)
  if label in ('teeth_top','teeth_bot'):return skin_explicit_part(plan,label,len(v),bounded=bounded)
  if label not in ('eyes','eye_caps','eye_cards','tongue'):raise ValueError('unknown face part')
  s=plan['auxiliary'].get(label) if label in plan['auxiliary'] else next(s for s in plan['shapes'] if s['label']==label)
  if s is None:raise ValueError('auxiliary eye shape missing')
  hh=np.array(s['heads'],'f4').reshape(-1,3);tt=np.array(s['tails'],'f4').reshape(-1,3)
  return skin(v,s['names'],hh,tt,rigid=label!='tongue',bounded=bounded)
 for idx,(_,v,f,e) in enumerate(meshes):
  if bounded and not 0<len(v)<=4096:raise ValueError('part mesh outside research bounds')
  c=components(len(v),e,bounded=bounded)
  sv=v if skin_vertices is None else skin_vertices[idx]
  wv=v if whole_meshes is None else whole_meshes[idx][1]
  if tongue is not None and tongue['mesh']==idx:
   ii=tongue['indices']
   if len(ii)==len(wv):
    parts.append((idx,'tongue',0.,write('tongue',sv,c)));continue
  hit=None if idx in body_parts else classify(wv,plan['shapes'],eye_radii=plan['eye_radii'],bounded=bounded)
  if hit is not None and hit[0]=='tongue' and surface_guard(runtime,wv,f,np.arange(len(wv))):hit=classify(wv,plan['shapes'],eye_radii=plan['eye_radii'],exclude='tongue',bounded=bounded)
  if hit is not None:
   label,score=hit;parts.append((idx,label,float(score),write(label,sv,c)));continue
  if not len(v):continue
  chosen_tongue=tongue['indices'].astype('i8') if tongue is not None and tongue['mesh']==idx else np.empty(0,'i8')
  groups=[np.flatnonzero(c==j).astype('i8') for j in range(int(c.max())+1)]
  largest=max(range(len(groups)),key=lambda j:(len(groups[j]),j));counts=[];selected={};residual_tongue=[]
  for j,ii in enumerate(groups):
   if j==largest or len(ii)<12 or (bounded and np.intersect1d(ii,chosen_tongue).size):continue
   hit=classify(v[ii],plan['shapes'],eye_radii=plan['eye_radii'],bounded=bounded)
   if hit is not None and hit[0]=='tongue' and surface_guard(runtime,v,f,ii):hit=classify(v[ii],plan['shapes'],eye_radii=plan['eye_radii'],exclude='tongue',bounded=bounded)
   if hit is None:continue
   if hit[0]=='tongue' and tongue is not None:
    if not bounded:residual_tongue.extend(ii.tolist())
    continue
   label,score=hit;counts.append((label,len(ii),float(score)));selected.setdefault(label,[]).extend(ii.tolist())
  if len(chosen_tongue):
   counts.append(('tongue',len(chosen_tongue),0.));selected.setdefault('tongue',[]).extend(chosen_tongue.tolist())
  if residual_tongue:
   # 0x120894: after the ray-selected tongue owns its vertices, reconsider
   # remaining tongue-classified regions together, excluding tongue itself.
   ii=np.array(sorted(set(residual_tongue)-set(chosen_tongue.tolist())),'i8')
   hit=classify(v[ii],plan['shapes'],eye_radii=plan['eye_radii'],exclude='tongue',bounded=False) if len(ii) else None
   if hit is not None:
    label,_=hit;counts.append((label,len(ii),0.));selected.setdefault(label,[]).extend(ii.tolist())
  if not selected:continue
  skins=[]
  for label,indices in selected.items():
   ii=np.array(indices,'i8');sn,sw,si,description=write(label,sv[ii],c[ii]);replace=label.startswith('teeth')
   if label=='teeth':
    total=len(ii);upper=int(np.count_nonzero(si==0))
    sn=[plan['head'],plan['jaw']]
    description=f'{total} rigid on {sn[0]}' if upper==total else (f'{total} rigid on {sn[1]}' if upper==0 else f'{upper} rigid on {sn[0]}, {total-upper} rigid on {sn[1]}')
   elif not replace:description='blended over '+', '.join(sn)
   skins.append((label,ii,(sn,sw,si,description),replace))
  touched=np.array(sorted({i for indices in selected.values() for i in indices}),'i8');regions.append((idx,touched,counts,skins))
 # The original only reserves generic eye bones if an eye-family surface was
 # actually accepted. Connected eye geometry otherwise stays in the main solve.
 labels={part[1] for part in parts}|{skin[0] for region in regions for skin in region[3]}
 reserved=plan['part_bones']
 if not labels.intersection(('eyes','eye_caps','eye_cards')):
  roles,_=runtime.detect_face(n,p.tolist(),d.tolist(),-1);eye_names={n[i] for i,role,_ in roles if role=='eye'}
  reserved=[name for name in reserved if name not in eye_names]
 return dict(parts=parts,regions=regions,part_bones=reserved,prepared=plan,tongue_selection=tongue)
