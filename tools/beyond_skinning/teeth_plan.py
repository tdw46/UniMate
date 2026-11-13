"""Independent bounded teeth-plan preparation. Geometry classification remains open.

Recovered from 0x115ff0: reserve semantic teeth bones, prepare role segments,
resolve deforming head/jaw owners and honor the explicit keep selection. No
original solver, capture records, or inferred mesh-name classification is used.
"""
import numpy as np
from .rig_mapping import validate,normalize

def prepare(runtime,names,parents,deform,heads,tails,keep=None,*,full_context=False,rig_context=None):
 bounded=rig_context is None
 if bounded:_,p,d=validate(names,parents,deform)
 else:p=list(map(int,parents));d=list(map(bool,deform))
 h=np.asarray(heads,'f4');t=np.asarray(tails,'f4')
 if h.shape!=(len(names),3) or t.shape!=h.shape or (bounded and (len(set(names))!=len(names) or len(names)>256 or not np.isfinite(h).all() or not np.isfinite(t).all())):raise ValueError('invalid teeth skeleton')
 if full_context and not np.any(d):
  return dict(split_hint=None,shapes=[],upper=None,lower=None,head=None,jaw=None,part_bones=[],complete_part_planner=False,eye_radii=None,auxiliary={'eye_caps':None,'eye_cards':None})
 roles,family=runtime.detect_face(list(names),list(p),list(d),-1) if bounded else ([],"")
 if rig_context is not None:
  family=rig_context['face_family'];roles=[]
  for _,kind,role,side,_,ids,status,_ in rig_context['elements']:
   if kind not in rig_context['kinds']['face'] or status=='absent':continue
   if kind.startswith('eyelid_corner'):role='eyelid_corner'
   roles.extend((i,role,side) for i in ids)
  roles.sort(key=lambda r:r[0])
 if not roles:return dict(split_hint=None,shapes=[],upper=None,lower=None,head=None,jaw=None,part_bones=[],complete_part_planner=False,eye_radii=None,auxiliary={'eye_caps':None,'eye_cards':None})
 if family!='arp':raise ValueError('other part-planner rig families remain unqualified')
 if any(r in (('eye_card','eye_cap') if full_context else ('tongue','eye_card','eye_cap')) for _,r,_ in roles):raise ValueError('remaining part shape families are not qualified')
 candidates=[i for i,name in enumerate(names) if normalize(name)=='j_bip_c_head']
 if rig_context is not None:
  candidates=[i for _,kind,_,_,_,ids,status,_ in rig_context['elements'] if kind=='head' and status!='absent' for i in ids]
 elif len(candidates)!=1:raise ValueError('part-planner head inference remains unqualified')
 head=candidates[0] if candidates else -1
 if full_context and head>=0 and not d[head]:
  # The named VRM torso fallback chooses the first deforming head/skull
  # candidate in skeleton order. That candidate becomes the main head role.
  fallback=[i for i,name in enumerate(names) if d[i] and ('head' in normalize(name) or 'skull' in normalize(name)) and normalize(name) not in ('headtop','headend')]
  if not fallback:raise ValueError('unnamed nondeform head fallback remains unqualified')
  head=fallback[0]
 while head>=0 and not d[head]:head=p[head]
 allowed=None if keep is None else set(keep)
 eligible=lambda i:i>=0 and d[i] and (allowed is None or names[i] in allowed)
 if not eligible(head):head=-1
 jaws=[i for i,r,_ in roles if r=='jaw' and eligible(i)]
 if len(jaws)>1 and not full_context:raise ValueError('multiple part-planner jaws remain unqualified')
 jaw=jaws[0] if jaws else -1
 if full_context and head>=0 and jaw>=0:
  from .teeth_geometry import segment_distance
  skulls=[i for i,r,_ in roles if r=='skull' and i!=head and eligible(i)]
  if skulls:head=min(skulls,key=lambda i:float(segment_distance(h[jaw],h[i],t[i])))
 groups={role:sorted(i for i,r,_ in roles if r==role and eligible(i)) for role in ('teeth_top','teeth_bot')}
 if head<0:groups['teeth_top']=[]
 if jaw<0:groups['teeth_bot']=[]
 eyes=[i for side in ('l','r','x') for i,r,s in roles if r=='eye' and s==side and eligible(i)]
 cards=[]
 if full_context:
  eyes=[]
  # The full role map assigns one main and one reflection-card owner per side.
  # Select before the plan's keep filter; a removed main is not replaced by an
  # auxiliary eye. Stable name-length ties retain the first skeleton entry.
  for side in ('l','r','x'):
   ids=[i for i,r,s in roles if r=='eye' and s==side]
   if not ids:continue
   shortest=min(len(names[i].encode('utf8')) for i in ids)
   main=[i for i in ids if len(names[i].encode('utf8'))==shortest]
   reflection=[i for i in ids if 'reflect' in normalize(names[i])]
   # Stable shortest-name selection: original skeleton order breaks equal lengths.
   card=min(reflection,key=lambda i:len(names[i].encode('utf8'))) if reflection else main[0]
   if rig_context is not None:
    mapped=lambda kind:[i for _,k,_,sd,_,ids,status,_ in rig_context['elements'] if k==kind and sd==side and status!='absent' for i in ids]
    main=mapped('eye_main') or main;card=(mapped('eye_card') or [card])[0]
   cards.append(card)
   if eligible(main[0]):eyes.append(main[0])
 def shape(label,ids):return dict(label=label,names=[names[i] for i in ids],heads=h[ids].reshape(-1).tolist(),tails=t[ids].reshape(-1).tolist())
 shapes=[shape(role,ids) for role,ids in groups.items() if ids]
 if full_context:
  tongues=sorted(i for i,r,_ in roles if r=='tongue' and eligible(i))
  if tongues:shapes.append(shape('tongue',tongues))
 if eyes:shapes.append(shape('eyes',eyes))
 upper=lower=None
 if all(groups.values()):
  shapes=[s for s in shapes if s['label'] not in groups]+[shape('teeth',groups['teeth_top']+groups['teeth_bot'])]
  upper=h[groups['teeth_top']].reshape(-1).tolist();lower=h[groups['teeth_bot']].reshape(-1).tolist()
 hint=None
 centers=[]
 for role in ('lip_top','lip_bot'):
  ids=[i for i,r,_ in roles if r==role]
  if not ids:break
  center=np.zeros(3,'f4')
  for i in ids:center+=((h[i]+t[i])*np.float32(.5))/np.float32(len(ids))
  centers.append(center)
 if len(centers)==2:hint=((centers[0]+centers[1])*np.float32(.5)).tolist()
 result=dict(split_hint=hint,shapes=shapes,upper=upper,lower=lower,head=names[head] if head>=0 else None,jaw=names[jaw] if jaw>=0 else None,
             part_bones=sorted(set(names[i] for i,r,_ in roles if r in (('teeth_top','teeth_bot','eye','tongue') if full_context else ('teeth_top','teeth_bot')) and eligible(i))),
             complete_part_planner=False)
 if full_context:
  from .face_geometry import norm
  radii=[]
  lids=[i for i,r,_ in roles if r in ('eyelid_top','eyelid_bot','eyelid_corner')]
  for eye in eyes:
   distances=[]
   for lid in lids:
    distance=norm(t[lid]-h[eye])
    if all(other==eye or distance<norm(t[lid]-h[other]) for other in eyes):distances.append(distance)
   if not distances:radii=[];break
   distances.sort();mid=len(distances)//2
   radii.append(float(distances[mid] if len(distances)%2 else np.float32(np.float32(distances[mid-1]+distances[mid])*np.float32(.5))))
  result['eye_radii']=radii or None
  result['auxiliary']={label:shape(label,ids) if radii else None for label,ids in (('eye_caps',eyes),('eye_cards',cards))}
 return result

def rigid_skin(vertex_count,owner,*,bounded=True):
 if not isinstance(vertex_count,int) or vertex_count<0 or (bounded and vertex_count>4096) or not owner:raise ValueError('invalid rigid part')
 return ([owner],np.ones((vertex_count,1),'f4'),np.zeros((vertex_count,1),'i4'),f'rigid on {owner}')

def skin_explicit_part(plan,label,vertex_count,*,bounded=True):
 """Only an already-classified upper/lower part. Never classify from mesh names."""
 if label not in ('teeth_top','teeth_bot'):raise ValueError('mixed arch routing requires separate qualification')
 if not any(s['label']==label for s in plan['shapes']):raise ValueError('part not admitted by the plan')
 return rigid_skin(vertex_count,plan['head' if label=='teeth_top' else 'jaw'],bounded=bounded)

def compact_arch_mask(vertices,upper,lower,components):
 """Recovered compact single-component branch of 0x1125b0.

Reject wide/multiple components rather than approximate its adaptive split.
Negative component labels remain excluded. Geometry classification is separate.
 """
 v=np.asarray(vertices,'f4');u=np.asarray(upper,'f4');l=np.asarray(lower,'f4');c=np.asarray(components,'i4')
 if v.ndim!=2 or v.shape[1]!=3 or len(v)>4096 or c.shape!=(len(v),) or any(a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all() for a in (v,u,l)) or not len(u) or not len(l):raise ValueError('invalid compact arch inputs')
 if len(set(c[c>=0].tolist()))>1:raise ValueError('multiple-component arch splitting remains unqualified')
 def center(a):
  result=np.zeros(3,'f8')
  for row in a:result+=row.astype('f8')/float(len(a))
  return result
 cu=center(u);cl=center(l);delta=cu-cl;length=float(np.sqrt((delta[0]*delta[0]+delta[1]*delta[1])+delta[2]*delta[2]))
 result=np.zeros(len(v),'u1')
 if length<1e-9:return result
 axis=delta/length
 dot=lambda a:(a[0]*axis[0]+a[1]*axis[1])+a[2]*axis[2]
 values=np.array([dot(row.astype('f8')) for row in v]);valid=c>=0
 if len(v) and float(np.ptp(values))>=length:raise ValueError('wide arch splitting remains unqualified')
 if np.any(valid):
  mean=sum(float(x) for x in values[valid])/int(np.count_nonzero(valid))
  result[valid]=abs(mean-dot(cu))<abs(mean-dot(cl))
 return result

def skin_compact_arch(plan,vertices,components):
 if plan['upper'] is None or plan['lower'] is None or not plan['head'] or not plan['jaw']:raise ValueError('combined teeth plan required')
 mask=compact_arch_mask(vertices,np.asarray(plan['upper'],'f4').reshape(-1,3),np.asarray(plan['lower'],'f4').reshape(-1,3),components)
 count=len(mask);num_upper=int(mask.sum())
 if num_upper==count:return rigid_skin(count,plan['head'])
 return ([plan['head'],plan['jaw']],np.ones((count,1),'f4'),(1-mask.astype('i4')).reshape(-1,1),f"rigid, upper arch ({num_upper} verts) on {plan['head']}, lower arch ({count-num_upper}) on {plan['jaw']}")
