"""Public spine mapper recovered from 0x15f09c and 0x1610b4.

Private bounded bind helpers retain their earlier interface until integrated.
"""
import re,math
import numpy as np
if __package__:
 from .rig_mapping import TORSO_FAMILIES,_FINGER_DECIMAL_CLASS
else:
 from .rig_mapping import TORSO_FAMILIES,_FINGER_DECIMAL_CLASS

def pat(s):return re.compile(s.replace(r'\d',_FINGER_DECIMAL_CLASS))
SHOULDER=pat(r'.*(shoulder|clavicle|collar).*')
ARM=pat(r'.*(^|[._ -]|left|right)(upper[._ -]?)?arm(\d|[._ -]|$).*')
LEG=pat(r'.*(thigh|upleg|upper[._ -]?leg|up[._ -]?leg|femur).*')
NECK=pat(r'.*(neck|collo|kubi|首).*');HEAD=pat(r'.*(head|skull|頭).*')
HEAD_EXCLUDE=('headtop','head_top','headend','head_end','forehead','headgear')

def detect_spine_owned(names,nn,parents,d,segments):
 class Checked(list):
  def __getitem__(self,i):
   if i<0 or i>=len(self):
    from . import _bvt_solver as runtime
    raise runtime.PanicException(f'index out of bounds: the len is {len(self)} but the index is {i}')
   return super().__getitem__(i)
 p=Checked(parents);s=None if segments is None else Checked(np.asarray(segments,'f8'))
 def ancestors(i):
  seen={i};at=p[i];out=[]
  while at>=0:
   v=p[at]
   if at in seen:break
   out.append(at);seen.add(at);at=v
  return out
 def matches(pattern,deforming=True):
  rx=pat(pattern);return [i for i,n in enumerate(nn) if (not deforming or d[i]) and rx.fullmatch(n)]
 def all_matches(patterns,deforming=True):
  return [i for i,n in enumerate(nn) if (not deforming or d[i]) and any(pat(x).fullmatch(n) for x in patterns)]
 def excluded(n):return bool(LEG.fullmatch(n) or (('forearm' not in n and 'armature' not in n) and (SHOULDER.fullmatch(n) or ARM.fullmatch(n))))
 def trim_torso(chain):
  for which in (0,1):
   armids=[i for i,n in enumerate(nn) if (bool(SHOULDER.fullmatch(n)) if which==0 else ('forearm' not in n and 'armature' not in n and (bool(ARM.fullmatch(n)) or 'humerus' in n)))]
   owned=set(j for i in armids for j in ancestors(i));cut=[j for j,i in enumerate(chain) if i in owned]
   if cut:return chain[:max(cut)+1]
  return chain
 def head_neck(last):
  def candidates(rx,exclusions=()):
   ids=[i for i,n in enumerate(nn) if d[i] and rx.fullmatch(n) and not any(x in n for x in exclusions) and not any(excluded(nn[j]) for j in ancestors(i))]
   return sorted(ids,key=lambda i:len(ancestors(i)))
  neck=[]
  for i in candidates(NECK):
   if not neck or neck[-1] in ancestors(i) or p[i]<0:neck.append(i)
  heads=candidates(HEAD,HEAD_EXCLUDE);head=heads[0] if heads else -1
  if head>=0 and neck:return neck,head
  children=Checked([[] for _ in p])
  for i,v in enumerate(p):
   if v>=0:children[v].append(i)
  def child(i):
   ids=[j for j in children[i] if d[j] and not excluded(nn[j])]
   return ids[0] if len(ids)==1 else -1
  a=child(last);b=child(a) if a>=0 else -1
  if a>=0 and b>=0:
   if not neck:neck=[a]
   if head<0:head=b
  return neck,head
 result=dict(family='',label='',pelvis=-1,spine=[],neck=[],head=-1,verified=False,reason='no known family signature')
 fam=next((f for f in TORSO_FAMILIES if all(matches(x,False) for x in f['required']) and not any(matches(x,False) for x in f['exclude'])),None)
 if fam is None:return result
 result.update(family=fam['key'],label=fam['label']);hips=all_matches(fam['hips']);pelvis=hips[0] if hips else -1
 if fam['key']=='english':
  spine=sorted(all_matches(fam['spine']),key=lambda i:len(ancestors(i)))
  if pelvis<0:
   legs=matches(fam['required'][0]);pelvis=next((i for i in spine if sum(p[j]==i for j in legs)>=2),-1)
  if pelvis<0:result['reason']='no pelvis bone and no torso bone parents two legs';return result
  result['pelvis']=pelvis;spine=[i for i in spine if i!=pelvis];chain=[]
  for i in spine:
   if not chain or chain[-1] in ancestors(i):chain.append(i)
  spine=trim_torso(chain);result['spine']=spine
  if not spine:result['reason']='torso bones do not form a chain';return result
  first=spine[0]
  if p[first]!=pelvis and p[pelvis]!=first and not(p[pelvis]>=0 and p[first]==p[pelvis]):result['reason']=f'hierarchy: "{names[pelvis]}" is not adjacent to "{names[first]}"';return result
  result['neck'],result['head']=head_neck(spine[-1]);result.update(verified=True,reason='');return result
 if fam['numeric']:
  found=[];rx=pat(fam['spine'][0])
  for i,n in enumerate(nn):
   m=rx.fullmatch(n)
   if not d[i] or not m:continue
   v=(m.group(1) or '0').lstrip('0') or '0';number=int(v) if v.isascii() and v.isdigit() and len(v)<20 else 0
   if number>2**63-1:number=0
   found.append((number,i))
  seen=set();spine=[]
  for number,i in sorted(found):
   if number not in seen:seen.add(number);spine.append(i)
  spine=trim_torso(spine)
 else:spine=[ids[0] for pattern in fam['spine'] if (ids:=matches(pattern))]
 if pelvis<0:result['reason']='signature matched but the pelvis bone is missing or does not deform';return result
 result['pelvis']=pelvis
 if not spine:
  enough=sum(v==pelvis for v in p)>=2;result.update(verified=enough,reason='no torso bones; pelvis only' if enough else 'no torso bones and the pelvis parents nothing');return result
 if fam['extra']:
  check=[]
  for pattern in fam['extra']:
   ids=matches(pattern,False)
   if not ids:break
   check.append(ids[0])
  if len(check)<2:result['spine']=spine;result['reason']='verify chain bones missing';return result
 else:check=[pelvis]+spine
 continuity=0
 tol=0.
 if s is not None and len(s):
  z=np.asarray(s)[:,:,2];lo=np.fmin.reduce(z.reshape(-1),initial=np.inf);hi=np.fmax.reduce(z.reshape(-1),initial=-np.inf);tol=max(float(hi-lo),1e-9)*.03
 for a,b in zip(check,check[1:]):
  if a in ancestors(b):continue
  valid=False
  if p[b]<0 and p[a]<0 and s is not None:
   delta=s[a][1]-s[b][0];dist=math.sqrt((delta[0]*delta[0]+delta[1]*delta[1])+delta[2]*delta[2]);valid=dist<=tol
  if not valid:result['spine']=spine;result['reason']=f'hierarchy: "{names[a]}" is not an ancestor of "{names[b]}"';return result
  continuity+=1
 result['spine']=spine;result['neck'],result['head']=head_neck(spine[-1]);result.update(verified=True,reason=f'{continuity} link(s) verified by continuity (flat rig)' if continuity else '');return result
